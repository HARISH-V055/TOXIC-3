"""
GNN Inference Service — EQ-KA-GCN Real PyTorch Model Integration

Loads the trained PyTorch EQ-KA-GCN model and executes live toxicity predictions
and GNNExplainer attribution for SMILES inputs.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional

import torch

# Ensure EQ-KA-GCN directory is on python path for importing model components
BASE_DIR = Path(__file__).resolve().parents[3]
EQ_KA_GCN_DIR = BASE_DIR / "EQ-KA-GCN"
if str(EQ_KA_GCN_DIR) not in sys.path:
    sys.path.insert(0, str(EQ_KA_GCN_DIR))

from config import get_config
from graph.graph_builder import smiles_to_graph
from models.baseline_gcn import BaselineGCN
from models.ka_gcn import KAGCN
from quantization.adaptive_qat import AdaptiveQATManager
from explainability import (
    GNNExplainerModule,
    rank_atom_importance,
)

from app.core.config import get_settings
from app.models.schemas import (
    PredictRequest,
    PredictResponse,
    ExplainRequest,
    ExplainResponse,
    BondAttention,
    AtomAttention,
    BondAttentionDetail,
)

logger = logging.getLogger(__name__)
settings = get_settings()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPTIMAL_THRESHOLD = 0.75  # Optimal threshold calibrated for EQ-KA-GCN


class GNNService:
    """
    Singleton service class for EQ-KA-GCN inference.
    Executes live PyTorch model prediction and GNNExplainer attribution.
    """

    _instance: Optional["GNNService"] = None
    _model_loaded: bool = False
    model: Optional[torch.nn.Module] = None
    model_type: str = "None"

    def __new__(cls) -> "GNNService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize model loading on startup."""
        logger.info("Initializing GNN Service...")
        eq_config = get_config()

        # Checkpoints in priority order: QAT best -> Weighted KA-GCN best -> Baseline GCN best -> eq_ka_gcn_best.pt
        checkpoints = [
            (eq_config.paths.checkpoints_dir / eq_config.quantization.qat_save_filename, "Adaptive Quantized KA-GCN"),
            (eq_config.paths.checkpoints_dir / eq_config.fourier_kan.weighted_save_filename, "Weighted KA-GCN"),
            (eq_config.paths.checkpoints_dir / eq_config.model.save_filename, "Baseline GCN"),
            (eq_config.paths.checkpoints_dir / "ka_gcn_best.pt", "KA-GCN"),
        ]

        loaded_checkpoint = None
        loaded_name = None

        for ckpt_path, name in checkpoints:
            if ckpt_path.exists():
                loaded_checkpoint = ckpt_path
                loaded_name = name
                break

        if loaded_checkpoint is None:
            logger.warning(
                f"⚠️ No model checkpoint found in {eq_config.paths.checkpoints_dir}. Running in fallback mode."
            )
            self._model_loaded = False
            return

        try:
            logger.info(f"Loading checkpoint: {loaded_checkpoint.name} ({loaded_name})")

            if eq_config.fourier_kan.enabled:
                model = KAGCN(
                    input_dim=eq_config.model.input_dim,
                    hidden_dim=eq_config.model.hidden_dim,
                    output_dim=eq_config.model.output_dim,
                    gcn_dropout=eq_config.model.dropout,
                    kan_hidden_dim=eq_config.fourier_kan.hidden_dim,
                    fourier_order=eq_config.fourier_kan.fourier_order,
                    kan_dropout=eq_config.fourier_kan.dropout,
                    kan_activation=eq_config.fourier_kan.activation,
                )
                if "qat" in loaded_checkpoint.name:
                    qat_mgr = AdaptiveQATManager(
                        target_layers=["conv1", "conv2", "fourier_kan", "fc_out"],
                        supported_bits=eq_config.quantization.supported_bits,
                    )
                    bit_assignments = {"fourier_kan": 8, "conv2": 8, "conv1": 6, "fc_out": 4}
                    model = qat_mgr.prepare_qat_model(model, bit_assignments)
            else:
                model = BaselineGCN(
                    input_dim=eq_config.model.input_dim,
                    hidden_dim=eq_config.model.hidden_dim,
                    output_dim=eq_config.model.output_dim,
                    dropout=eq_config.model.dropout,
                )

            state_dict = torch.load(loaded_checkpoint, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()

            self.model = model
            self.model_type = loaded_name
            self._model_loaded = True
            logger.info(f"✅ GNN Model ({loaded_name}) successfully loaded onto {DEVICE}")
        except Exception as e:
            logger.error(f"❌ Failed to load GNN model: {str(e)}", exc_info=True)
            self._model_loaded = False

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Run toxicity prediction for a given SMILES string using trained model."""
        start_time = time.time()
        smiles = request.smiles

        if not self._model_loaded or self.model is None:
            raise ValueError("GNN model is not loaded. Cannot process prediction request.")

        # 1. Parse SMILES & Build Graph
        graph = smiles_to_graph(smiles=smiles, label=0)
        if graph is None:
            raise ValueError(f"Invalid SMILES string: '{smiles}'. Unable to construct molecular graph.")

        graph = graph.to(DEVICE)
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)

        # 2. PyTorch Model Forward Pass
        with torch.no_grad():
            logits = self.model(
                x=graph.x,
                edge_index=graph.edge_index,
                batch=batch,
                return_logits=True,
            )
            prob = torch.sigmoid(logits).item()

        is_toxic = prob >= OPTIMAL_THRESHOLD
        prediction_label = "toxic" if is_toxic else "non-toxic"
        confidence = prob if is_toxic else (1.0 - prob)

        # 3. GNNExplainer attribution
        important_atoms = []
        important_bonds = []

        try:
            explainer = GNNExplainerModule(epochs=40, lr=0.01)
            node_importance, edge_importance = explainer.explain_graph(
                model=self.model,
                graph=graph,
                device=DEVICE,
            )

            top_atoms, top_bonds = rank_atom_importance(
                graph=graph,
                node_importance=node_importance,
                edge_importance=edge_importance,
                top_k_atoms=5,
                top_k_bonds=5,
            )

            important_atoms = [atom["atom_index"] for atom in top_atoms]
            important_bonds = [
                BondAttention(
                    atom_a=bond["u"],
                    atom_b=bond["v"],
                    weight=round(float(bond["importance_score"]), 3),
                )
                for bond in top_bonds
            ]
        except Exception as e:
            logger.warning(f"GNNExplainer failed during predict fallback: {str(e)}")

        execution_time = (time.time() - start_time) * 1000

        return PredictResponse(
            prediction=prediction_label,
            probability=round(float(prob), 4),
            confidence=round(float(confidence), 4),
            important_atoms=important_atoms,
            important_bonds=important_bonds,
            execution_time=round(execution_time, 2),
        )

    def explain(self, request: ExplainRequest) -> ExplainResponse:
        """Generate GNN attention-based explanation for a SMILES prediction."""
        if not self._model_loaded or self.model is None:
            raise ValueError("GNN model is not loaded. Cannot process explain request.")

        graph = smiles_to_graph(smiles=request.smiles, label=0)
        if graph is None:
            raise ValueError(f"Invalid SMILES string: '{request.smiles}'. Unable to construct graph.")

        graph = graph.to(DEVICE)

        try:
            explainer = GNNExplainerModule(epochs=50, lr=0.01)
            node_importance, edge_importance = explainer.explain_graph(
                model=self.model,
                graph=graph,
                device=DEVICE,
            )

            atom_attentions = [
                AtomAttention(atom_index=i, weight=round(float(score), 4))
                for i, score in enumerate(node_importance)
            ]

            bond_attentions = []
            if graph.edge_index is not None and graph.edge_index.size(1) > 0:
                edges = graph.edge_index.cpu().numpy()
                for i in range(edges.shape[1]):
                    u, v = int(edges[0, i]), int(edges[1, i])
                    if u < v:  # Add unique undirected edge
                        weight = float(edge_importance[i]) if i < len(edge_importance) else 0.5
                        bond_attentions.append(
                            BondAttentionDetail(
                                bond_index=len(bond_attentions),
                                weight=round(weight, 4),
                                atoms=(u, v),
                            )
                        )

            return ExplainResponse(
                atom_attentions=atom_attentions,
                bond_attentions=bond_attentions,
                saliency_map=[],
            )
        except Exception as e:
            logger.error(f"Explanation extraction failed: {str(e)}")
            raise RuntimeError(f"GNNExplainer execution failed: {str(e)}")


# Singleton instance
gnn_service = GNNService()

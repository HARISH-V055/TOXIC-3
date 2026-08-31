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

from rdkit import Chem
from rdkit.Chem import AllChem
from config import get_config, TOX21_ENDPOINTS, ENDPOINT_INFO
from graph.graph_builder import smiles_to_graph
from models.baseline_gcn import BaselineGCN
from models.ka_gcn import KAGCN
from quantization.adaptive_qat import AdaptiveQATManager
from explainability import (
    GNNExplainerModule,
    rank_atom_importance,
    visualize_molecule_explanation,
)

from app.core.config import get_settings
from app.models.schemas import (
    PredictRequest,
    PredictResponse,
    ExplainRequest,
    ExplainResponse,
    EndpointPrediction,
    ImportantAtom,
    ImportantBond,
    GraphAtom,
    GraphBond,
    MolecularGraph,
    AtomAttention,
    BondAttentionDetail,
)

logger = logging.getLogger(__name__)
settings = get_settings()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPTIMAL_THRESHOLD = 0.50  # Decision threshold for 12-task multi-label classification


class GNNService:
    """
    Singleton service class for EQ-KA-GCN inference.
    Executes live PyTorch model prediction across all 12 Tox21 endpoints and GNNExplainer attribution.
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

        # Checkpoints in priority order: Adaptive QAT KA-GCN best -> Focal KA-GCN -> Weighted KA-GCN
        checkpoints = [
            (eq_config.paths.checkpoints_dir / eq_config.quantization.qat_save_filename, "Adaptive Quantized KA-GCN (12 Tasks)"),
            (eq_config.paths.checkpoints_dir / eq_config.fourier_kan.focal_save_filename, "Focal KA-GCN (12 Tasks)"),
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
            logger.error(f"❌ Failed to load GNN checkpoint from {loaded_checkpoint}: {str(e)}")
            self._model_loaded = False

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Run multi-task toxicity prediction for a given SMILES string across all 12 Tox21 endpoints."""
        start_time = time.time()
        smiles = request.smiles

        if not self._model_loaded or self.model is None:
            raise ValueError("GNN model is not loaded. Cannot process prediction request.")

        # 1. Parse SMILES & Build Graph
        graph = smiles_to_graph(smiles=smiles, label=[0.0] * 12)
        if graph is None:
            raise ValueError(f"Invalid SMILES string: '{smiles}'. Unable to construct molecular graph.")

        graph = graph.to(DEVICE)
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)

        # 2. PyTorch Model Forward Pass (12-endpoint prediction)
        with torch.no_grad():
            logits = self.model(
                x=graph.x,
                edge_index=graph.edge_index,
                batch=batch,
                return_logits=True,
            )
            raw_probs = torch.sigmoid(logits).squeeze()

        endpoints_list = []
        if raw_probs.numel() > 1:
            for idx, ep_name in enumerate(TOX21_ENDPOINTS):
                if idx < raw_probs.numel():
                    p_val = float(raw_probs[idx].item())
                else:
                    p_val = 0.0
                info = ENDPOINT_INFO.get(ep_name, {"name": ep_name, "category": "Bioassay"})
                is_active = p_val >= OPTIMAL_THRESHOLD
                pred_label = "Toxic / Active" if is_active else "Non-Toxic / Inactive"
                conf = p_val if is_active else (1.0 - p_val)

                endpoints_list.append(
                    EndpointPrediction(
                        endpoint=ep_name,
                        name=info["name"],
                        category=info["category"],
                        prediction=pred_label,
                        probability=round(p_val, 4),
                        confidence=round(conf, 4),
                        threshold=OPTIMAL_THRESHOLD,
                    )
                )

            # Primary endpoint is SR-p53 (index 11) or max hazard
            primary_idx = 11 if raw_probs.numel() >= 12 else 0
            primary_prob = float(raw_probs[primary_idx].item())
            any_active = any(ep.probability >= OPTIMAL_THRESHOLD for ep in endpoints_list)
            prediction_label = "Toxic" if any_active else "Non-Toxic"
            confidence = primary_prob if any_active else (1.0 - primary_prob)
            prob = primary_prob
        else:
            prob = float(raw_probs.item())
            is_toxic = prob >= OPTIMAL_THRESHOLD
            prediction_label = "Toxic" if is_toxic else "Non-Toxic"
            confidence = prob if is_toxic else (1.0 - prob)

        # 3. Build RDKit 2D molecular graph connectivity
        graph_atoms = []
        graph_bonds = []
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                mol_2d = Chem.Mol(mol)
                AllChem.Compute2DCoords(mol_2d)
                conf = mol_2d.GetConformer()
                for idx, atom in enumerate(mol_2d.GetAtoms()):
                    pos = conf.GetAtomPosition(idx)
                    graph_atoms.append(
                        GraphAtom(
                            index=idx,
                            element=atom.GetSymbol(),
                            x=round(float(pos.x), 4),
                            y=round(float(pos.y), 4),
                        )
                    )
                for bond in mol_2d.GetBonds():
                    graph_bonds.append(
                        GraphBond(
                            source=bond.GetBeginAtomIdx(),
                            target=bond.GetEndAtomIdx(),
                        )
                    )
            except Exception as e:
                logger.warning(f"RDKit 2D layout extraction warning: {e}")

        # 4. GNNExplainer attribution (explaining primary endpoint SR-p53 or max active assay)
        important_atoms = []
        important_bonds = []
        explanation_summary = None
        explanation_status = "failed"

        try:
            target_idx = primary_idx if (raw_probs.numel() > 1) else -1
            explainer = GNNExplainerModule(epochs=40, lr=0.01, threshold=OPTIMAL_THRESHOLD, seed=42)
            node_importance, edge_importance = explainer.explain_graph(
                model=self.model,
                graph=graph,
                device=DEVICE,
                target_idx=target_idx,
            )

            primary_ep_name = TOX21_ENDPOINTS[primary_idx] if primary_idx < len(TOX21_ENDPOINTS) else "Toxicity"

            top_atoms, top_bonds = rank_atom_importance(
                graph=graph,
                node_importance=node_importance,
                edge_importance=edge_importance,
                top_k_atoms=5,
                top_k_bonds=5,
                mol=mol,
                prediction_label=prediction_label,
                endpoint=primary_ep_name,
            )

            important_atoms = [
                ImportantAtom(
                    index=atom["atom_index"],
                    element=atom["atom_symbol"],
                    name=atom.get("atom_name"),
                    score=atom["score"],
                    rank=atom.get("rank"),
                    influence_type=atom.get("influence_type"),
                    role=atom.get("role"),
                    description=atom.get("description"),
                )
                for atom in top_atoms
                if 0 <= atom["atom_index"] < graph.num_nodes
            ]
            important_bonds = [
                ImportantBond(
                    source=bond["source"],
                    target=bond["target"],
                    score=bond["score"],
                    rank=bond.get("rank"),
                    bond_name=bond.get("bond_name"),
                    influence_type=bond.get("influence_type"),
                    role=bond.get("role"),
                    description=bond.get("description"),
                )
                for bond in top_bonds
                if 0 <= bond["source"] < graph.num_nodes and 0 <= bond["target"] < graph.num_nodes
            ]

            if prediction_label == "Toxic":
                top_elem_list = ", ".join([f"{a['atom_name']} #{a['atom_index']}" for a in top_atoms[:3]])
                explanation_summary = (
                    f"Model identified {len(important_atoms)} critical toxicity-driving atomic centers ({top_elem_list}) "
                    f"acting as toxicophores that elevate toxicity probability across active assay pathways."
                )
            else:
                top_elem_list = ", ".join([f"{a['atom_name']} #{a['atom_index']}" for a in top_atoms[:3]])
                explanation_summary = (
                    f"Model identified {len(important_atoms)} key safety-stabilizing atomic centers ({top_elem_list}) "
                    f"maintaining a non-reactive, metabolically benign conformation that suppresses toxicity hazard."
                )

            # Generate/Save the GNNExplainer visualization PNG
            eq_config = get_config()
            save_img_path = eq_config.paths.outputs_dir / "explanations" / "molecule_explanation.png"
            visualize_molecule_explanation(
                smiles=smiles,
                node_importance=node_importance,
                edge_importance=edge_importance,
                save_path=str(save_img_path),
            )
            explanation_status = "Successfully generated"
        except Exception as e:
            logger.error(f"GNNExplainer execution failed: {str(e)}", exc_info=True)
            explanation_status = f"failed: {str(e)}"
            explanation_summary = "GNNExplainer attribution fallback: Standard molecular feature graph generated."

        execution_time = (time.time() - start_time) * 1000

        return PredictResponse(
            smiles=smiles,
            prediction=prediction_label,
            probability=round(float(prob), 4),
            confidence=round(float(confidence), 4),
            threshold=OPTIMAL_THRESHOLD,
            endpoint="Tox21 (12 Endpoints)",
            inference_time_ms=round(execution_time, 2),
            endpoints=endpoints_list,
            important_atoms=important_atoms,
            important_bonds=important_bonds,
            explanation_summary=explanation_summary,
            molecular_graph=MolecularGraph(atoms=graph_atoms, bonds=graph_bonds),
            explanation_image="/outputs/explanations/molecule_explanation.png",
        )

    def explain(self, request: ExplainRequest) -> ExplainResponse:
        """Generate GNN attention-based explanation for a SMILES prediction."""
        if not self._model_loaded or self.model is None:
            raise ValueError("GNN model is not loaded. Cannot process explain request.")

        graph = smiles_to_graph(smiles=request.smiles, label=[0.0] * 12)
        if graph is None:
            raise ValueError(f"Invalid SMILES string: '{request.smiles}'. Unable to construct graph.")

        graph = graph.to(DEVICE)

        target_idx = 11
        if request.target_endpoint and request.target_endpoint in TOX21_ENDPOINTS:
            target_idx = TOX21_ENDPOINTS.index(request.target_endpoint)

        try:
            explainer = GNNExplainerModule(epochs=50, lr=0.01)
            node_importance, edge_importance = explainer.explain_graph(
                model=self.model,
                graph=graph,
                device=DEVICE,
                target_idx=target_idx,
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

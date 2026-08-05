"""
EQ-KA-GCN Demonstration & Interactive Prediction Script

Loads the trained Adaptive Quantized EQ-KA-GCN model checkpoint,
predicts molecular toxicity using optimal decision thresholding (0.75),
and automatically runs GNNExplainer to generate atom/bond attribution explanations.
"""

import json
import time
from pathlib import Path
import torch
from rdkit import Chem

from config import get_config
from graph.graph_builder import smiles_to_graph
from models.baseline_gcn import BaselineGCN
from models.ka_gcn import KAGCN
from quantization.adaptive_qat import AdaptiveQATManager
from explainability import (
    GNNExplainerModule,
    rank_atom_importance,
    visualize_molecule_explanation,
    generate_explanation_report,
)

# ==========================================================
# CONFIGURATION
# ==========================================================
config = get_config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint priority: QAT best -> Weighted KA-GCN best -> Baseline GCN best
QAT_CHECKPOINT = config.paths.checkpoints_dir / config.quantization.qat_save_filename
WEIGHTED_CHECKPOINT = config.paths.checkpoints_dir / config.fourier_kan.weighted_save_filename
BASELINE_CHECKPOINT = config.paths.checkpoints_dir / config.model.save_filename

DATASET_INFO_PATH = config.paths.processed_dir / config.data.info_filename
EVALUATION_PATH = config.paths.outputs_dir / "weighted_evaluation_report.json"
OPTIMAL_THRESHOLD = 0.75  # Optimal threshold determined in Phase 12


def load_json(file_path: Path):
    """Load JSON file safely."""
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_header():
    print("=" * 60)
    print("EQ-KA-GCN DEMONSTRATION & EXPLAINABILITY ENGINE")
    print("=" * 60)


def load_model():
    """Loads the best available trained model checkpoint."""
    print("\nLoading trained model...")

    if QAT_CHECKPOINT.exists():
        checkpoint_path = QAT_CHECKPOINT
        model_type = "Adaptive Quantized KA-GCN"
    elif WEIGHTED_CHECKPOINT.exists():
        checkpoint_path = WEIGHTED_CHECKPOINT
        model_type = "Weighted KA-GCN"
    else:
        checkpoint_path = BASELINE_CHECKPOINT
        model_type = "Baseline GCN"

    print(f"Detected Model Checkpoint: {checkpoint_path.name} ({model_type})")

    if config.fourier_kan.enabled:
        model = KAGCN(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.output_dim,
            gcn_dropout=config.model.dropout,
            kan_hidden_dim=config.fourier_kan.hidden_dim,
            fourier_order=config.fourier_kan.fourier_order,
            kan_dropout=config.fourier_kan.dropout,
            kan_activation=config.fourier_kan.activation,
        )
        if "qat" in checkpoint_path.name:
            qat_mgr = AdaptiveQATManager(
                target_layers=["conv1", "conv2", "fourier_kan", "fc_out"],
                supported_bits=config.quantization.supported_bits,
            )
            # Default bit assignment matching Phase 13 calibration
            bit_assignments = {"fourier_kan": 8, "conv2": 8, "conv1": 6, "fc_out": 4}
            model = qat_mgr.prepare_qat_model(model, bit_assignments)
    else:
        model = BaselineGCN(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.output_dim,
            dropout=config.model.dropout,
        )

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print(f"[OK] {model_type} Loaded Successfully")
    return model, model_type


def predict(model, graph):
    """Executes model inference and returns class prediction and confidence."""
    graph = graph.to(DEVICE)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(
            x=graph.x,
            edge_index=graph.edge_index,
            batch=batch,
            return_logits=True,
        )
        prob = torch.sigmoid(logits).item()
    end = time.perf_counter()

    inference_ms = (end - start) * 1000
    prediction = "Toxic" if prob >= OPTIMAL_THRESHOLD else "Non-Toxic"
    confidence = prob if prediction == "Toxic" else (1.0 - prob)

    return prediction, confidence, prob, inference_ms


def get_smiles():
    """Prompts user for SMILES string input."""
    while True:
        smiles = input("\nEnter SMILES (or press Enter for default 'CC(=O)Oc1ccccc1C(=O)O'): ").strip()
        if smiles == "":
            smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Default aspirin
            print(f"Using default SMILES: {smiles}")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Invalid SMILES string. Please try again.")
            continue
        return smiles


def main():
    print_header()

    # Load Model
    model, model_type = load_model()

    # Get User Input SMILES
    smiles = get_smiles()

    # Build Graph
    graph = smiles_to_graph(smiles=smiles, label=0)
    if graph is None:
        print("[ERROR] Could not build graph from SMILES.")
        return

    # Execute Prediction
    prediction, confidence, prob, inference_time = predict(model, graph)

    print("\n=================================================")
    print("PREDICTION")
    print("=================================================")
    print(f"Predicted Class : {prediction}")
    print(f"Confidence      : {confidence * 100:.2f} %")
    print(f"Probability     : {prob:.4f}")
    print(f"Inference Time  : {inference_time:.3f} ms")
    print("=================================================")

    # Run GNNExplainer
    print("\nExecuting GNNExplainer for molecular toxicity attribution...")
    explainer = GNNExplainerModule(epochs=100, lr=0.01)
    node_importance, edge_importance = explainer.explain_graph(
        model=model,
        graph=graph,
        device=DEVICE,
    )

    # Rank Atoms and Bonds
    top_atoms, top_bonds = rank_atom_importance(
        graph=graph,
        node_importance=node_importance,
        edge_importance=edge_importance,
        top_k_atoms=config.explainability.top_k_atoms,
        top_k_bonds=config.explainability.top_k_bonds,
    )

    # Save Visualization & Report
    vis_path = config.paths.outputs_dir / "explanations" / "molecule_explanation.png"
    report_path = config.paths.outputs_dir / "explanation_report.json"

    visualize_molecule_explanation(
        smiles=smiles,
        node_importance=node_importance,
        edge_importance=edge_importance,
        save_path=str(vis_path),
    )

    generate_explanation_report(
        smiles=smiles,
        prediction=prediction,
        confidence=confidence,
        top_atoms=top_atoms,
        top_bonds=top_bonds,
        node_importance=node_importance,
        edge_importance=edge_importance,
        inference_time_ms=inference_time,
        save_path=str(report_path),
    )

    # Console Output Requirements
    print("\n=================================================")
    print("EXPLANATION")
    print("=================================================")
    print("Most Important Atoms:")
    for atom in top_atoms:
        print(f"  Rank {atom['rank']} | Atom {atom['atom_index']:<2} ({atom['atom_name']:<10}) : Score {atom['importance_score']:.4f}")

    print("\nMost Important Bonds:")
    for bond in top_bonds:
        print(f"  Rank {bond['rank']} | {bond['bond_name']:<18} : Score {bond['importance_score']:.4f}")

    print(f"\nExplanation Image Saved  : {vis_path}")
    print(f"Explanation Report Saved : {report_path}")
    print("=================================================")


if __name__ == "__main__":
    main()
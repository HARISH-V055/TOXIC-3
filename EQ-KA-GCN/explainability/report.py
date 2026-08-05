"""
Explanation Report Generator Module for EQ-KA-GCN

Saves JSON reports containing prediction metadata, confidence scores,
and ranked atom/bond GNNExplainer attribution scores.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

logger = logging.getLogger("EQ-KA-GCN.explainability.report")


def generate_explanation_report(
    smiles: str,
    prediction: str,
    confidence: float,
    top_atoms: List[Dict[str, Any]],
    top_bonds: List[Dict[str, Any]],
    node_importance: np.ndarray,
    edge_importance: np.ndarray,
    inference_time_ms: float = 0.0,
    save_path: str = "outputs/explanation_report.json",
) -> None:
    """
    Creates and saves explanation_report.json containing explanation metadata.

    Args:
        smiles (str): Input SMILES string.
        prediction (str): Predicted class label ("Toxic" or "Non-Toxic").
        confidence (float): Prediction probability / confidence value.
        top_atoms (List[Dict[str, Any]]): Top-k ranked atoms list.
        top_bonds (List[Dict[str, Any]]): Top-k ranked bonds list.
        node_importance (np.ndarray): Full array of atom importance scores.
        edge_importance (np.ndarray): Full array of bond importance scores.
        inference_time_ms (float): Latency for explanation generation.
        save_path (str): File path where JSON report is exported.
    """
    logger.info(f"Generating explanation report at: {save_path}")
    try:
        report_data = {
            "smiles": smiles,
            "prediction": prediction,
            "confidence": float(confidence),
            "inference_time_ms": float(inference_time_ms),
            "explanation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "important_atoms": top_atoms,
            "important_bonds": top_bonds,
            "atom_importance_scores": [float(v) for v in node_importance],
            "edge_importance_scores": [float(v) for v in edge_importance],
        }

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"JSON explanation report successfully saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save explanation report to {save_path}: {str(e)}")

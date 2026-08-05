"""
Explainability package for EQ-KA-GCN.

Provides GNNExplainer module for atom and bond attribution,
ranking functions, RDKit/matplotlib visualisations, and JSON report generation.
"""

from explainability.explainer import GNNExplainerModule
from explainability.atom_importance import rank_atom_importance, decode_atom_info
from explainability.visualization import visualize_molecule_explanation, plot_explainability_figures
from explainability.report import generate_explanation_report

__all__ = [
    "GNNExplainerModule",
    "rank_atom_importance",
    "decode_atom_info",
    "visualize_molecule_explanation",
    "plot_explainability_figures",
    "generate_explanation_report",
]

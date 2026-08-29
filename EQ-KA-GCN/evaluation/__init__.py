"""
Evaluation module for EQ-KA-GCN.
Tracks validation runs, computes performance metrics (ROC-AUC, Precision, Recall, F1),
handles plot generation, and writes test summaries to disk.
"""

from evaluation.calibration import (
    PlattScaler,
    TemperatureScaler,
    compute_calibration_metrics,
    export_calibration_report,
    plot_calibration_curves,
)
from evaluation.evaluator import Evaluator
from evaluation.plots import (
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from evaluation.report import generate_json_report, generate_text_report
from evaluation.threshold_optimizer import ThresholdOptimizer

__all__ = [
    "Evaluator",
    "ThresholdOptimizer",
    "TemperatureScaler",
    "PlattScaler",
    "compute_calibration_metrics",
    "plot_calibration_curves",
    "export_calibration_report",
    "plot_loss_curve",
    "plot_accuracy_curve",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_confusion_matrix",
    "generate_json_report",
    "generate_text_report",
]

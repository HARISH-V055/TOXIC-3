"""
Reporting Utilities Module for EQ-KA-GCN

Provides functions to format and save textual classification report summaries
and metadata evaluation results in JSON structures.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("EQ-KA-GCN.evaluation.report")


def generate_json_report(
    metrics: Dict[str, Any],
    save_path: str,
    model_name: str = "BaselineGCN",
    dataset_name: str = "Tox21 (12 Endpoints)",
) -> None:
    """
    Creates and saves evaluation_report.json containing all calculated
    performance scores alongside runtime metadata.

    Args:
        metrics (Dict[str, Any]): Dictionary output from the Evaluator.
        save_path (str): File path where the JSON report should be saved.
        model_name (str): Evaluated model architecture identifier.
        dataset_name (str): Evaluated dataset identifier.
    """
    logger.info(f"Generating JSON evaluation report at: {save_path}")
    try:
        report_data = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": metrics.get("accuracy", 0.0),
            "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1_score": metrics.get("f1_score", 0.0),
            "roc_auc": metrics.get("roc_auc", 0.0),
            "macro_roc_auc": metrics.get("macro_roc_auc", metrics.get("roc_auc", 0.0)),
            "macro_f1": metrics.get("macro_f1", metrics.get("f1_score", 0.0)),
            "mcc": metrics.get("mcc", 0.0),
            "inference_time_per_sample_ms": metrics.get("inference_time_per_sample_ms", 0.0),
        }
        if "confusion_matrix" in metrics:
            report_data["confusion_matrix"] = metrics["confusion_matrix"]
        if "classification_report_dict" in metrics:
            report_data["classification_report_dict"] = metrics["classification_report_dict"]
        if "per_task_metrics" in metrics:
            report_data["per_task_metrics"] = metrics["per_task_metrics"]
        if "optimal_threshold" in metrics:
            report_data["optimal_threshold"] = metrics["optimal_threshold"]
        if "optimized_metrics" in metrics:
            report_data["optimized_metrics"] = metrics["optimized_metrics"]

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"JSON evaluation report successfully written to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate JSON report: {str(e)}")


def generate_text_report(metrics: Dict[str, Any], save_path: str) -> None:
    """
    Saves a formatted classification summary report text file.

    Args:
        metrics (Dict[str, Any]): Dictionary output from the Evaluator.
        save_path (str): File path where the TXT report should be saved.
    """
    logger.info(f"Generating text classification report at: {save_path}")
    try:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write("======================================================================\n")
            f.write("TOX21 MULTI-TASK EVALUATION CLASSIFICATION REPORT\n")
            f.write("======================================================================\n")
            f.write(f"Macro ROC-AUC          : {metrics.get('macro_roc_auc', metrics.get('roc_auc', 0.0)):.4f}\n")
            f.write(f"Macro F1 Score         : {metrics.get('macro_f1', metrics.get('f1_score', 0.0)):.4f}\n")
            f.write(f"Macro Accuracy         : {metrics.get('macro_accuracy', metrics.get('accuracy', 0.0)):.4f}\n")
            f.write(f"Macro Balanced Accuracy: {metrics.get('macro_balanced_accuracy', metrics.get('balanced_accuracy', 0.0)):.4f}\n")
            f.write(f"Macro MCC              : {metrics.get('macro_mcc', metrics.get('mcc', 0.0)):.4f}\n")
            f.write("======================================================================\n")

            if "per_task_metrics" in metrics:
                f.write("\nINDIVIDUAL PER-ENDPOINT METRICS:\n")
                f.write(f"{'Endpoint':<16} | {'ROC-AUC':<8} | {'F1':<8} | {'Precision':<10} | {'Recall':<8} | {'Accuracy':<8}\n")
                f.write("-" * 70 + "\n")
                for name, tm in metrics["per_task_metrics"].items():
                    f.write(
                        f"{name:<16} | {tm.get('roc_auc', 0.0):<8.4f} | {tm.get('f1_score', 0.0):<8.4f} | "
                        f"{tm.get('precision', 0.0):<10.4f} | {tm.get('recall', 0.0):<8.4f} | {tm.get('accuracy', 0.0):<8.4f}\n"
                    )
                f.write("======================================================================\n")
            elif "classification_report_str" in metrics:
                f.write(metrics["classification_report_str"])
                f.write("\n======================================================================\n")

        logger.info(f"Text classification report successfully written to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate text classification report: {str(e)}")

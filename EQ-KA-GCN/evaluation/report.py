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
    dataset_name: str = "Tox21",
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
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "mcc": metrics["mcc"],
            "confusion_matrix": metrics["confusion_matrix"],
            "inference_time_per_sample_ms": metrics["inference_time_per_sample_ms"],
            "classification_report_dict": metrics["classification_report_dict"],
        }
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
        class_report_str = metrics.get("classification_report_str", "")
        
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("BASELINE GCN CLASSIFICATION REPORT\n")
            f.write("==================================================\n")
            f.write(class_report_str)
            f.write("==================================================\n")
            
        logger.info(f"Text classification report successfully written to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate text classification report: {str(e)}")

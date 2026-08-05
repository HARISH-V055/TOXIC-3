"""
Training History Tracking Module for EQ-KA-GCN

Manages the persistence of historical epoch training losses and evaluation metrics,
exporting them to standard CSV files for future plotting and publication results.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

logger = logging.getLogger("EQ-KA-GCN.training.history")


class History:
    """
    Accumulates epoch-level metrics and loss logs during training.
    Provides utility methods to save the records to a CSV file.
    """

    def __init__(self) -> None:
        """Initializes an empty history tracking container."""
        self.records: List[Dict[str, Any]] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Dict[str, Any],
        lr: float,
    ) -> None:
        """
        Logs a single epoch's metrics.

        Args:
            epoch (int): The current epoch number.
            train_loss (float): Average training loss for the epoch.
            val_loss (float): Average validation loss for the epoch.
            metrics (Dict[str, Any]): Dictionary containing classification metrics
                                      (accuracy, precision, recall, f1_score, roc_auc).
            lr (float): Current optimizer learning rate.
        """
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": metrics.get("accuracy", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1_score": metrics.get("f1_score", 0.0),
            "roc_auc": metrics.get("roc_auc", 0.0),
            "learning_rate": lr,
        }
        self.records.append(record)
        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Acc: {record['accuracy']:.4f} | Prec: {record['precision']:.4f} | "
            f"Rec: {record['recall']:.4f} | F1: {record['f1_score']:.4f} | "
            f"ROC-AUC: {record['roc_auc']:.4f} | LR: {lr:.6f}"
        )

    def export_csv(self, export_path: str = "outputs/history.csv") -> None:
        """
        Exports the logged history records to a CSV file.

        Args:
            export_path (str): Target file path for the CSV output.
        """
        try:
            path = Path(export_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(self.records)
            df.to_csv(path, index=False)
            logger.info(f"Successfully exported training history to: {path}")
        except Exception as e:
            logger.error(f"Failed to export training history to {export_path}: {str(e)}")

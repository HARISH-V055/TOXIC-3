"""
Threshold Optimization Module for EQ-KA-GCN

Provides the ThresholdOptimizer class to evaluate and select the optimal binary classification
decision threshold on validation datasets, maximizing F1 score and MCC for imbalanced data.
Also exports CSV analysis and 300 DPI publication plots.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("EQ-KA-GCN.evaluation.threshold_optimizer")


class ThresholdOptimizer:
    """
    Evaluates binary classification decision thresholds to optimize performance metrics on validation data.
    """

    def __init__(
        self,
        search_start: float = 0.05,
        search_end: float = 0.95,
        step: float = 0.05,
        selection_metric: str = "f1",
    ) -> None:
        """
        Initializes the ThresholdOptimizer.

        Args:
            search_start (float): Starting threshold value (inclusive).
            search_end (float): Ending threshold value (inclusive).
            step (float): Step size for threshold grid search.
            selection_metric (str): Primary metric to maximize ('f1' or 'mcc').
        """
        self.search_start = search_start
        self.search_end = search_end
        self.step = step
        self.selection_metric = selection_metric

    def grid_search(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
        """
        Evaluates thresholds on validation ground truth and predicted probabilities.

        Args:
            y_true (np.ndarray): Binary ground truth labels [num_samples].
            y_prob (np.ndarray): Predicted probabilities [num_samples].

        Returns:
            Tuple[float, Dict[str, Any], pd.DataFrame]:
                - Optimal threshold value.
                - Dictionary of metrics at the optimal threshold.
                - DataFrame containing results for all evaluated thresholds.
        """
        logger.info(
            f"Running threshold grid search from {self.search_start:.2f} to {self.search_end:.2f} "
            f"with step {self.step:.2f}..."
        )

        thresholds = np.arange(self.search_start, self.search_end + 1e-5, self.step)

        # Calculate ROC-AUC once (constant for reference)
        if len(np.unique(y_true)) > 1:
            auc_val = float(roc_auc_score(y_true, y_prob))
        else:
            auc_val = 0.5

        records: List[Dict[str, Any]] = []

        for thresh in thresholds:
            t = float(np.round(thresh, 4))
            y_pred = (y_prob >= t).astype(int)

            acc = float(accuracy_score(y_true, y_pred))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            mcc = float(matthews_corrcoef(y_true, y_pred))

            records.append({
                "Threshold": t,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "Balanced Accuracy": bal_acc,
                "MCC": mcc,
                "ROC-AUC": auc_val,
            })

        df_results = pd.DataFrame(records)

        # Primary sorting by F1 descending, secondary by MCC descending
        sorted_df = df_results.sort_values(by=["F1", "MCC"], ascending=[False, False])
        best_row = sorted_df.iloc[0]

        optimal_threshold = float(best_row["Threshold"])
        optimal_metrics = {
            "threshold": optimal_threshold,
            "accuracy": float(best_row["Accuracy"]),
            "precision": float(best_row["Precision"]),
            "recall": float(best_row["Recall"]),
            "f1_score": float(best_row["F1"]),
            "balanced_accuracy": float(best_row["Balanced Accuracy"]),
            "mcc": float(best_row["MCC"]),
            "roc_auc": float(best_row["ROC-AUC"]),
        }

        logger.info(
            f"Optimal threshold identified: {optimal_threshold:.2f} | "
            f"F1: {optimal_metrics['f1_score']:.4f} | MCC: {optimal_metrics['mcc']:.4f} | "
            f"Recall: {optimal_metrics['recall']:.4f} | Prec: {optimal_metrics['precision']:.4f}"
        )

        return optimal_threshold, optimal_metrics, df_results

    def save_csv(self, df_results: pd.DataFrame, csv_path: str = "outputs/threshold_analysis.csv") -> None:
        """
        Saves threshold grid search results to a CSV file.

        Args:
            df_results (pd.DataFrame): DataFrame containing threshold analysis results.
            csv_path (str): Target file path for the CSV output.
        """
        try:
            path = Path(csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            df_results.to_csv(path, index=False)
            logger.info(f"Threshold analysis CSV saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save threshold analysis CSV to {csv_path}: {str(e)}")

    def plot_curves(self, df_results: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
        """
        Generates and saves publication-quality 300 DPI plots for Threshold vs Metrics.

        Args:
            df_results (pd.DataFrame): DataFrame containing threshold analysis results.
            output_dir (str): Directory where PNG plots will be saved.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        metrics_to_plot = [
            ("F1", "F1 Score vs Decision Threshold", "threshold_vs_f1.png", "#1f77b4"),
            ("Precision", "Precision vs Decision Threshold", "threshold_vs_precision.png", "#2ca02c"),
            ("Recall", "Recall vs Decision Threshold", "threshold_vs_recall.png", "#ff7f0e"),
            ("MCC", "Matthews Correlation Coefficient (MCC) vs Decision Threshold", "threshold_vs_mcc.png", "#9467bd"),
        ]

        thresholds = df_results["Threshold"].values

        for col_name, title, filename, color in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            values = df_results[col_name].values

            # Plot main metric curve
            ax.plot(thresholds, values, marker="o", color=color, linewidth=2.5, label=col_name)

            # Highlight max value point
            max_idx = int(np.argmax(values))
            best_t = thresholds[max_idx]
            best_val = values[max_idx]
            ax.plot(best_t, best_val, marker="*", markersize=14, color="crimson", label=f"Optimal ({best_t:.2f}, {best_val:.3f})")

            ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
            ax.set_xlabel("Decision Threshold", fontsize=12, labelpad=8)
            ax.set_ylabel(col_name, fontsize=12, labelpad=8)
            ax.set_xlim(self.search_start - 0.02, self.search_end + 0.02)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=11, loc="best")

            plt.tight_layout()
            plot_file = out_path / filename
            plt.savefig(plot_file, dpi=300)
            plt.close(fig)
            logger.info(f"Saved publication threshold plot (300 DPI): {plot_file}")

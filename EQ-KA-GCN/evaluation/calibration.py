"""
Probability Calibration Module for EQ-KA-GCN

Provides Temperature Scaling and Platt Scaling (Logistic Calibration)
to align predicted model confidence with empirical ground-truth probabilities,
minimizing Mean Squared Error (Brier Score) and Expected Calibration Error (ECE)
without altering ROC-AUC discrimination ranking.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

logger = logging.getLogger("EQ-KA-GCN.evaluation.calibration")


class TemperatureScaler(nn.Module):
    """
    Temperature Scaling for post-hoc probability calibration.
    Learns a single scalar temperature T > 0 such that:
        p_calibrated = sigmoid(logits / T)
    Preserves prediction ranking (ROC-AUC invariant).
    """

    def __init__(self, init_temp: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales raw logits by learned temperature."""
        # Ensure temperature stays positive
        temp = torch.clamp(self.temperature, min=1e-3)
        return logits / temp

    def fit(
        self,
        val_logits: np.ndarray,
        val_targets: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Learns optimal temperature T via NLL minimization on validation data.

        Args:
            val_logits (np.ndarray): Raw validation logits [N].
            val_targets (np.ndarray): Binary validation labels [N].

        Returns:
            float: Optimal learned temperature T.
        """
        logger.info("Fitting Temperature Scaling parameter on validation set...")
        t_logits = torch.tensor(val_logits, dtype=torch.float32).view(-1, 1)
        t_targets = torch.tensor(val_targets, dtype=torch.float32).view(-1, 1)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def eval_step():
            optimizer.zero_grad()
            scaled_logits = self.forward(t_logits)
            loss = criterion(scaled_logits, t_targets)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        optimal_temp = float(self.temperature.item())
        logger.info(f"Temperature Scaling fit complete. Optimal T = {optimal_temp:.4f}")
        return optimal_temp

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Applies learned temperature and returns calibrated probabilities."""
        with torch.no_grad():
            t_logits = torch.tensor(logits, dtype=torch.float32).view(-1, 1)
            scaled = self.forward(t_logits)
            probs = torch.sigmoid(scaled).view(-1).numpy()
        return probs


class PlattScaler(nn.Module):
    """
    Platt Scaling (Affine / Logistic Calibration).
    Learns affine parameters (a, b) on raw logits:
        z_cal = a * z + b
        p_calibrated = sigmoid(a * z + b)
    """

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies affine transformation to raw logits."""
        return self.a * logits + self.b

    def fit(
        self,
        val_logits: np.ndarray,
        val_targets: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 150,
    ) -> Tuple[float, float]:
        """
        Fits (a, b) on validation data using L-BFGS optimizer.

        Args:
            val_logits (np.ndarray): Raw validation logits [N].
            val_targets (np.ndarray): Binary validation labels [N].

        Returns:
            Tuple[float, float]: (a, b) parameters.
        """
        logger.info("Fitting Platt Scaling parameters (a, b) on validation set...")
        t_logits = torch.tensor(val_logits, dtype=torch.float32).view(-1, 1)
        t_targets = torch.tensor(val_targets, dtype=torch.float32).view(-1, 1)

        optimizer = torch.optim.LBFGS([self.a, self.b], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def eval_step():
            optimizer.zero_grad()
            scaled = self.forward(t_logits)
            loss = criterion(scaled, t_targets)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        val_a = float(self.a.item())
        val_b = float(self.b.item())
        logger.info(f"Platt Scaling fit complete. a = {val_a:.4f}, b = {val_b:.4f}")
        return val_a, val_b

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Applies Platt transformation and returns calibrated probabilities."""
        with torch.no_grad():
            t_logits = torch.tensor(logits, dtype=torch.float32).view(-1, 1)
            scaled = self.forward(t_logits)
            probs = torch.sigmoid(scaled).view(-1).numpy()
        return probs


def compute_calibration_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes calibration metrics: Brier Score (MSE), ECE, MCE, and NLL (Log Loss).

    Args:
        y_true (np.ndarray): Ground truth binary labels (0 or 1).
        y_prob (np.ndarray): Predicted probabilities in [0, 1].
        n_bins (int): Number of equal-width bins for reliability diagram.

    Returns:
        Dict[str, Any]: Dictionary of calibration metrics and bin statistics.
    """
    y_true = np.array(y_true).astype(int)
    y_prob = np.clip(np.array(y_prob), 1e-7, 1.0 - 1e-7)

    # 1. Brier Score (MSE) & RMSE
    mse = float(brier_score_loss(y_true, y_prob))
    rmse = float(np.sqrt(mse))

    # 2. Log Loss (NLL)
    nll = float(log_loss(y_true, y_prob))

    # 3. ROC-AUC
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5

    # 4. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0
    mce = 0.0

    total_samples = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            err = abs(bin_acc - bin_conf)
            ece += (count / total_samples) * err
            mce = max(mce, err)
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(count)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((bins[i] + bins[i + 1]) / 2.0)
            bin_counts.append(0)

    return {
        "brier_score_mse": mse,
        "rmse": rmse,
        "nll_log_loss": nll,
        "roc_auc": auc,
        "ece": float(ece),
        "mce": float(mce),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "n_bins": n_bins,
    }


def plot_calibration_curves(
    y_true: np.ndarray,
    uncal_prob: np.ndarray,
    temp_prob: np.ndarray,
    platt_prob: np.ndarray,
    save_path: str,
    n_bins: int = 10,
) -> None:
    """
    Generates and saves publication-quality 300 DPI Reliability Diagrams
    comparing Uncalibrated, Temperature Scaled, and Platt Scaled probabilities.

    Args:
        y_true (np.ndarray): Binary ground truth.
        uncal_prob (np.ndarray): Uncalibrated probabilities.
        temp_prob (np.ndarray): Temperature scaled probabilities.
        platt_prob (np.ndarray): Platt scaled probabilities.
        save_path (str): Filepath to save 300 DPI plot.
        n_bins (int): Number of bins for reliability diagram.
    """
    try:
        from sklearn.calibration import calibration_curve

        uncal_metrics = compute_calibration_metrics(y_true, uncal_prob, n_bins=n_bins)
        temp_metrics = compute_calibration_metrics(y_true, temp_prob, n_bins=n_bins)
        platt_metrics = compute_calibration_metrics(y_true, platt_prob, n_bins=n_bins)

        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, figsize=(8, 9), dpi=300, gridspec_kw={"height_ratios": [3, 1]}
        )

        # 1. Main Calibration / Reliability Plot
        ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", linewidth=1.5)

        # Uncalibrated curve
        frac_pos_uncal, mean_pred_uncal = calibration_curve(y_true, uncal_prob, n_bins=n_bins, strategy="uniform")
        ax1.plot(
            mean_pred_uncal,
            frac_pos_uncal,
            "s-",
            color="#d62728",
            linewidth=2.0,
            label=f"Uncalibrated (MSE: {uncal_metrics['brier_score_mse']:.4f}, ECE: {uncal_metrics['ece']:.4f})",
        )

        # Temperature Scaled curve
        frac_pos_temp, mean_pred_temp = calibration_curve(y_true, temp_prob, n_bins=n_bins, strategy="uniform")
        ax1.plot(
            mean_pred_temp,
            frac_pos_temp,
            "o-",
            color="#1f77b4",
            linewidth=2.0,
            label=f"Temperature Scaled (MSE: {temp_metrics['brier_score_mse']:.4f}, ECE: {temp_metrics['ece']:.4f})",
        )

        # Platt Scaled curve
        frac_pos_platt, mean_pred_platt = calibration_curve(y_true, platt_prob, n_bins=n_bins, strategy="uniform")
        ax1.plot(
            mean_pred_platt,
            frac_pos_platt,
            "^-",
            color="#2ca02c",
            linewidth=2.0,
            label=f"Platt Scaled (MSE: {platt_metrics['brier_score_mse']:.4f}, ECE: {platt_metrics['ece']:.4f})",
        )

        ax1.set_title("Probability Calibration Reliability Diagram", fontsize=14, fontweight="bold", pad=12)
        ax1.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax1.set_ylabel("Empirical Fraction of Positives", fontsize=11)
        ax1.set_xlim([-0.02, 1.02])
        ax1.set_ylim([-0.02, 1.02])
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", fontsize=9.5)

        # 2. Probability Distribution Histogram
        ax2.hist(
            [uncal_prob, temp_prob, platt_prob],
            bins=15,
            label=["Uncalibrated", "Temperature", "Platt"],
            color=["#d62728", "#1f77b4", "#2ca02c"],
            alpha=0.7,
            histtype="bar",
        )
        ax2.set_xlabel("Predicted Probability", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_xlim([-0.02, 1.02])
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", fontsize=9.5)

        plt.tight_layout()
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, dpi=300)
        plt.close(fig)
        logger.info(f"Calibration reliability diagram saved at: {save_file}")
    except Exception as e:
        logger.error(f"Failed to generate calibration plot: {str(e)}")


def export_calibration_report(
    report_data: Dict[str, Any], save_path: str = "outputs/calibration_report.json"
) -> None:
    """Saves calibration evaluation results to a JSON report."""
    try:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Calibration JSON report saved to: {path}")
    except Exception as e:
        logger.error(f"Failed to save calibration report: {str(e)}")

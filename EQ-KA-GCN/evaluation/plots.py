"""
Plotting and Visualization Module for EQ-KA-GCN

Provides functions to generate high-resolution, publication-quality figures (300 DPI)
including: Loss Curves, Accuracy Curves, ROC Curves, Precision-Recall Curves,
and Confusion Matrix heatmaps.
"""

import logging
from pathlib import Path
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

# Use 'Agg' non-interactive backend to avoid window rendering blocks
matplotlib.use("Agg")

logger = logging.getLogger("EQ-KA-GCN.evaluation.plots")


def plot_loss_curve(history_path: str, save_path: str) -> None:
    """
    Plots the training and validation loss curves over epochs.

    Args:
        history_path (str): Path to history.csv log file.
        save_path (str): Target path to save loss_curve.png.
    """
    logger.info(f"Generating Loss Curve from: {history_path}")
    try:
        df = pd.read_csv(history_path)
        
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["train_loss"], label="Training Loss", color="#1f77b4", linewidth=2)
        plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2, linestyle="--")
        
        plt.title("Model Training and Validation Loss", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss (BCE)", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        # Ensure directories exist
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_file, dpi=300)
        plt.close()
        logger.info(f"Loss Curve successfully saved at: {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate Loss Curve: {str(e)}")


def plot_accuracy_curve(history_path: str, save_path: str) -> None:
    """
    Plots validation accuracy progression over epochs.

    Args:
        history_path (str): Path to history.csv log file.
        save_path (str): Target path to save accuracy_curve.png.
    """
    logger.info(f"Generating Accuracy Curve from: {history_path}")
    try:
        df = pd.read_csv(history_path)
        
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["accuracy"] * 100, label="Validation Accuracy", color="#2ca02c", linewidth=2)
        
        plt.title("Validation Accuracy Progression", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_file, dpi=300)
        plt.close()
        logger.info(f"Accuracy Curve successfully saved at: {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate Accuracy Curve: {str(e)}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str) -> None:
    """
    Plots test Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true (np.ndarray): True target binary labels.
        y_prob (np.ndarray): Predicted probabilities.
        save_path (str): Target path to save roc_curve.png.
    """
    logger.info("Generating ROC Curve...")
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = 0.5
        if len(np.unique(y_true)) > 1:
            from sklearn.metrics import roc_auc_score
            auc_score = roc_auc_score(y_true, y_prob)
            
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color="#d62728", label=f"Baseline GCN (AUC = {auc_score:.4f})", linewidth=2.5)
        plt.plot([0, 1], [0, 1], color="#7f7f7f", linestyle=":", label="Random Guess (AUC = 0.5000)")
        
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("False Positive Rate (FPR)", fontsize=11)
        plt.ylabel("True Positive Rate (TPR)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_file, dpi=300)
        plt.close()
        logger.info(f"ROC Curve successfully saved at: {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate ROC Curve: {str(e)}")


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str) -> None:
    """
    Plots the test Precision-Recall (PR) curve.

    Args:
        y_true (np.ndarray): True target binary labels.
        y_prob (np.ndarray): Predicted probabilities.
        save_path (str): Target path to save precision_recall_curve.png.
    """
    logger.info("Generating Precision-Recall Curve...")
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(6, 6))
        plt.plot(recall_vals, precision_vals, color="#9467bd", label=f"Baseline GCN (AP = {avg_precision:.4f})", linewidth=2.5)
        
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.title("Precision-Recall (PR) Curve", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Recall", fontsize=11)
        plt.ylabel("Precision", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
        
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_file, dpi=300)
        plt.close()
        logger.info(f"PR Curve successfully saved at: {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate PR Curve: {str(e)}")


def plot_confusion_matrix(cm: List[List[int]], save_path: str) -> None:
    """
    Plots a Confusion Matrix heatmap.

    Args:
        cm (List[List[int]]): 2x2 confusion matrix array representation.
        save_path (str): Target path to save confusion_matrix.png.
    """
    logger.info("Generating Confusion Matrix Heatmap...")
    try:
        cm_array = np.array(cm)
        plt.figure(figsize=(6, 5))
        
        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            xticklabels=["Non-Toxic", "Toxic"],
            yticklabels=["Non-Toxic", "Toxic"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        
        plt.title("Confusion Matrix Heatmap", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Predicted Class", fontsize=11)
        plt.ylabel("True Class", fontsize=11)
        plt.tight_layout()
        
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_file, dpi=300)
        plt.close()
        logger.info(f"Confusion Matrix successfully saved at: {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate Confusion Matrix Heatmap: {str(e)}")

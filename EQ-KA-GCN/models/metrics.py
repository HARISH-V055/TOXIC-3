"""
Evaluation Metrics Module for EQ-KA-GCN

Provides reusable classification metrics functions (Accuracy, Precision, Recall,
F1 Score, ROC-AUC, Confusion Matrix) using scikit-learn.
"""

from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score as sklearn_f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes prediction accuracy."""
    return float(accuracy_score(y_true, y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes precision with safe handling of division by zero."""
    return float(precision_score(y_true, y_pred, zero_division=0))


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes recall with safe handling of division by zero."""
    return float(recall_score(y_true, y_pred, zero_division=0))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes F1-score with safe handling of division by zero."""
    return float(sklearn_f1_score(y_true, y_pred, zero_division=0))


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes Area Under the ROC Curve (ROC-AUC).
    Returns 0.5 if ROC-AUC is undefined (e.g. only one class present in split).
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> List[List[int]]:
    """Computes the confusion matrix, returned as a nested list."""
    matrix = sklearn_confusion_matrix(y_true, y_pred)
    return matrix.tolist()


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Computes all classification metrics and returns them in a formatted dictionary.

    Args:
        y_true (np.ndarray): True target binary labels.
        y_pred (np.ndarray): Predicted binary labels (0 or 1).
        y_prob (np.ndarray): Predicted class probabilities.

    Returns:
        Dict[str, Any]: Dictionary of evaluation results.
    """
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

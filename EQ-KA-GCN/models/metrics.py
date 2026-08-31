"""
Evaluation Metrics Module for EQ-KA-GCN

Provides reusable classification metrics functions (Accuracy, Precision, Recall,
F1 Score, ROC-AUC, Confusion Matrix) using scikit-learn.
"""

from typing import Any, Dict, List, Optional
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    task_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes classification metrics for 1D (single-task) or 2D (multi-task) arrays.

    Args:
        y_true (np.ndarray): True target binary labels [N] or [N, num_tasks].
        y_pred (np.ndarray): Predicted binary labels [N] or [N, num_tasks].
        y_prob (np.ndarray): Predicted class probabilities [N] or [N, num_tasks].
        task_names (Optional[List[str]]): List of task names if multi-task.

    Returns:
        Dict[str, Any]: Evaluation results including macro averages and per-task metrics.
    """
    # Single-task 1D case (or [N, 1])
    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
        yt = y_true.ravel()
        yp = y_pred.ravel()
        ypr = y_prob.ravel()
        return {
            "accuracy": accuracy(yt, yp),
            "precision": precision(yt, yp),
            "recall": recall(yt, yp),
            "f1_score": f1_score(yt, yp),
            "roc_auc": roc_auc(yt, ypr),
            "confusion_matrix": confusion_matrix(yt, yp),
        }

    # Multi-task 2D case: [N, num_tasks]
    num_tasks = y_true.shape[1]
    if task_names is None or len(task_names) != num_tasks:
        task_names = [f"Task_{i}" for i in range(num_tasks)]

    per_task_metrics: Dict[str, Dict[str, float]] = {}
    auc_list, f1_list, acc_list, prec_list, rec_list = [], [], [], [], []

    for i, name in enumerate(task_names):
        yt_i = y_true[:, i]
        yp_i = y_pred[:, i]
        ypr_i = y_prob[:, i]

        acc_i = accuracy(yt_i, yp_i)
        prec_i = precision(yt_i, yp_i)
        rec_i = recall(yt_i, yp_i)
        f1_i = f1_score(yt_i, yp_i)
        auc_i = roc_auc(yt_i, ypr_i)

        per_task_metrics[name] = {
            "accuracy": acc_i,
            "precision": prec_i,
            "recall": rec_i,
            "f1_score": f1_i,
            "roc_auc": auc_i,
        }

        acc_list.append(acc_i)
        prec_list.append(prec_i)
        rec_list.append(rec_i)
        f1_list.append(f1_i)
        auc_list.append(auc_i)

    macro_acc = float(np.mean(acc_list))
    macro_prec = float(np.mean(prec_list))
    macro_rec = float(np.mean(rec_list))
    macro_f1 = float(np.mean(f1_list))
    macro_auc = float(np.mean(auc_list))

    return {
        "accuracy": macro_acc,
        "precision": macro_prec,
        "recall": macro_rec,
        "f1_score": macro_f1,
        "roc_auc": macro_auc,
        "macro_roc_auc": macro_auc,
        "macro_f1": macro_f1,
        "macro_accuracy": macro_acc,
        "per_task_metrics": per_task_metrics,
    }

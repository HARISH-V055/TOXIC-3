"""
Evaluation Engine Module for EQ-KA-GCN

Provides the Evaluator class to execute model inference over test datasets,
evaluating key metrics like accuracy, balanced accuracy, recall, precision, F1,
Matthews Correlation Coefficient (MCC), ROC-AUC, and inference latency.
"""

import logging
import time
from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("EQ-KA-GCN.evaluation.evaluator")


class Evaluator:
    """
    Evaluates trained GCN models on testing datasets.
    Controls forward passes, prediction thresholding, performance metric calculations,
    and computational profiling.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        """
        Initializes the model Evaluator.

        Args:
            model (nn.Module): GCN model instance to evaluate.
            device (torch.device): Computing device to use (CUDA or CPU).
        """
        self.model = model
        self.device = device

    def load_model(self, checkpoint_path: str) -> None:
        """
        Loads the best performing trained weights into the model.

        Args:
            checkpoint_path (str): File path to model state dict checkpoint (.pt).
        """
        logger.info(f"Loading GCN model state dict checkpoint from: {checkpoint_path}")
        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("Successfully loaded GCN checkpoint.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
            raise e

    def evaluate(
        self, loader: DataLoader, threshold: float = 0.5
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs model evaluation over the test dataset.

        Args:
            loader (DataLoader): DataLoader for testing graphs.
            threshold (float): Decision threshold for binary classification.

        Returns:
            Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
                - Dictionary containing all computed evaluation metrics.
                - Ground truth labels (y_true).
                - Class predictions (y_pred).
                - Toxicity probabilities (y_prob).
        """
        logger.info(f"Evaluating GCN model on test dataset (Decision threshold: {threshold})...")
        self.model.eval()

        all_targets = []
        all_probs = []

        total_samples = 0
        inference_start = time.time()

        with torch.no_grad():
            for batch in loader:
                batch_x = batch.x.to(self.device)
                batch_edge_index = batch.edge_index.to(self.device)
                batch_indicator = batch.batch.to(self.device)
                targets = batch.y.to(self.device).float().reshape(batch.num_graphs, -1)
                batch_fp = getattr(batch, "fp", None)
                if batch_fp is not None:
                    batch_fp = batch_fp.to(self.device).float().reshape(batch.num_graphs, -1)

                # Model forward pass (obtaining logits)
                logits = self.model(
                    x=batch_x,
                    edge_index=batch_edge_index,
                    batch=batch_indicator,
                    fp=batch_fp,
                    return_logits=True,
                )

                # Convert raw logits to probabilities
                probs = torch.sigmoid(logits)

                # Accumulate values
                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                total_samples += batch.num_graphs

        total_inference_time = time.time() - inference_start
        avg_inference_time_ms = (
            (total_inference_time / total_samples) * 1000 if total_samples > 0 else 0.0
        )
        logger.info(
            f"Inference complete. Processed {total_samples} graphs. "
            f"Average latency per sample: {avg_inference_time_ms:.3f} ms"
        )

        # Concatenate batches
        y_true_raw = np.concatenate(all_targets, axis=0)
        y_prob_raw = np.concatenate(all_probs, axis=0)

        from config import TOX21_ENDPOINTS

        # Single-task case
        if y_true_raw.ndim == 1 or y_true_raw.shape[1] == 1:
            y_true = y_true_raw.flatten().astype(int)
            y_prob = y_prob_raw.flatten()
            y_pred = (y_prob >= threshold).astype(int)

            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
            cm = confusion_matrix(y_true, y_pred).tolist()
            class_report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            class_report_str = classification_report(y_true, y_pred, zero_division=0)

            metrics = {
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "roc_auc": float(auc),
                "mcc": float(mcc),
                "confusion_matrix": cm,
                "classification_report_dict": class_report_dict,
                "classification_report_str": class_report_str,
                "inference_time_per_sample_ms": float(avg_inference_time_ms),
            }
        else:
            # Multi-task case: shape [N, num_tasks]
            num_tasks = y_true_raw.shape[1]
            task_names = TOX21_ENDPOINTS[:num_tasks] if len(TOX21_ENDPOINTS) >= num_tasks else [f"Task_{i}" for i in range(num_tasks)]
            y_true = y_true_raw.astype(int)
            y_prob = y_prob_raw
            y_pred = (y_prob >= threshold).astype(int)

            per_task_metrics = {}
            auc_list, f1_list, acc_list, bal_acc_list, prec_list, rec_list, mcc_list = [], [], [], [], [], [], []

            for i, name in enumerate(task_names):
                yt_i = y_true[:, i]
                yp_i = y_pred[:, i]
                ypr_i = y_prob[:, i]

                acc_i = float(accuracy_score(yt_i, yp_i))
                bal_acc_i = float(balanced_accuracy_score(yt_i, yp_i))
                prec_i = float(precision_score(yt_i, yp_i, zero_division=0))
                rec_i = float(recall_score(yt_i, yp_i, zero_division=0))
                f1_i = float(f1_score(yt_i, yp_i, zero_division=0))
                mcc_i = float(matthews_corrcoef(yt_i, yp_i))
                auc_i = float(roc_auc_score(yt_i, ypr_i)) if len(np.unique(yt_i)) > 1 else 0.5

                per_task_metrics[name] = {
                    "accuracy": acc_i,
                    "balanced_accuracy": bal_acc_i,
                    "precision": prec_i,
                    "recall": rec_i,
                    "f1_score": f1_i,
                    "roc_auc": auc_i,
                    "mcc": mcc_i,
                }

                acc_list.append(acc_i)
                bal_acc_list.append(bal_acc_i)
                prec_list.append(prec_i)
                rec_list.append(rec_i)
                f1_list.append(f1_i)
                auc_list.append(auc_i)
                mcc_list.append(mcc_i)

            macro_acc = float(np.mean(acc_list))
            macro_bal_acc = float(np.mean(bal_acc_list))
            macro_prec = float(np.mean(prec_list))
            macro_rec = float(np.mean(rec_list))
            macro_f1 = float(np.mean(f1_list))
            macro_auc = float(np.mean(auc_list))
            macro_mcc = float(np.mean(mcc_list))

            # Primary task metrics (SR-p53 is the last task)
            primary_name = task_names[-1] if task_names else "Primary"
            primary_m = per_task_metrics.get(primary_name, {})

            metrics = {
                "accuracy": macro_acc,
                "balanced_accuracy": macro_bal_acc,
                "precision": macro_prec,
                "recall": macro_rec,
                "f1_score": macro_f1,
                "roc_auc": macro_auc,
                "macro_roc_auc": macro_auc,
                "macro_f1": macro_f1,
                "macro_accuracy": macro_acc,
                "macro_balanced_accuracy": macro_bal_acc,
                "macro_mcc": macro_mcc,
                "mcc": macro_mcc,
                "per_task_metrics": per_task_metrics,
                "primary_endpoint": primary_name,
                "primary_endpoint_metrics": primary_m,
                "inference_time_per_sample_ms": float(avg_inference_time_ms),
            }

        logger.info(f"Evaluation metrics computed successfully. Accuracy: {metrics['accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        return metrics, y_true, y_pred, y_prob

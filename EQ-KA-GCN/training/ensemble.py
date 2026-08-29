"""
5-Fold Cross-Validation Ensembling Module for EQ-KA-GCN

Implements stratified 5-fold cross-validation training and ensembling
for Kolmogorov-Arnold Graph Convolutional Networks (KA-GCN),
reducing model variance and boosting ROC-AUC, Balanced Accuracy, and F1.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from config import Config, get_config
from evaluation.calibration import PlattScaler
from models.ka_gcn import KAGCN
from models.loss import FocalLoss
from training.early_stopping import EarlyStopping
from training.history import History
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.trainer import Trainer

logger = logging.getLogger("EQ-KA-GCN.training.ensemble")


class KFoldEnsemble:
    """
    Manages 5-Fold Stratified Cross-Validation training, checkpointing,
    and aggregated ensemble prediction.
    """

    def __init__(self, config: Optional[Config] = None, n_splits: int = 5) -> None:
        self.config = config or get_config()
        self.n_splits = n_splits
        self.models: List[KAGCN] = []
        self.platt_scalers: List[PlattScaler] = []
        self.fold_metrics: List[Dict[str, Any]] = []

    def split_stratified_folds(
        self, graphs: List[Data], seed: int = 42
    ) -> List[Tuple[List[Data], List[Data]]]:
        """
        Splits graph list into K stratified folds.

        Args:
            graphs (List[Data]): Molecular graph datasets.
            seed (int): Random seed for reproducible splitting.

        Returns:
            List[Tuple[List[Data], List[Data]]]: List of (train_fold, val_fold) pairs.
        """
        labels = [int(g.y.item()) if isinstance(g.y, torch.Tensor) else int(g.y) for g in graphs]
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)

        folds = []
        for train_idx, val_idx in skf.split(graphs, labels):
            train_sub = [graphs[i] for i in train_idx]
            val_sub = [graphs[i] for i in val_idx]
            folds.append((train_sub, val_sub))

        logger.info(f"Created {self.n_splits} stratified folds from {len(graphs)} graphs.")
        return folds

    def build_model(self) -> KAGCN:
        """Instantiates a fresh KA-GCN model instance."""
        return KAGCN(
            input_dim=self.config.model.input_dim,
            hidden_dim=self.config.model.hidden_dim,
            output_dim=self.config.model.output_dim,
            gcn_dropout=self.config.model.dropout,
            fourier_order=self.config.fourier_kan.fourier_order,
            kan_hidden_dim=self.config.fourier_kan.hidden_dim,
            kan_dropout=self.config.fourier_kan.dropout,
            kan_activation=self.config.fourier_kan.activation,
        )

    def train_fold(
        self,
        fold_idx: int,
        train_graphs: List[Data],
        val_graphs: List[Data],
        device: torch.device,
        epochs: int = 35,
    ) -> Tuple[KAGCN, PlattScaler, Dict[str, Any]]:
        """
        Trains an individual fold model using Focal Loss.

        Args:
            fold_idx (int): Current fold index (1-indexed).
            train_graphs (List[Data]): Training graphs for this fold.
            val_graphs (List[Data]): Validation graphs for this fold.
            device (torch.device): Computing device.
            epochs (int): Max epochs to train.

        Returns:
            Tuple[KAGCN, PlattScaler, Dict[str, Any]]: Trained model, fitted Platt scaler, and fold history info.
        """
        logger.info(f"--- Training Fold {fold_idx}/{self.n_splits} ---")
        train_loader = DataLoader(
            train_graphs, batch_size=self.config.training.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_graphs, batch_size=self.config.training.batch_size, shuffle=False
        )

        model = self.build_model().to(device)
        criterion = FocalLoss(
            alpha=self.config.fourier_kan.focal_alpha,
            gamma=self.config.fourier_kan.focal_gamma,
        )
        optimizer = create_optimizer(
            model=model,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        scheduler = create_scheduler(optimizer, factor=0.5, patience=5, min_lr=1e-5)

        ckpt_path = self.config.paths.checkpoints_dir / f"ensemble_fold_{fold_idx}.pt"
        early_stopping = EarlyStopping(
            patience=12,
            save_path=str(ckpt_path),
        )
        history = History()

        trainer = Trainer(
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            history=history,
        )

        best_loss, best_epoch, duration = trainer.fit(
            train_loader=train_loader, val_loader=val_loader, epochs=epochs
        )

        # Load best weights
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        model.eval()

        # Fit Platt Scaler on validation predictions
        val_logits_list, val_y_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    batch.x.to(device),
                    batch.edge_index.to(device),
                    batch.batch.to(device),
                    return_logits=True,
                )
                val_logits_list.extend(out.view(-1).cpu().numpy())
                val_y_list.extend(batch.y.view(-1).cpu().numpy())

        val_logits = np.array(val_logits_list)
        val_y = np.array(val_y_list)

        platt_scaler = PlattScaler()
        platt_scaler.fit(val_logits, val_y)

        val_probs = platt_scaler.predict_proba(val_logits)
        val_auc = float(roc_auc_score(val_y, val_probs))

        logger.info(
            f"Fold {fold_idx} Complete | Best Epoch: {best_epoch} | "
            f"Val Loss: {best_loss:.4f} | Val ROC-AUC: {val_auc:.4f}"
        )

        return model, platt_scaler, {
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "val_roc_auc": val_auc,
            "checkpoint": str(ckpt_path),
        }

    def train_all(
        self, dev_graphs: List[Data], device: torch.device, epochs: int = 35
    ) -> List[Dict[str, Any]]:
        """
        Trains all K folds across the development dataset.
        """
        folds = self.split_stratified_folds(dev_graphs, seed=self.config.training.seed)
        self.models.clear()
        self.platt_scalers.clear()
        self.fold_metrics.clear()

        for fold_idx, (train_sub, val_sub) in enumerate(folds, start=1):
            model, platt_scaler, info = self.train_fold(
                fold_idx=fold_idx,
                train_graphs=train_sub,
                val_graphs=val_sub,
                device=device,
                epochs=epochs,
            )
            self.models.append(model)
            self.platt_scalers.append(platt_scaler)
            self.fold_metrics.append(info)

        return self.fold_metrics

    def predict(
        self, test_loader: DataLoader, device: torch.device, use_calibration: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Computes 5-Fold ensemble predictions over the test DataLoader.

        Args:
            test_loader (DataLoader): Test dataset loader.
            device (torch.device): Computing device.
            use_calibration (bool): Whether to apply Platt scaling to each fold.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
                - True ground-truth labels (test_y).
                - Ensembled probabilities (y_prob_ensemble).
                - List of individual fold probability arrays.
        """
        all_fold_probs: List[np.ndarray] = []
        test_y = None

        for idx, model in enumerate(self.models):
            model.eval()
            logits_list = []
            targets_list = []

            with torch.no_grad():
                for batch in test_loader:
                    out = model(
                        batch.x.to(device),
                        batch.edge_index.to(device),
                        batch.batch.to(device),
                        return_logits=True,
                    )
                    logits_list.extend(out.view(-1).cpu().numpy())
                    if test_y is None:
                        targets_list.extend(batch.y.view(-1).cpu().numpy())

            if test_y is None:
                test_y = np.array(targets_list)

            fold_logits = np.array(logits_list)

            if use_calibration and idx < len(self.platt_scalers):
                probs = self.platt_scalers[idx].predict_proba(fold_logits)
            else:
                probs = torch.sigmoid(torch.tensor(fold_logits)).numpy()

            all_fold_probs.append(probs)

        # Average probabilities across all 5 folds
        y_prob_ensemble = np.mean(all_fold_probs, axis=0)
        return test_y, y_prob_ensemble, all_fold_probs

    def evaluate_ensemble(
        self, test_loader: DataLoader, device: torch.device, threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Evaluates individual fold models and the final aggregated 5-Fold ensemble on the test set.
        """
        test_y, ens_probs, fold_probs_list = self.predict(test_loader, device, use_calibration=True)

        # Individual fold test ROC-AUCs
        individual_fold_aucs = [
            float(roc_auc_score(test_y, p)) for p in fold_probs_list
        ]
        individual_fold_mses = [
            float(brier_score_loss(test_y, p)) for p in fold_probs_list
        ]

        # Ensemble metrics
        ens_preds = (ens_probs >= threshold).astype(int)
        ens_acc = float(accuracy_score(test_y, ens_preds))
        ens_bal_acc = float(balanced_accuracy_score(test_y, ens_preds))
        ens_prec = float(precision_score(test_y, ens_preds, zero_division=0))
        ens_rec = float(recall_score(test_y, ens_preds, zero_division=0))
        ens_f1 = float(f1_score(test_y, ens_preds, zero_division=0))
        ens_mcc = float(matthews_corrcoef(test_y, ens_preds))
        ens_auc = float(roc_auc_score(test_y, ens_probs))
        ens_mse = float(brier_score_loss(test_y, ens_probs))

        return {
            "ensemble_roc_auc": ens_auc,
            "ensemble_mse_brier": ens_mse,
            "ensemble_accuracy": ens_acc,
            "ensemble_balanced_accuracy": ens_bal_acc,
            "ensemble_precision": ens_prec,
            "ensemble_recall": ens_rec,
            "ensemble_f1": ens_f1,
            "ensemble_mcc": ens_mcc,
            "fold_test_aucs": individual_fold_aucs,
            "fold_test_mses": individual_fold_mses,
            "mean_fold_auc": float(np.mean(individual_fold_aucs)),
            "ensemble_gain_auc": ens_auc - float(np.mean(individual_fold_aucs)),
        }

    def plot_ensemble_roc(
        self,
        test_y: np.ndarray,
        ens_probs: np.ndarray,
        fold_probs_list: List[np.ndarray],
        save_path: str = "outputs/figures/ensemble_roc_curve.png",
    ) -> None:
        """
        Generates 300 DPI publication ROC comparison plot.
        """
        try:
            plt.figure(figsize=(7, 7), dpi=300)
            plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.5000)", linewidth=1.2)

            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
            for i, p in enumerate(fold_probs_list):
                fpr, tpr, _ = roc_curve(test_y, p)
                auc_val = roc_auc_score(test_y, p)
                plt.plot(
                    fpr,
                    tpr,
                    color=colors[i % len(colors)],
                    linestyle=":",
                    alpha=0.65,
                    label=f"Fold {i+1} (AUC = {auc_val:.4f})",
                )

            # Ensemble ROC Curve
            fpr_ens, tpr_ens, _ = roc_curve(test_y, ens_probs)
            auc_ens = roc_auc_score(test_y, ens_probs)
            plt.plot(
                fpr_ens,
                tpr_ens,
                color="#d62728",
                linewidth=2.8,
                label=f"5-Fold Ensemble (AUC = {auc_ens:.4f})",
            )

            plt.title("5-Fold Cross-Validation Ensemble ROC Curve", fontsize=14, fontweight="bold", pad=12)
            plt.xlabel("False Positive Rate (FPR)", fontsize=11)
            plt.ylabel("True Positive Rate (TPR)", fontsize=11)
            plt.xlim([-0.02, 1.02])
            plt.ylim([-0.02, 1.02])
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend(loc="lower right", fontsize=10)
            plt.tight_layout()

            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_file, dpi=300)
            plt.close()
            logger.info(f"Saved ensemble ROC curve: {save_file}")
        except Exception as e:
            logger.error(f"Failed to plot ensemble ROC: {str(e)}")

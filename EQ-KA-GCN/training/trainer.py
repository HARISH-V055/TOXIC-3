"""
Training Orchestration Module for EQ-KA-GCN

Defines the Trainer class which encapsulates epoch loops, forward passes,
backpropagation, validation logging, and model checkpoints tracking.
"""

import logging
import time
from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from models.metrics import calculate_metrics
from training.early_stopping import EarlyStopping
from training.history import History

logger = logging.getLogger("EQ-KA-GCN.training.trainer")


class Trainer:
    """
    Coordinates baseline GCN training cycles: handles execution of epochs,
    gradients update, validation metrics, early stopping, and history accumulation.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        early_stopping: EarlyStopping,
        history: History,
    ) -> None:
        """
        Initializes the GCN pipeline Trainer.

        Args:
            model (nn.Module): The model architecture to train.
            device (torch.device): CUDA or CPU computation device.
            criterion (nn.Module): BCE loss module.
            optimizer (torch.optim.Optimizer): Model parameters optimizer.
            scheduler (Any): ReduceLROnPlateau learning rate scheduler.
            early_stopping (EarlyStopping): Early stopping manager.
            history (History): History recorder.
        """
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.history = history

    def train_epoch(self, loader: DataLoader) -> float:
        """
        Executes one epoch of training over the loader dataset.

        Args:
            loader (DataLoader): The training DataLoader split.

        Returns:
            float: Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        total_graphs = 0

        for batch in loader:
            # Move batch tensors to device
            batch_x = batch.x.to(self.device)
            batch_edge_index = batch.edge_index.to(self.device)
            batch_indicator = batch.batch.to(self.device)
            targets = batch.y.to(self.device).view(-1, 1).float()  # Align shapes for BCE

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(
                x=batch_x,
                edge_index=batch_edge_index,
                batch=batch_indicator,
                return_logits=True,
            )

            # Compute loss
            loss = self.criterion(logits, targets)

            # Backward pass & Optimizer update
            loss.backward()
            self.optimizer.step()

            # Track statistics
            num_graphs = batch.num_graphs
            total_loss += loss.item() * num_graphs
            total_graphs += num_graphs

        return total_loss / total_graphs if total_graphs > 0 else 0.0

    def validate_epoch(self, loader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluates the model over the validation loader dataset.

        Args:
            loader (DataLoader): The validation DataLoader split.

        Returns:
            Tuple[float, Dict[str, Any]]: Average validation loss and classification metrics dictionary.
        """
        self.model.eval()
        total_loss = 0.0
        total_graphs = 0

        all_targets = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                # Move batch tensors to device
                batch_x = batch.x.to(self.device)
                batch_edge_index = batch.edge_index.to(self.device)
                batch_indicator = batch.batch.to(self.device)
                targets = batch.y.to(self.device).view(-1, 1).float()

                # Forward pass
                logits = self.model(
                    x=batch_x,
                    edge_index=batch_edge_index,
                    batch=batch_indicator,
                    return_logits=True,
                )

                # Compute loss
                loss = self.criterion(logits, targets)

                # Track metrics
                num_graphs = batch.num_graphs
                total_loss += loss.item() * num_graphs
                total_graphs += num_graphs

                # Sigmoid probabilities and binary predictions
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                # Accumulate lists for metric calculation
                all_targets.append(targets.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        avg_loss = total_loss / total_graphs if total_graphs > 0 else 0.0

        # Concatenate arrays for metrics calculation
        if total_graphs > 0:
            y_true = np.concatenate(all_targets, axis=0)
            y_pred = np.concatenate(all_preds, axis=0)
            y_prob = np.concatenate(all_probs, axis=0)
            metrics = calculate_metrics(y_true, y_pred, y_prob)
        else:
            metrics = {}

        return avg_loss, metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> Tuple[float, int, float]:
        """
        Fits the GCN model on the dataset loaders for a specified number of epochs.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            epochs (int): Number of epochs to train.

        Returns:
            Tuple[float, int, float]: Best validation loss, best epoch, and total training duration (in seconds).
        """
        logger.info(f"Starting GCN training pipeline for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # 1. Train Step
            train_loss = self.train_epoch(train_loader)

            # 2. Validation Step
            val_loss, val_metrics = self.validate_epoch(val_loader)

            # 3. Fetch current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 4. Log to history
            self.history.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=val_metrics,
                lr=current_lr,
            )

            # 5. Step Scheduler with validation loss
            self.scheduler.step(val_loss)

            # 6. Step EarlyStopping with validation loss
            self.early_stopping(val_loss=val_loss, model=self.model, epoch=epoch)

            epoch_time = time.time() - epoch_start
            logger.debug(f"Epoch {epoch:03d} finished in {epoch_time:.2f} seconds.")

            # 7. Check early stopping trigger
            if self.early_stopping.early_stop:
                logger.info(f"Early termination triggered at epoch {epoch}.")
                break

        training_duration = time.time() - start_time
        logger.info(
            f"Training finished in {training_duration:.2f} seconds. "
            f"Best Validation Loss: {self.early_stopping.best_loss:.6f} at epoch {self.early_stopping.best_epoch}."
        )

        return self.early_stopping.best_loss, self.early_stopping.best_epoch, training_duration

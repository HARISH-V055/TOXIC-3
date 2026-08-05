"""
Early Stopping Module for EQ-KA-GCN

Provides checkpointing logic based on validation loss convergence to prevent overfitting
and save model states cleanly to disk.
"""

import logging
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger("EQ-KA-GCN.training.early_stopping")


class EarlyStopping:
    """
    Tracks validation loss and triggers training termination when validation
    loss stops improving for a specified patience threshold.
    Also manages saving the best performing model checkpoint to disk.
    """

    def __init__(
        self,
        patience: int = 15,
        save_path: str = "checkpoints/best_model.pt",
        min_delta: float = 0.0,
    ) -> None:
        """
        Initializes EarlyStopping monitor.

        Args:
            patience (int): Number of epochs to wait for improvement before early stopping.
            save_path (str): File path where the best model checkpoint should be stored.
            min_delta (float): Minimum change in validation loss to qualify as an improvement.
        """
        self.patience = patience
        self.save_path = Path(save_path)
        self.min_delta = min_delta
        
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_epoch = -1

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> None:
        """
        Evaluates validation loss at the end of an epoch.

        Args:
            val_loss (float): Validation loss for the current epoch.
            model (nn.Module): The active model being trained.
            epoch (int): Current epoch number.
        """
        if val_loss < self.best_loss - self.min_delta:
            logger.info(
                f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). "
                f"Saving best model state..."
            )
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            logger.info(
                f"Validation loss did not improve. "
                f"EarlyStopping counter: {self.counter} out of {self.patience}."
            )
            if self.counter >= self.patience:
                logger.info(f"EarlyStopping triggered! Convergence reached at epoch {epoch}.")
                self.early_stop = True

    def save_checkpoint(self, model: nn.Module) -> None:
        """Serializes the best performing model state dict to disk."""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            logger.info(f"Successfully checkpointed best model at: {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to save model checkpoint to {self.save_path}: {str(e)}")

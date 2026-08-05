"""
Learning Rate Scheduler Module for EQ-KA-GCN

Provides functions to configure adaptive learning rate schedulers, specifically
ReduceLROnPlateau to automatically throttle LR upon optimization stalls.
"""

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


def create_scheduler(
    optimizer: optim.Optimizer,
    factor: float = 0.5,
    patience: int = 5,
    min_lr: float = 1e-6,
) -> ReduceLROnPlateau:
    """
    Creates a ReduceLROnPlateau learning rate scheduler.

    Args:
        optimizer (optim.Optimizer): The active optimizer to schedule.
        factor (float): Factor by which the learning rate will be reduced.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        min_lr (float): A lower bound on the learning rate of all param groups.

    Returns:
        ReduceLROnPlateau: The configured scheduler.
    """
    return ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )

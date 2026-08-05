"""
Optimizer Construction Module for EQ-KA-GCN

Provides functions to construct AdamW optimization routines with custom weight decays.
"""

import torch.nn as nn
import torch.optim as optim


def create_optimizer(
    model: nn.Module,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
) -> optim.AdamW:
    """
    Constructs an AdamW optimizer for the model parameters.

    Args:
        model (nn.Module): The model to optimize.
        lr (float): Base learning rate.
        weight_decay (float): Weight decay penalty coefficient.

    Returns:
        optim.AdamW: The constructed optimizer.
    """
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

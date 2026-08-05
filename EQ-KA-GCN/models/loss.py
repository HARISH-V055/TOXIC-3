"""
Loss Functions Module for EQ-KA-GCN

Provides standard Binary Cross Entropy (BCE) loss criterion wrapped using PyTorch
BCEWithLogitsLoss with supporting parameters for class weights handling.
"""

from typing import Optional
import torch
import torch.nn as nn


def get_loss_criterion(positive_class_weight: Optional[float] = None) -> nn.BCEWithLogitsLoss:
    """
    Constructs and returns a Binary Cross Entropy loss function with logits.

    Args:
        positive_class_weight (Optional[float]): Optional weight multiplier for positive samples,
                                                 enabling handling of highly imbalanced datasets.

    Returns:
        nn.BCEWithLogitsLoss: The configured PyTorch loss module.
    """
    if positive_class_weight is not None:
        # pos_weight parameter in BCEWithLogitsLoss expects a 1D tensor of length [num_classes]
        pos_weight_tensor = torch.tensor([positive_class_weight], dtype=torch.float)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()

    return criterion

"""
Class Imbalance Utility Module for EQ-KA-GCN

Provides functions to compute class weights for imbalanced graph dataset splits.
"""

import logging
from typing import List, Tuple
import torch
from torch_geometric.data import Data

logger = logging.getLogger("EQ-KA-GCN.training.class_weights")


def compute_positive_class_weight(train_graphs: List[Data]) -> Tuple[float, int, int]:
    """
    Computes the positive class weight for BCEWithLogitsLoss based on training split class balance.

    pos_weight = negative_samples / positive_samples

    Args:
        train_graphs (List[Data]): List of PyTorch Geometric Data objects for the training set.

    Returns:
        Tuple[float, int, int]: Computed positive class weight ratio, positive count, and negative count.
    """
    pos_count = 0
    neg_count = 0

    for graph in train_graphs:
        if graph.y is not None:
            val = graph.y.item() if isinstance(graph.y, torch.Tensor) else graph.y
            if val == 1:
                pos_count += 1
            elif val == 0:
                neg_count += 1

    if pos_count == 0:
        logger.warning("No positive samples found in training set. Defaulting pos_weight to 1.0.")
        return 1.0, pos_count, neg_count

    pos_weight = float(neg_count) / float(pos_count)
    logger.info(
        f"Class balance analysis | Positive: {pos_count} | Negative: {neg_count} | "
        f"Computed pos_weight: {pos_weight:.4f}"
    )

    return pos_weight, pos_count, neg_count

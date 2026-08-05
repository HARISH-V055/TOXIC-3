"""
Dataset Splitting Module for EQ-KA-GCN

Provides functions to perform stratified splits (Train/Val/Test) of molecular graph datasets
to preserve toxicity label distributions across validation subsets.
"""

import logging
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# Use child logger under the main project logger
logger = logging.getLogger("EQ-KA-GCN.training.dataset_split")


def split_graph_dataset(
    graphs: List[Data],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Performs a stratified split of a list of PyTorch Geometric Data graphs into
    Train, Validation, and Test subsets based on the configured ratios and random seed.

    Args:
        graphs (List[Data]): List of PyTorch Geometric graph Data objects.
        train_ratio (float): Fraction of dataset for training (default: 0.8).
        val_ratio (float): Fraction of dataset for validation (default: 0.1).
        test_ratio (float): Fraction of dataset for testing (default: 0.1).
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[List[Data], List[Data], List[Data]]: List of graphs for train, val, and test splits.
    """
    total_graphs = len(graphs)
    logger.info(
        f"Initiating stratified split of {total_graphs} graphs. "
        f"Ratios: Train {train_ratio} | Val {val_ratio} | Test {test_ratio}"
    )

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        err_msg = (
            f"Split ratios must sum to 1.0. "
            f"Got: Train={train_ratio}, Val={val_ratio}, Test={test_ratio} (Sum={train_ratio + val_ratio + test_ratio})"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    # 1. Extract labels for stratification
    labels = [int(g.y.item()) for g in graphs]

    # 2. First Split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    logger.info(f"Splitting train subset (ratio={train_ratio}) from temp subset (ratio={temp_ratio:.2f})")
    
    train_graphs, temp_graphs, _, temp_labels = train_test_split(
        graphs,
        labels,
        test_size=temp_ratio,
        random_state=seed,
        stratify=labels,
    )

    # 3. Second Split: Val vs Test within the Temp subset
    val_relative_ratio = val_ratio / temp_ratio
    logger.info(f"Splitting validation (ratio={val_ratio}) and test (ratio={test_ratio}) within temp subset")
    
    val_graphs, test_graphs = train_test_split(
        temp_graphs,
        test_size=(1.0 - val_relative_ratio),
        random_state=seed,
        stratify=temp_labels,
    )

    logger.info(
        f"Stratified split complete. Sizes: "
        f"Train={len(train_graphs)} | Val={len(val_graphs)} | Test={len(test_graphs)}"
    )

    return train_graphs, val_graphs, test_graphs

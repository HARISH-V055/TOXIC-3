"""
DataLoader Ingestion Module for EQ-KA-GCN

Constructs PyTorch Geometric compatible DataLoaders for train, validation, and test splits
with configured shuffling and batch sizes.
"""

import logging
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Use child logger under the main project logger
logger = logging.getLogger("EQ-KA-GCN.training.dataloader")


def create_dataloaders(
    train_graphs: List[Data],
    val_graphs: List[Data],
    test_graphs: List[Data],
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wraps the lists of graphs into PyTorch Geometric DataLoaders.

    Args:
        train_graphs (List[Data]): Graphs for the training split.
        val_graphs (List[Data]): Graphs for the validation split.
        test_graphs (List[Data]): Graphs for the test split.
        batch_size (int): Mini-batch size.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test loaders.
    """
    logger.info(f"Creating PyTorch Geometric DataLoaders (Batch Size: {batch_size})")

    # 1. Train Loader: shuffle = True
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
    )

    # 2. Validation Loader: shuffle = False
    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False,
    )

    # 3. Test Loader: shuffle = False
    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False,
    )

    logger.info("Successfully initialized all DataLoaders.")
    return train_loader, val_loader, test_loader

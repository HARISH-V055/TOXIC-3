"""
Graph Dataset Module for EQ-KA-GCN

Defines a custom PyTorch/PyTorch Geometric compatible dataset wrapper for pre-computed
molecular graphs serialized using torch.save.
"""

import logging
from pathlib import Path
from typing import List, Union
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# Use child logger under the main project logger
logger = logging.getLogger("EQ-KA-GCN.graph.graph_dataset")


class GraphDataset(Dataset):
    """
    GraphDataset wraps pre-compiled PyTorch Geometric graph Data objects.
    Compatible with torch_geometric.loader.DataLoader.
    """

    def __init__(self, graphs_path: Union[str, Path]) -> None:
        """
        Loads the pre-saved list of graphs from the specified path.

        Args:
            graphs_path (Union[str, Path]): Path to the serialized PyG graphs (.pt file).
        """
        self.path = Path(graphs_path)
        logger.info(f"Loading GraphDataset from: {self.path}")

        if not self.path.exists():
            err_msg = f"Graph dataset file not found at: {self.path.absolute()}"
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        try:
            # Safe loading with weights_only=False since Data instances are custom PyG classes
            self.graphs: List[Data] = torch.load(self.path, weights_only=False)
            logger.info(f"Loaded {len(self.graphs)} graphs successfully.")
        except Exception as e:
            err_msg = f"Failed to load serialized graph list: {str(e)}"
            logger.error(err_msg)
            raise ValueError(err_msg) from e

    def __len__(self) -> int:
        """
        Returns the total number of graphs in the dataset.
        """
        return len(self.graphs)

    def __getitem__(self, idx: Union[int, slice, torch.Tensor]) -> Union[Data, List[Data]]:
        """
        Retrieves a single graph object or list of graphs by index.

        Args:
            idx: Index, slice, or tensor of indices.

        Returns:
            Union[Data, List[Data]]: PyTorch Geometric Data object or list of objects.
        """
        if isinstance(idx, (slice, list, torch.Tensor)):
            # Handle slice/list indexing
            if isinstance(idx, slice):
                return self.graphs[idx]
            elif isinstance(idx, torch.Tensor):
                indices = idx.tolist()
                return [self.graphs[i] for i in indices]
            else:
                return [self.graphs[i] for i in idx]
        
        # Single element lookup
        return self.graphs[idx]

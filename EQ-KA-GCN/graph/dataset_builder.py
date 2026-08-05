"""
Dataset Builder Module for EQ-KA-GCN

Responsible for converting all SMILES strings in a preprocessed CSV dataset
into a list of PyTorch Geometric Data graph objects, managing invalid entries
safely, and saving the graph dataset to disk.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from graph.graph_builder import smiles_to_graph

# Use child logger under the main project logger
logger = logging.getLogger("EQ-KA-GCN.graph.dataset_builder")


class DatasetBuilder:
    """
    Builder class to orchestrate full dataset conversion from chemical SMILES to PyG graphs.
    """

    def __init__(self) -> None:
        self.skipped_count = 0
        self.valid_count = 0

    def build_dataset(
        self, csv_path: Union[str, Path], smiles_column: str, target_column: str
    ) -> List[Data]:
        """
        Loads the preprocessed dataset, converts each row to a PyTorch Geometric Graph,
        and returns the list of successfully constructed graph objects.

        Args:
            csv_path (Union[str, Path]): Path to the processed CSV dataset.
            smiles_column (str): Name of the column containing SMILES strings.
            target_column (str): Name of the column containing target endpoint labels.

        Returns:
            List[Data]: A list of valid PyTorch Geometric Data objects.
        """
        path = Path(csv_path)
        logger.info(f"Loading cleaned dataset for graph construction from: {path}")

        try:
            df = pd.read_csv(path)
        except Exception as e:
            err_msg = f"Failed to load processed dataset file: {str(e)}"
            logger.error(err_msg)
            raise ValueError(err_msg) from e

        total_samples = len(df)
        logger.info(f"Dataset loaded. Total rows to process: {total_samples}")

        self.skipped_count = 0
        self.valid_count = 0
        graphs: List[Data] = []

        # Process SMILES with progress bar
        print("Processing Molecules...")
        for _, row in tqdm(df.iterrows(), total=total_samples, desc="Generating Graphs"):
            smiles = str(row[smiles_column]).strip()
            label = int(row[target_column])

            # Convert SMILES to PyG Data graph object
            graph_data = smiles_to_graph(smiles, label)

            if graph_data is not None:
                graphs.append(graph_data)
                self.valid_count += 1
            else:
                self.skipped_count += 1

        logger.info(
            f"Graph dataset construction complete. "
            f"Valid Graphs: {self.valid_count} | Skipped/Invalid: {self.skipped_count}"
        )
        return graphs

    def save_dataset(self, graphs: List[Data], output_path: Union[str, Path]) -> None:
        """
        Saves the list of generated graph Data objects to disk.

        Args:
            graphs (List[Data]): List of PyTorch Geometric graphs to save.
            output_path (Union[str, Path]): Target output file path (typically .pt file).
        """
        path = Path(output_path)
        logger.info(f"Saving {len(graphs)} graphs to: {path}")
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            torch.save(graphs, path)
            logger.info(f"Successfully saved graphs dataset at: {path}")
        except Exception as e:
            err_msg = f"Failed to save graphs dataset: {str(e)}"
            logger.error(err_msg)
            raise IOError(err_msg) from e

"""
Graph Dataset Statistics Module for EQ-KA-GCN

Computes, logs, and returns structural and biological properties of a pre-computed
molecular graph dataset, such as average node/edge distributions and target label splits.
"""

import logging
from typing import Any, Dict, Union
import numpy as np
from graph.graph_dataset import GraphDataset

# Use child logger under the main project logger
logger = logging.getLogger("EQ-KA-GCN.graph.dataset_statistics")


def compute_and_log_dataset_statistics(
    dataset: GraphDataset,
    skipped_count: int,
    dataset_name: str,
    target_column: str,
) -> Dict[str, Any]:
    """
    Computes graph metrics and label splits across the entire graph dataset.

    Args:
        dataset (GraphDataset): Wrapped GraphDataset instance.
        skipped_count (int): Count of invalid molecules skipped during build.
        dataset_name (str): Friendly name of the chemical dataset (e.g. Tox21).
        target_column (str): Toxicological target endpoint name.

    Returns:
        Dict[str, Any]: A dictionary containing all computed statistics.
    """
    logger.info("Computing graph dataset statistics...")

    valid_count = len(dataset)
    total_samples = valid_count + skipped_count

    if valid_count == 0:
        logger.warning("Empty graph dataset. Cannot compute structural statistics.")
        return {
            "dataset": dataset_name,
            "target": target_column,
            "total_samples": total_samples,
            "valid_graphs": 0,
            "invalid_graphs": skipped_count,
            "average_atoms": 0.0,
            "average_edges": 0.0,
            "positive": 0,
            "negative": 0,
        }

    # Arrays to accumulate structural metrics
    node_counts = np.array([g.num_nodes for g in dataset.graphs])
    edge_counts = np.array([g.num_edges for g in dataset.graphs])
    
    # In PyG undirected graphs, each chemical bond adds two directed edges: (u,v) and (v,u)
    bond_counts = edge_counts // 2

    # Calculate statistics
    avg_atoms = round(float(np.mean(node_counts)), 2)
    avg_edges = round(float(np.mean(edge_counts)), 2)
    avg_bonds = round(float(np.mean(bond_counts)), 2)
    avg_graph_size = round(float(np.mean(node_counts + edge_counts)), 2)

    max_nodes = int(np.max(node_counts))
    max_edges = int(np.max(edge_counts))
    min_nodes = int(np.min(node_counts))
    min_edges = int(np.min(edge_counts))

    # Calculate label splits
    labels = np.array([g.y.item() for g in dataset.graphs])
    pos_count = int(np.sum(labels == 1))
    neg_count = int(np.sum(labels == 0))

    pos_pct = round((pos_count / valid_count) * 100, 2)
    neg_pct = round((neg_count / valid_count) * 100, 2)

    # Log statistics
    logger.info("==================================================================")
    logger.info(f"DATASET GRAPH STATISTICS - {dataset_name}")
    logger.info("==================================================================")
    logger.info(f"Total Molecules (Raw):                 {total_samples}")
    logger.info(f"Valid Graphs Generated:                {valid_count}")
    logger.info(f"Invalid Molecules Skipped:             {skipped_count}")
    logger.info(f"Average Atoms (Nodes) per Molecule:    {avg_atoms}")
    logger.info(f"Average Bonds per Molecule:            {avg_bonds}")
    logger.info(f"Average Graph Size (Nodes + Edges):    {avg_graph_size}")
    logger.info(f"Max Graph Size (Nodes: {max_nodes}, Edges: {max_edges})")
    logger.info(f"Min Graph Size (Nodes: {min_nodes}, Edges: {min_edges})")
    logger.info(f"Positive (Toxic) Samples:              {pos_count} ({pos_pct}%)")
    logger.info(f"Negative (Non-Toxic) Samples:          {neg_count} ({neg_pct}%)")
    logger.info("==================================================================")

    # Return dictionary structured for dataset_info.json
    return {
        "dataset": dataset_name,
        "target": target_column,
        "total_samples": total_samples,
        "valid_graphs": valid_count,
        "invalid_graphs": skipped_count,
        "average_atoms": avg_atoms,
        "average_edges": avg_edges,
        "positive": pos_count,
        "negative": neg_count,
    }

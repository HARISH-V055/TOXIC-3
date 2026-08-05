"""
Molecular Visualization Module for EQ-KA-GCN

Provides functions to depict 2D structures of molecules from SMILES representations
and print structure summaries of PyTorch Geometric graphs.
"""

import logging
from typing import Optional
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data

# Use child logger under the main project logger
logger = logging.getLogger("EQ-KA-GCN.graph.visualizer")


def draw_molecule(smiles: str) -> Optional[Image.Image]:
    """
    Renders a 2D image of a molecule from its SMILES string.

    Args:
        smiles (str): SMILES string representation.

    Returns:
        Optional[Image.Image]: PIL Image object of the molecule, or None if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Unable to draw molecule: invalid SMILES '{smiles}'")
            return None
        
        # Depict 2D image
        img = Draw.MolToImage(mol)
        return img
    except Exception as e:
        logger.error(f"Failed to generate molecule image: {str(e)}")
        return None


def print_graph_info(graph: Data) -> None:
    """
    Logs metadata and structural parameters of a PyTorch Geometric Graph object.

    Args:
        graph (Data): The PyTorch Geometric graph Data object.
    """
    if graph is None:
        logger.warning("Cannot print graph info: Graph object is None.")
        return

    # Extract information
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    feature_dim = graph.num_node_features
    label = graph.y.item() if graph.y is not None else "N/A"

    logger.info("==================================================================")
    logger.info("MOLECULAR GRAPH REPRESENTATION SUMMARY")
    logger.info("==================================================================")
    logger.info(f"Number of Nodes (Atoms):       {num_nodes}")
    logger.info(f"Number of Edges (Directed):    {num_edges}")
    logger.info(f"Feature Dimension (Node):      {feature_dim}")
    logger.info(f"Toxicity Label:                {label}")
    logger.info("==================================================================")

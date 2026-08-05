"""
Graph Builder Module for EQ-KA-GCN

Parses molecular SMILES representations using RDKit and constructs
undirected graph objects compatible with PyTorch Geometric.
"""

import logging
from typing import Optional
import torch
from torch_geometric.data import Data
from rdkit import Chem

from graph.feature_extractor import extract_node_features

# Use child logger under the main project logger
logger = logging.getLogger("EQ-KA-GCN.graph.graph_builder")


def smiles_to_graph(smiles: str, label: int) -> Optional[Data]:
    """
    Parses a SMILES string, extracts node features and bond connectivity,
    and returns a PyTorch Geometric Data object representing the molecule.

    Args:
        smiles (str): SMILES string representation of the molecule.
        label (int): Target classification label for toxicity.

    Returns:
        Optional[Data]: PyTorch Geometric Data object with attributes:
                        - x: Node feature tensor [num_atoms, 5]
                        - edge_index: Graph connectivity tensor [2, 2 * num_bonds]
                        - y: Label tensor [1]
                        Returns None if the SMILES string is invalid or cannot be parsed.
    """
    # 1. Parse SMILES
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        logger.warning(f"Exception raised parsing SMILES '{smiles}': {str(e)}")
        return None

    # 2. Validate parsed molecule
    if mol is None:
        logger.warning(f"Invalid SMILES string: '{smiles}'. RDKit failed to parse.")
        return None

    # 3. Extract node features
    x = extract_node_features(mol)

    # 4. Extract undirected edge indices (bond connectivity)
    edge_list = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        # Add both directions to represent undirected connectivity
        edge_list.append((u, v))
        edge_list.append((v, u))

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # Handle molecules with zero bonds (e.g. single-atom compounds like Methane with implicit H's only)
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 5. Construct label tensor
    y = torch.tensor([label], dtype=torch.long)

    # 6. Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data

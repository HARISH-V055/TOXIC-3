"""
Graph Builder Module for EQ-KA-GCN

Parses molecular SMILES representations using RDKit and constructs
undirected graph objects compatible with PyTorch Geometric.

Upgrade: Now includes 6-dimensional edge (bond) features:
    - Bond type one-hot (Single, Double, Triple, Aromatic)
    - Conjugation flag
    - Ring membership flag
"""

import logging
from typing import Optional
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem

from graph.feature_extractor import extract_node_features

# Use child logger under the main project logger
logger = logging.getLogger("EQ-KA-GCN.graph.graph_builder")


# Bond type vocabulary
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def get_bond_features(bond: Chem.Bond) -> list:
    """
    Extracts 6-dimensional features from a single RDKit bond.

    Features:
        Bond type one-hot : 4 dims (SINGLE, DOUBLE, TRIPLE, AROMATIC)
        Is Conjugated     : 1 dim
        Is In Ring        : 1 dim

    Args:
        bond (Chem.Bond): RDKit bond object.

    Returns:
        list: 6-dimensional bond feature list.
    """
    bt = bond.GetBondType()
    return [
        1.0 if bt == BOND_TYPES[0] else 0.0,   # SINGLE
        1.0 if bt == BOND_TYPES[1] else 0.0,   # DOUBLE
        1.0 if bt == BOND_TYPES[2] else 0.0,   # TRIPLE
        1.0 if bt == BOND_TYPES[3] else 0.0,   # AROMATIC
        1.0 if bond.GetIsConjugated() else 0.0, # Conjugated system
        1.0 if bond.IsInRing() else 0.0,        # In ring
    ]


def smiles_to_graph(smiles: str, label: Optional[object] = 0) -> Optional[Data]:
    """
    Parses a SMILES string, extracts node features, bond features, ECFP4 fingerprint,
    and connectivity, and returns a PyTorch Geometric Data object representing the molecule.

    Args:
        smiles (str): SMILES string representation of the molecule.
        label (Optional[object]): Target classification label(s). Can be a scalar (int/float),
                                  list of floats (multi-task), or numpy array.

    Returns:
        Optional[Data]: PyTorch Geometric Data object with attributes:
                        - x: Node feature tensor [num_atoms, 32]
                        - edge_index: Graph connectivity tensor [2, 2 * num_bonds]
                        - edge_attr: Bond feature tensor [2 * num_bonds, 6]
                        - fp: 1024-bit ECFP4 fingerprint tensor [1024]
                        - y: Label tensor [1] or [12]
                        - smiles: SMILES string representation
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

    # 3. Extract node (atom) features [N, 32]
    x = extract_node_features(mol)

    # 4. Extract undirected edge indices and bond features
    edge_list = []
    edge_feat_list = []

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_feats = get_bond_features(bond)

        # Add both directions (undirected graph)
        edge_list.append((u, v))
        edge_list.append((v, u))
        edge_feat_list.append(bond_feats)
        edge_feat_list.append(bond_feats)

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feat_list, dtype=torch.float)
    else:
        # Handle molecules with zero bonds (single-atom compounds)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)

    # 5. Extract 1024-bit Morgan ECFP4 fingerprint (radius=2)
    try:
        fp_arr = np.zeros((1024,), dtype=np.float32)
        bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        for bit_idx in bit_vect.GetOnBits():
            fp_arr[bit_idx] = 1.0
        fp = torch.tensor(fp_arr, dtype=torch.float).unsqueeze(0)  # Shape [1, 1024]
    except Exception as e:
        logger.warning(f"Error computing ECFP4 fingerprint for '{smiles}': {e}")
        fp = torch.zeros((1, 1024), dtype=torch.float)

    # 6. Construct label tensor (supports scalar or multi-task 12-task vector)
    if label is None:
        y = torch.tensor([0.0], dtype=torch.float)
    elif isinstance(label, (list, tuple)):
        y = torch.tensor(label, dtype=torch.float)
    elif hasattr(label, "tolist"):
        y = torch.tensor(label.tolist(), dtype=torch.float)
    elif isinstance(label, torch.Tensor):
        y = label.float()
    else:
        y = torch.tensor([float(label)], dtype=torch.float)

    # 7. Create PyTorch Geometric Data object with edge features and fingerprint
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, fp=fp, y=y, smiles=smiles)

    return data

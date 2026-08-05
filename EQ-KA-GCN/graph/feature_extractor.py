"""
Molecular Feature Extractor Module for EQ-KA-GCN

Extracts chemical node (atom) features from RDKit atom representations and formats
them as PyTorch float tensors.
"""

from typing import List
import torch
from rdkit import Chem


def get_atom_features(atom: Chem.Atom) -> List[float]:
    """
    Extracts numerical features from a single RDKit atom.

    Features:
        1. Atomic Number
        2. Atom Degree (number of bonded neighbors)
        3. Formal Charge
        4. Is Aromatic (1.0 if aromatic, 0.0 otherwise)
        5. Total Number of Hydrogens (implicit + explicit)

    Args:
        atom (Chem.Atom): RDKit atom object.

    Returns:
        List[float]: A list containing the 5 atom features.
    """
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        1.0 if atom.GetIsAromatic() else 0.0,
        float(atom.GetTotalNumHs()),
    ]


def extract_node_features(mol: Chem.Mol) -> torch.FloatTensor:
    """
    Extracts feature representations for all atoms in a molecule.

    Args:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        torch.FloatTensor: Tensor of shape [N, 5] where N is number of atoms in molecule,
                           and 5 is the size of the feature vector.
    """
    features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    return torch.tensor(features, dtype=torch.float)

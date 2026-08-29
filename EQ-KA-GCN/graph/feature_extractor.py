"""
Molecular Feature Extractor Module for EQ-KA-GCN

Extracts rich chemical node (atom) features from RDKit atom representations
and formats them as PyTorch float tensors.

Feature Expansion (5 → 32 dimensions):
    - One-hot atomic symbol (13 dims)
    - One-hot hybridization state (6 dims)
    - One-hot degree (6 dims)
    - Scalar chemical properties (7 dims)

Total: 32 node features per atom (upgraded from 5).
This richer representation exposes key toxicophore substructures
(e.g. nitro groups, quinones, reactive epoxides) that govern SR-p53 toxicity.
"""

from typing import List
import torch
from rdkit import Chem


# Atom symbol vocabulary
ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'Other']

# Hybridization vocabulary
HYBRIDIZATIONS = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER']

# Degree vocabulary (capped at 5)
DEGREES = [0, 1, 2, 3, 4, 5]


def get_atom_features(atom: Chem.Atom) -> List[float]:
    """
    Extracts 32-dimensional numerical features from a single RDKit atom.

    Features (32 total):
        One-hot Atom Symbol     : 13 dims (C, N, O, S, F, P, Cl, Br, I, B, Si, Se, Other)
        One-hot Hybridization   :  6 dims (SP, SP2, SP3, SP3D, SP3D2, OTHER)
        One-hot Degree          :  6 dims (0, 1, 2, 3, 4, 5)
        Formal Charge           :  1 dim
        Total Hydrogens         :  1 dim
        Is Aromatic             :  1 dim
        Is In Ring              :  1 dim
        Has Chiral Tag          :  1 dim
        Num Radical Electrons   :  1 dim
        Is H Donor              :  1 dim  (pharmacophore)

    Args:
        atom (Chem.Atom): RDKit atom object.

    Returns:
        List[float]: A list of 32 atom features.
    """
    # 1. One-hot Atom Symbol (13 dims)
    sym = atom.GetSymbol()
    atom_one_hot = [1.0 if sym == a else 0.0 for a in ATOM_SYMBOLS[:-1]]
    atom_one_hot.append(0.0 if sym in ATOM_SYMBOLS[:-1] else 1.0)  # 'Other'

    # 2. One-hot Hybridization (6 dims)
    hyb_str = str(atom.GetHybridization()).split('.')[-1]
    hyb_one_hot = [1.0 if hyb_str == h else 0.0 for h in HYBRIDIZATIONS[:-1]]
    hyb_one_hot.append(0.0 if hyb_str in HYBRIDIZATIONS[:-1] else 1.0)  # 'OTHER'

    # 3. One-hot Degree (6 dims, capped at 5)
    degree = min(atom.GetDegree(), 5)
    deg_one_hot = [1.0 if degree == d else 0.0 for d in DEGREES]

    # 4. Scalar Chemical Features (7 dims)
    scalar_features = [
        float(atom.GetFormalCharge()),               # Formal charge (e.g. -1, 0, +1)
        float(atom.GetTotalNumHs()),                  # Total hydrogens (implicit + explicit)
        1.0 if atom.GetIsAromatic() else 0.0,         # Aromatic flag (ring systems)
        1.0 if atom.IsInRing() else 0.0,              # Ring membership
        1.0 if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0.0,  # Chirality
        float(atom.GetNumRadicalElectrons()),         # Radical electrons (reactive species)
        float(atom.GetNoImplicit()),                  # No implicit H flag
    ]

    return atom_one_hot + hyb_one_hot + deg_one_hot + scalar_features  # 13+6+6+7 = 32 features


def extract_node_features(mol: Chem.Mol) -> torch.FloatTensor:
    """
    Extracts 32-dimensional feature representations for all atoms in a molecule.

    Args:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        torch.FloatTensor: Tensor of shape [N, 32] where N is number of atoms.
    """
    features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    return torch.tensor(features, dtype=torch.float)

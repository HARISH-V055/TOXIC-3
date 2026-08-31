"""
Atom & Bond Importance Ranking Module for EQ-KA-GCN

Maps atomic representations and RDKit structures to chemical element names,
parses GNNExplainer masks, and ranks top-k influential atoms and chemical bonds
with explicit directional influence (Toxicity Drivers vs Safety/Non-Toxicity Stabilizers).
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem

logger = logging.getLogger("EQ-KA-GCN.explainability.atom_importance")

ATOM_SYMBOLS_LIST = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'Other']

ELEMENT_NAMES: Dict[str, str] = {
    "H": "Hydrogen",
    "C": "Carbon",
    "N": "Nitrogen",
    "O": "Oxygen",
    "S": "Sulfur",
    "F": "Fluorine",
    "P": "Phosphorus",
    "Cl": "Chlorine",
    "Br": "Bromine",
    "I": "Iodine",
    "B": "Boron",
    "Si": "Silicon",
    "Se": "Selenium",
    "Other": "Heteroatom",
}

ATOMIC_NUMBERS: Dict[str, int] = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Se": 34,
    "Br": 35,
    "I": 53,
    "Other": 0,
}


def decode_atom_info(
    feat_vector: torch.Tensor,
    atom_idx: int,
    mol: Optional[Chem.Mol] = None,
) -> Tuple[str, str, int]:
    """
    Decodes atom information accurately from RDKit molecule or one-hot feature vector.

    Args:
        feat_vector (torch.Tensor): 32-dimensional node feature vector.
        atom_idx (int): 0-indexed position of the atom.
        mol (Optional[Chem.Mol]): RDKit molecule if available.

    Returns:
        Tuple[str, str, int]: (Element symbol, Element name, Atomic number).
    """
    if mol is not None and 0 <= atom_idx < mol.GetNumAtoms():
        rd_atom = mol.GetAtomWithIdx(atom_idx)
        sym = rd_atom.GetSymbol()
        num = rd_atom.GetAtomicNum()
        name = ELEMENT_NAMES.get(sym, f"Element {sym}")
        return sym, name, num

    # Decode from 13-dim one-hot vector (features 0..12)
    one_hot = feat_vector[:13].cpu().numpy() if isinstance(feat_vector, torch.Tensor) else np.array(feat_vector[:13])
    symbol_idx = int(np.argmax(one_hot))
    if symbol_idx < len(ATOM_SYMBOLS_LIST):
        sym = ATOM_SYMBOLS_LIST[symbol_idx]
    else:
        sym = "C"

    name = ELEMENT_NAMES.get(sym, f"Element {sym}")
    num = ATOMIC_NUMBERS.get(sym, 0)
    return sym, name, num


def generate_atom_mechanism(
    symbol: str,
    atom_idx: int,
    score: float,
    is_toxic: bool,
    endpoint: str = "Toxicity",
) -> str:
    """
    Generates a concise chemical mechanistic explanation for an atom's directional influence.
    """
    if is_toxic:
        if symbol in ["O", "N"]:
            return f"Electrophilic / hydrogen-bonding {symbol} center acting as a primary toxicophore driver for {endpoint}."
        elif symbol in ["Cl", "F", "Br", "I"]:
            return f"Halogen ({symbol}) substituent enhancing lipophilicity and metabolic reactive partitioning."
        elif symbol in ["S", "P"]:
            return f"Reactive {symbol} center susceptible to metabolic bioactivation and oxidative stress."
        elif symbol == "C":
            return f"Core carbon framework scaffolding the active toxicophoric pharmacophore."
        return f"Key atomic center with high attribution score ({score:.3f}) elevating toxicity risk."
    else:
        if symbol == "C":
            return f"Stable carbon scaffold maintaining non-reactive, metabolically benign conformation."
        elif symbol in ["O", "N"]:
            return f"Polar {symbol} group promoting aqueous solubility and safe physiological clearance."
        elif symbol in ["S", "P"]:
            return f"Stable {symbol} moiety incorporated into a non-toxic, shielded chemical environment."
        return f"Stabilizing atomic feature with high attribution score ({score:.3f}) confirming non-toxic safety."


def generate_bond_mechanism(
    u_sym: str,
    v_sym: str,
    score: float,
    is_toxic: bool,
) -> str:
    """
    Generates a concise explanation for a chemical bond's directional influence.
    """
    bond_str = f"{u_sym}-{v_sym}"
    if is_toxic:
        if "O" in [u_sym, v_sym] or "N" in [u_sym, v_sym]:
            return f"Polar {bond_str} linkage facilitating toxicophore receptor interaction (Score: {score:.3f})."
        return f"Reactive {bond_str} bond pathway propagating toxicity attention weights (Score: {score:.3f})."
    else:
        return f"Stabilizing {bond_str} covalent bond maintaining structural integrity and safety (Score: {score:.3f})."


def rank_atom_importance(
    graph: Data,
    node_importance: np.ndarray,
    edge_importance: np.ndarray,
    top_k_atoms: int = 5,
    top_k_bonds: int = 5,
    mol: Optional[Chem.Mol] = None,
    prediction_label: str = "Non-Toxic",
    endpoint: str = "Tox21 Assay",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Ranks atoms and chemical bonds according to GNNExplainer importance scores,
    annotating whether they act as Toxicity Drivers or Safety / Non-Toxicity Stabilizers.

    Args:
        graph (Data): PyTorch Geometric graph containing node features x and edge_index.
        node_importance (np.ndarray): Node importance array [num_nodes].
        edge_importance (np.ndarray): Edge importance array [num_edges].
        top_k_atoms (int): Number of top atoms to retrieve.
        top_k_bonds (int): Number of top bonds to retrieve.
        mol (Optional[Chem.Mol]): RDKit molecule instance.
        prediction_label (str): "Toxic" or "Non-Toxic".
        endpoint (str): Target endpoint name.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            - Top-k atom rankings with directional influence metadata.
            - Top-k bond rankings with directional influence metadata.
    """
    x = graph.x
    edge_index = graph.edge_index
    num_nodes = x.size(0)

    is_toxic = "toxic" in prediction_label.lower() and "non-toxic" not in prediction_label.lower()
    influence_type = "Toxicity" if is_toxic else "Non-Toxicity"
    atom_role = "Toxicity Driver (Toxicophore)" if is_toxic else "Safety / Non-Toxicity Stabilizer"
    bond_role = "Toxicity-Propagating Bond" if is_toxic else "Structural Safety Stabilizer"

    # 1. Rank Atoms
    atom_records = []
    for idx in range(num_nodes):
        sym, name, atomic_num = decode_atom_info(x[idx], idx, mol=mol)
        score = float(node_importance[idx])
        mechanism = generate_atom_mechanism(sym, idx, score, is_toxic, endpoint=endpoint)

        atom_records.append({
            "atom_index": idx,
            "atomic_number": atomic_num,
            "atom_symbol": sym,
            "atom_name": name,
            "importance_score": score,
            "score": round(score, 4),
            "influence_type": influence_type,
            "role": atom_role,
            "description": mechanism,
        })

    # Sort descending by importance score
    atom_records.sort(key=lambda r: r["importance_score"], reverse=True)

    ranked_atoms = []
    for rank_idx, rec in enumerate(atom_records[:top_k_atoms], start=1):
        rec["rank"] = rank_idx
        ranked_atoms.append(rec)

    # 2. Rank Chemical Bonds (Map directed edges u->v and v->u to undirected bond scores)
    pair_scores: Dict[Tuple[int, int], List[float]] = {}
    if edge_index is not None and edge_index.size(1) > 0:
        for edge_i in range(edge_index.size(1)):
            u = int(edge_index[0][edge_i].item())
            v = int(edge_index[1][edge_i].item())
            pair = tuple(sorted((u, v)))
            if pair not in pair_scores:
                pair_scores[pair] = []
            if edge_i < len(edge_importance):
                pair_scores[pair].append(float(edge_importance[edge_i]))

    bond_records = []
    for (u, v), scores in pair_scores.items():
        avg_score = float(np.mean(scores)) if scores else 0.5
        u_sym, _, _ = decode_atom_info(x[u], u, mol=mol)
        v_sym, _, _ = decode_atom_info(x[v], v, mol=mol)

        bond_label = f"{u_sym}(#{u}) — {v_sym}(#{v})"
        mechanism = generate_bond_mechanism(u_sym, v_sym, avg_score, is_toxic)

        bond_records.append({
            "source": u,
            "target": v,
            "u": u,
            "v": v,
            "u_symbol": u_sym,
            "v_symbol": v_sym,
            "bond_name": bond_label,
            "importance_score": avg_score,
            "score": round(avg_score, 4),
            "influence_type": influence_type,
            "role": bond_role,
            "description": mechanism,
        })

    bond_records.sort(key=lambda r: r["importance_score"], reverse=True)

    ranked_bonds = []
    for rank_idx, rec in enumerate(bond_records[:top_k_bonds], start=1):
        rec["rank"] = rank_idx
        ranked_bonds.append(rec)

    top_atom_desc = f"{ranked_atoms[0]['atom_name']} #{ranked_atoms[0]['atom_index']}" if ranked_atoms else "None"
    logger.info(
        f"Atom and bond importance ranking complete ({influence_type}). Top atom: {top_atom_desc} "
        f"(Score: {ranked_atoms[0]['importance_score']:.4f} - {atom_role})"
    )

    return ranked_atoms, ranked_bonds


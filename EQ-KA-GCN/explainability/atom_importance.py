"""
Atom & Bond Importance Ranking Module for EQ-KA-GCN

Maps atomic numbers to chemical element names, parses GNNExplainer masks,
and ranks top-k influential atoms and chemical bonds for model toxicity explanations.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger("EQ-KA-GCN.explainability.atom_importance")

ATOM_SYMBOLS: Dict[int, str] = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}

ATOM_NAMES: Dict[int, str] = {
    1: "Hydrogen",
    5: "Boron",
    6: "Carbon",
    7: "Nitrogen",
    8: "Oxygen",
    9: "Fluorine",
    15: "Phosphorus",
    16: "Sulfur",
    17: "Chlorine",
    35: "Bromine",
    53: "Iodine",
}


def decode_atom_info(atomic_num: int) -> Tuple[str, str]:
    """
    Decodes atomic number to element symbol and full element name.

    Args:
        atomic_num (int): Atomic number (e.g. 6 for Carbon).

    Returns:
        Tuple[str, str]: (Element symbol, Element name).
    """
    symbol = ATOM_SYMBOLS.get(atomic_num, f"Elem{atomic_num}")
    name = ATOM_NAMES.get(atomic_num, f"Atomic Number {atomic_num}")
    return symbol, name


def rank_atom_importance(
    graph: Data,
    node_importance: np.ndarray,
    edge_importance: np.ndarray,
    top_k_atoms: int = 5,
    top_k_bonds: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Ranks atoms and chemical bonds according to GNNExplainer importance scores.

    Args:
        graph (Data): PyTorch Geometric graph containing node features x and edge_index.
        node_importance (np.ndarray): Node importance array [num_nodes].
        edge_importance (np.ndarray): Edge importance array [num_edges].
        top_k_atoms (int): Number of top atoms to retrieve.
        top_k_bonds (int): Number of top bonds to retrieve.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            - Top-k atom rankings list.
            - Top-k bond rankings list.
    """
    x = graph.x
    edge_index = graph.edge_index
    num_nodes = x.size(0)

    # 1. Rank Atoms
    atom_records = []
    for idx in range(num_nodes):
        atomic_num = int(x[idx][0].item())
        symbol, name = decode_atom_info(atomic_num)
        score = float(node_importance[idx])

        atom_records.append({
            "atom_index": idx,
            "atomic_number": atomic_num,
            "atom_symbol": symbol,
            "atom_name": name,
            "importance_score": score,
        })

    # Sort descending by importance score
    atom_records.sort(key=lambda r: r["importance_score"], reverse=True)

    ranked_atoms = []
    for rank_idx, rec in enumerate(atom_records[:top_k_atoms], start=1):
        rec["rank"] = rank_idx
        ranked_atoms.append(rec)

    # 2. Rank Chemical Bonds
    bond_seen = set()
    bond_records = []

    for edge_i in range(edge_index.size(1)):
        u = int(edge_index[0][edge_i])
        v = int(edge_index[1][edge_i])
        pair = tuple(sorted((u, v)))

        if pair in bond_seen:
            continue
        bond_seen.add(pair)

        score = float(edge_importance[edge_i])
        u_atomic = int(x[u][0].item())
        v_atomic = int(x[v][0].item())

        u_sym, _ = decode_atom_info(u_atomic)
        v_sym, _ = decode_atom_info(v_atomic)

        bond_label = f"{u_sym}({u}) -- {v_sym}({v})"

        bond_records.append({
            "u": u,
            "v": v,
            "u_symbol": u_sym,
            "v_symbol": v_sym,
            "bond_name": bond_label,
            "importance_score": score,
        })

    bond_records.sort(key=lambda r: r["importance_score"], reverse=True)

    ranked_bonds = []
    for rank_idx, rec in enumerate(bond_records[:top_k_bonds], start=1):
        rec["rank"] = rank_idx
        ranked_bonds.append(rec)

    logger.info(
        f"Atom and bond importance ranking complete. Top atom: {ranked_atoms[0]['atom_name']} "
        f"(Index {ranked_atoms[0]['atom_index']}, Score: {ranked_atoms[0]['importance_score']:.4f})"
    )

    return ranked_atoms, ranked_bonds

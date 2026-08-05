"""
Molecular Explanation Visualization Module for EQ-KA-GCN

Provides functions to generate RDKit / Matplotlib molecular graph visualisations,
highlighting influential atoms and bonds, and saving 300 DPI publication plots.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data

from explainability.atom_importance import decode_atom_info

logger = logging.getLogger("EQ-KA-GCN.explainability.visualization")


def visualize_molecule_explanation(
    smiles: str,
    node_importance: np.ndarray,
    edge_importance: np.ndarray,
    save_path: str = "outputs/explanations/molecule_explanation.png",
) -> None:
    """
    Renders molecular structure highlighting atom importance using RDKit or Matplotlib.

    Args:
        smiles (str): SMILES representation of the molecule.
        node_importance (np.ndarray): Array of atom importance scores [num_nodes].
        edge_importance (np.ndarray): Array of bond importance scores [num_edges].
        save_path (str): Target file path for the output PNG image.
    """
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            # Prepare RDKit atom highlights
            num_atoms = mol.GetNumAtoms()
            atom_colors = {}
            highlight_atoms = []

            for idx in range(min(num_atoms, len(node_importance))):
                score = float(node_importance[idx])
                if score > 0.4:
                    highlight_atoms.append(idx)
                    # Color map from yellow/orange to dark red
                    color = (1.0, 1.0 - score * 0.7, 1.0 - score)
                    atom_colors[idx] = color

            img = Draw.MolToImage(
                mol,
                size=(600, 600),
                highlightAtoms=highlight_atoms,
                highlightColor=(1.0, 0.4, 0.4),
            )
            img.save(out_file)
            logger.info(f"Saved RDKit molecular explanation plot to: {out_file}")
            return
        except Exception as e:
            logger.warning(f"RDKit rendering fallback to Matplotlib graph plot: {e}")

    # Matplotlib NetworkX fallback rendering
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.set_title(f"Molecular Toxicity Explanation ({smiles})", fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_explainability_figures(
    graph: Data,
    node_importance: np.ndarray,
    edge_importance: np.ndarray,
    top_atoms: List[Dict[str, Any]],
    top_bonds: List[Dict[str, Any]],
    output_dir: str = "outputs/figures",
) -> None:
    """
    Generates publication-quality 300 DPI figures for explainability analysis:
      1. atom_importance_heatmap.png
      2. edge_importance_heatmap.png
      3. molecule_explanation.png

    Args:
        graph (Data): PyG Data object.
        node_importance (np.ndarray): Array of atom importance scores.
        edge_importance (np.ndarray): Array of bond importance scores.
        top_atoms (List[Dict[str, Any]]): Top-k ranked atoms list.
        top_bonds (List[Dict[str, Any]]): Top-k ranked bonds list.
        output_dir (str): Output directory for plots.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Atom Importance Heatmap Plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    atom_labels = [
        f"{rec['atom_symbol']}({rec['atom_index']})" for rec in top_atoms
    ]
    atom_scores = [rec["importance_score"] for rec in top_atoms]

    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.95, len(atom_scores)))
    bars = ax.bar(atom_labels, atom_scores, color=colors, edgecolor="black", width=0.5)

    ax.set_xlabel("Atom (Element & Index)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Importance Score", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("Atom Importance Ranking (GNNExplainer)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plot1_path = out_path / "atom_importance_heatmap.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved figure: {plot1_path}")

    # 2. Edge Importance Heatmap Plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bond_labels = [rec["bond_name"] for rec in top_bonds]
    bond_scores = [rec["importance_score"] for rec in top_bonds]

    bond_colors = plt.cm.plasma(np.linspace(0.4, 0.9, len(bond_scores)))
    bars = ax.bar(bond_labels, bond_scores, color=bond_colors, edgecolor="black", width=0.5)

    ax.set_xlabel("Chemical Bond (u -- v)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Importance Score", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("Chemical Bond Importance Ranking (GNNExplainer)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plot2_path = out_path / "edge_importance_heatmap.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved figure: {plot2_path}")

    # 3. Network Structure Graph Visualization Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    G = nx.Graph()

    x = graph.x
    edge_index = graph.edge_index
    num_nodes = x.size(0)

    for i in range(num_nodes):
        atomic_num = int(x[i][0].item())
        sym, _ = decode_atom_info(atomic_num)
        G.add_node(i, label=f"{sym}{i}", score=node_importance[i])

    seen_e = set()
    for e_i in range(edge_index.size(1)):
        u = int(edge_index[0][e_i])
        v = int(edge_index[1][e_i])
        pair = tuple(sorted((u, v)))
        if pair in seen_e:
            continue
        seen_e.add(pair)
        G.add_edge(u, v, weight=edge_importance[e_i])

    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n]["score"] for n in G.nodes()]
    node_sizes = [300 + 700 * G.nodes[n]["score"] for n in G.nodes()]

    nodes_drawn = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, cmap=plt.cm.YlOrRd, node_size=node_sizes, edgecolors="black"
    )
    nx.draw_networkx_edges(G, pos, ax=ax, width=2.0, alpha=0.7, edge_color="gray")
    node_labels = {n: G.nodes[n]["label"] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=10, font_weight="bold")

    cbar = plt.colorbar(nodes_drawn, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Atom Importance Score", fontsize=11, fontweight="bold")

    ax.set_title("GNNExplainer Molecular Subgraph Explanation", fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")

    plt.tight_layout()
    plot3_path = out_path / "molecule_explanation.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved figure: {plot3_path}")

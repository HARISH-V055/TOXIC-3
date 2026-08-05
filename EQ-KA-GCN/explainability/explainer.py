"""
GNNExplainer Module for EQ-KA-GCN

Implements GNNExplainer using PyTorch and PyTorch Geometric logic to compute
atom (node) and bond (edge) importance masks by maximizing mutual information.
"""

import logging
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

logger = logging.getLogger("EQ-KA-GCN.explainability.explainer")


class GNNExplainerModule:
    """
    Wrapper for GNNExplainer to extract node feature masks and edge importance masks
    for individual molecular graph toxicity predictions.
    """

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.01,
        feat_mask_type: str = "feature",
        num_hops: int = 2,
    ) -> None:
        """
        Initializes the GNNExplainerModule.

        Args:
            epochs (int): Number of optimization iterations.
            lr (float): Learning rate for mask optimization.
            feat_mask_type (str): Type of node feature mask.
            num_hops (int): Computation graph hop distance.
        """
        self.epochs = epochs
        self.lr = lr
        self.feat_mask_type = feat_mask_type
        self.num_hops = num_hops

    def explain_graph(
        self,
        model: nn.Module,
        graph: Data,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs GNNExplainer optimization on a molecular graph to extract node and edge importance masks.

        Args:
            model (nn.Module): Trained GCN / KA-GCN model.
            graph (Data): PyTorch Geometric Data graph instance.
            device (torch.device): Computing device.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Node importance scores array of shape [num_nodes].
                - Edge importance scores array of shape [num_edges].
        """
        logger.info(f"Running GNNExplainer on molecular graph with {graph.num_nodes} nodes and {graph.num_edges} edges...")
        model.eval()

        x = graph.x.to(device).float()
        edge_index = graph.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        # 1. Forward pass to get target class prediction logit
        with torch.no_grad():
            orig_logits = model(x=x, edge_index=edge_index, batch=batch, return_logits=True)
            orig_prob = torch.sigmoid(orig_logits).item()
            target_class = 1 if orig_prob >= 0.5 else 0

        num_nodes, num_features = x.size()
        num_edges = edge_index.size(1)

        # 2. Initialize learnable mask parameters
        edge_mask_param = nn.Parameter(torch.randn(num_edges, device=device) * 0.1)
        node_mask_param = nn.Parameter(torch.randn(num_nodes, num_features, device=device) * 0.1)

        optimizer = torch.optim.Adam([edge_mask_param, node_mask_param], lr=self.lr)

        # Loss weights
        size_weight = 0.005
        entropy_weight = 1.0

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Sigmoid activation maps parameters to [0, 1] masks
            edge_mask = torch.sigmoid(edge_mask_param)
            node_mask = torch.sigmoid(node_mask_param)

            # Apply node mask to input features
            x_masked = x * node_mask

            # Forward pass with masked features and edge weights
            # Pass edge_mask as edge weights if model supports or via edge_index weighting
            try:
                masked_logits = model(
                    x=x_masked,
                    edge_index=edge_index,
                    batch=batch,
                    return_logits=True,
                    edge_weight=edge_mask,
                )
            except TypeError:
                # If model forward does not take edge_weight directly, pass x_masked
                masked_logits = model(
                    x=x_masked,
                    edge_index=edge_index,
                    batch=batch,
                    return_logits=True,
                )

            masked_prob = torch.sigmoid(masked_logits)

            # Target prediction loss (cross-entropy or log-loss with target class)
            if target_class == 1:
                pred_loss = -torch.log(masked_prob + 1e-8)
            else:
                pred_loss = -torch.log(1.0 - masked_prob + 1e-8)

            # Regularization losses
            edge_size_loss = torch.sum(edge_mask)
            node_size_loss = torch.sum(node_mask)

            edge_entropy_loss = -torch.sum(
                edge_mask * torch.log(edge_mask + 1e-8)
                + (1 - edge_mask) * torch.log(1 - edge_mask + 1e-8)
            )

            total_loss = (
                pred_loss.squeeze()
                + size_weight * (edge_size_loss + node_size_loss)
                + entropy_weight * edge_entropy_loss
            )

            total_loss.backward()
            optimizer.step()

        # 3. Final Importance Scores Extraction
        with torch.no_grad():
            final_edge_mask = torch.sigmoid(edge_mask_param).cpu().numpy()
            final_node_mask = torch.sigmoid(node_mask_param).mean(dim=-1).cpu().numpy()

        # Normalize importance scores to range [0, 1]
        node_max = np.max(final_node_mask) if np.max(final_node_mask) > 0 else 1.0
        edge_max = np.max(final_edge_mask) if np.max(final_edge_mask) > 0 else 1.0

        node_importance = (final_node_mask / node_max).clip(0.0, 1.0)
        edge_importance = (final_edge_mask / edge_max).clip(0.0, 1.0)

        logger.info(
            f"GNNExplainer complete. Node importance range: [{node_importance.min():.3f}, {node_importance.max():.3f}] | "
            f"Edge importance range: [{edge_importance.min():.3f}, {edge_importance.max():.3f}]"
        )

        return node_importance, edge_importance

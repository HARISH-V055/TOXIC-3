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
    GNNExplainer module for extracting node feature masks and edge importance masks
    for individual molecular graph toxicity predictions on trained KA-GCN models.
    """

    def __init__(
        self,
        epochs: int = 40,
        lr: float = 0.01,
        threshold: float = 0.75,
        seed: int = 42,
    ) -> None:
        """
        Initializes GNNExplainerModule.

        Args:
            epochs (int): Number of mask optimization epochs.
            lr (float): Learning rate for mask optimizer.
            threshold (float): Optimized decision threshold for toxicity classification (default 0.75).
            seed (int): Random seed for reproducible mask initialization.
        """
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold
        self.seed = seed

    def explain_graph(
        self,
        model: nn.Module,
        graph: Data,
        device: torch.device,
        target_idx: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs GNNExplainer optimization on a molecular graph using edge_weight tensor.

        Args:
            model (nn.Module): Trained GCN / KA-GCN model instance.
            graph (Data): PyTorch Geometric Data graph instance.
            device (torch.device): Computing device (cpu or cuda).
            target_idx (int): Endpoint index to explain if multi-task (default: -1 for SR-p53).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Node importance scores array of shape [num_nodes].
                - Edge importance scores array of shape [num_edges].
        """
        logger.info(
            f"Running GNNExplainer on molecular graph ({graph.num_nodes} nodes, {graph.num_edges} edges, threshold {self.threshold}, seed {self.seed}, target_idx {target_idx})..."
        )

        # 1. Freeze Model Parameters & Set Evaluation Mode
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Set controlled seeds for reproducible mask initialization
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if graph is None or graph.x is None or graph.edge_index is None:
            raise ValueError("Invalid PyG Data object passed to explain_graph: 'x' and 'edge_index' tensors must be non-None.")

        x: torch.Tensor = graph.x.to(device).float()
        edge_index: torch.Tensor = graph.edge_index.to(device)
        batch: torch.Tensor = torch.zeros(x.size(0), dtype=torch.long, device=device)

        # 2. Get Original Model Prediction with Configured Threshold
        with torch.no_grad():
            try:
                orig_logits = model(x=x, edge_index=edge_index, batch=batch, return_logits=True)
            except Exception as e:
                logger.error(f"Failed forward pass during GNNExplainer initialization: {e}")
                raise RuntimeError(f"Model forward pass failed: {e}") from e

            if orig_logits.numel() > 1:
                t_idx = target_idx if target_idx >= 0 else (orig_logits.shape[-1] - 1)
                orig_prob = torch.sigmoid(orig_logits).squeeze()[t_idx].item()
            else:
                orig_prob = torch.sigmoid(orig_logits).squeeze().item()
            target_class = 1 if orig_prob >= self.threshold else 0

        num_nodes, num_features = x.size()
        num_edges = edge_index.size(1)

        # 3. Initialize Learnable Mask Parameters
        edge_mask_param = nn.Parameter(torch.randn(num_edges, device=device) * 0.1)
        node_mask_param = nn.Parameter(torch.randn(num_nodes, num_features, device=device) * 0.1)

        optimizer = torch.optim.Adam([edge_mask_param, node_mask_param], lr=self.lr)

        size_weight = 0.005
        entropy_weight = 0.1

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            edge_mask = torch.sigmoid(edge_mask_param)
            node_mask = torch.sigmoid(node_mask_param)

            x_masked = x * node_mask

            # Pass edge_mask as edge_weight into model forward pass (NO TypeError fallback)
            try:
                masked_logits = model(
                    x=x_masked,
                    edge_index=edge_index,
                    batch=batch,
                    return_logits=True,
                    edge_weight=edge_mask,
                )
            except TypeError as te:
                logger.error(f"Model forward pass does not support edge_weight parameter: {te}")
                raise ValueError("Model does not support edge_weight parameter in forward pass.") from te

            if masked_logits.numel() > 1:
                t_idx = target_idx if target_idx >= 0 else (masked_logits.shape[-1] - 1)
                masked_prob = torch.sigmoid(masked_logits).squeeze()[t_idx]
            else:
                masked_prob = torch.sigmoid(masked_logits).squeeze()

            # Target prediction log-loss objective
            if target_class == 1:
                pred_loss = -torch.log(masked_prob.clamp(1e-7, 1.0 - 1e-7))
            else:
                pred_loss = -torch.log((1.0 - masked_prob).clamp(1e-7, 1.0 - 1e-7))

            # Sparsity Regularization
            edge_size_loss = torch.sum(edge_mask)
            node_size_loss = torch.sum(node_mask)

            # Entropy Regularization
            edge_entropy = -torch.sum(
                edge_mask * torch.log(edge_mask + 1e-7) + (1.0 - edge_mask) * torch.log(1.0 - edge_mask + 1e-7)
            )
            node_entropy = -torch.sum(
                node_mask * torch.log(node_mask + 1e-7) + (1.0 - node_mask) * torch.log(1.0 - node_mask + 1e-7)
            )

            total_loss = pred_loss + size_weight * (edge_size_loss + node_size_loss) + entropy_weight * (edge_entropy + node_entropy)

            total_loss.backward()
            optimizer.step()

        # 4. Extract Final Importance Scores
        with torch.no_grad():
            final_edge_mask = torch.sigmoid(edge_mask_param).cpu().numpy()
            final_node_mask = torch.sigmoid(node_mask_param).mean(dim=-1).cpu().numpy()

        # Relative Normalization
        node_max = float(np.max(final_node_mask)) if np.max(final_node_mask) > 0 else 1.0
        edge_max = float(np.max(final_edge_mask)) if np.max(final_edge_mask) > 0 else 1.0

        node_importance = (final_node_mask / node_max).clip(0.0, 1.0)
        edge_importance = (final_edge_mask / edge_max).clip(0.0, 1.0)

        logger.info(
            f"GNNExplainer complete. Node relative importance range: [{node_importance.min():.4f}, {node_importance.max():.4f}] | "
            f"Edge relative importance range: [{edge_importance.min():.4f}, {edge_importance.max():.4f}]"
        )

        return node_importance, edge_importance

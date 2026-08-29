"""
Baseline Graph Convolutional Network (GCN) Module for EQ-KA-GCN

Implements a standard GCN baseline model using PyTorch and PyTorch Geometric.
This architecture serves as a comparison benchmark for KA-GCN enhancements
such as Fourier-KAN, quantization, and explainability mechanisms.

Upgrades:
    - Multi-scale readout pooling (mean + max + sum concatenation)
    - Residual skip connection from GCN layer 1 → layer 2
"""

from typing import Optional
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool


class BaselineGCN(nn.Module):
    """
    Baseline Graph Convolutional Network (GCN) for binary graph classification.

    Architecture:
        Input Node Features
        ↓
        GCNConv (input_dim → hidden_dim)
        ↓
        BatchNorm1d + ReLU + Dropout
        ↓
        GCNConv (hidden_dim → hidden_dim)   ← residual skip connection
        ↓
        BatchNorm1d + ReLU
        ↓
        Multi-Scale Readout: cat([mean_pool, max_pool, sum_pool])
        ↓
        Linear projection (3×hidden_dim → hidden_dim)
        ↓
        Linear FC (hidden_dim → output_dim)
        ↓
        Sigmoid (optional for inference)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        dropout: float = 0.3,
    ) -> None:
        """
        Initializes GCN baseline layers.

        Args:
            input_dim (int): Dimensionality of input node features.
            hidden_dim (int): Dimensionality of GCN hidden layers.
            output_dim (int): Output dimension (typically 1 for binary classification).
            dropout (float): Dropout probability.
        """
        super().__init__()

        # First GCN Layer
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

        # Second GCN Layer with residual connection
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim)
        self.relu2 = nn.ReLU()

        # Multi-scale pooling projection: 3×hidden_dim → hidden_dim
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Fully Connected Layer (Classifier head)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_logits: bool = True,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs forward propagation on input molecular graphs.

        Args:
            x (torch.Tensor): Node feature tensor of shape [num_nodes, input_dim].
            edge_index (torch.Tensor): Graph edge indices of shape [2, num_edges].
            batch (torch.Tensor): Graph batch indicators of shape [num_nodes].
            return_logits (bool): If True, returns raw linear logits (for training with BCEWithLogitsLoss).
                                 If False, returns sigmoid-activated probabilities.
            edge_weight (Optional[torch.Tensor]): Optional edge weight tensor for GNNExplainer.

        Returns:
            torch.Tensor: Prediction values of shape [batch_size, output_dim].
        """
        # 1. First GCN Block
        h1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        h1 = self.bn1(h1)
        h1 = self.relu1(h1)
        h1 = self.drop(h1)

        # 2. Second GCN Block with residual skip connection
        h2 = self.conv2(h1, edge_index, edge_weight=edge_weight)
        h2 = self.bn2(h2)
        h2 = self.relu2(h2 + h1)  # ← Residual connection

        # 3. Multi-Scale Readout Pooling
        h_mean = global_mean_pool(h2, batch)
        h_max  = global_max_pool(h2, batch)
        h_sum  = global_add_pool(h2, batch)
        out = torch.cat([h_mean, h_max, h_sum], dim=-1)  # [B, 3H]

        # 4. Project to hidden_dim
        out = self.pool_proj(out)  # [B, H]

        # 5. Classification Projection
        logits = self.fc(out)

        # 6. Output activation logic
        if return_logits:
            return logits

        return torch.sigmoid(logits)

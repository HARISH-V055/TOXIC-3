"""
Kolmogorov-Arnold Graph Convolutional Network (KA-GCN) Module for EQ-KA-GCN

Implements KA-GCN by replacing the standard MLP classifier of the GCN baseline
with a Fourier-based Kolmogorov-Arnold Network (FourierKAN) classifier head.

Upgrades over Baseline:
    - Multi-scale readout pooling (mean + max + sum concatenation)
    - Residual skip connection from GCN layer 1 → layer 2
    - Richer 32-dim atom features + optional edge features
"""

from typing import Optional
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

from models.fourier_kan import FourierKANLayer


class KAGCN(nn.Module):
    """
    Kolmogorov-Arnold Graph Convolutional Network (KA-GCN) for binary graph classification.

    Architecture:
        Input Node Features (32 dims)
        ↓
        GCNConv (input_dim → hidden_dim)
        ↓
        BatchNorm1d + ReLU + Dropout
        ↓
        GCNConv (hidden_dim → hidden_dim)   ← residual: h2 = conv2(h1) + h1
        ↓
        BatchNorm1d + ReLU
        ↓
        Multi-Scale Readout: cat([global_mean_pool, global_max_pool, global_add_pool])
        ↓                       → [batch, 3 × hidden_dim]
        Linear projection (3×hidden_dim → hidden_dim)
        ↓
        FourierKAN Layer (hidden_dim → kan_hidden_dim)
        ↓
        Activation + Dropout
        ↓
        Linear Output Layer (kan_hidden_dim → output_dim)
        ↓
        Sigmoid (optional for inference)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        gcn_dropout: float = 0.3,
        kan_hidden_dim: int = 64,
        fourier_order: int = 5,
        kan_dropout: float = 0.2,
        kan_activation: str = "silu",
    ) -> None:
        """
        Initializes KA-GCN model layers.

        Args:
            input_dim (int): Dimensionality of input node features.
            hidden_dim (int): Dimensionality of GCN hidden layers.
            output_dim (int): Output dimension (1 for binary classification).
            gcn_dropout (float): Dropout probability for GCN layers.
            kan_hidden_dim (int): Hidden dimension for the FourierKAN classifier.
            fourier_order (int): Order/grid size of Fourier harmonics.
            kan_dropout (float): Dropout probability after FourierKAN layer.
            kan_activation (str): Activation function after FourierKAN layer.
        """
        super().__init__()

        # First GCN Layer
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout(p=gcn_dropout)

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

        # FourierKAN Layer
        self.fourier_kan = FourierKANLayer(
            in_features=hidden_dim,
            out_features=kan_hidden_dim,
            fourier_order=fourier_order,
        )

        act_lower = kan_activation.lower()
        self.kan_act: nn.Module
        if act_lower == "silu":
            self.kan_act = nn.SiLU()
        elif act_lower == "relu":
            self.kan_act = nn.ReLU()
        elif act_lower == "tanh":
            self.kan_act = nn.Tanh()
        elif act_lower == "gelu":
            self.kan_act = nn.GELU()
        else:
            self.kan_act = nn.Identity()

        self.kan_drop = nn.Dropout(p=kan_dropout)

        # Final Output Layer Projection
        self.fc_out = nn.Linear(in_features=kan_hidden_dim, out_features=output_dim)

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
            edge_weight (Optional[torch.Tensor]): Optional edge weight tensor of shape [num_edges] for GNNExplainer.

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

        # 3. Multi-Scale Readout Pooling (captures mean chemistry + local hotspots + size)
        h_mean = global_mean_pool(h2, batch)   # [B, H] average atom signature
        h_max  = global_max_pool(h2, batch)    # [B, H] most activated atom (toxic hotspot)
        h_sum  = global_add_pool(h2, batch)    # [B, H] size-scaled molecular fingerprint
        out = torch.cat([h_mean, h_max, h_sum], dim=-1)  # [B, 3H]

        # 4. Project back to hidden_dim
        out = self.pool_proj(out)  # [B, H]

        # 5. FourierKAN Classifier Transformation
        out = self.fourier_kan(out)
        out = self.kan_act(out)
        out = self.kan_drop(out)

        # 6. Output Projection Layer
        logits = self.fc_out(out)

        # 7. Output activation logic
        if return_logits:
            return logits

        return torch.sigmoid(logits)

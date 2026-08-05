"""
Baseline Graph Convolutional Network (GCN) Module for EQ-KA-GCN

Implements a standard GCN baseline model using PyTorch and PyTorch Geometric.
This architecture will serve as a comparison benchmark for future enhancements
such as Fourier-KAN, quantization, and explainability mechanisms.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class BaselineGCN(nn.Module):
    """
    Baseline Graph Convolutional Network (GCN) for binary graph classification.
    
    Architecture:
        Input Node Features
        ↓
        GCNConv (input_dim -> hidden_dim)
        ↓
        BatchNorm1d
        ↓
        ReLU
        ↓
        Dropout (rate)
        ↓
        GCNConv (hidden_dim -> hidden_dim)
        ↓
        BatchNorm1d
        ↓
        ReLU
        ↓
        Global Mean Pool
        ↓
        Linear FC (hidden_dim -> output_dim)
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

        # Second GCN Layer
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim)
        self.relu2 = nn.ReLU()

        # Fully Connected Layer (Classifier head)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, return_logits: bool = True
    ) -> torch.Tensor:
        """
        Performs forward propagation on input molecular graphs.

        Args:
            x (torch.Tensor): Node feature tensor of shape [num_nodes, input_dim].
            edge_index (torch.Tensor): Graph edge indices of shape [2, num_edges].
            batch (torch.Tensor): Graph batch indicators of shape [num_nodes].
            return_logits (bool): If True, returns raw linear logits (for training with BCEWithLogitsLoss).
                                 If False, returns sigmoid-activated probabilities.

        Returns:
            torch.Tensor: Prediction values of shape [batch_size, output_dim].
        """
        # 1. First GCN Block
        out = self.conv1(x, edge_index)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop(out)

        # 2. Second GCN Block
        out = self.conv2(out, edge_index)
        out = self.bn2(out)
        out = self.relu2(out)

        # 3. Global Mean Pooling (Node aggregation to graph level)
        out = global_mean_pool(out, batch)

        # 4. Classification Projection
        logits = self.fc(out)

        # 5. Output activation logic
        if return_logits:
            return logits
        
        return torch.sigmoid(logits)

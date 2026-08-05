"""
Fourier Kolmogorov-Arnold Network (FourierKAN) Module for EQ-KA-GCN

Provides PyTorch implementations of FourierKANLayer and FourierKAN.
Uses Fourier basis functions (sine and cosine harmonics) to parameterize 1D edge activation
functions in Kolmogorov-Arnold Networks, enabling smooth, fully differentiable, non-linear function approximation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class FourierKANLayer(nn.Module):
    """
    Fourier Kolmogorov-Arnold Network Layer.

    Replaces fixed activation functions with learnable 1D Fourier series expansions:
    phi(x) = w * x + sum_{k=1}^N (a_k * cos(k * x) + b_k * sin(k * x)) + bias

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        fourier_order (int): Order/grid size of Fourier harmonics (N). Default: 5.
        bias (bool): Whether to include a learnable bias term. Default: True.
    """

    k: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fourier_order: int = 5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fourier_order = fourier_order

        # 1. Base Linear Transformation Weights: [out_features, in_features]
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))

        # 2. Learnable Fourier Coefficients for Cosine and Sine Terms: [out_features, in_features, fourier_order]
        self.fourier_coeffs_cos = nn.Parameter(torch.empty(out_features, in_features, fourier_order))
        self.fourier_coeffs_sin = nn.Parameter(torch.empty(out_features, in_features, fourier_order))

        # 3. Optional Bias Term: [out_features]
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # 4. Harmonic Multipliers Buffer: [1, 1, fourier_order] containing [1.0, 2.0, ..., fourier_order]
        k_values = torch.arange(1, fourier_order + 1, dtype=torch.float32).reshape(1, 1, fourier_order)
        self.register_buffer("k", k_values)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initializes weights using Kaiming Uniform for linear base and decaying harmonic scaling for Fourier coefficients."""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

        # Decaying harmonic initialization for numerical stability at higher order frequencies
        scale = 1.0 / (self.in_features * torch.sqrt(self.k))
        cos_init = torch.randn_like(self.fourier_coeffs_cos) * scale * 0.1
        sin_init = torch.randn_like(self.fourier_coeffs_sin) * scale * 0.1
        self.fourier_coeffs_cos.data.copy_(cos_init)
        self.fourier_coeffs_sin.data.copy_(sin_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FourierKANLayer.

        Args:
            x (torch.Tensor): Input feature tensor of shape [..., in_features].

        Returns:
            torch.Tensor: Transformed output tensor of shape [..., out_features].
        """
        # Linear base output: [..., out_features]
        base_out = torch.matmul(x, self.base_weight.t())

        # Expand input tensor for harmonic computation: [..., in_features, 1]
        x_expanded = x.unsqueeze(-1)

        # Compute harmonic phase angles (k * x): [..., in_features, fourier_order]
        k_x = x_expanded * self.k

        # Compute cosine and sine basis functions
        cos_x = torch.cos(k_x)
        sin_x = torch.sin(k_x)

        # Vectorised tensor contraction using einsum
        fourier_cos_out = torch.einsum("...in,oin->...o", cos_x, self.fourier_coeffs_cos)
        fourier_sin_out = torch.einsum("...in,oin->...o", sin_x, self.fourier_coeffs_sin)

        out = base_out + fourier_cos_out + fourier_sin_out

        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"fourier_order={self.fourier_order}, bias={self.bias is not None}"
        )


class FourierKAN(nn.Module):
    """
    Multi-layer Fourier Kolmogorov-Arnold Network (FourierKAN).

    Sequence of FourierKANLayers with configurable hidden dimensions, activations, and dropout.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        fourier_order: int = 5,
        dropout: float = 0.2,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.kan1 = FourierKANLayer(in_features, hidden_dim, fourier_order=fourier_order)

        act_lower = activation.lower()
        self.act: nn.Module
        if act_lower == "silu":
            self.act = nn.SiLU()
        elif act_lower == "relu":
            self.act = nn.ReLU()
        elif act_lower == "tanh":
            self.act = nn.Tanh()
        elif act_lower == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

        self.drop = nn.Dropout(p=dropout)
        self.kan2 = FourierKANLayer(hidden_dim, out_features, fourier_order=fourier_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.kan1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.kan2(out)
        return out

"""
Activation Statistics Observer Module for EQ-KA-GCN

Provides MinMaxObserver to record running minimum, maximum, dynamic activation range,
and variance for target layers during calibration prior to Quantization-Aware Training (QAT).
"""

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

logger = logging.getLogger("EQ-KA-GCN.quantization.observer")


class MinMaxObserver(nn.Module):
    """
    Monitors tensor activation statistics during calibration forward passes.
    Tracks running min, running max, dynamic range, and variance.
    """

    def __init__(self, layer_name: str = "") -> None:
        """
        Initializes the MinMaxObserver.

        Args:
            layer_name (str): Identifier name for the target layer being observed.
        """
        super().__init__()
        self.layer_name = layer_name
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("sum_val", torch.tensor(0.0))
        self.register_buffer("sum_sq_val", torch.tensor(0.0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

        self.min_val: torch.Tensor
        self.max_val: torch.Tensor
        self.sum_val: torch.Tensor
        self.sum_sq_val: torch.Tensor
        self.count: torch.Tensor

    def reset(self) -> None:
        """Resets tracked activation statistics."""
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))
        self.sum_val.zero_()
        self.sum_sq_val.zero_()
        self.count.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Observes the input tensor and updates running statistics.

        Args:
            x (torch.Tensor): Tensor passed through the observed layer.

        Returns:
            torch.Tensor: Unmodified input tensor (passthrough).
        """
        if self.training or True:  # Observe during calibration
            with torch.no_grad():
                detached_x = x.detach().float()
                batch_min = torch.min(detached_x)
                batch_max = torch.max(detached_x)
                batch_count = torch.numel(detached_x)

                self.min_val = torch.minimum(self.min_val, batch_min)
                self.max_val = torch.maximum(self.max_val, batch_max)
                self.sum_val += torch.sum(detached_x)
                self.sum_sq_val += torch.sum(detached_x ** 2)
                self.count += batch_count

        return x

    def get_statistics(self) -> Dict[str, Any]:
        """
        Computes summary statistics of observed activations.

        Returns:
            Dict[str, Any]: Dictionary containing min, max, dynamic range, mean, and variance.
        """
        cnt = max(1, self.count.item())
        min_v = float(self.min_val.item()) if self.count.item() > 0 else 0.0
        max_v = float(self.max_val.item()) if self.count.item() > 0 else 0.0
        mean_v = float((self.sum_val / cnt).item())
        var_v = float((self.sum_sq_val / cnt - mean_v ** 2).clamp(min=0.0).item())
        dynamic_range = max(0.0, max_v - min_v)

        return {
            "layer_name": self.layer_name,
            "min_val": min_v,
            "max_val": max_v,
            "dynamic_range": dynamic_range,
            "mean": mean_v,
            "variance": var_v,
            "count": int(cnt),
        }

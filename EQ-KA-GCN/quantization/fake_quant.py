"""
Fake Quantization Module for EQ-KA-GCN

Provides FakeQuantize with Straight-Through Estimator (STE) gradient pass
supporting dynamic bit-widths (4-bit, 6-bit, 8-bit) during Quantization-Aware Training (QAT).
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger("EQ-KA-GCN.quantization.fake_quant")


class FakeQuantize(nn.Module):
    """
    Simulates quantization effects in full-precision floating point tensors
    with Straight-Through Estimator (STE) for autograd backpropagation.
    """

    def __init__(self, bits: int = 8, symmetric: bool = True) -> None:
        """
        Initializes the FakeQuantize module.

        Args:
            bits (int): Bit-width for quantization (e.g., 4, 6, 8).
            symmetric (bool): Whether to use symmetric quantization bounds.
        """
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric

        if symmetric:
            self.qmin = -(1 << (bits - 1))
            self.qmax = (1 << (bits - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (1 << bits) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies fake quantization to tensor x with STE.

        Args:
            x (torch.Tensor): Input tensor (weights or activations).

        Returns:
            torch.Tensor: Fake-quantized tensor with STE gradient path.
        """
        if not self.training:
            # During evaluation, apply deterministic fake quantization
            return self._quantize(x)

        # During training, apply fake quantization with STE gradient pass
        quant_x = self._quantize(x)
        # STE trick: quant_x - x detached from autograd graph, + x passes gradients directly
        return (quant_x - x).detach() + x

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Internal uniform quantization scaling and clamping logic.
        """
        max_abs = torch.max(torch.abs(x))
        if max_abs == 0 or torch.isnan(max_abs):
            return x

        scale = max_abs / float(self.qmax)
        if scale == 0:
            return x

        # Quantize and clamp
        x_scaled = x / scale
        x_clamped = torch.clamp(torch.round(x_scaled), self.qmin, self.qmax)
        x_dequantized = x_clamped * scale

        return x_dequantized

    def extra_repr(self) -> str:
        return f"bits={self.bits}, symmetric={self.symmetric}, qmin={self.qmin}, qmax={self.qmax}"

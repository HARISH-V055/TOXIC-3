"""
Quantization package for EQ-KA-GCN.

Responsible for Adaptive Layer-wise Quantization-Aware Training (QAT),
activation statistics calibration, fake quantization (STE), size profiling, and figure plotting.
"""

from quantization.observer import MinMaxObserver
from quantization.fake_quant import FakeQuantize
from quantization.quant_utils import calculate_model_size_bytes, plot_quantization_figures
from quantization.adaptive_qat import AdaptiveQATManager

__all__ = [
    "MinMaxObserver",
    "FakeQuantize",
    "calculate_model_size_bytes",
    "plot_quantization_figures",
    "AdaptiveQATManager",
]

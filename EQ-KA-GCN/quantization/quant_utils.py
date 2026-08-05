"""
Quantization Utility Module for EQ-KA-GCN

Provides functions for model parameter size profiling, compression ratio calculations,
and generating 300 DPI publication-quality figures for QAT analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Any, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("EQ-KA-GCN.quantization.quant_utils")


def calculate_model_size_bytes(
    model: nn.Module, bit_assignments: Dict[str, int]
) -> Tuple[float, float, float, float]:
    """
    Calculates FP32 model size, layer-wise quantized model size, compression ratio, and memory reduction.

    Args:
        model (nn.Module): The model instance to profile.
        bit_assignments (Dict[str, int]): Dictionary mapping layer names to assigned bit-widths.

    Returns:
        Tuple[float, float, float, float]:
            - Original FP32 size in KB.
            - Quantized model size in KB.
            - Compression ratio (FP32 / Quantized).
            - Memory reduction percentage (%).
    """
    total_params = 0
    fp32_bits = 0
    quant_bits = 0

    for name, param in model.named_parameters():
        num_p = param.numel()
        total_params += num_p
        fp32_bits += num_p * 32

        # Check if parameter belongs to a quantized layer
        matched_bits = 32  # Default to FP32 for non-target parameters
        for layer_key, bits in bit_assignments.items():
            if layer_key in name:
                matched_bits = bits
                break

        quant_bits += num_p * matched_bits

    fp32_size_kb = (fp32_bits / 8.0) / 1024.0
    quant_size_kb = (quant_bits / 8.0) / 1024.0

    compression_ratio = fp32_size_kb / quant_size_kb if quant_size_kb > 0 else 1.0
    memory_reduction = (1.0 - (quant_size_kb / fp32_size_kb)) * 100.0 if fp32_size_kb > 0 else 0.0

    logger.info(
        f"Model Size Profiling | FP32: {fp32_size_kb:.2f} KB | Quantized: {quant_size_kb:.2f} KB | "
        f"Compression Ratio: {compression_ratio:.2f}x | Memory Reduction: {memory_reduction:.2f}%"
    )

    return fp32_size_kb, quant_size_kb, compression_ratio, memory_reduction


def plot_quantization_figures(
    bit_assignments: Dict[str, int],
    fp32_size_kb: float,
    quant_size_kb: float,
    fp32_latency_ms: float,
    quant_latency_ms: float,
    output_dir: str = "outputs/figures",
) -> None:
    """
    Generates and saves publication-quality 300 DPI plots for QAT analysis:
      1. model_size_comparison.png
      2. inference_time_comparison.png
      3. layer_bitwidth_distribution.png

    Args:
        bit_assignments (Dict[str, int]): Layer-wise assigned bit-widths.
        fp32_size_kb (float): Original FP32 size in KB.
        quant_size_kb (float): Quantized size in KB.
        fp32_latency_ms (float): FP32 inference latency per sample in ms.
        quant_latency_ms (float): Quantized inference latency per sample in ms.
        output_dir (str): Destination directory for PNG plots.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Model Size Comparison Plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    categories = ["Original FP32", "Adaptive QAT"]
    sizes = [fp32_size_kb, quant_size_kb]
    colors = ["#4c72b0", "#55a868"]

    bars = ax.bar(categories, sizes, color=colors, width=0.5, edgecolor="black")
    ax.set_ylabel("Model Memory Size (KB)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("Model Size Reduction (Adaptive QAT)", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f} KB",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plot1_path = out_path / "model_size_comparison.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved figure: {plot1_path}")

    # 2. Inference Time Comparison Plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    latencies = [fp32_latency_ms, quant_latency_ms]
    bar_colors = ["#c44e52", "#8172b0"]

    bars = ax.bar(categories, latencies, color=bar_colors, width=0.5, edgecolor="black")
    ax.set_ylabel("Average Latency (ms/sample)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("Inference Latency Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plot2_path = out_path / "inference_time_comparison.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved figure: {plot2_path}")

    # 3. Layer Bit-width Distribution Plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    layer_names = list(bit_assignments.keys())
    bit_widths = list(bit_assignments.values())

    cmap = plt.get_cmap("viridis")
    norm_colors = [cmap(b / 8.0) for b in bit_widths]

    bars = ax.bar(layer_names, bit_widths, color=norm_colors, width=0.5, edgecolor="black")
    ax.set_xlabel("Target Layer", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("Assigned Bit-Width (bits)", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("Adaptive Layer-wise Bit-width Allocation", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, 10)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}-bit",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plot3_path = out_path / "layer_bitwidth_distribution.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved figure: {plot3_path}")

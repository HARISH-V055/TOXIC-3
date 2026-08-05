"""
Adaptive Layer-wise QAT Manager Module for EQ-KA-GCN

Orchestrates calibration statistics collection, dynamic layer-wise bit-width allocation,
fake quantization wrapping (STE), and exporting JSON quantization reports.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from quantization.observer import MinMaxObserver
from quantization.fake_quant import FakeQuantize

logger = logging.getLogger("EQ-KA-GCN.quantization.adaptive_qat")


class AdaptiveQATManager:
    """
    Manages calibration, adaptive layer-wise bit-width assignment, and QAT module insertion.
    """

    def __init__(
        self,
        target_layers: Optional[List[str]] = None,
        supported_bits: Optional[List[int]] = None,
    ) -> None:
        """
        Initializes the AdaptiveQATManager.

        Args:
            target_layers (Optional[List[str]]): Target layer names to quantize.
            supported_bits (Optional[List[int]]): Candidate bit-widths for layer assignment.
        """
        self.target_layers = target_layers or ["conv1", "conv2", "fourier_kan", "fc_out"]
        self.supported_bits = supported_bits or [4, 6, 8]
        self.observers: Dict[str, MinMaxObserver] = {}

    def calibrate_and_assign_bits(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        calibration_batches: int = 10,
    ) -> Dict[str, int]:
        """
        Runs calibration over DataLoader batches to collect activation statistics and assign bit-widths.

        Args:
            model (nn.Module): The model to calibrate.
            dataloader (DataLoader): DataLoader split for calibration.
            device (torch.device): Computing device.
            calibration_batches (int): Number of batches to pass during calibration.

        Returns:
            Dict[str, int]: Dictionary mapping target layer names to assigned bit-widths.
        """
        logger.info(f"Starting activation calibration across {calibration_batches} batches...")
        model.eval()

        # 1. Attach Observers
        hooks = []
        for name, module in model.named_modules():
            for target in self.target_layers:
                if name == target or name.endswith(target):
                    observer = MinMaxObserver(layer_name=name)
                    self.observers[name] = observer

                    # Forward hook to record activation statistics
                    def make_hook(obs: MinMaxObserver):
                        def hook_fn(mod, inp, out):
                            if isinstance(out, torch.Tensor):
                                obs(out)
                            elif isinstance(out, tuple) and isinstance(out[0], torch.Tensor):
                                obs(out[0])
                        return hook_fn

                    hooks.append(module.register_forward_hook(make_hook(observer)))

        # 2. Run Calibration Forward Passes
        batch_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if batch_count >= calibration_batches:
                    break
                batch_x = batch.x.to(device)
                batch_edge = batch.edge_index.to(device)
                batch_ind = batch.batch.to(device)

                model(x=batch_x, edge_index=batch_edge, batch=batch_ind, return_logits=True)
                batch_count += 1

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # 3. Collect statistics and assign bit-widths adaptively
        stats_list = []
        for name, observer in self.observers.items():
            st = observer.get_statistics()
            stats_list.append((name, st))

        # Sort layers by dynamic range (larger range = higher sensitivity = higher bit-width)
        stats_list.sort(key=lambda item: item[1]["dynamic_range"], reverse=True)

        bit_assignments: Dict[str, int] = {}
        sorted_supported = sorted(self.supported_bits, reverse=True)  # [8, 6, 4]
        num_layers = len(stats_list)

        for idx, (layer_name, st) in enumerate(stats_list):
            # Assign bits according to dynamic range percentile/rank
            if idx == 0 or idx < num_layers * 0.35:
                assigned_bit = sorted_supported[0]  # 8-bit
            elif idx < num_layers * 0.70:
                assigned_bit = sorted_supported[min(1, len(sorted_supported) - 1)]  # 6-bit
            else:
                assigned_bit = sorted_supported[-1]  # 4-bit

            bit_assignments[layer_name] = assigned_bit
            logger.info(
                f"Layer '{layer_name}' | Dynamic Range: {st['dynamic_range']:.4f} | "
                f"Variance: {st['variance']:.4f} --> Assigned Bit-Width: {assigned_bit}-bit"
            )

        return bit_assignments

    def prepare_qat_model(self, model: nn.Module, bit_assignments: Dict[str, int]) -> nn.Module:
        """
        Wraps target layer parameters with FakeQuantize modules according to bit assignments.

        Args:
            model (nn.Module): Model instance to convert.
            bit_assignments (Dict[str, int]): Assigned bit-widths.

        Returns:
            nn.Module: Model with FakeQuantize modules attached.
        """
        logger.info("Injecting FakeQuantize modules into target layers for QAT...")

        for name, module in model.named_modules():
            for target_key, bits in bit_assignments.items():
                if name == target_key or name.endswith(target_key):
                    # Attach weight fake quantizer attribute
                    module.weight_fake_quant = FakeQuantize(bits=bits, symmetric=True)
                    logger.info(f"Attached {bits}-bit FakeQuantize to layer '{name}'")

        # Define custom forward wrap to execute fake quantizer on weights during forward pass
        original_forward = model.forward

        def qat_forward(*args, **kwargs):
            # Apply weight fake quantization before execution
            for name, module in model.named_modules():
                if hasattr(module, "weight_fake_quant") and hasattr(module, "weight") and module.weight is not None:
                    # Apply fake quantization to weight
                    module.weight.data.copy_(module.weight_fake_quant(module.weight))
            return original_forward(*args, **kwargs)

        model.forward = qat_forward
        return model

    def export_quantization_report(
        self,
        report_data: Dict[str, Any],
        export_path: str = "outputs/quantization_report.json",
    ) -> None:
        """
        Saves quantization report metadata to a JSON file.

        Args:
            report_data (Dict[str, Any]): Quantization report dictionary.
            export_path (str): File path for JSON export.
        """
        try:
            path = Path(export_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            logger.info(f"Quantization report successfully saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save quantization report to {export_path}: {str(e)}")

"""
Helper Utilities for EQ-KA-GCN

Provides reusable helper functions for model evaluation, parameter counting,
and other utility metrics for the EQ-KA-GCN scientific pipeline.
"""

from typing import Dict, Any
import torch


def get_model_parameters_count(model: torch.nn.Module) -> Dict[str, int]:
    """
    Computes the total and trainable parameters of a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch neural network.

    Returns:
        Dict[str, int]: A dictionary containing total and trainable parameters.
    """
    # Placeholder implementation
    # In practice:
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": 0,
        "trainable_parameters": 0,
    }


def format_execution_time(seconds: float) -> str:
    """
    Converts execution duration in seconds to a human-readable format.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Formatted duration (e.g. "1h 23m 45s" or "0.45s").
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    
    minutes, secs = divmod(seconds, 60)
    hours, mins = divmod(minutes, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(mins)}m {int(secs)}s"
    return f"{int(mins)}m {int(secs)}s"


def save_metrics_to_json(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Saves a dictionary of evaluation metrics to a JSON file.

    Args:
        metrics (Dict[str, Any]): Evaluation metrics.
        filepath (str): Target filepath.
    """
    # Placeholder for metric persistence
    pass

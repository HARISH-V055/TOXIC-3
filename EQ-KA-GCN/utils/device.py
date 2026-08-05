"""
Device Detection Utility for EQ-KA-GCN

Detects hardware availability (such as CUDA GPU) and configures PyTorch devices accordingly.
"""

import torch


def get_device(device_setting: str = "auto") -> torch.device:
    """
    Returns the appropriate PyTorch device based on auto-detection or setting.

    Args:
        device_setting (str): The configuration setting. If 'auto', detects CUDA first,
                             otherwise defaults to CPU. If 'cuda' or 'cpu', forces that device.

    Returns:
        torch.device: The selected PyTorch device.
    """
    if device_setting == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise RuntimeError("CUDA was explicitly requested, but CUDA is not available.")
            
    if device_setting == "cpu":
        return torch.device("cpu")
        
    # Auto-detection
    if torch.cuda.is_available():
        return torch.device("cuda")
        
    return torch.device("cpu")

"""
Seeding Utility for EQ-KA-GCN

Ensures scientific reproducibility by fixing random seeds across all libraries
including standard python random, numpy, PyTorch, and CUDA-specific settings.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Sets the random seed for Python, NumPy, PyTorch, and CUDA to ensure reproducibility.

    Args:
        seed (int): The seed value to be used.
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Configure PyTorch deterministic behaviors
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set standard environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)

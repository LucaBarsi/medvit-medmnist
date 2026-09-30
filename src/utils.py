"""General utilities: reproducibility and device selection."""
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds to make experiments reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available device (CUDA, Apple MPS or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

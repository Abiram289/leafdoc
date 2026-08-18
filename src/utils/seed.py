import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Make runs reproducible across random, numpy, and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

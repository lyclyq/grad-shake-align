from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_rank() -> int:
    return int(os.environ.get('RANK', '0'))


def is_main_process() -> bool:
    return get_rank() == 0


def infer_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

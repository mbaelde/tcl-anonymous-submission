"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Parameters
    ----------
    seed
        Integer seed shared across RNGs.
    deterministic_torch
        If True, force deterministic CuDNN algorithms (slower but reproducible).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Note: agents use np.random.default_rng(seed) directly; the legacy
    # np.random.seed() would only seed the global legacy RNG, which is
    # independent of Generator instances and is not used in this codebase.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

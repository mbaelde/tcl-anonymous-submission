"""Shared parsing and scheduling utilities for all SAC agents."""

from __future__ import annotations

import numpy as np


def parse_vector(s: str, k: int, name: str) -> np.ndarray:
    """Parse a comma-separated string into a length-k float array.

    A single value is broadcast to all k positions.
    """
    parts = [float(x) for x in s.split(",") if x.strip()]
    if len(parts) == 1:
        parts = parts * k
    if len(parts) != k:
        raise ValueError(f"{name} has {len(parts)} entries, env has K={k}")
    return np.asarray(parts, dtype=np.float32)


def parse_weights(s: str, k: int) -> np.ndarray:
    """Parse a comma-separated string into a length-k non-negative weight array."""
    parts = [float(x) for x in s.split(",") if x.strip()]
    if len(parts) == 1:
        parts = parts * k
    if len(parts) != k:
        raise ValueError(f"cost_weights has {len(parts)} entries, env has K={k}")
    w = np.asarray(parts, dtype=np.float32)
    if (w < 0).any():
        raise ValueError(f"cost_weights must be non-negative; got {parts}")
    return w


def current_betas(
    step: int,
    betas_init: np.ndarray,
    betas_final: np.ndarray | None,
    schedule: str,
    anneal_steps: int,
) -> np.ndarray:
    """Return the current beta values under a linear or exponential annealing schedule."""
    if betas_final is None or anneal_steps <= 0:
        return betas_init
    if step >= anneal_steps:
        return betas_final
    t = float(step) / float(anneal_steps)
    if schedule == "linear":
        return betas_init + t * (betas_final - betas_init)
    if schedule == "exp":
        # Geometric interpolation in log-space (positive betas only).
        log_b = (1.0 - t) * np.log(betas_init) + t * np.log(betas_final)
        return np.exp(log_b).astype(np.float32)
    raise ValueError(f"unknown beta_schedule: {schedule}")

"""Base Args dataclass shared by all SAC agents targeting SinusoidalCMDP."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SACBaseArgs:
    """Common hyperparameters for all SAC + SinusoidalCMDP agents.

    Each agent subclasses this and overrides ``exp_name`` plus adds its own
    algorithm-specific fields.
    """

    # bookkeeping
    exp_name: str = "sac"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    log_dir: str = "runs"

    # environment (SinusoidalCMDP)
    horizon: int = 144
    b0: float = 0.5
    amplitude: float = 0.2
    omega: float = 2.0 * math.pi / 144.0
    random_phase_at_reset: bool = True

    # SAC core
    total_timesteps: int = 100_000
    buffer_size: int = 200_000
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 1_000
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 2
    target_network_frequency: int = 1
    autotune_alpha: bool = True
    alpha_init: float = 0.2

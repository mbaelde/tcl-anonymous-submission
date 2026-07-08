"""Shared replay buffer for all SAC agents."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """FIFO numpy replay buffer storing (s, a, r, costs, s', done).

    Parameters
    ----------
    k_costs:
        Number of cost dimensions. Use 1 (default) for single-constraint agents.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, k_costs: int = 1) -> None:
        self.capacity = capacity
        self.k_costs = k_costs
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.costs = np.zeros((capacity, k_costs), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(
        self,
        s: np.ndarray,
        a: np.ndarray,
        r: float,
        c: float | np.ndarray,
        s_next: np.ndarray,
        done: bool,
    ) -> None:
        i = self.idx
        self.obs[i] = s
        self.actions[i] = a
        self.rewards[i, 0] = r
        self.costs[i] = c
        self.next_obs[i] = s_next
        self.dones[i, 0] = float(done)
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        ix = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[ix]),
            "actions": torch.from_numpy(self.actions[ix]),
            "rewards": torch.from_numpy(self.rewards[ix]),
            "costs": torch.from_numpy(self.costs[ix]),
            "next_obs": torch.from_numpy(self.next_obs[ix]),
            "dones": torch.from_numpy(self.dones[ix]),
        }

"""Shared neural network architectures for all SAC agents."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class SoftQNetwork(nn.Module):
    """Critic Q(s, a) -> R."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    """Tanh-squashed Gaussian policy with action rescaling."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mean = nn.Linear(hidden, act_dim)
        self.fc_logstd = nn.Linear(hidden, act_dim)
        scale = torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32)
        bias = torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32)
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std) + 1.0)
        return mean, log_std

    def sample(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(s)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Tanh squash Jacobian correction.
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

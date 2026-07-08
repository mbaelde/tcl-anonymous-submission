"""Constrained Pendulum — Pendulum-v1 with an angle-limit constraint.

Wraps gymnasium's Pendulum-v1 with a single constraint:
    g_1(s) = |theta| - theta_max  (positive if violated)

The constraint is angle-based: the pendulum must stay within theta_max
radians of the upright position. This provides a standard benchmark
(non-AdCraft) for evaluating TCL and baseline agents on constrained RL.

Observation:  [cos(theta), sin(theta), theta_dot]  -- same as Pendulum-v1
Action:       torque in [-max_torque, max_torque]
Reward:       -(theta^2 + 0.1*theta_dot^2 + 0.001*torque^2)  -- native Pendulum reward
Cost:         [|theta| - theta_max]  -- positive when outside the cone

info["costs"] is a list [g_1] matching the interface of MultiConstraintAdCraft.

Usage:
    env = ConstrainedPendulum(theta_max=0.5)
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(action)
    print(info["costs"])  # [0.12]  -- violation magnitude, or negative if satisfied
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np


class ConstrainedPendulum(gym.Wrapper):
    """Pendulum-v1 augmented with a cone-constraint on the angle.

    Args:
        theta_max: Maximum allowed absolute angle (radians). Default 0.5 (~28 deg).
        render_mode: Passed to gymnasium.make.
    """

    def __init__(
        self,
        theta_max: float = 0.5,
        render_mode: str | None = None,
    ) -> None:
        env = gym.make("Pendulum-v1", render_mode=render_mode)
        super().__init__(env)
        self.theta_max = float(theta_max)

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        # obs = [cos(theta), sin(theta), theta_dot]
        theta = math.atan2(float(obs[1]), float(obs[0]))
        cost = abs(theta) - self.theta_max  # positive if |theta| > theta_max
        info = dict(info)
        info["costs"] = [float(cost)]
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> tuple:
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        theta = math.atan2(float(obs[1]), float(obs[0]))
        cost = abs(theta) - self.theta_max
        info["costs"] = [float(cost)]
        return obs, info

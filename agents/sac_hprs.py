"""SAC with Hierarchical Potential-based Reward Shaping (HPRS) — single-file.

Baseline based on Berducci et al., "Hierarchical Potential-based Reward
Shaping from Task Specifications" (2024). The shaped reward applied to
each transition (s_t, a_t, s_{t+1}) is

    r_HPRS(s_t, a_t) = r(s_t, a_t) + gamma * Phi(s_{t+1}) - Phi(s_t),

a Ng et al. (1999) potential-based shaping that preserves the optimal
policy. The potential

    Phi(state) = - sum_k w_k * ReLU(g_k - tau_k)

penalises violations of constraint k hierarchically: w_k is set
exponentially so the more important constraints dominate the
preference (we use w_k = base_weight * decay**k, decay > 1 by default
since k=0 is the most important).

Practical detail. The "state" entering Phi is taken to be the most
recent observed cost vector, because Phi must be a function of the
underlying state and the cost is the only state-dependent quantity we
have available at runtime. The shaping is computed *online* in the
rollout (so the buffer stores the already-shaped scalar reward) and
the per-episode running potential is reset to Phi(0) at episode start.
This is the standard online formulation used in HPRS implementations.

Usage:
    uv run python agents/sac_hprs.py \\
        --total-timesteps 200000 \\
        --thresholds "0.0" --base-weight 1.0 --weight-decay 1.0
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import tyro
from torch.utils.tensorboard import SummaryWriter

from agents.common.args import SACBaseArgs
from agents.common.sac_core import SACTrainer, make_sinusoidal_env, probe_k_costs
from agents.common.utils import parse_vector

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


@dataclass
class Args(SACBaseArgs):
    """Hyperparameters for SAC + HPRS shaping."""

    exp_name: str = "sac_hprs"

    # HPRS shaping
    # Thresholds tau_k below which constraint k is considered satisfied.
    thresholds: str = "0.0"
    # Hierarchical weights: w_k = base_weight * weight_decay ** k.
    # weight_decay > 1 gives k=0 highest priority (steepest decay first).
    base_weight: float = 1.0
    weight_decay: float = 1.0


def build_weights(base: float, decay: float, k: int) -> np.ndarray:
    return np.asarray([base * (decay ** i) for i in range(k)], dtype=np.float32)


def potential(costs: np.ndarray, weights: np.ndarray, thresholds: np.ndarray) -> float:
    """Phi(state) = - sum_k w_k * ReLU(c_k - tau_k)."""
    viol = np.maximum(costs - thresholds, 0.0)
    return float(-(weights * viol).sum())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    args: Args,
    env_factory: Callable[[Args], gym.Env] | None = None,
) -> dict[str, float]:
    env_f = env_factory or make_sinusoidal_env
    k_costs = probe_k_costs(args, env_f)

    thresholds_np = parse_vector(args.thresholds, k_costs, "thresholds")
    weights_np = build_weights(args.base_weight, args.weight_decay, k_costs)
    zeros_costs = np.zeros(k_costs, dtype=np.float32)
    phi_zero = potential(zeros_costs, weights_np, thresholds_np)

    # Mutable state shared by step_hook and episode_end_fn.
    prev_state = [phi_zero]   # prev_state[0] = Phi(s_t)
    last_phi = [phi_zero]     # last_phi[0] = Phi(s_{t+1}) at most recent step

    def step_hook(r: float, costs: np.ndarray, done: bool) -> float:
        """Online PBRS: r_shaped = r + gamma * Phi(s') - Phi(s)."""
        phi_next = potential(costs, weights_np, thresholds_np)
        r_shaped = r + args.gamma * phi_next - prev_state[0]
        last_phi[0] = phi_next
        prev_state[0] = phi_zero if done else phi_next
        return r_shaped

    def episode_end_fn(writer: SummaryWriter, global_step: int) -> None:
        writer.add_scalar("shaping/phi_end_of_episode", last_phi[0], global_step)

    return SACTrainer(
        args, env_f, k_costs,
        step_hook=step_hook,
        episode_end_fn=episode_end_fn,
    ).run()


if __name__ == "__main__":
    args = tyro.cli(Args)
    result = train(args)
    print(f"Training complete. episodes_done={int(result['episodes_done'])}")
    print(f"Logs: {result['log_path']}")

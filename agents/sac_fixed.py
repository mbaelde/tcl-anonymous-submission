"""SAC with a fixed weighted linear combination of reward and costs — single-file.

Baseline shaping for the TCL paper. The critic is trained on the
*scalar* shaped reward

    r_shaped(s, a) = r(s, a) - sum_k w_k * g_k(s, a),

where w = (w_1, ..., w_K) are user-supplied non-negative weights and
g_k are the per-constraint costs read from info["costs"]. This is the
simplest possible CMDP baseline: no Lagrangian update, no potential
shaping, just a hand-picked tradeoff between reward and constraint
violation.

Hyperparameters are exposed via tyro. The script logs episodic return,
per-constraint episodic costs, and the SAC losses to TensorBoard.

Usage:
    uv run python agents/sac_fixed.py \\
        --total-timesteps 200000 --cost-weights "0.5"
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import torch
import tyro

from agents.common.args import SACBaseArgs
from agents.common.sac_core import SACTrainer, make_sinusoidal_env, probe_k_costs
from agents.common.utils import parse_weights

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


@dataclass
class Args(SACBaseArgs):
    """Hyperparameters for SAC + fixed-weighted reward shaping."""

    exp_name: str = "sac_fixed"

    # Comma-separated non-negative weights w_k, one per constraint.
    # A single value broadcasts to all K constraints.
    cost_weights: str = "0.5"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    args: Args,
    env_factory: Callable[[Args], gym.Env] | None = None,
) -> dict[str, float]:
    env_f = env_factory or make_sinusoidal_env
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    k_costs = probe_k_costs(args, env_f)
    weights_np = parse_weights(args.cost_weights, k_costs)
    weights = torch.as_tensor(weights_np, device=device)

    def reward_fn(batch: dict, _: int) -> torch.Tensor:
        return batch["rewards"] - (batch["costs"] * weights).sum(dim=-1, keepdim=True)

    return SACTrainer(args, env_f, k_costs, reward_fn=reward_fn).run()


if __name__ == "__main__":
    args = tyro.cli(Args)
    result = train(args)
    print(f"Training complete. episodes_done={int(result['episodes_done'])}")
    print(f"Logs: {result['log_path']}")

"""SAC + multi-constraint Lagrangian (RCPO, K-generic) — single-file.

Generalizes :mod:`agents.sac_lagrangian` to K > 1 constraints by keeping
one Lagrangian multiplier per constraint. The augmented critic reward is

    r_aug(s, a) = r(s, a) - sum_k lambda_k * g_k(s, a),

where ``g_k = info["costs"][k]`` is the per-step cost of constraint k and
each :math:`\\lambda_k \\ge 0` is updated by projected primal-dual ascent:

    lambda_k <- ReLU(lambda_k + eta_lambda * mean_batch(g_k - cost_baseline)).

This is the §7.1 baseline that competes against the proposed TCL agent on
:class:`tcl.envs.MultiConstraintAdCraft` (K=3). The K=1 sibling
:mod:`agents.sac_lagrangian` is kept intact for §4.3 / Proposition 2,
where the scalar trajectory of :math:`\\lambda(t)` is the primary signal.

Usage:
    uv run python agents/sac_lagrangian_multi.py \\
        --total-timesteps 200000 --lambda-lr 1e-3

References:
    Haarnoja et al., Soft Actor-Critic, ICML 2018.
    Tessler, Mankowitz, Mannor, RCPO, ICLR 2019.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from agents.common.args import SACBaseArgs
from agents.common.sac_core import SACTrainer, make_sinusoidal_env, probe_k_costs

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


@dataclass
class Args(SACBaseArgs):
    """Hyperparameters for SAC + multi-constraint Lagrangian."""

    exp_name: str = "sac_lagrangian_multi"

    # Lagrangian (scalar values broadcast to all K components at init)
    lambda_init: float = 0.0
    lambda_lr: float = 1e-3
    lambda_update_frequency: int = 1  # in gradient steps
    cost_baseline: float = 0.0  # per-component subtractive baseline in dual update


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

    lam = torch.full((k_costs,), float(args.lambda_init), device=device)

    def reward_fn(batch: dict, _: int) -> torch.Tensor:
        penalty = (batch["costs"] * lam.unsqueeze(0)).sum(dim=-1, keepdim=True)
        return batch["rewards"] - penalty

    def dual_update_fn(batch: dict, gradient_steps: int) -> None:
        nonlocal lam
        if gradient_steps % args.lambda_update_frequency == 0:
            mean_cost_vec = batch["costs"].mean(dim=0) - args.cost_baseline
            lam = torch.clamp(lam + args.lambda_lr * mean_cost_vec, min=0.0)

    def extra_log_fn(
        writer: SummaryWriter, batch: dict, global_step: int, gradient_steps: int
    ) -> None:
        if global_step % 500 == 0:
            for k in range(k_costs):
                writer.add_scalar(f"train/lambda_{k}", lam[k].item(), global_step)
        for k in range(k_costs):
            writer.add_scalar(f"dual/lambda_{k}", lam[k].item(), global_step)
            writer.add_scalar(
                f"dual/cost_batch_mean_{k}",
                batch["costs"][:, k].mean().item(),
                global_step,
            )

    def extra_return_fn() -> dict:
        return {f"lambda_final_k{k}": lam[k].item() for k in range(k_costs)}

    return SACTrainer(
        args, env_f, k_costs,
        reward_fn=reward_fn,
        dual_update_fn=dual_update_fn,
        extra_log_fn=extra_log_fn,
        extra_return_fn=extra_return_fn,
    ).run()


if __name__ == "__main__":
    args = tyro.cli(Args)
    result = train(args)
    final_lams = [v for k, v in result.items() if k.startswith("lambda_final_k")]
    print(f"Training complete. lambda_final={final_lams}")
    print(f"Logs: {result['log_path']}")

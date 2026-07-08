"""SAC + PID-Lagrangian (Stooke et al. 2020) — K-generic, single-file.

Replaces the standard gradient-ascent dual update of :mod:`agents.sac_lagrangian_multi`
with a discrete PID controller on the constraint violation signal:

    e_k(t)   = mean_batch(g_k)              -- proportional error
    I_k(t)   = sum_{s<=t} e_k(s)            -- integral (accumulated error)
    D_k(t)   = e_k(t) - e_k(t-1)           -- discrete derivative

    lambda_k(t+1) = ReLU(lambda_k(t) + K_P * e_k + K_I * I_k + K_D * D_k)

The integral term accumulates historical violations, providing a restoring force
that prevents the oscillations documented in Proposition 2. The derivative term
damps overshoot. Pure P-control (K_I=K_D=0) recovers the standard Lagrangian.

Reference: Stooke, Achiam, Abbeel (2020). "Responsive Safety in Reinforcement
Learning by PID Lagrangian Methods." ICML 2020. arXiv:2007.03964.

Usage:
    uv run python agents/sac_pid_lagrangian.py \\
        --total-timesteps 200000 --pid-kp 1e-3 --pid-ki 1e-4
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


@dataclass
class Args(SACBaseArgs):
    """Hyperparameters for SAC + PID Lagrangian."""

    exp_name: str = "sac_pid_lagrangian"

    # PID Lagrangian gains (K_I = K_D = 0 recovers standard Lagrangian)
    lambda_init: float = 0.0
    pid_kp: float = 1e-3   # proportional gain (≈ lambda_lr in standard Lag)
    pid_ki: float = 1e-4   # integral gain (Stooke 2020: typically 0.1*K_P)
    pid_kd: float = 0.0    # derivative gain (0 = no damping term)
    lambda_update_frequency: int = 1  # in gradient steps


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
    cost_integral = torch.zeros(k_costs, device=device)
    prev_cost_mean = torch.zeros(k_costs, device=device)

    def reward_fn(batch: dict, _: int) -> torch.Tensor:
        penalty = (batch["costs"] * lam.unsqueeze(0)).sum(dim=-1, keepdim=True)
        return batch["rewards"] - penalty

    def dual_update_fn(batch: dict, gradient_steps: int) -> None:
        nonlocal lam, cost_integral, prev_cost_mean
        if gradient_steps % args.lambda_update_frequency == 0:
            e = batch["costs"].mean(dim=0)
            cost_integral = cost_integral + e
            d = e - prev_cost_mean
            delta = args.pid_kp * e + args.pid_ki * cost_integral + args.pid_kd * d
            lam = torch.clamp(lam + delta, min=0.0)
            prev_cost_mean = e.clone()

    def extra_log_fn(
        writer: SummaryWriter, batch: dict, global_step: int, gradient_steps: int
    ) -> None:
        if global_step % 500 == 0:
            for k in range(k_costs):
                writer.add_scalar(f"train/lambda_{k}", lam[k].item(), global_step)
                writer.add_scalar(
                    f"train/cost_integral_{k}", cost_integral[k].item(), global_step
                )
        for k in range(k_costs):
            writer.add_scalar(f"dual/lambda_{k}", lam[k].item(), global_step)

    def extra_return_fn() -> dict:
        return {f"lambda_final_{k}": lam[k].item() for k in range(k_costs)}

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
    print(f"Training complete. log: {result['log_path']}")

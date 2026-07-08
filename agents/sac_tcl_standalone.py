"""SAC with the standalone (A) TCL reward — single-file.

Variant of `agents/sac_tcl.py` that consumes the additive cumulative-lex
reward of Formulation (A) instead of the multiplicative shaped reward (B):

    R_TCL^(A)(s, a; beta) = sum_{k=1}^{K} w_k(s, a; beta) * R_k(s, a; beta),

with R_k = sigma_beta(tau_k - g_k) and cumulative-lex gates
w_k = prod_{j<k} sigma_beta(R_j - 1/2), w_1 = 1.

Unlike (B), this reward is computed from costs only — the base reward r_b
is optional and controlled by `--rb-mode`:
  - "ignore"     : pure (A), bounded in [0, K]. r_b is unused.
  - "bonus"      : R_TCL^(A) + rb_weight * r_b. r_b is not gated.
  - "last_layer" : R_TCL^(A) + (prod_k sigma_beta(R_k - 1/2)) * rb_weight * r_b.
                   The principled cumulative-lex extension: r_b contributes
                   only once all K constraints have crossed half-satisfaction.

beta annealing and the Gaussian-gate variant (R_k = exp(-kappa_k * relu(g-tau)^2))
are supported via the same flags as in `sac_tcl.py`. The buffer keeps raw costs
and rewards, so the shaped reward is re-evaluated on every gradient step.

Usage:
    uv run python agents/sac_tcl_standalone.py \\
        --total-timesteps 200000 \\
        --thresholds "0.0" --betas-init "10.0" --rb-mode "ignore"
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from agents.common.args import SACBaseArgs
from agents.common.sac_core import SACTrainer, make_sinusoidal_env, probe_k_costs
from agents.common.utils import current_betas, parse_vector
from tcl.rewards.standalone import (
    tcl_standalone_reward,
    tcl_standalone_reward_gaussian,
)


@dataclass
class Args(SACBaseArgs):
    """Hyperparameters for SAC + standalone (A) TCL reward."""

    exp_name: str = "sac_tcl_standalone"

    thresholds: str = "0.0"
    betas_init: str = "10.0"
    betas_final: str = ""
    beta_schedule: str = "linear"
    beta_anneal_steps: int = 0
    kappas: str = ""

    rb_mode: str = "ignore"
    rb_weight: float = 1.0


def train(
    args: Args,
    env_factory: Callable[[Args], gym.Env] | None = None,
) -> dict[str, float]:
    if args.rb_mode not in ("ignore", "bonus", "last_layer"):
        raise ValueError(
            f"--rb-mode must be one of ignore|bonus|last_layer, got {args.rb_mode!r}"
        )

    env_f = env_factory or make_sinusoidal_env
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    k_costs = probe_k_costs(args, env_f)

    thresholds_np = parse_vector(args.thresholds, k_costs, "thresholds")
    betas_init_np = parse_vector(args.betas_init, k_costs, "betas_init")
    if (betas_init_np <= 0).any():
        raise ValueError(f"betas_init must be strictly positive; got {betas_init_np}")

    betas_final_np: np.ndarray | None
    if args.betas_final.strip():
        betas_final_np = parse_vector(args.betas_final, k_costs, "betas_final")
        if (betas_final_np <= 0).any():
            raise ValueError(f"betas_final must be strictly positive; got {betas_final_np}")
    else:
        betas_final_np = None

    kappas_np: np.ndarray | None
    if args.kappas.strip():
        kappas_np = parse_vector(args.kappas, k_costs, "kappas")
        if (kappas_np <= 0).any():
            raise ValueError(f"kappas must be strictly positive; got {kappas_np}")
    else:
        kappas_np = None

    thresholds_t = torch.as_tensor(thresholds_np, device=device)
    kappas_t = torch.as_tensor(kappas_np, device=device) if kappas_np is not None else None

    last_r_shaped_mean = [0.0]

    def reward_fn(batch: dict, global_step: int) -> torch.Tensor:
        betas_np = current_betas(
            global_step, betas_init_np, betas_final_np,
            args.beta_schedule, args.beta_anneal_steps,
        )
        betas_t = torch.as_tensor(betas_np, device=batch["rewards"].device)
        if kappas_t is not None:
            r = tcl_standalone_reward_gaussian(
                batch["rewards"], batch["costs"],
                thresholds_t, betas_t, kappas_t,
                rb_mode=args.rb_mode, rb_weight=args.rb_weight,
            )
        else:
            r = tcl_standalone_reward(
                batch["rewards"], batch["costs"],
                thresholds_t, betas_t,
                rb_mode=args.rb_mode, rb_weight=args.rb_weight,
            )
        last_r_shaped_mean[0] = r.mean().item()
        return r

    def extra_log_fn(
        writer: SummaryWriter, batch: dict, global_step: int, gradient_steps: int
    ) -> None:
        if global_step % 500 == 0:
            writer.add_scalar(
                "train/r_shaped_mean", last_r_shaped_mean[0], global_step
            )
            betas = current_betas(
                global_step, betas_init_np, betas_final_np,
                args.beta_schedule, args.beta_anneal_steps,
            )
            for k in range(k_costs):
                writer.add_scalar(f"train/beta_{k}", float(betas[k]), global_step)
                if kappas_np is not None:
                    writer.add_scalar(f"train/kappa_{k}", float(kappas_np[k]), global_step)

    return SACTrainer(
        args, env_f, k_costs,
        reward_fn=reward_fn,
        extra_log_fn=extra_log_fn,
    ).run()


if __name__ == "__main__":
    args = tyro.cli(Args)
    result = train(args)
    print(f"Training complete. episodes_done={int(result['episodes_done'])}")
    print(f"Logs: {result['log_path']}")

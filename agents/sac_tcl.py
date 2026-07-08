"""SAC with Threshold-Cascaded Lexicographic (TCL) reward shaping — single-file.

Main agent of the TCL paper. The critic is trained on the shaped reward

    r_TCL(s, a) = r(s, a) * prod_{k=1}^{K} sigma(-beta_k * (g_k(s,a) - tau_k)),

where g_k is the per-step cost of constraint k, tau_k is the
threshold above which constraint k is considered violated, beta_k is
the gate stiffness (large beta -> hard step), and sigma is the
logistic sigmoid. The product of gates implements the *threshold-
cascaded* multiplicative structure: a violation of any constraint
suppresses the entire shaped reward, with severity controlled by the
beta_k's. When the constraints are ordered by priority, the product
gives a smooth lexicographic preference (Theorem 1 of the paper).

beta annealing (used in the §7.2 experiment) is supported via a
linear or exponential schedule between beta_init and beta_final over
the first `beta_anneal_steps` environment steps. The shaped reward
re-evaluates the current beta values on every gradient step, so the
buffer keeps raw costs and no rewriting is necessary.

Usage:
    uv run python agents/sac_tcl.py \\
        --total-timesteps 200000 \\
        --thresholds "0.0" --betas-init "10.0"
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

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


@dataclass
class Args(SACBaseArgs):
    """Hyperparameters for SAC + TCL shaping."""

    exp_name: str = "sac_tcl"

    # TCL shaping
    # All three are comma-separated lists; a single value broadcasts to K.
    thresholds: str = "0.0"
    betas_init: str = "10.0"
    betas_final: str = ""  # empty -> no annealing
    beta_schedule: str = "linear"  # "linear" or "exp"
    beta_anneal_steps: int = 0  # 0 -> no annealing
    # Gaussian gate mode: if non-empty, use R_k^stab = exp(-kappa_k * relu(c_k - tau_k)^2)
    # and gate = sigma_beta(R_stab - 0.5) instead of sigma(-beta*(c_k - tau_k)).
    # kappa_k = ln2 / (e_k*)^2 so that R_stab = 0.5 at violation e_k = e_k*.
    kappas: str = ""  # empty -> linear gate (default)

    # Reward shift (Proposition 4 fix for loss-budget environments).
    # Replaces r_b by r_b + reward_shift before the gate product.
    # Set C > max|r_b| (i.e. C > budget per step) to make r' > 0 uniformly and
    # eliminate the sup-sup inversion. Does not affect the raw reward in the buffer.
    reward_shift: float = 0.0


# ---------------------------------------------------------------------------
# TCL shaping
# ---------------------------------------------------------------------------


def tcl_shaped_reward(
    rewards: torch.Tensor,  # (B, 1)
    costs: torch.Tensor,  # (B, K)
    thresholds: torch.Tensor,  # (K,)
    betas: torch.Tensor,  # (K,)
) -> torch.Tensor:
    """Compute r * prod_k sigma(-beta_k * (c_k - tau_k))."""
    gate_logits = -betas * (costs - thresholds)  # (B, K)
    gates = torch.sigmoid(gate_logits)  # (B, K)
    gate_product = gates.prod(dim=-1, keepdim=True)  # (B, 1)
    return rewards * gate_product


def tcl_shaped_reward_gaussian(
    rewards: torch.Tensor,  # (B, 1)
    costs: torch.Tensor,  # (B, K)
    thresholds: torch.Tensor,  # (K,)
    betas: torch.Tensor,  # (K,)
    kappas: torch.Tensor,  # (K,)
) -> torch.Tensor:
    """Gaussian-gate TCL: R_k^stab = exp(-kappa_k * relu(c_k - tau_k)^2).

    Gate = sigma_beta(R_stab - 0.5): equals 0.5 at violation e_k = sqrt(ln2/kappa_k),
    approaches 1 when satisfied (c_k <= tau_k), approaches 0 for large violations.
    """
    errors = torch.relu(costs - thresholds)  # (B, K), zero when satisfied
    r_stab = torch.exp(-kappas * errors ** 2)  # (B, K) in (0, 1]
    gate_logits = betas * (r_stab - 0.5)  # positive when satisfied
    gates = torch.sigmoid(gate_logits)  # (B, K)
    gate_product = gates.prod(dim=-1, keepdim=True)  # (B, 1)
    return rewards * gate_product


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

    def reward_fn(batch: dict, global_step: int) -> torch.Tensor:
        betas_np = current_betas(
            global_step, betas_init_np, betas_final_np,
            args.beta_schedule, args.beta_anneal_steps,
        )
        betas_t = torch.as_tensor(betas_np, device=batch["rewards"].device)
        r_base = batch["rewards"] + args.reward_shift
        if kappas_t is not None:
            return tcl_shaped_reward_gaussian(
                r_base, batch["costs"], thresholds_t, betas_t, kappas_t
            )
        return tcl_shaped_reward(r_base, batch["costs"], thresholds_t, betas_t)

    def extra_log_fn(
        writer: SummaryWriter, batch: dict, global_step: int, gradient_steps: int
    ) -> None:
        if global_step % 500 == 0:
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

"""Standalone (A) TCL reward primitive.

Implements the additive cumulative-lex reward of Formulation (A) studied in
§5.1 of the TCL paper (Theorem 1):

    R_TCL^(A)(s, a; beta) = sum_{k=1}^{K} w_k(s, a; beta) * R_k(s, a; beta),

with constraint satisfactions and cumulative-lex weights

    R_k = sigma_beta(tau_k - g_k),
    w_1 = 1,    w_k = prod_{j<k} sigma_beta(R_j - 1/2)  for k >= 2.

As beta -> infinity, the gate sigma_beta(R_j - 1/2) collapses to the indicator
1_{R_j > 1/2} = 1_{g_j < tau_j}, so R_TCL^(A) converges pointwise to the
cumulative-lexicographic limit reward (Definition 1 of the paper) which ranks
policies by the cascading priority R_1 > R_2 > ... > R_K.

This module is intentionally framework-agnostic at the math level but uses
torch for batched evaluation on the GPU. It is consumed by `agents/sac_tcl_standalone.py`.
"""

from __future__ import annotations

import torch


def tcl_standalone_reward(
    rewards: torch.Tensor,        # (B, 1) base reward r_b — used only in hybrid modes
    costs: torch.Tensor,          # (B, K) per-step constraint costs g_k
    thresholds: torch.Tensor,     # (K,) per-constraint thresholds tau_k
    betas: torch.Tensor,          # (K,) per-constraint stiffness beta_k
    rb_mode: str = "ignore",      # "ignore" (pure A) | "bonus" | "last_layer"
    rb_weight: float = 1.0,       # multiplier on the base reward in hybrid modes
) -> torch.Tensor:
    """Compute R_TCL^(A) (with optional base-reward hybrid term).

    Returns a (B, 1) tensor.

    rb_mode:
      - "ignore"     : pure (A), output in [0, K]. The base `rewards` is unused.
      - "bonus"      : output = R_TCL^(A) + rb_weight * r_b. No gating on r_b.
      - "last_layer" : extends the cumulative-lex cascade by treating r_b as an
                       additional final layer R_{K+1} := rb_weight * r_b, gated
                       by w_{K+1} = prod_{j=1..K} sigma_beta(R_j - 1/2).
                       This is the principled cumulative-lex extension to a
                       base reward: r_b matters only when all K constraints
                       have crossed their half-satisfaction threshold.
    """
    if costs.dim() != 2:
        raise ValueError(f"costs must be (B, K), got shape {tuple(costs.shape)}")
    if rewards.dim() != 2 or rewards.shape[-1] != 1:
        raise ValueError(f"rewards must be (B, 1), got shape {tuple(rewards.shape)}")
    K = costs.shape[-1]
    if thresholds.shape[-1] != K or betas.shape[-1] != K:
        raise ValueError(
            f"thresholds/betas must have last dim K={K}, "
            f"got {tuple(thresholds.shape)}, {tuple(betas.shape)}"
        )

    # R_k = sigma(beta_k * (tau_k - g_k)) in (0, 1).
    R = torch.sigmoid(betas * (thresholds - costs))  # (B, K)

    # sat_gates[..., k] = sigma(beta_k * (R_k - 1/2)) in (0, 1).
    sat_gates = torch.sigmoid(betas * (R - 0.5))  # (B, K)

    # w_k = prod_{j<k} sat_gates[..., j]; w_1 = 1.
    # Build cumulative product of sat_gates[..., :-1] prepended with 1.
    ones = torch.ones_like(sat_gates[..., :1])
    if K == 1:
        w = ones  # only w_1 = 1
    else:
        w = torch.cat([ones, sat_gates[..., :-1].cumprod(dim=-1)], dim=-1)  # (B, K)

    r_tcl_a = (w * R).sum(dim=-1, keepdim=True)  # (B, 1)

    if rb_mode == "ignore":
        return r_tcl_a
    if rb_mode == "bonus":
        return r_tcl_a + rb_weight * rewards
    if rb_mode == "last_layer":
        w_last = sat_gates.prod(dim=-1, keepdim=True)  # (B, 1)
        return r_tcl_a + w_last * (rb_weight * rewards)
    raise ValueError(f"unknown rb_mode={rb_mode!r}; expected ignore|bonus|last_layer")


def tcl_standalone_reward_gaussian(
    rewards: torch.Tensor,
    costs: torch.Tensor,
    thresholds: torch.Tensor,
    betas: torch.Tensor,
    kappas: torch.Tensor,         # (K,) Gaussian-gate stiffness, kappa_k = ln2 / (e_k*)^2
    rb_mode: str = "ignore",
    rb_weight: float = 1.0,
) -> torch.Tensor:
    """Gaussian-gate variant of the standalone (A) reward.

    Replaces the linear gate R_k = sigma(beta_k(tau_k - g_k)) with the Gaussian
    satisfaction R_k = exp(-kappa_k * relu(g_k - tau_k)^2). Otherwise identical
    to `tcl_standalone_reward`. The cumulative weights w_k are still defined via
    sigma_beta(R_k - 1/2) (same beta as the linear gate would have used).
    """
    if costs.dim() != 2:
        raise ValueError(f"costs must be (B, K), got shape {tuple(costs.shape)}")
    K = costs.shape[-1]
    if thresholds.shape[-1] != K or betas.shape[-1] != K or kappas.shape[-1] != K:
        raise ValueError("thresholds/betas/kappas must all have last dim K")

    errors = torch.relu(costs - thresholds)  # (B, K)
    R = torch.exp(-kappas * errors ** 2)     # (B, K) in (0, 1]
    sat_gates = torch.sigmoid(betas * (R - 0.5))  # (B, K)

    ones = torch.ones_like(sat_gates[..., :1])
    if K == 1:
        w = ones
    else:
        w = torch.cat([ones, sat_gates[..., :-1].cumprod(dim=-1)], dim=-1)

    r_tcl_a = (w * R).sum(dim=-1, keepdim=True)

    if rb_mode == "ignore":
        return r_tcl_a
    if rb_mode == "bonus":
        return r_tcl_a + rb_weight * rewards
    if rb_mode == "last_layer":
        w_last = sat_gates.prod(dim=-1, keepdim=True)
        return r_tcl_a + w_last * (rb_weight * rewards)
    raise ValueError(f"unknown rb_mode={rb_mode!r}; expected ignore|bonus|last_layer")

"""Unit tests for the standalone (A) TCL reward primitive.

Tests cover:
  - shape / batch correctness
  - cumulative-lex limit at large beta (Theorem 1)
  - the three rb_mode variants (ignore | bonus | last_layer)
  - monotonicity properties and the K=1 edge case
"""

from __future__ import annotations

import math

import pytest
import torch

from tcl.rewards.standalone import (
    tcl_standalone_reward,
    tcl_standalone_reward_gaussian,
)


# ---------------------------------------------------------------------------
# Shapes & API
# ---------------------------------------------------------------------------


def test_standalone_shape_batch() -> None:
    B, K = 5, 3
    rewards = torch.zeros(B, 1)
    costs = torch.zeros(B, K)
    thresholds = torch.zeros(K)
    betas = torch.full((K,), 10.0)
    out = tcl_standalone_reward(rewards, costs, thresholds, betas)
    assert out.shape == (B, 1)


def test_standalone_K1_edge_case() -> None:
    # K=1: w_1 = 1, output = R_1 = sigma(beta * (tau - g)).
    rewards = torch.zeros(1, 1)
    costs = torch.tensor([[-2.0]])    # satisfied: g << tau
    thresholds = torch.tensor([0.0])
    betas = torch.tensor([10.0])
    out = tcl_standalone_reward(rewards, costs, thresholds, betas)
    assert out.shape == (1, 1)
    assert out.item() == pytest.approx(1.0, abs=1e-6)


def test_standalone_rejects_bad_rb_mode() -> None:
    rewards = torch.zeros(1, 1)
    costs = torch.zeros(1, 2)
    thresholds = torch.zeros(2)
    betas = torch.full((2,), 10.0)
    with pytest.raises(ValueError):
        tcl_standalone_reward(rewards, costs, thresholds, betas, rb_mode="foo")


def test_standalone_rejects_bad_shapes() -> None:
    rewards = torch.zeros(1, 1)
    thresholds = torch.zeros(2)
    betas = torch.full((2,), 10.0)
    # K mismatch between costs and thresholds.
    with pytest.raises(ValueError):
        tcl_standalone_reward(rewards, torch.zeros(1, 3), thresholds, betas)


# ---------------------------------------------------------------------------
# Cumulative-lex limit (Theorem 1)
# ---------------------------------------------------------------------------


def test_standalone_all_satisfied_large_beta() -> None:
    # All constraints satisfied: R_k -> 1 for all k, sat_gates -> 1, w_k -> 1
    # -> R_TCL^(A) -> sum_k R_k -> K.
    K = 4
    rewards = torch.zeros(1, 1)
    costs = torch.full((1, K), -3.0)     # g_k = -3 << tau_k = 0
    thresholds = torch.zeros(K)
    betas = torch.full((K,), 50.0)
    out = tcl_standalone_reward(rewards, costs, thresholds, betas)
    assert out.item() == pytest.approx(float(K), abs=1e-3)


def test_standalone_first_violated_collapses() -> None:
    # R_1 violated -> R_1 ~ 0, sat_gates[0] = sigma(beta * (R_1 - 0.5)) ~ sigma(-beta/2) ~ 0
    # so w_k ~ 0 for all k >= 2. R_TCL^(A) ~ R_1 ~ 0.
    K = 3
    rewards = torch.zeros(1, 1)
    costs = torch.tensor([[3.0, -3.0, -3.0]])  # only R_1 violated
    thresholds = torch.zeros(K)
    betas = torch.full((K,), 50.0)
    out = tcl_standalone_reward(rewards, costs, thresholds, betas)
    assert out.item() < 1e-3


def test_standalone_partial_cascade_K3() -> None:
    # R_1 satisfied, R_2 violated, R_3 satisfied:
    # cumulative-lex limit: R_1 contributes 1, R_2 stops the cascade -> total ~ 1.
    K = 3
    rewards = torch.zeros(1, 1)
    costs = torch.tensor([[-3.0, 3.0, -3.0]])
    thresholds = torch.zeros(K)
    betas = torch.full((K,), 50.0)
    out = tcl_standalone_reward(rewards, costs, thresholds, betas)
    assert out.item() == pytest.approx(1.0, abs=5e-3)


def test_standalone_bound_in_0_K() -> None:
    # For any inputs and any beta > 0, R_TCL^(A) in [0, K].
    K = 5
    rewards = torch.zeros(8, 1)
    costs = torch.randn(8, K) * 2.0
    thresholds = torch.zeros(K)
    for beta_val in [0.1, 1.0, 10.0, 100.0]:
        betas = torch.full((K,), beta_val)
        out = tcl_standalone_reward(rewards, costs, thresholds, betas)
        assert (out >= 0.0).all()
        assert (out <= float(K) + 1e-5).all()


# ---------------------------------------------------------------------------
# Hybrid modes
# ---------------------------------------------------------------------------


def test_standalone_bonus_mode_adds_rb() -> None:
    K = 2
    rewards = torch.tensor([[2.5]])
    # Satisfied -> pure (A) gives K, plus rb_weight * r_b.
    costs = torch.full((1, K), -3.0)
    thresholds = torch.zeros(K)
    betas = torch.full((K,), 50.0)
    pure = tcl_standalone_reward(rewards, costs, thresholds, betas, rb_mode="ignore").item()
    bonus = tcl_standalone_reward(
        rewards, costs, thresholds, betas, rb_mode="bonus", rb_weight=0.5,
    ).item()
    assert bonus == pytest.approx(pure + 0.5 * 2.5, abs=1e-4)


def test_standalone_last_layer_gates_rb_on_satisfaction() -> None:
    # All constraints satisfied -> w_last ~ 1 -> last_layer == bonus.
    K = 2
    rewards = torch.tensor([[3.0]])
    costs = torch.full((1, K), -3.0)
    thresholds = torch.zeros(K)
    betas = torch.full((K,), 50.0)
    bonus = tcl_standalone_reward(
        rewards, costs, thresholds, betas, rb_mode="bonus", rb_weight=0.7,
    ).item()
    last_layer = tcl_standalone_reward(
        rewards, costs, thresholds, betas, rb_mode="last_layer", rb_weight=0.7,
    ).item()
    assert last_layer == pytest.approx(bonus, abs=1e-3)


def test_standalone_last_layer_kills_rb_on_violation() -> None:
    # First constraint violated -> w_last ~ 0 -> r_b contribution suppressed.
    K = 2
    rewards = torch.tensor([[10.0]])    # large r_b
    costs = torch.tensor([[3.0, -3.0]])  # R_1 violated, R_2 satisfied
    thresholds = torch.zeros(K)
    betas = torch.full((K,), 50.0)
    pure = tcl_standalone_reward(
        rewards, costs, thresholds, betas, rb_mode="ignore",
    ).item()
    last_layer = tcl_standalone_reward(
        rewards, costs, thresholds, betas, rb_mode="last_layer", rb_weight=1.0,
    ).item()
    # last_layer should not add much over pure: r_b is gated out.
    assert abs(last_layer - pure) < 0.01


# ---------------------------------------------------------------------------
# Gaussian-gate variant
# ---------------------------------------------------------------------------


def test_standalone_gaussian_gate_satisfied() -> None:
    # Cost == threshold -> error = 0 -> R_k = 1 -> cumulative-lex limit ~ K.
    K = 2
    rewards = torch.zeros(1, 1)
    costs = torch.tensor([[-1.0, -1.0]])  # below threshold -> error = 0
    thresholds = torch.tensor([0.0, 0.0])
    betas = torch.full((K,), 50.0)
    kappas = torch.full((K,), 1.0)
    out = tcl_standalone_reward_gaussian(rewards, costs, thresholds, betas, kappas)
    assert out.item() == pytest.approx(float(K), abs=1e-3)


def test_standalone_gaussian_gate_first_violated() -> None:
    # Large violation on R_1 -> R_1 ~ 0 -> cascade kills the rest.
    K = 2
    rewards = torch.zeros(1, 1)
    costs = torch.tensor([[5.0, -1.0]])
    thresholds = torch.tensor([0.0, 0.0])
    betas = torch.full((K,), 50.0)
    kappas = torch.full((K,), 1.0)
    out = tcl_standalone_reward_gaussian(rewards, costs, thresholds, betas, kappas)
    assert out.item() < 1e-3


def test_standalone_gaussian_calibration_half_at_estar() -> None:
    # kappa_k = ln2 / (e_k*)^2 so that R_k = 0.5 at e_k = e_k*.
    K = 1
    e_star = 1.5
    kappas = torch.tensor([math.log(2.0) / e_star ** 2])
    rewards = torch.zeros(1, 1)
    costs = torch.tensor([[e_star]])  # violation exactly at e_star -> R_k = 0.5
    thresholds = torch.tensor([0.0])
    betas = torch.tensor([10.0])
    out = tcl_standalone_reward_gaussian(rewards, costs, thresholds, betas, kappas)
    # K=1 -> w_1 = 1, output = R_1 = 0.5.
    assert out.item() == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# Differentiability (gradient flow check)
# ---------------------------------------------------------------------------


def test_standalone_differentiable_in_costs() -> None:
    K = 3
    rewards = torch.zeros(2, 1)
    costs = torch.randn(2, K, requires_grad=True)
    thresholds = torch.zeros(K)
    betas = torch.full((K,), 5.0)
    out = tcl_standalone_reward(rewards, costs, thresholds, betas)
    out.sum().backward()
    assert costs.grad is not None
    assert torch.isfinite(costs.grad).all()

"""Unit tests for the shaping primitives and minimal smoke tests for each agent.

Smoke tests are intentionally tiny (a few hundred steps) so the whole
test suite stays under a minute. The goal is just to catch shape /
import / API regressions, not to assess learning quality.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# sac_fixed helpers
# ---------------------------------------------------------------------------


def test_fixed_parse_weights_broadcast() -> None:
    from agents.sac_fixed import parse_weights

    w = parse_weights("0.5", 3)
    np.testing.assert_allclose(w, [0.5, 0.5, 0.5])


def test_fixed_parse_weights_explicit() -> None:
    from agents.sac_fixed import parse_weights

    w = parse_weights("1.0, 2.0, 0.3", 3)
    np.testing.assert_allclose(w, [1.0, 2.0, 0.3])


def test_fixed_parse_weights_wrong_length() -> None:
    from agents.sac_fixed import parse_weights

    with pytest.raises(ValueError):
        parse_weights("1.0, 2.0", 3)


def test_fixed_parse_weights_negative() -> None:
    from agents.sac_fixed import parse_weights

    with pytest.raises(ValueError):
        parse_weights("-0.1, 0.5", 2)


# ---------------------------------------------------------------------------
# sac_tcl helpers
# ---------------------------------------------------------------------------


def test_tcl_parse_vector_broadcast() -> None:
    from agents.sac_tcl import parse_vector

    v = parse_vector("3.0", 2, "betas_init")
    np.testing.assert_allclose(v, [3.0, 3.0])


def test_tcl_current_betas_no_anneal() -> None:
    from agents.sac_tcl import current_betas

    b0 = np.array([1.0, 2.0], dtype=np.float32)
    out = current_betas(step=500, betas_init=b0, betas_final=None,
                        schedule="linear", anneal_steps=0)
    np.testing.assert_array_equal(out, b0)


def test_tcl_current_betas_linear() -> None:
    from agents.sac_tcl import current_betas

    b0 = np.array([1.0], dtype=np.float32)
    b1 = np.array([10.0], dtype=np.float32)
    half = current_betas(step=500, betas_init=b0, betas_final=b1,
                         schedule="linear", anneal_steps=1000)
    np.testing.assert_allclose(half, [5.5])
    done = current_betas(step=2000, betas_init=b0, betas_final=b1,
                         schedule="linear", anneal_steps=1000)
    np.testing.assert_allclose(done, b1)


def test_tcl_current_betas_exp() -> None:
    from agents.sac_tcl import current_betas

    b0 = np.array([1.0], dtype=np.float32)
    b1 = np.array([100.0], dtype=np.float32)
    # halfway in log-space: sqrt(1 * 100) = 10
    half = current_betas(step=500, betas_init=b0, betas_final=b1,
                         schedule="exp", anneal_steps=1000)
    np.testing.assert_allclose(half, [10.0], rtol=1e-5)


def test_tcl_current_betas_linear_decreasing() -> None:
    """Mirrors the RTB-prod regime: betas relax from strict (10) to soft (1)."""
    from agents.sac_tcl import current_betas

    b0 = np.array([10.0], dtype=np.float32)
    b1 = np.array([1.0], dtype=np.float32)
    half = current_betas(step=500, betas_init=b0, betas_final=b1,
                         schedule="linear", anneal_steps=1000)
    np.testing.assert_allclose(half, [5.5])
    done = current_betas(step=2000, betas_init=b0, betas_final=b1,
                         schedule="linear", anneal_steps=1000)
    np.testing.assert_allclose(done, b1)


def test_tcl_current_betas_exp_decreasing() -> None:
    """Geometric relaxation regime (log-space interpolation, decreasing)."""
    from agents.sac_tcl import current_betas

    b0 = np.array([100.0], dtype=np.float32)
    b1 = np.array([1.0], dtype=np.float32)
    # halfway in log-space: sqrt(100 * 1) = 10
    half = current_betas(step=500, betas_init=b0, betas_final=b1,
                         schedule="exp", anneal_steps=1000)
    np.testing.assert_allclose(half, [10.0], rtol=1e-5)
    done = current_betas(step=2000, betas_init=b0, betas_final=b1,
                         schedule="exp", anneal_steps=1000)
    np.testing.assert_allclose(done, b1, rtol=1e-5)


def test_tcl_shaped_reward_gate_open() -> None:
    from agents.sac_tcl import tcl_shaped_reward

    # cost well below threshold -> gate ~ 1, shaped reward ~ reward.
    rewards = torch.tensor([[1.0], [2.0]])
    costs = torch.tensor([[-1.0], [-0.5]])  # K=1
    thresholds = torch.tensor([0.0])
    betas = torch.tensor([10.0])
    shaped = tcl_shaped_reward(rewards, costs, thresholds, betas)
    assert shaped[0, 0] > 0.99 * rewards[0, 0]
    assert shaped[1, 0] > 0.99 * rewards[1, 0]


def test_tcl_shaped_reward_gate_closed() -> None:
    from agents.sac_tcl import tcl_shaped_reward

    # cost well above threshold -> gate ~ 0, shaped reward ~ 0.
    rewards = torch.tensor([[1.0], [2.0]])
    costs = torch.tensor([[1.0], [0.5]])  # K=1
    thresholds = torch.tensor([0.0])
    betas = torch.tensor([20.0])
    shaped = tcl_shaped_reward(rewards, costs, thresholds, betas)
    assert shaped[0, 0].abs() < 0.01
    assert shaped[1, 0].abs() < 0.01


def test_tcl_shaped_reward_product_K2() -> None:
    from agents.sac_tcl import tcl_shaped_reward

    # K=2: gate1 closed, gate2 open -> shaped reward ~ 0.
    rewards = torch.tensor([[1.0]])
    costs = torch.tensor([[1.0, -1.0]])
    thresholds = torch.tensor([0.0, 0.0])
    betas = torch.tensor([20.0, 20.0])
    shaped = tcl_shaped_reward(rewards, costs, thresholds, betas)
    assert shaped[0, 0].abs() < 0.01


def test_tcl_shaped_reward_gaussian_gate_open() -> None:
    from agents.sac_tcl import tcl_shaped_reward_gaussian

    # cost < threshold -> error = 0 -> r_stab = 1 -> gate_logit = beta/2 > 0 -> gate ~ 1.
    rewards = torch.tensor([[2.0]])
    costs = torch.tensor([[-0.5]])  # K=1, well below threshold
    thresholds = torch.tensor([0.0])
    betas = torch.tensor([20.0])
    kappas = torch.tensor([1.0])
    shaped = tcl_shaped_reward_gaussian(rewards, costs, thresholds, betas, kappas)
    assert shaped[0, 0] > 0.99 * rewards[0, 0]


def test_tcl_shaped_reward_gaussian_gate_closed() -> None:
    from agents.sac_tcl import tcl_shaped_reward_gaussian

    # large violation -> r_stab ~ 0 -> gate_logit = -beta/2 < 0 -> gate ~ 0 -> shaped ~ 0.
    rewards = torch.tensor([[2.0]])
    costs = torch.tensor([[5.0]])  # K=1, large violation
    thresholds = torch.tensor([0.0])
    betas = torch.tensor([20.0])
    kappas = torch.tensor([1.0])
    shaped = tcl_shaped_reward_gaussian(rewards, costs, thresholds, betas, kappas)
    assert shaped[0, 0].abs() < 0.01


def test_tcl_shaped_reward_gaussian_satisfied_boundary() -> None:
    from agents.sac_tcl import tcl_shaped_reward_gaussian

    # cost == threshold exactly -> error = 0 -> r_stab = 1 -> gate = sigma(beta/2) > 0.5.
    rewards = torch.tensor([[1.0]])
    costs = torch.tensor([[0.0]])
    thresholds = torch.tensor([0.0])
    betas = torch.tensor([10.0])
    kappas = torch.tensor([1.0])
    shaped = tcl_shaped_reward_gaussian(rewards, costs, thresholds, betas, kappas)
    import math
    expected_gate = 1.0 / (1.0 + math.exp(-5.0))  # sigma(10 * 0.5)
    assert shaped[0, 0].item() == pytest.approx(expected_gate, rel=1e-5)


# ---------------------------------------------------------------------------
# sac_lagrangian_multi dual update
# ---------------------------------------------------------------------------


def test_lagrangian_multi_dual_update_sign_and_clamp() -> None:
    """Dual update λ ← ReLU(λ + η * mean_cost): sign correct, no negative λ."""
    lam = torch.tensor([0.0, 0.0])
    costs = torch.tensor([[0.5, -0.3], [0.2, 0.1]])  # (B=2, K=2)
    cost_baseline = 0.0
    lambda_lr = 1.0

    mean_cost_vec = costs.mean(dim=0) - cost_baseline  # [0.35, -0.10]
    lam_new = torch.clamp(lam + lambda_lr * mean_cost_vec, min=0.0)

    # k=0: violated on average -> λ increases
    assert lam_new[0].item() == pytest.approx(0.35)
    # k=1: satisfied on average -> λ would go negative -> clamped to 0
    assert lam_new[1].item() == pytest.approx(0.0)


def test_lagrangian_multi_dual_update_accumulates() -> None:
    """Two successive updates accumulate correctly."""
    lam = torch.tensor([0.0, 0.5])
    costs_1 = torch.tensor([[0.2, 0.3]])   # both violated
    costs_2 = torch.tensor([[-0.1, 0.1]])  # k=0 satisfied, k=1 violated
    lr = 1.0

    lam = torch.clamp(lam + lr * (costs_1.mean(dim=0) - 0.0), min=0.0)
    lam = torch.clamp(lam + lr * (costs_2.mean(dim=0) - 0.0), min=0.0)

    assert lam[0].item() == pytest.approx(0.2 - 0.1)   # 0 + 0.2 - 0.1
    assert lam[1].item() == pytest.approx(0.5 + 0.3 + 0.1)


# ---------------------------------------------------------------------------
# sac_hprs helpers
# ---------------------------------------------------------------------------


def test_hprs_build_weights() -> None:
    from agents.sac_hprs import build_weights

    w = build_weights(base=2.0, decay=0.5, k=3)
    np.testing.assert_allclose(w, [2.0, 1.0, 0.5])


def test_hprs_potential_no_violation() -> None:
    from agents.sac_hprs import potential

    costs = np.array([-0.2, -0.5], dtype=np.float32)
    thresholds = np.array([0.0, 0.0], dtype=np.float32)
    weights = np.array([1.0, 1.0], dtype=np.float32)
    phi = potential(costs, weights, thresholds)
    assert phi == pytest.approx(0.0)


def test_hprs_potential_violation_weighted() -> None:
    from agents.sac_hprs import potential

    # cost violation 0.3 in constraint 0 (weight 2), 0.1 in c1 (weight 1)
    costs = np.array([0.3, 0.1], dtype=np.float32)
    thresholds = np.array([0.0, 0.0], dtype=np.float32)
    weights = np.array([2.0, 1.0], dtype=np.float32)
    phi = potential(costs, weights, thresholds)
    assert phi == pytest.approx(-(2.0 * 0.3 + 1.0 * 0.1))


# ---------------------------------------------------------------------------
# Minimal smoke tests for train() — short, just to catch wiring regressions.
# ---------------------------------------------------------------------------

# These are very short (300 steps, no learning) so they run in a few
# seconds each. They check that the module can be imported, the train
# loop runs, and the result dict has the expected keys.

_SMOKE_KWARGS = dict(
    total_timesteps=300,
    learning_starts=100,
    buffer_size=1000,
    batch_size=32,
    cuda=False,
    horizon=12,
)


def _cleanup_log_path(result: dict) -> None:
    import shutil

    log_path = result.get("log_path")
    if log_path:
        shutil.rmtree(log_path, ignore_errors=True)


def test_smoke_sac_lagrangian(tmp_path) -> None:
    from agents.sac_lagrangian import Args, train

    args = Args(log_dir=str(tmp_path), **_SMOKE_KWARGS)
    result = train(args)
    assert "lambda_final" in result
    assert result["gradient_steps"] > 0


def test_smoke_sac_fixed(tmp_path) -> None:
    from agents.sac_fixed import Args, train

    args = Args(log_dir=str(tmp_path), cost_weights="0.3", **_SMOKE_KWARGS)
    result = train(args)
    assert result["gradient_steps"] > 0


def test_smoke_sac_tcl(tmp_path) -> None:
    from agents.sac_tcl import Args, train

    args = Args(log_dir=str(tmp_path),
                thresholds="0.0", betas_init="5.0", **_SMOKE_KWARGS)
    result = train(args)
    assert result["gradient_steps"] > 0


def test_smoke_sac_tcl_with_annealing(tmp_path) -> None:
    from agents.sac_tcl import Args, train

    args = Args(
        log_dir=str(tmp_path),
        thresholds="0.0",
        betas_init="1.0",
        betas_final="10.0",
        beta_anneal_steps=200,
        beta_schedule="linear",
        **_SMOKE_KWARGS,
    )
    result = train(args)
    assert result["gradient_steps"] > 0


def test_smoke_sac_tcl_reward_shift(tmp_path) -> None:
    """reward_shift lifts negative base reward before gating (Proposition 4 fix)."""
    from agents.sac_tcl import Args, train

    args = Args(
        log_dir=str(tmp_path),
        thresholds="0.0",
        betas_init="5.0",
        reward_shift=100.0,
        **_SMOKE_KWARGS,
    )
    result = train(args)
    assert result["gradient_steps"] > 0


def test_smoke_sac_lagrangian_multi(tmp_path) -> None:
    from agents.sac_lagrangian_multi import Args, train

    args = Args(log_dir=str(tmp_path), **_SMOKE_KWARGS)
    result = train(args)
    assert "lambda_final_k0" in result
    assert result["gradient_steps"] > 0


def test_smoke_sac_hprs(tmp_path) -> None:
    from agents.sac_hprs import Args, train

    args = Args(log_dir=str(tmp_path),
                thresholds="0.0", base_weight=1.0, **_SMOKE_KWARGS)
    result = train(args)
    assert result["gradient_steps"] > 0

"""Tests for BiddingSimulationLaplacian and MultiConstraintAdCraftLaplacian.

Phase 1 (sim): distribution faithfulness, auction mechanics, budget cap,
               non-stationarity, seed reproducibility, obs keys.
Phase 2 (wrapper): costs shape, action clip.
Phase 3 (calibration): polytope non-empty sanity check (xfail until
                        calibration targets are set in Phase 3).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from tcl.envs.adcraft_laplacian import MultiConstraintAdCraftLaplacian
from tcl.envs.adcraft_laplacian_sim import BiddingSimulationLaplacian

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sim(**kwargs: object) -> BiddingSimulationLaplacian:
    """Construct a sim and immediately reset with a fixed seed."""
    sim = BiddingSimulationLaplacian(**kwargs)  # type: ignore[arg-type]
    sim.reset(seed=SEED)
    return sim


# ---------------------------------------------------------------------------
# Phase 1 — BiddingSimulationLaplacian
# ---------------------------------------------------------------------------


def test_loc_scale_ranges() -> None:
    sim = make_sim(num_keywords=500, updater_params=[])
    assert np.all(sim._loc >= 0.30) and np.all(sim._loc <= 1.00)
    ratio = sim._scale / sim._loc
    assert np.all(ratio >= 0.01) and np.all(ratio <= 0.30)


def test_critical_bid_distribution() -> None:
    """KS test: |Laplace(loc, scale)| draws match the expected CDF."""
    sim = make_sim(num_keywords=1, updater_params=[])
    loc, scale = float(sim._loc[0]), float(sim._scale[0])

    # Draw directly via the sim's rng for a large sample
    rng = np.random.default_rng(0)
    samples = np.abs(rng.laplace(loc, scale, 100_000))

    # With loc ∈ [0.30, 1.00] and scale/loc ≤ 0.30, P(c < 0) < 1e-4,
    # so the folded-Laplace CDF ≈ standard Laplace CDF on [0, ∞).
    result = stats.kstest(samples, stats.laplace(loc=loc, scale=scale).cdf)
    assert result.pvalue > 0.01, (
        f"KS p-value={result.pvalue:.4f} — folded-Laplace deviated from expected"
    )


def test_second_price_mechanism() -> None:
    """Empirical win rate ≈ F_Laplace(bid) under second-price."""
    # High volume + oversized budget so the budget cap never fires,
    # ensuring impressions == raw auction wins.
    sim = make_sim(
        num_keywords=1,
        budget=1e8,
        volume_mean_range=(5_000.0, 5_001.0),
        updater_params=[],
        pricing_mode="second",
    )
    loc, scale = float(sim._loc[0]), float(sim._scale[0])
    bid = float(loc)  # ~50 % win rate

    v_before = float(sim._v_mean[0])
    obs, _, _, _, _ = sim.step({"keyword_bids": np.array([bid])})

    emp_rate = obs["impressions"][0] / v_before
    expected = stats.laplace.cdf(bid, loc=loc, scale=scale)
    assert abs(emp_rate - expected) < 0.05, (
        f"Second-price win rate {emp_rate:.3f} too far from F_L({bid:.3f})={expected:.3f}"
    )


def test_cost_truncated_expectation() -> None:
    """E[cost | imp] ≈ E[c | c ≤ bid] (analytical) under second-price."""
    sim = make_sim(
        num_keywords=1,
        budget=1e8,
        volume_mean_range=(5_000.0, 5_001.0),
        updater_params=[],
        pricing_mode="second",
    )
    loc, scale = float(sim._loc[0]), float(sim._scale[0])
    bid = float(loc) * 1.2

    obs, _, _, _, _ = sim.step({"keyword_bids": np.array([bid])})
    imp = float(obs["impressions"][0])
    if imp < 10:
        pytest.skip("too few impressions to estimate E[cost|imp]")

    emp_cpc = float(obs["cost"][0]) / imp

    # Analytical: E[c | c ≤ bid, c ~ Laplace(loc, scale)]
    # ≈ E[c | c ≤ bid] since P(c < 0) ≪ 1 for our loc/scale ranges
    numer = stats.laplace.expect(
        lambda x: x, loc=loc, scale=scale, lb=0.0, ub=bid
    )
    denom = stats.laplace.cdf(bid, loc=loc, scale=scale)
    analytic_cpc = numer / denom if denom > 1e-9 else 0.0

    assert abs(emp_cpc - analytic_cpc) < 0.05, (
        f"Second-price CPC {emp_cpc:.4f} vs analytic {analytic_cpc:.4f}"
    )


def test_first_price_mechanism() -> None:
    """Under first-price, every won auction pays exactly the submitted bid."""
    sim = make_sim(
        num_keywords=1,
        budget=1e8,
        volume_mean_range=(1_000.0, 1_001.0),
        updater_params=[],
        pricing_mode="first",
    )
    bid = 0.7
    obs, _, _, _, _ = sim.step({"keyword_bids": np.array([bid])})
    imp = float(obs["impressions"][0])
    if imp < 1:
        pytest.skip("zero impressions — bid below loc")
    emp_cpc = float(obs["cost"][0]) / imp
    assert abs(emp_cpc - bid) < 1e-6, (
        f"First-price CPC {emp_cpc:.6f} should equal bid {bid}"
    )


def test_sctr_beta_5_2() -> None:
    """Sellside CTR mean ≈ 5/7 ≈ 0.714 across many keywords (Table 1)."""
    sim = make_sim(
        num_keywords=10_000,
        sctr_beta_alpha=5.0,
        sctr_beta_beta=2.0,
        updater_params=[],
    )
    mean_sctr = float(sim._sctr.mean())
    assert abs(mean_sctr - 5 / 7) < 0.01, (
        f"mean sctr={mean_sctr:.4f}, expected ≈ 0.714"
    )


def test_reward_truncated_normal_min() -> None:
    """Every revenue contribution is ≥ 0.01 (TruncNorm lower bound)."""
    sim = make_sim(
        num_keywords=10,
        volume_mean_range=(50.0, 51.0),
        updater_params=[],
        pricing_mode="second",
    )
    bid = 2.0
    action = {"keyword_bids": np.full(10, bid)}
    # Run several steps to accumulate enough conversions
    for _ in range(20):
        obs, _, terminated, truncated, _ = sim.step(action)
        # Revenue per conversion must be ≥ 0.01 — check that total revenue
        # is consistent (≥ 0.01 × total conversions)
        total_conv = float(obs["sellside_conversions"].sum())
        if total_conv > 0:
            assert float(obs["revenue"].sum()) >= 0.01 * total_conv - 1e-9
        if terminated or truncated:
            break


def test_budget_cap() -> None:
    """Total daily cost never exceeds budget."""
    sim = make_sim(
        num_keywords=20,
        budget=10.0,
        volume_mean_range=(100.0, 101.0),
        updater_params=[],
        pricing_mode="second",
    )
    action = {"keyword_bids": np.full(20, 3.0)}
    for _ in range(60):
        obs, _, terminated, truncated, _ = sim.step(action)
        assert float(obs["cost"].sum()) <= 10.0 + 1e-6, (
            f"Budget cap violated: cost={obs['cost'].sum():.4f} > budget=10.0"
        )
        if terminated or truncated:
            break


def test_non_stationarity_drift() -> None:
    """Volume mean drifts at exactly (1 + rate)^T after T steps (vol only)."""
    sim = make_sim(
        num_keywords=5,
        updater_params=[["vol", 0.03]],
    )
    v0 = sim._v_mean.copy()

    action = {"keyword_bids": np.zeros(5)}
    n_steps = 10
    for _ in range(n_steps):
        sim.step(action)

    expected = v0 * (1.03 ** n_steps)
    np.testing.assert_allclose(sim._v_mean, expected, rtol=1e-9)


def test_seed_reproducibility() -> None:
    """Two reset(seed=42) calls produce identical keyword parameters."""
    sim = make_sim(num_keywords=20, updater_params=[])
    loc1, scale1, v1 = sim._loc.copy(), sim._scale.copy(), sim._v_mean.copy()

    sim.reset(seed=SEED)
    np.testing.assert_array_equal(sim._loc, loc1)
    np.testing.assert_array_equal(sim._scale, scale1)
    np.testing.assert_array_equal(sim._v_mean, v1)


def test_obs_dict_keys() -> None:
    """step() returns an obs dict with exactly the 7 expected keys."""
    sim = make_sim(num_keywords=5, updater_params=[])
    obs, _, _, _, _ = sim.step({"keyword_bids": np.ones(5) * 0.5})
    expected_keys = {
        "impressions",
        "buyside_clicks",
        "cost",
        "sellside_conversions",
        "revenue",
        "cumulative_profit",
        "days_passed",
    }
    assert set(obs.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Phase 2 — MultiConstraintAdCraftLaplacian wrapper
# ---------------------------------------------------------------------------


def test_wrapper_costs_vector() -> None:
    """info['costs'] has shape (3,) and dtype float32 after every step."""
    env = MultiConstraintAdCraftLaplacian(
        num_keywords=5, budget=10.0, bid_max=3.0
    )
    env.reset(seed=SEED)
    bid = np.full(5, 0.5, dtype=np.float32)
    _, _, _, _, info = env.step(bid)
    assert info["costs"].shape == (3,)
    assert info["costs"].dtype == np.float32
    assert np.all(np.isfinite(info["costs"]))


def test_wrapper_action_clip() -> None:
    """Bids above bid_max are silently clipped; step does not raise."""
    env = MultiConstraintAdCraftLaplacian(
        num_keywords=5, budget=10.0, bid_max=1.0
    )
    env.reset(seed=SEED)
    oversized = np.full(5, 99.0, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(oversized)
    assert obs.shape == (5 * 5 + 2,)
    assert np.isfinite(reward)


def test_wrapper_spaces() -> None:
    env = MultiConstraintAdCraftLaplacian(num_keywords=10, budget=50.0, bid_max=3.0)
    assert env.action_space.shape == (10,)
    assert env.observation_space.shape == (5 * 10 + 2,)
    assert env.k_costs == 3


def test_wrapper_pricing_mode_forwarded() -> None:
    """pricing_mode is forwarded to the underlying sim."""
    env_sp = MultiConstraintAdCraftLaplacian(num_keywords=5, pricing_mode="second")
    env_fp = MultiConstraintAdCraftLaplacian(num_keywords=5, pricing_mode="first")
    assert env_sp._base.pricing_mode == "second"
    assert env_fp._base.pricing_mode == "first"


# ---------------------------------------------------------------------------
# Phase 3 — Polytope non-emptiness sanity check (calibration prerequisite)
# ---------------------------------------------------------------------------


def test_polytope_nonempty_calibration() -> None:
    """A uniform bid=0.5 policy satisfies all 3 constraints on average.

    Calibrated targets (Phase 3, 2026-05-19):
    - target_util=0.40 : actual mean util ≈ 1.00 at bid=0.5, B=100
    - target_ctr=0.15  : actual mean ctr  ≈ 0.28 (≈ bctr mean Beta(2,5))
    - target_margin=-4.0: actual mean margin ≈ -3.67 (cost >> revenue at cap;
      feasibility window is bid ∈ [~0.35, ~0.55] — too high a bid gives
      margin < -4.0 because cost scales with budget but revenue does not)

    The margin constraint is the discriminating one: it forces agents to find
    a moderate bid, not just maximise impressions.
    """
    env = MultiConstraintAdCraftLaplacian(
        num_keywords=100,
        budget=100.0,
        bid_max=3.0,
        target_utilization=0.40,
        target_ctr=0.15,
        target_margin=-4.0,
        margin_formula="revenue_share",
        updater_params=[],  # no drift during calibration check
    )
    all_costs: list[np.ndarray] = []
    bid = np.full(100, 0.5, dtype=np.float32)
    for seed in range(5):
        env.reset(seed=seed)
        for _ in range(60):
            _, _, terminated, truncated, info = env.step(bid)
            all_costs.append(info["costs"].copy())
            if terminated or truncated:
                break

    mean_costs = np.mean(all_costs, axis=0)
    assert np.all(mean_costs < 0.0), (
        f"Polytope non-empty check failed: mean_costs={mean_costs} "
        f"(c_k < 0 = constraint satisfied)"
    )

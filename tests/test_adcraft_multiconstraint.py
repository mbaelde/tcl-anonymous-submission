"""Sanity checks for the MultiConstraintAdCraft wrapper.

These tests require the (Rust-backed) ``adcraft`` package; they are
skipped automatically if the import fails.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("adcraft.gymnasium_kw_env")

from tcl.envs import MultiConstraintAdCraft


def test_spaces() -> None:
    env = MultiConstraintAdCraft(num_keywords=4, budget=100.0, bid_max=5.0)
    assert env.action_space.shape == (4,)
    assert env.observation_space.shape == (5 * 4 + 2,)
    assert env.k_costs == 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_keywords": 0}, "num_keywords"),
        ({"budget": -1.0}, "budget"),
        ({"bid_max": 0.0}, "bid_max"),
        ({"target_utilization": 0.0}, "target_utilization"),
        ({"target_utilization": 1.1}, "target_utilization"),
        ({"target_ctr": 1.5}, "target_ctr"),
        ({"target_ctr": 0.0}, "target_ctr"),
    ],
)
def test_invalid_args(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MultiConstraintAdCraft(**kwargs)


def test_reset_returns_flat_obs_and_zero_costs() -> None:
    env = MultiConstraintAdCraft(num_keywords=3, budget=50.0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (5 * 3 + 2,)
    assert obs.dtype == np.float32
    assert "costs" in info
    np.testing.assert_array_equal(info["costs"], np.zeros(3, dtype=np.float32))


def test_step_costs_vector_shape_and_finite() -> None:
    env = MultiConstraintAdCraft(num_keywords=5, budget=200.0, bid_max=3.0)
    env.reset(seed=1)
    action = np.full((5,), 1.0, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (5 * 5 + 2,)
    assert info["costs"].shape == (3,)
    assert np.all(np.isfinite(info["costs"]))
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_action_shape_mismatch_raises() -> None:
    env = MultiConstraintAdCraft(num_keywords=4, budget=100.0)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="action shape"):
        env.step(np.zeros(3, dtype=np.float32))


def test_underspend_triggers_c1_positive() -> None:
    """With a generous budget and tiny bids, c1 (utilization shortfall) > 0.

    AdCraft hard-caps spend at the per-step budget, so the budget
    constraint is framed as a utilization floor: c1 > 0 iff the agent
    leaves budget on the table.
    """
    env = MultiConstraintAdCraft(
        num_keywords=10,
        budget=1000.0,
        bid_max=5.0,
        target_utilization=0.8,
        max_days=30,
    )
    env.reset(seed=2)
    # Bid almost nothing → cost ≈ 0 → c1 ≈ target_utilization (0.8).
    c1_values = []
    for _ in range(5):
        _, _, term, trunc, info = env.step(
            np.full((10,), 0.01, dtype=np.float32)
        )
        c1_values.append(float(info["costs"][0]))
        if term or trunc:
            break
    assert max(c1_values) > 0.0, f"c1 never positive; got {c1_values}"


def test_zero_bids_zero_traffic_costs_safe() -> None:
    """With zero bids: c1=target_util (underspend), c2=0 (no impressions),
    c3=target_margin (margin=0 since no revenue/cost → full violation)."""
    env = MultiConstraintAdCraft(
        num_keywords=4, budget=100.0, bid_max=2.0, target_utilization=0.8
    )
    env.reset(seed=3)
    _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    # cost=0 → c1=target_utilization; no impressions → c2=target_ctr (limit imp→0⁺);
    # cost=0 → margin=0 (guarded denominator) → c3=target_margin (default 0.10)
    assert info["costs"][0] == pytest.approx(0.8, abs=1e-6)
    assert info["costs"][1] == pytest.approx(0.05, abs=1e-6)
    assert info["costs"][2] == pytest.approx(0.10, abs=1e-6)


@pytest.mark.xfail(
    reason=(
        "AdCraft's Rust extension uses thread_rng() (see "
        "adcraft/src/lib.rs lines 25, 43, 61, 75, 320) for auction "
        "sampling, so the trajectory is not reproducible from the "
        "gymnasium seed. Upstream TODO at lib.rs:316 acknowledges this. "
        "Determinism only holds for keyword setup and the drift update."
    ),
    strict=True,
)
def test_seed_determinism() -> None:
    """Same seed → identical trajectory (currently broken upstream)."""
    def roll(seed: int) -> list[float]:
        env = MultiConstraintAdCraft(num_keywords=3, budget=20.0, bid_max=2.0)
        env.reset(seed=seed)
        rewards: list[float] = []
        rng = np.random.default_rng(seed)
        for _ in range(5):
            a = rng.uniform(0.0, 2.0, size=3).astype(np.float32)
            _, r, _, _, _ = env.step(a)
            rewards.append(r)
        return rewards

    np.testing.assert_allclose(roll(7), roll(7), rtol=0.0, atol=0.0)


def test_non_stationarity_active() -> None:
    """Default updater drifts keyword params: ctr after 20 steps should
    differ from initial buyside_ctr for at least one keyword."""
    env = MultiConstraintAdCraft(num_keywords=5, budget=100.0, bid_max=2.0)
    env.reset(seed=4)
    initial_ctrs = [kw.buyside_ctr for kw in env._base.keywords]
    for _ in range(20):
        env.step(np.full((5,), 1.0, dtype=np.float32))
    final_ctrs = [kw.buyside_ctr for kw in env._base.keywords]
    drift = max(abs(a - b) for a, b in zip(initial_ctrs, final_ctrs, strict=True))
    assert drift > 0.0, "expected non-stationary drift, got static CTR"

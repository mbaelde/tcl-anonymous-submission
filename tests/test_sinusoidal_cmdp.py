"""Sanity checks for the SinusoidalCMDP toy environment."""

from __future__ import annotations

import numpy as np
import pytest

from tcl.envs import SinusoidalCMDP


def test_spaces() -> None:
    env = SinusoidalCMDP(horizon=10)
    assert env.action_space.shape == (1,)
    assert env.observation_space.shape == (2,)


def test_invalid_b0() -> None:
    with pytest.raises(ValueError):
        SinusoidalCMDP(b0=-0.1)
    with pytest.raises(ValueError):
        SinusoidalCMDP(b0=1.5)


def test_invalid_amplitude_out_of_range() -> None:
    with pytest.raises(ValueError):
        SinusoidalCMDP(b0=0.5, amplitude=0.7)  # 0.5 + 0.7 > 1


def test_reset_and_step_basic() -> None:
    env = SinusoidalCMDP(horizon=4, b0=0.5, amplitude=0.1, omega=np.pi / 2,
                         random_phase_at_reset=False, phase_offset=0.0)
    obs, info = env.reset()
    assert obs.shape == (2,)
    np.testing.assert_allclose(obs, [0.0, 1.0], atol=1e-6)
    assert pytest.approx(info["budget_cap"], abs=1e-9) == 0.5  # sin(0) = 0

    obs, r, term, trunc, info = env.step(np.array([0.5]))
    assert not term
    assert not trunc
    # reward = 0.5 - 0.5 * 0.25 = 0.375
    assert pytest.approx(r, abs=1e-9) == 0.375


def test_episode_truncates_at_horizon() -> None:
    env = SinusoidalCMDP(horizon=3)
    env.reset()
    for _ in range(2):
        _, _, term, trunc, _ = env.step(np.array([0.3]))
        assert not term and not trunc
    _, _, term, trunc, _ = env.step(np.array([0.3]))
    assert not term and trunc


def test_budget_cap_periodic() -> None:
    env = SinusoidalCMDP(horizon=8, b0=0.5, amplitude=0.2, omega=np.pi / 2,
                         random_phase_at_reset=False, phase_offset=0.0)
    env.reset()
    caps = []
    for _ in range(8):
        _, _, _, _, info = env.step(np.array([0.0]))
        caps.append(info["budget_cap"])
    # phase advances by pi/2 per step ; one full cycle = 4 steps
    assert pytest.approx(caps[0], abs=1e-6) == 0.5
    assert pytest.approx(caps[1], abs=1e-6) == 0.7  # sin(pi/2)=1
    assert pytest.approx(caps[2], abs=1e-6) == 0.5
    assert pytest.approx(caps[3], abs=1e-6) == 0.3
    assert pytest.approx(caps[4], abs=1e-6) == 0.5  # back to start


def test_random_phase_diversity() -> None:
    env = SinusoidalCMDP(horizon=4, random_phase_at_reset=True)
    phases = set()
    for seed in range(20):
        _, info = env.reset(seed=seed)
        phases.add(round(info["phase"], 6))
    assert len(phases) > 5  # not all identical


def test_predicted_amplitude_property() -> None:
    env = SinusoidalCMDP(horizon=144, b0=0.5, amplitude=0.2,
                         omega=2.0 * np.pi / 144.0)
    # 2 * A / omega = 2 * 0.2 / (2*pi/144) ≈ 9.167
    assert pytest.approx(env.predicted_dual_amplitude, rel=1e-4) == 2 * 0.2 * 144 / (2 * np.pi)


def test_optimal_lagrangian() -> None:
    env = SinusoidalCMDP(b0=0.3)
    assert env.optimal_lagrangian == pytest.approx(0.7)

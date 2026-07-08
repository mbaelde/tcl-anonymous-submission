"""Single-constraint contextual MDP with sinusoidally non-stationary budget cap.

This environment is the experimental counterpart of Proposition 2 in the
TCL paper. It is the minimal setting that exhibits the dual-oscillation
mechanism analyzed there:

    A campaign-like agent must choose, at each of `horizon` steps in an
    episode, an action $a_t \\in [0, 1]$ (a "bid intensity"). The
    instantaneous reward is concave in the action,

        r(s, a) = a - 0.5 a^2,

    so the *unconstrained* optimum is $a^\\star = 1/2$. The agent is
    subject to a periodically non-stationary budget constraint with
    angular frequency $\\omega$ (radians per step):

        cost(s, a) = a - b(\\phi_t),
        b(\\phi)   = b_0 + A \\sin(\\phi),
        \\phi_t   = \\omega t,

    and the episodic constraint $\\sum_t \\text{cost}_t \\le 0$ enforces
    the average budget $b_0$.

The observation is the 2-vector $(\\sin\\phi_t, \\cos\\phi_t)$, which embeds
the phase smoothly on the unit circle and makes the environment Markov
without exposing a raw, discontinuous time index.

Under a Lagrangian-augmented policy gradient method, the dual variable
$\\lambda(t)$ tracks $a_t - b(\\phi_t)$ with rate $\\alpha$ (the dual
learning rate). In steady state, $\\lambda(t)$ oscillates with peak-to-
peak amplitude

    $\\Delta\\lambda \\approx \\frac{2 \\alpha A}{\\sqrt{\\alpha^2 + \\omega^2}}
                          \\approx \\frac{2 \\alpha A}{\\omega}$
    when $\\omega \\gg \\alpha$,

which is the $\\mathcal{O}(\\alpha/\\omega)$ behaviour stated by
Proposition 2. The amplitude *grows* as $\\omega$ shrinks — i.e. slow
non-stationarity is the worst regime for Lagrangian stability — until
saturating at $A$ when $\\omega \\lesssim \\alpha$.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SinusoidalCMDP(gym.Env[np.ndarray, np.ndarray]):
    """Periodic single-constraint CMDP for Proposition 2 validation.

    Parameters
    ----------
    horizon
        Number of steps per episode.
    b0
        Baseline (mean) budget cap, in $(0, 1)$.
    amplitude
        Amplitude $A$ of the sinusoidal variation of the budget cap.
        Must satisfy $b_0 \\pm A \\subset (0, 1)$ for the cap to remain
        physically meaningful at every phase.
    omega
        Angular frequency in radians per environment step. One full
        period over an episode of length $H$ corresponds to
        $\\omega = 2\\pi / H$.
    phase_offset
        Initial phase $\\phi_0$ at episode start. Defaults to 0.0 for
        deterministic non-stationarity; set to a non-zero value (or
        randomize via `random_phase_at_reset`) to add diversity.
    random_phase_at_reset
        If True, the initial phase at each `reset()` is drawn uniformly
        in $[0, 2\\pi)$. This is the recommended setting when training:
        the agent then has to learn the *function* $a^\\star(\\phi)$
        rather than a fixed schedule.
    reward_shift
        Constant added to the reward at each step. Defaults to 0.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        horizon: int = 144,
        b0: float = 0.5,
        amplitude: float = 0.2,
        omega: float = 2.0 * np.pi / 144.0,
        phase_offset: float = 0.0,
        random_phase_at_reset: bool = True,
        reward_shift: float = 0.0,
    ) -> None:
        super().__init__()

        if not 0.0 < b0 < 1.0:
            raise ValueError(f"b0 must be in (0, 1); got {b0}")
        if amplitude < 0.0:
            raise ValueError(f"amplitude must be non-negative; got {amplitude}")
        if b0 - amplitude < 0.0 or b0 + amplitude > 1.0:
            raise ValueError(
                f"b0 +/- amplitude must lie in [0, 1]; got b0={b0}, A={amplitude}"
            )
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {horizon}")
        if omega < 0.0:
            raise ValueError(f"omega must be non-negative; got {omega}")

        self.horizon = int(horizon)
        self.b0 = float(b0)
        self.amplitude = float(amplitude)
        self.omega = float(omega)
        self.phase_offset = float(phase_offset)
        self.random_phase_at_reset = bool(random_phase_at_reset)
        self.reward_shift = float(reward_shift)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._step: int = 0
        self._phase0: float = self.phase_offset

    def _phase(self, t: int) -> float:
        return float((self._phase0 + self.omega * t) % (2.0 * np.pi))

    def _budget_cap(self, phase: float) -> float:
        return self.b0 + self.amplitude * float(np.sin(phase))

    def _obs(self, phase: float) -> np.ndarray:
        return np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step = 0
        if self.random_phase_at_reset:
            self._phase0 = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        else:
            self._phase0 = self.phase_offset
        phase = self._phase(0)
        info = {"phase": phase, "budget_cap": self._budget_cap(phase)}
        return self._obs(phase), info

    def step(
        self, action: np.ndarray | float
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = float(np.clip(np.asarray(action, dtype=np.float64).item(), 0.0, 1.0))

        phase = self._phase(self._step)
        cap = self._budget_cap(phase)
        reward = a - 0.5 * a * a + self.reward_shift
        cost = a - cap

        self._step += 1
        terminated = False
        truncated = self._step >= self.horizon

        next_phase = self._phase(self._step)
        info: dict[str, Any] = {
            "phase": phase,
            "next_phase": next_phase,
            "budget_cap": cap,
            "cost": cost,
            "costs": np.array([cost], dtype=np.float32),
            "action": a,
        }
        return self._obs(next_phase), float(reward), terminated, truncated, info

    @property
    def optimal_lagrangian(self) -> float:
        r"""Closed-form $\lambda^\star$ for the stationary problem.

        With reward $a - 0.5 a^2$ and binding budget $a = b_0$ on average,
        the inner-loop Lagrangian optimum is $a^\star = 1 - \lambda$, so
        $\lambda^\star = 1 - b_0$.
        """
        return 1.0 - self.b0

    @property
    def predicted_dual_amplitude(self) -> float:
        r"""Predicted steady-state peak-to-peak amplitude of $\lambda(t)$ ...

        ... under continuous-time dual ascent of rate $\alpha$, assuming
        the policy tracks $a_t = 1 - \lambda_t$ on a faster timescale.
        Returns the *asymptotic* expression that depends only on the
        environment (multiplied later by $\alpha$ to get the actual
        amplitude):

            amplitude($\alpha$) = $\frac{2 \alpha A}{\sqrt{\alpha^2 + \omega^2}}$.

        This property returns just $2 A / \sqrt{1 + (\omega/\alpha)^2}$,
        i.e. the amplitude divided by $\alpha$, leaving callers to scale
        by $\alpha$ explicitly when comparing to empirical measurements.
        """
        # Returns the geometric factor; caller multiplies by alpha.
        # amplitude = 2 * alpha * A / sqrt(alpha^2 + omega^2)
        # so geometric factor = 2 * A / sqrt(1 + (omega/alpha)^2) is alpha-dependent.
        # We return the omega-only factor 2*A/omega valid in the omega >> alpha regime.
        return 2.0 * self.amplitude / max(self.omega, 1e-12)

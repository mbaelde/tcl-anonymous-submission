"""Multi-constraint wrapper around AdCraft's BiddingSimulation.

Adapts the SEM keyword auction benchmark of Gomrokchi et al. (2023,
`github.com/Mikata-Project/adcraft`) so it exposes the K-generic
constraint API consumed by the agents in ``agents/``:

* a flat ``Box`` action space (per-keyword bids, budget held fixed);
* a flat ``Box`` observation space (per-keyword arrays concatenated);
* a vector ``info["costs"]`` of K=3 constraint violations,

    .. math::

        c_1 = \\tau_{\\mathrm{util}} - \\frac{\\sum_i \\mathrm{cost}_i}
                                              {\\mathrm{budget}},
        \\qquad
        c_2 = \\tau_{\\mathrm{ctr}} - \\frac{\\sum_i \\mathrm{clicks}_i}
                                              {\\sum_i \\mathrm{impressions}_i},
        \\qquad
        c_3 = \\tau_{\\mathrm{margin}} - m,

where the margin :math:`m` is one of (selected via ``margin_formula``):

* ``"revenue_share"`` (recommended): :math:`m = (\\mathrm{revenue} -
  \\mathrm{cost}) / \\mathrm{revenue}`. Matches the anonymous institution convention
  (target ~0.70).
* ``"cost_markup"`` (legacy, default): :math:`m = (\\mathrm{revenue} -
  \\mathrm{cost}) / \\mathrm{cost}`. Default for pre-2026-05 runs
  (target was 0.10); retained for backward compatibility on existing
  artefacts.

The convention ``c_k > 0`` ⇔ constraint k is violated at step t
matches the threshold convention in :func:`agents.sac_tcl.tcl_shaped_reward`
and its baselines with default ``tau_k = 0``.

Note: AdCraft's :class:`BiddingSimulation` hard-caps per-step spend at
``budget`` (see ``bidding_simulation.simulate_epoch_of_bidding_on_campaign``),
so ``cost/budget`` is in ``[0, 1]``. We therefore frame the budget
constraint as a *utilization floor* (under-spending is the violation, not
overspending), which is the only spend-side constraint the underlying
simulator can express.

The native non-stationarity of AdCraft (``BiddingSimulation.update_keywords``)
is left untouched, so the dynamics-as-non-stationarity character of
the original benchmark is preserved for §7.1 of the TCL paper.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Numerical guard for divisions on near-zero denominators.
_EPS = 1e-8


class MultiConstraintAdCraft(gym.Env[np.ndarray, np.ndarray]):
    """K=3 multi-constraint wrapper around AdCraft ``BiddingSimulation``.

    Parameters
    ----------
    num_keywords
        Number of keywords bid upon at each step.
    budget
        Per-step budget passed to ``BiddingSimulation``. Held fixed
        across the episode (not part of the action).
    bid_max
        Upper bound of the per-keyword bid action space. Bids are
        clipped to ``[0, bid_max]`` before being forwarded to AdCraft.
    target_utilization
        Threshold for the budget-utilization floor constraint $c_1$.
        A violation is any step at which spent share of budget falls
        below this value (i.e. the agent under-bids and leaves budget
        on the table). Must lie in $(0, 1]$.
    target_ctr
        Threshold for the CTR-floor constraint $c_2$. A violation is
        any step at which the aggregate keyword CTR falls below this
        value.
    target_margin
        Threshold for the margin-floor constraint $c_3$. A violation is
        any step at which the realized margin falls below this value.
        See ``margin_formula`` for the definition of margin.
    margin_formula
        Definition of "margin" used for $c_3$:
        - ``"revenue_share"`` (default): $m = (\\mathrm{revenue} -
          \\mathrm{cost}) / \\mathrm{revenue}$. Matches the convention
          used at anonymous institution (target ~0.70). Range $(-\\infty, 1]$.
        - ``"cost_markup"`` (legacy): $m = (\\mathrm{revenue} -
          \\mathrm{cost}) / \\mathrm{cost}$. The default of pre-2026-05
          runs (target was 0.10). Kept for backward compatibility on
          existing artefacts; not recommended for new experiments.
    max_days
        Episode horizon in steps.
    loss_threshold
        Cumulative-loss truncation threshold (AdCraft default 10000).
    updater_params
        AdCraft non-stationarity parameters, see
        ``BiddingSimulation.update_keywords``. Defaults to
        ``[["vol", 0.03], ["ctr", 0.03], ["cvr", 0.03]]`` (the AdCraft
        default).
    updater_mask
        Per-keyword boolean mask controlling which keywords drift.
        Defaults to all ``True``.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        num_keywords: int = 10,
        budget: float = 1000.0,
        bid_max: float = 10.0,
        target_utilization: float = 0.8,
        target_ctr: float = 0.05,
        target_margin: float = 0.10,
        margin_formula: str = "cost_markup",
        max_days: int = 60,
        loss_threshold: float = 10000.0,
        updater_params: list[list[Any]] | None = None,
        updater_mask: list[bool] | None = None,
    ) -> None:
        super().__init__()

        if num_keywords < 1:
            raise ValueError(f"num_keywords must be >= 1; got {num_keywords}")
        if budget <= 0.0:
            raise ValueError(f"budget must be > 0; got {budget}")
        if bid_max <= 0.0:
            raise ValueError(f"bid_max must be > 0; got {bid_max}")
        if not 0.0 < target_utilization <= 1.0:
            raise ValueError(
                f"target_utilization must be in (0, 1]; got {target_utilization}"
            )
        if not 0.0 < target_ctr < 1.0:
            raise ValueError(f"target_ctr must be in (0, 1); got {target_ctr}")
        if margin_formula not in ("revenue_share", "cost_markup"):
            raise ValueError(
                f"margin_formula must be 'revenue_share' or 'cost_markup'; "
                f"got {margin_formula!r}"
            )

        # Import locally so this module is importable even on machines
        # where AdCraft (which has a Rust extension) is not installed
        # yet; the failure surfaces only when the env is instantiated.
        from adcraft.gymnasium_kw_env import BiddingSimulation

        if updater_params is None:
            updater_params = [["vol", 0.03], ["ctr", 0.03], ["cvr", 0.03]]
        if updater_mask is None:
            updater_mask = [True] * num_keywords

        self.num_keywords = int(num_keywords)
        self.budget = float(budget)
        self.bid_max = float(bid_max)
        self.target_utilization = float(target_utilization)
        self.target_ctr = float(target_ctr)
        self.target_margin = float(target_margin)
        self.margin_formula = str(margin_formula)
        self.max_days = int(max_days)
        self.k_costs = 3

        self._base = BiddingSimulation(
            num_keywords=self.num_keywords,
            budget=self.budget,
            max_days=self.max_days,
            loss_threshold=loss_threshold,
            updater_params=updater_params,
            updater_mask=updater_mask,
        )

        n = self.num_keywords
        self.action_space = spaces.Box(
            low=0.0, high=self.bid_max, shape=(n,), dtype=np.float32
        )
        # 5 per-keyword fields + 2 scalars = 5n + 2.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5 * n + 2,), dtype=np.float32
        )

    def _flatten_obs(self, obs: dict[str, Any]) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(obs["impressions"], dtype=np.float32).ravel(),
                np.asarray(obs["buyside_clicks"], dtype=np.float32).ravel(),
                np.asarray(obs["cost"], dtype=np.float32).ravel(),
                np.asarray(obs["sellside_conversions"], dtype=np.float32).ravel(),
                np.asarray(obs["revenue"], dtype=np.float32).ravel(),
                np.atleast_1d(np.asarray(obs["cumulative_profit"], dtype=np.float32)).ravel(),
                np.atleast_1d(np.asarray(obs["days_passed"], dtype=np.float32)).ravel(),
            ]
        )

    def _costs_vector(self, obs: dict[str, Any]) -> np.ndarray:
        cost = float(np.sum(obs["cost"]))
        impressions = float(np.sum(obs["impressions"]))
        clicks = float(np.sum(obs["buyside_clicks"]))
        revenue = float(np.sum(obs["revenue"]))

        c1 = self.target_utilization - cost / self.budget
        c2 = self.target_ctr - clicks / impressions if impressions > _EPS else self.target_ctr
        if self.margin_formula == "revenue_share":
            margin = (revenue - cost) / revenue if revenue > _EPS else 0.0
        else:  # cost_markup (legacy)
            margin = (revenue - cost) / cost if cost > _EPS else 0.0
        c3 = self.target_margin - margin
        return np.array([c1, c2, c3], dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = self._base.reset(seed=seed, options=options)
        out_info: dict[str, Any] = dict(info)
        out_info["costs"] = np.zeros(self.k_costs, dtype=np.float32)
        return self._flatten_obs(obs), out_info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != self.num_keywords:
            raise ValueError(
                f"action shape {a.shape} != expected ({self.num_keywords},)"
            )
        a = np.clip(a, 0.0, self.bid_max).astype(np.float32)
        action_dict: dict[str, Any] = {"keyword_bids": a}
        obs, reward, terminated, truncated, info = self._base.step(action_dict)
        out_info: dict[str, Any] = dict(info)
        out_info["costs"] = self._costs_vector(obs)
        return self._flatten_obs(obs), float(reward), terminated, truncated, out_info

    def render(self) -> Any:
        return self._base.render()

    def close(self) -> None:
        self._base.close()

"""Multi-constraint Gymnasium wrapper around BiddingSimulationLaplacian.

Drop-in replacement for MultiConstraintAdCraft: same K=3 constraint API,
same spaces formula (5n + 2), same info["costs"] convention. Swaps the
upstream Rust-backed BiddingSimulation for the pure-Python Laplacian sim.

See MultiConstraintAdCraft for the constraint definitions (c_1=util,
c_2=ctr, c_3=margin) and the c_k > 0 ⟺ violation convention.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tcl.envs.adcraft_laplacian_sim import BiddingSimulationLaplacian

_EPS = 1e-8


class MultiConstraintAdCraftLaplacian(gym.Env[np.ndarray, np.ndarray]):
    """K=3 multi-constraint wrapper around BiddingSimulationLaplacian.

    Parameters
    ----------
    num_keywords : int
        Number of keywords. Paper default: 100.
    budget : float
        Per-step budget passed to the sim.
    bid_max : float
        Upper bound of the per-keyword bid action space.
        Paper §G.1 bid grid: [0.01, 3.00] → default bid_max=3.0.
    target_utilization : float
        Budget-utilization floor threshold (must be in (0, 1]).
    target_ctr : float
        CTR floor threshold (must be in (0, 1)).
    target_margin : float
        Margin floor threshold (can be negative for loss-budget formulation).
    margin_formula : {"revenue_share", "cost_markup"}
        Definition of margin for c_3.
    max_days : int
        Episode horizon in steps.
    loss_threshold : float
        Early-termination cumulative-loss threshold.
    pricing_mode : {"second", "first"}
        Passed to BiddingSimulationLaplacian.
        ``"second"`` matches the AdCraft paper (§G.1 default).
        ``"first"`` matches the Anonymous Institution production first-price auction.
    updater_params, updater_mask
        Non-stationarity spec forwarded to the sim.
    loc_range, scale_loc_ratio_range, bctr_beta_alpha, bctr_beta_beta,
    sctr_beta_alpha, sctr_beta_beta, reward_mean_range,
    reward_std_to_mean_ratio, volume_mean_range
        §G.1 distribution parameters forwarded to the sim.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        num_keywords: int = 100,
        budget: float = 100.0,
        bid_max: float = 3.0,
        target_utilization: float = 0.5,
        target_ctr: float = 0.05,
        target_margin: float = -0.5,
        margin_formula: str = "revenue_share",
        max_days: int = 60,
        loss_threshold: float = 10000.0,
        pricing_mode: str = "second",
        updater_params: list[list[Any]] | None = None,
        updater_mask: list[bool] | None = None,
        loc_range: tuple[float, float] = (0.30, 1.00),
        scale_loc_ratio_range: tuple[float, float] = (0.01, 0.30),
        bctr_beta_alpha: float = 2.0,
        bctr_beta_beta: float = 5.0,
        sctr_beta_alpha: float = 5.0,
        sctr_beta_beta: float = 2.0,
        reward_mean_range: tuple[float, float] = (0.30, 0.50),
        reward_std_to_mean_ratio: float = 0.30,
        volume_mean_range: tuple[float, float] = (5.0, 30.0),
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

        self._base = BiddingSimulationLaplacian(
            num_keywords=self.num_keywords,
            budget=self.budget,
            max_days=self.max_days,
            loss_threshold=loss_threshold,
            updater_params=updater_params,
            updater_mask=updater_mask,
            loc_range=loc_range,
            scale_loc_ratio_range=scale_loc_ratio_range,
            bctr_beta_alpha=bctr_beta_alpha,
            bctr_beta_beta=bctr_beta_beta,
            sctr_beta_alpha=sctr_beta_alpha,
            sctr_beta_beta=sctr_beta_beta,
            reward_mean_range=reward_mean_range,
            reward_std_to_mean_ratio=reward_std_to_mean_ratio,
            volume_mean_range=volume_mean_range,
            pricing_mode=pricing_mode,
        )

        n = self.num_keywords
        self.action_space = spaces.Box(
            low=0.0, high=self.bid_max, shape=(n,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5 * n + 2,), dtype=np.float32
        )

    def _flatten_obs(self, obs: dict[str, Any]) -> np.ndarray:
        return np.concatenate([
            np.asarray(obs["impressions"], dtype=np.float32).ravel(),
            np.asarray(obs["buyside_clicks"], dtype=np.float32).ravel(),
            np.asarray(obs["cost"], dtype=np.float32).ravel(),
            np.asarray(obs["sellside_conversions"], dtype=np.float32).ravel(),
            np.asarray(obs["revenue"], dtype=np.float32).ravel(),
            np.atleast_1d(np.asarray(obs["cumulative_profit"], dtype=np.float32)).ravel(),
            np.atleast_1d(np.asarray(obs["days_passed"], dtype=np.float32)).ravel(),
        ])

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
        obs, reward, terminated, truncated, info = self._base.step(
            {"keyword_bids": a}
        )
        out_info: dict[str, Any] = dict(info)
        out_info["costs"] = self._costs_vector(obs)
        return self._flatten_obs(obs), float(reward), terminated, truncated, out_info

    def render(self) -> Any:
        return None

    def close(self) -> None:
        self._base.close()

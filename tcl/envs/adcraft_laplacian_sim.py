"""Pure-Python reimplementation of the AdCraft BiddingSimulation.

Faithful to §G.1 + Table 1 of Gomrokchi et al. (arXiv:2306.11971).
Replaces Mikata-Project/adcraft (abandoned Aug 2023, deviates from the
paper on three independent points — see decision_option_b.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np

_EPS = 1e-9


class BiddingSimulationLaplacian:
    """SEM bidding simulator matching §G.1 + Table 1 (pure Python / NumPy).

    Parameters
    ----------
    num_keywords : int
        Number of keywords. Paper default: 100.
    budget : float
        Per-step spend cap; daily cost is scaled proportionally if exceeded.
    max_days : int
        Episode horizon in days (steps).
    loss_threshold : float
        Episode terminates early when cumulative profit < −threshold.
    updater_params : list of [str, float]
        Non-stationarity spec. Each (feature, rate) multiplies the
        corresponding per-keyword parameter by ``(1 + rate)`` every step.
        Supported features: ``"vol"``, ``"ctr"``, ``"cvr"``.
    updater_mask : list of bool
        Per-keyword mask; only masked-True keywords undergo drift.
    loc_range : (float, float)
        Uniform range for Laplace ``loc_k`` (§G.1: [0.30, 1.00]).
    scale_loc_ratio_range : (float, float)
        Uniform range for ``scale_k / loc_k`` (§G.1: [0.01, 0.30]).
    bctr_beta_alpha, bctr_beta_beta : float
        Beta(α, β) parameters for buyside CTR per keyword.
    sctr_beta_alpha, sctr_beta_beta : float
        Beta(α, β) for sellside paid CTR. Paper Table 1: Beta(5, 2),
        mean ≈ 5/7 ≈ 0.714.
    reward_mean_range : (float, float)
        Uniform range for per-keyword reward mean μ_R (Table 1: ~0.40).
    reward_std_to_mean_ratio : float
        σ_R = ratio × μ_R. Paper Table 1: 0.30.
    volume_mean_range : (float, float)
        Uniform range for per-keyword daily Poisson volume mean v̄_k.
    pricing_mode : {"second", "first"}
        Auction pricing rule:
        - ``"second"`` (default): agent wins when b ≥ c, **pays c**
          (paper §G.1 standard).
        - ``"first"``: agent wins when b ≥ c, **pays b**
          (Anonymous Institution production first-price auction).
    """

    def __init__(
        self,
        num_keywords: int = 100,
        budget: float = 100.0,
        max_days: int = 60,
        loss_threshold: float = 10000.0,
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
        pricing_mode: str = "second",
    ) -> None:
        if pricing_mode not in ("second", "first"):
            raise ValueError(
                f"pricing_mode must be 'second' or 'first'; got {pricing_mode!r}"
            )

        self.num_keywords = int(num_keywords)
        self.budget = float(budget)
        self.max_days = int(max_days)
        self.loss_threshold = float(loss_threshold)
        self.pricing_mode = pricing_mode

        self._updater_params: list[list[Any]] = (
            updater_params
            if updater_params is not None
            else [["vol", 0.03], ["ctr", 0.03], ["cvr", 0.03]]
        )
        _mask = updater_mask if updater_mask is not None else [True] * self.num_keywords
        self._updater_mask: np.ndarray = np.asarray(_mask, dtype=bool)

        # Keyword hyper-parameters (fixed across episodes)
        self._loc_range = loc_range
        self._scale_loc_ratio_range = scale_loc_ratio_range
        self._bctr_params = (bctr_beta_alpha, bctr_beta_beta)
        self._sctr_params = (sctr_beta_alpha, sctr_beta_beta)
        self._reward_mean_range = reward_mean_range
        self._reward_std_ratio = reward_std_to_mean_ratio
        self._volume_mean_range = volume_mean_range

        # Mutable episode state — initialised by reset()
        self._rng: np.random.Generator = np.random.default_rng(None)
        self._loc: np.ndarray = np.empty(self.num_keywords)
        self._scale: np.ndarray = np.empty(self.num_keywords)
        self._bctr: np.ndarray = np.empty(self.num_keywords)
        self._sctr: np.ndarray = np.empty(self.num_keywords)
        self._reward_mean: np.ndarray = np.empty(self.num_keywords)
        self._reward_std: np.ndarray = np.empty(self.num_keywords)
        self._v_mean: np.ndarray = np.empty(self.num_keywords)
        self._days_passed: int = 0
        self._cumulative_profit: float = 0.0

        self.reset()

    # ------------------------------------------------------------------
    # Keyword parameter sampling
    # ------------------------------------------------------------------

    def _sample_keyword_params(self) -> None:
        """Sample per-keyword parameters from §G.1 distributions."""
        rng = self._rng
        n = self.num_keywords

        loc_lo, loc_hi = self._loc_range
        self._loc = rng.uniform(loc_lo, loc_hi, n)

        ratio_lo, ratio_hi = self._scale_loc_ratio_range
        self._scale = rng.uniform(ratio_lo, ratio_hi, n) * self._loc

        alpha_b, beta_b = self._bctr_params
        self._bctr = rng.beta(alpha_b, beta_b, n)

        alpha_s, beta_s = self._sctr_params
        self._sctr = rng.beta(alpha_s, beta_s, n)

        mu_lo, mu_hi = self._reward_mean_range
        self._reward_mean = rng.uniform(mu_lo, mu_hi, n)
        self._reward_std = self._reward_std_ratio * self._reward_mean

        v_lo, v_hi = self._volume_mean_range
        self._v_mean = rng.uniform(v_lo, v_hi, n)

    # ------------------------------------------------------------------
    # Gymnasium-compatible API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._rng = np.random.default_rng(seed)
        self._sample_keyword_params()
        self._days_passed = 0
        self._cumulative_profit = 0.0
        return self._zero_obs(), {}

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        bids = np.asarray(action["keyword_bids"], dtype=np.float64)
        n = self.num_keywords
        rng = self._rng

        impressions = np.zeros(n, dtype=np.float64)
        cost_arr = np.zeros(n, dtype=np.float64)
        clicks_buy = np.zeros(n, dtype=np.float64)
        conv_arr = np.zeros(n, dtype=np.float64)
        revenue_arr = np.zeros(n, dtype=np.float64)

        for i in range(n):
            v_i = int(rng.poisson(self._v_mean[i]))
            if v_i == 0:
                continue

            # §G.1: critical bid ~ |Laplace(loc_k, scale_k)|
            c_bids = np.abs(rng.laplace(self._loc[i], self._scale[i], v_i))
            won = c_bids <= bids[i]
            imp_i = int(won.sum())
            if imp_i == 0:
                continue

            impressions[i] = imp_i
            if self.pricing_mode == "second":
                cost_arr[i] = float(c_bids[won].sum())
            else:  # first-price: agent pays its own bid
                cost_arr[i] = bids[i] * imp_i

            clk_i = int(rng.binomial(imp_i, float(np.clip(self._bctr[i], 0.0, 1.0))))
            clicks_buy[i] = clk_i
            if clk_i == 0:
                continue

            cnv_i = int(rng.binomial(clk_i, float(np.clip(self._sctr[i], 0.0, 1.0))))
            conv_arr[i] = cnv_i
            if cnv_i > 0:
                # TruncNormal(μ, σ, min=0.01) — clamp approximation valid
                # because min=0.01 << μ_R ∈ [0.30, 0.50] (tail mass < 0.01%)
                raw = rng.normal(self._reward_mean[i], self._reward_std[i], cnv_i)
                revenue_arr[i] = float(np.sum(np.maximum(raw, 0.01)))

        # Budget cap: scale all metrics proportionally so that CTR, margin,
        # and other per-impression ratios remain consistent after capping.
        total_cost = float(cost_arr.sum())
        if total_cost > self.budget + _EPS:
            scale = self.budget / total_cost
            cost_arr *= scale
            impressions *= scale
            clicks_buy *= scale
            conv_arr *= scale
            revenue_arr *= scale

        day_profit = float(revenue_arr.sum()) - float(cost_arr.sum())
        self._cumulative_profit += day_profit
        self._days_passed += 1

        self._update_keywords()

        terminated = bool(self._cumulative_profit < -self.loss_threshold)
        truncated = bool(self._days_passed >= self.max_days)

        obs: dict[str, Any] = {
            "impressions": impressions,
            "buyside_clicks": clicks_buy,
            "cost": cost_arr,
            "sellside_conversions": conv_arr,
            "revenue": revenue_arr,
            "cumulative_profit": self._cumulative_profit,
            "days_passed": self._days_passed,
        }
        return obs, day_profit, terminated, truncated, {}

    def close(self) -> None:
        pass

    def render(self) -> Any:
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zero_obs(self) -> dict[str, Any]:
        n = self.num_keywords
        return {
            "impressions": np.zeros(n, dtype=np.float64),
            "buyside_clicks": np.zeros(n, dtype=np.float64),
            "cost": np.zeros(n, dtype=np.float64),
            "sellside_conversions": np.zeros(n, dtype=np.float64),
            "revenue": np.zeros(n, dtype=np.float64),
            "cumulative_profit": 0.0,
            "days_passed": 0,
        }

    def _update_keywords(self) -> None:
        """Multiplicative drift on vol/ctr/cvr (§G.1: 3 %/day default)."""
        mask = self._updater_mask
        for feature, rate in self._updater_params:
            if feature == "vol":
                self._v_mean[mask] *= 1.0 + rate
            elif feature == "ctr":
                self._bctr[mask] = np.clip(self._bctr[mask] * (1.0 + rate), 0.0, 1.0)
            elif feature == "cvr":
                self._sctr[mask] = np.clip(self._sctr[mask] * (1.0 + rate), 0.0, 1.0)

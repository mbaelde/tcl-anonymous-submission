"""Audit of AdCraft default economics (§7.1 follow-up, 2026-05-18).

Goal: explain *quantitatively* why the polytope (util>=0.6, margin>=0) is
empty for MultiConstraintAdCraft under the default sample_random_keywords
distributions. Targets the question: "is there any per-keyword bid b* that
makes E[revenue(b*)] >= E[cost(b*)] for a typical keyword?"

Method.
1. Reset the env over many seeds; extract the actual per-keyword sampled
   parameters (bctr, sctr, mean_revenue, std_revenue, v_mean,
   imp_intercept, imp_slope) and aggregate empirical means + per-sample
   spread across seeds.
2. For a fine bid grid b in [0.01, 10], compute per-keyword:
   - impression_rate(b)  (deterministic, lib.rs:93-105 via Keyword.impression_rate)
   - E[cpc(b)]           (MC over rust.cost_create, N_CPC samples)
   - E[cost(b)] = E[V] * imp_rate(b) * bctr * E[cpc(b)]
   - E[rev(b)]  = E[V] * imp_rate(b) * bctr * sctr * E[R]
   - E[profit(b)] = E[rev(b)] - E[cost(b)]
3. Find b*_i = argmax_b E[profit_i(b)] per keyword; report distribution of
   b*, max profit, achievable margin = (E[rev(b*)] - E[cost(b*)]) / E[rev(b*)].
4. Pool over keywords and seeds: what *fraction* of keywords have a
   positive-profit bid at all, and at what bid?

Output:
  runs/pilot_adcraft_economics/keyword_params.csv  (per-seed-keyword raw)
  runs/pilot_adcraft_economics/optimal_bid.csv     (per-seed-keyword b*)
  runs/pilot_adcraft_economics/summary.txt         (aggregated readout)
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tcl.envs.adcraft_multiconstraint import MultiConstraintAdCraft  # noqa: E402


# --- Config -----------------------------------------------------------
NUM_KEYWORDS = 10
N_SEEDS = 50
BID_GRID = np.concatenate([
    np.linspace(0.01, 0.5, 50),     # fine near zero
    np.linspace(0.5, 2.0, 30)[1:],  # mid range
    np.linspace(2.0, 10.0, 17)[1:], # coarse high
])  # 96 points total
N_CPC_MC = 1000   # MC samples for E[cpc(b)] per (keyword, b) cell
N_REWARD_MC = 1000  # MC samples for E[reward] per keyword


def _mc_cpc(env: MultiConstraintAdCraft, kw, bid: float, n: int = N_CPC_MC) -> float:
    """Monte Carlo estimate of E[cpc(bid)] using rust.cost_create.

    Calls the explicit cost sampler n times. The sampler is a clipped
    Normal mean=sqrt(b)/4 + b/2, std=sqrt(b)/6, clipped to [0, b].
    """
    samples = np.array([kw.cost_per_buyside_click(bid) for _ in range(n)])
    return float(samples.mean())


def _mc_reward(kw, n: int = N_REWARD_MC) -> float:
    """E[reward] via MC over the reward_distribution_sampler."""
    return float(np.mean(kw.sample_reward(n)))


def run_audit() -> None:
    out_dir = _REPO_ROOT / "runs" / "pilot_adcraft_economics"
    out_dir.mkdir(parents=True, exist_ok=True)

    keyword_rows: list[dict] = []
    optimal_rows: list[dict] = []

    print(f"Running audit: {N_SEEDS} seeds x {NUM_KEYWORDS} keywords x {len(BID_GRID)} bids")
    print(f"  CPC MC: {N_CPC_MC} samples per (kw, b) cell")
    print(f"  reward MC: {N_REWARD_MC} samples per keyword")

    for seed in tqdm(range(N_SEEDS), desc="seeds"):
        env = MultiConstraintAdCraft(
            num_keywords=NUM_KEYWORDS,
            budget=100.0,
            bid_max=10.0,
            max_days=60,
            target_utilization=0.5,
            target_ctr=0.7,
            target_margin=-1.7,
            margin_formula="revenue_share",
        )
        env.reset(seed=seed)

        # E[V] per keyword via the volume_sampler (Poisson around v_mean).
        # We use v_mean directly from the keyword's stored params if available,
        # else MC sample.
        for kw_idx, kw in enumerate(env._base.keywords):
            # extract structural params (params is a dict stored at __init__).
            params = getattr(kw, "params", None) or {}
            bctr = float(kw.buyside_ctr)
            sctr = float(kw.sellside_paid_ctr)
            # E[V] via MC since v_mean isn't directly accessible (it's hidden in volume_sampler closure)
            vol_samples = np.array([kw.sample_volume(1)[0] for _ in range(500)])
            e_v = float(vol_samples.mean())
            v_std = float(vol_samples.std())
            # E[R]
            e_r = _mc_reward(kw)
            # imp_intercept and slope from kw if exposed
            imp_intercept = params.get("impression_bid_intercept")
            imp_slope = params.get("impression_slope")

            keyword_rows.append({
                "seed": seed,
                "kw": kw_idx,
                "bctr": bctr,
                "sctr": sctr,
                "e_volume": e_v,
                "v_std": v_std,
                "e_reward": e_r,
                "imp_intercept": float(imp_intercept) if imp_intercept is not None else float("nan"),
                "imp_slope": float(imp_slope) if imp_slope is not None else float("nan"),
            })

            # Per-bid: cost, rev, profit.
            best_profit = -np.inf
            best_bid = float("nan")
            best_rev = float("nan")
            best_cost = float("nan")
            best_imp_rate = float("nan")
            for b in BID_GRID:
                imp_rate = float(kw.impression_rate(b))
                if imp_rate < 1e-6:
                    cost = 0.0
                    rev = 0.0
                else:
                    e_cpc = _mc_cpc(env, kw, b)
                    cost = e_v * imp_rate * bctr * e_cpc
                    rev = e_v * imp_rate * bctr * sctr * e_r
                profit = rev - cost
                if profit > best_profit:
                    best_profit = profit
                    best_bid = float(b)
                    best_rev = rev
                    best_cost = cost
                    best_imp_rate = imp_rate
            margin_at_best = (best_rev - best_cost) / best_rev if best_rev > 1e-9 else float("nan")
            optimal_rows.append({
                "seed": seed,
                "kw": kw_idx,
                "b_star": best_bid,
                "imp_rate_at_b_star": best_imp_rate,
                "e_cost_at_b_star": best_cost,
                "e_rev_at_b_star": best_rev,
                "e_profit_at_b_star": best_profit,
                "margin_at_b_star": margin_at_best,
                "positive_profit": int(best_profit > 0),
            })

    # --- Write CSVs --------------------------------------------------
    kw_csv = out_dir / "keyword_params.csv"
    with kw_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(keyword_rows[0].keys()))
        w.writeheader(); w.writerows(keyword_rows)
    print(f"wrote {kw_csv} ({len(keyword_rows)} rows)")

    opt_csv = out_dir / "optimal_bid.csv"
    with opt_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(optimal_rows[0].keys()))
        w.writeheader(); w.writerows(optimal_rows)
    print(f"wrote {opt_csv} ({len(optimal_rows)} rows)")

    # --- Aggregate ---------------------------------------------------
    bctr = np.array([r["bctr"] for r in keyword_rows])
    sctr = np.array([r["sctr"] for r in keyword_rows])
    e_v = np.array([r["e_volume"] for r in keyword_rows])
    e_r = np.array([r["e_reward"] for r in keyword_rows])
    imp_int = np.array([r["imp_intercept"] for r in keyword_rows])
    imp_slp = np.array([r["imp_slope"] for r in keyword_rows])

    b_star = np.array([r["b_star"] for r in optimal_rows])
    profit_star = np.array([r["e_profit_at_b_star"] for r in optimal_rows])
    margin_star = np.array([r["margin_at_b_star"] for r in optimal_rows])
    is_pos = np.array([r["positive_profit"] for r in optimal_rows])

    lines = []
    lines.append(f"# AdCraft economics audit ({N_SEEDS} seeds x {NUM_KEYWORDS} kw)\n")
    lines.append("## Per-keyword param distributions (analytical vs empirical)\n")
    lines.append(f"  bctr           : mean={bctr.mean():.4f}  std={bctr.std():.4f}  (analytic Beta(2,5) mean=0.2857)")
    lines.append(f"  sctr           : mean={sctr.mean():.4f}  std={sctr.std():.4f}  (analytic Beta(5,2) mean=0.7143)")
    lines.append(f"  E[V]           : mean={e_v.mean():.4f}  std={e_v.std():.4f}  (analytic 15*E[2^X]-1 = 17.40 pre-int)")
    lines.append(f"  E[reward]      : mean={e_r.mean():.4f}  std={e_r.std():.4f}  (analytic 1.5*Beta(2,5) mean = 0.4286)")
    lines.append(f"  imp_intercept  : mean={np.nanmean(imp_int):.4f}  std={np.nanstd(imp_int):.4f}  (analytic U(0,1.5) mean=0.75)")
    lines.append(f"  imp_slope      : mean={np.nanmean(imp_slp):.4f}  std={np.nanstd(imp_slp):.4f}  (analytic 25*Beta(5,5) mean=12.50)")
    lines.append("")

    lines.append("## Optimal per-keyword bid (argmax E[profit])\n")
    lines.append(f"  b_star          : mean={b_star.mean():.4f}  median={np.median(b_star):.4f}  min={b_star.min():.4f}  max={b_star.max():.4f}")
    lines.append(f"  E[profit_star]  : mean={profit_star.mean():.4f}  median={np.median(profit_star):.4f}  min={profit_star.min():.4f}  max={profit_star.max():.4f}")
    lines.append(f"  margin_star     : mean={np.nanmean(margin_star):.4f}  median={np.nanmedian(margin_star):.4f}")
    lines.append(f"  positive_profit : {int(is_pos.sum())}/{len(is_pos)} keywords ({100*is_pos.mean():.1f} %)")
    lines.append("")

    lines.append("## Conditional on a profitable keyword (positive_profit==1)\n")
    if is_pos.sum() > 0:
        m_pos = margin_star[is_pos == 1]
        b_pos = b_star[is_pos == 1]
        p_pos = profit_star[is_pos == 1]
        lines.append(f"  b_star (cond)        : mean={b_pos.mean():.4f}  median={np.median(b_pos):.4f}")
        lines.append(f"  margin_star (cond)   : mean={m_pos.mean():.4f}  median={np.median(m_pos):.4f}")
        lines.append(f"  E[profit_star] (cond): mean={p_pos.mean():.4f}  median={np.median(p_pos):.4f}")
    else:
        lines.append("  No keyword has positive expected profit at any bid in the grid.")
    lines.append("")

    # Structural deficit ratio (b=b_max)
    e_cpc_bmax = math.sqrt(10) / 4 + 10 / 2
    e_rev_per_click = float(np.mean(sctr * e_r))
    lines.append("## Structural ratio at b=b_max=10\n")
    lines.append(f"  E[cpc(10)]         (analytic) = {e_cpc_bmax:.4f}")
    lines.append(f"  E[rev/click] (sctr*E[R] mean) = {e_rev_per_click:.4f}")
    lines.append(f"  Ratio                          = {e_cpc_bmax / e_rev_per_click:.4f}")
    lines.append(f"  Implied margin at saturated b  = {1.0 - e_cpc_bmax / e_rev_per_click:.4f}")
    lines.append("")

    summary_path = out_dir / "summary.txt"
    text = "\n".join(lines)
    summary_path.write_text(text, encoding="utf-8")
    print(f"wrote {summary_path}")
    print()
    print(text)


if __name__ == "__main__":
    run_audit()

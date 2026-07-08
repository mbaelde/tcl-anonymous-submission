"""Calibration sweep for MultiConstraintAdCraftLaplacian (Phase 3).

Sweeps (budget, bid) on the pure-Python Laplacian env (100 keywords,
bid_max=3.0) with constant per-keyword bids and records realized
util / ctr / margin per cell, so we can identify targets where a
uniform policy is feasible (polytope non-empty) and where the tradeoff
between constraints is non-trivial (discriminating for Phase 4).

No non-stationarity during calibration (updater_params=[]) — cleaner physics.

Output:
    runs/pilot_adcraft_calibration_laplacian/summary.csv

Usage::

    cd /path/to/tcl-code
    uv run python -m experiments.pilot_adcraft.calibrate_laplacian
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tcl.envs.adcraft_laplacian import MultiConstraintAdCraftLaplacian  # noqa: E402

_EPS = 1e-8


def run_constant_bid(
    *,
    num_keywords: int,
    budget: float,
    bid_max: float,
    bid: float,
    max_days: int,
    episodes: int,
) -> list[dict[str, float]]:
    """Run <episodes> episodes with a constant per-keyword bid.

    Targets are set to neutral values (1.0 util, 1.0 ctr, -100 margin)
    so that c_k is always negative — we only care about the raw realized
    metrics recorded from info, not the cost vector.
    """
    env = MultiConstraintAdCraftLaplacian(
        num_keywords=num_keywords,
        budget=budget,
        bid_max=bid_max,
        max_days=max_days,
        target_utilization=0.99,  # neutral — physics unchanged, just for API compliance
        target_ctr=0.99,
        target_margin=-100.0,
        margin_formula="revenue_share",
        updater_params=[],        # no drift during calibration
    )
    rows: list[dict[str, float]] = []
    action = np.full(num_keywords, bid, dtype=np.float32)

    for ep in range(episodes):
        env.reset(seed=ep)
        util_acc: list[float] = []
        ctr_acc: list[float] = []
        mgn_acc: list[float] = []
        rev_acc: list[float] = []
        cost_acc: list[float] = []
        imp_acc: list[float] = []
        clicks_acc: list[float] = []

        done = False
        while not done:
            obs_flat, reward, terminated, truncated, info = env.step(action)

            # Extract realized metrics directly from the obs (last step's obs)
            # We reconstruct from the flat obs:  [imp, clicks, cost, conv, rev, cum_profit, days]
            n = num_keywords
            imp   = float(obs_flat[:n].sum())
            click = float(obs_flat[n:2*n].sum())
            cost  = float(obs_flat[2*n:3*n].sum())
            rev   = float(obs_flat[4*n:5*n].sum())

            realized_util = cost / budget
            realized_ctr  = click / imp if imp > _EPS else 0.0
            realized_margin = (rev - cost) / rev if rev > _EPS else 0.0

            util_acc.append(realized_util)
            ctr_acc.append(realized_ctr)
            mgn_acc.append(realized_margin)
            rev_acc.append(rev)
            cost_acc.append(cost)
            imp_acc.append(imp)
            clicks_acc.append(click)

            done = terminated or truncated

        rows.append({
            "budget": budget,
            "bid": bid,
            "bid_rel": bid / bid_max,
            "episode": ep,
            "util_mean": float(np.mean(util_acc)),
            "ctr_mean": float(np.mean(ctr_acc)),
            "margin_mean": float(np.mean(mgn_acc)),
            "rev_mean": float(np.mean(rev_acc)),
            "cost_mean": float(np.mean(cost_acc)),
            "imp_mean": float(np.mean(imp_acc)),
            "clicks_mean": float(np.mean(clicks_acc)),
        })

    return rows


def main() -> None:
    # --- Grid ---------------------------------------------------------
    BUDGETS = [50.0, 100.0, 200.0]
    BID_MAX = 3.0
    # Absolute bids (fractions of BID_MAX)
    BIDS_REL = [0.10, 0.17, 0.25, 0.33, 0.50, 0.67, 1.00]
    NUM_KEYWORDS = 100
    EPISODES = 3
    MAX_DAYS = 60

    out_dir = _REPO_ROOT / "runs" / "pilot_adcraft_calibration_laplacian"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    combos = [(b, BID_MAX * br) for b in BUDGETS for br in BIDS_REL]
    for budget, bid in tqdm(combos, desc="calibrate_laplacian"):
        rows.extend(
            run_constant_bid(
                num_keywords=NUM_KEYWORDS,
                budget=budget,
                bid_max=BID_MAX,
                bid=bid,
                max_days=MAX_DAYS,
                episodes=EPISODES,
            )
        )

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path} ({len(rows)} rows)")

    # -- Aggregate (budget × bid) → mean over episodes ----------------
    keys = sorted({(r["budget"], r["bid"]) for r in rows})
    print(
        f"\n{'budget':>8} {'bid':>5} {'bid_rel':>7} | "
        f"{'util':>7} {'ctr':>7} {'margin':>8} | "
        f"{'rev':>7} {'cost':>7}"
    )
    print("-" * 70)
    for budget, bid in keys:
        rs = [r for r in rows if r["budget"] == budget and r["bid"] == bid]
        u = float(np.mean([r["util_mean"] for r in rs]))
        c = float(np.mean([r["ctr_mean"] for r in rs]))
        m = float(np.mean([r["margin_mean"] for r in rs]))
        rv = float(np.mean([r["rev_mean"] for r in rs]))
        co = float(np.mean([r["cost_mean"] for r in rs]))
        br = bid / 3.0
        feasible = (
            "✓ u>0.3 c>0.01 m>-0.5"
            if u > 0.3 and c > 0.01 and m > -0.5
            else ""
        )
        print(
            f"  {budget:>6.0f} {bid:>5.2f} {br:>7.2f} | "
            f"{u:>7.3f} {c:>7.4f} {m:>8.3f} | "
            f"{rv:>7.3f} {co:>7.3f}  {feasible}"
        )

    # -- Suggest calibrated targets -----------------------------------
    print("\n--- Calibration suggestion ---")
    print("Looking for cells where bid=0.5 policy satisfies util>T, ctr>T, margin>T")
    for budget, bid in keys:
        if abs(bid - 0.5) < 0.01:
            rs = [r for r in rows if r["budget"] == budget and r["bid"] == bid]
            u = float(np.mean([r["util_mean"] for r in rs]))
            c = float(np.mean([r["ctr_mean"] for r in rs]))
            m = float(np.mean([r["margin_mean"] for r in rs]))
            print(
                f"  B={budget:.0f}, bid=0.5 → "
                f"util={u:.3f}, ctr={c:.4f}, margin={m:.3f}"
            )
            # Suggest targets at 70% of realized values (safe margin)
            t_util   = round(u * 0.70, 2)
            t_ctr    = round(c * 0.70, 4)
            t_margin = round(m - 0.10, 2)  # 10 pp below realized margin
            print(
                f"    Suggested targets: util={t_util}, ctr={t_ctr}, "
                f"margin={t_margin}"
            )


if __name__ == "__main__":
    main()

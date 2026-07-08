"""Quick calibration sweep for MultiConstraintAdCraft.

The 2026-05-17 diagnostic showed the canonical env config
(budget=1000, bid_max=10, num_keywords=10) is physically infeasible at
the anonymous institution targets (target_utilization=0.8 unreachable: max realized
util = 14% even at bid_max). This script sweeps (budget, bid_max) at a
fixed grid of constant per-keyword bids and records the realized
utilization / CTR / margin per cell, so we can pick a config where
target_utilization and target_margin are jointly reachable and where
util ↔ margin show a non-trivial Pareto tradeoff (necessary to
discriminate constrained agents).

Constant-bid only — no SAC. Cheap (<5 min total). Output:

    runs/pilot_adcraft_calibration/summary.csv

Usage::

    py -3.14 -m uv run python -m experiments.pilot_adcraft.calibrate

Edit the ``BUDGETS``, ``BID_MAXES``, ``BIDS_REL``, ``NUM_KEYWORDS``,
``EPISODES`` constants at the top of ``main()`` for a different grid.
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

from tcl.envs.adcraft_multiconstraint import MultiConstraintAdCraft  # noqa: E402


def run_constant_bid(
    *,
    num_keywords: int,
    budget: float,
    bid_max: float,
    bid: float,
    target_utilization: float,
    target_ctr: float,
    target_margin: float,
    margin_formula: str,
    max_days: int,
    episodes: int,
) -> list[dict[str, float]]:
    env = MultiConstraintAdCraft(
        num_keywords=num_keywords,
        budget=budget,
        bid_max=bid_max,
        max_days=max_days,
        target_utilization=target_utilization,
        target_ctr=target_ctr,
        target_margin=target_margin,
        margin_formula=margin_formula,
    )
    rows: list[dict[str, float]] = []
    action = np.full(env.num_keywords, bid, dtype=np.float32)
    for ep in range(episodes):
        env.reset(seed=ep)
        util_acc: list[float] = []
        ctr_acc: list[float] = []
        mgn_acc: list[float] = []
        for _ in range(env.max_days):
            _, _, term, trunc, info = env.step(action)
            c = info["costs"]
            util_acc.append(env.target_utilization - float(c[0]))
            ctr_acc.append(env.target_ctr - float(c[1]))
            mgn_acc.append(env.target_margin - float(c[2]))
            if term or trunc:
                break
        rows.append(
            {
                "budget": budget,
                "bid_max": bid_max,
                "bid": bid,
                "episode": ep,
                "util_mean": float(np.mean(util_acc)),
                "ctr_mean": float(np.mean(ctr_acc)),
                "margin_mean": float(np.mean(mgn_acc)),
            }
        )
    return rows


def main() -> None:
    # --- Grid ---------------------------------------------------------
    BUDGETS = [50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0]
    BID_MAXES = [10.0]  # canonical; volume saturates already at bid=10
    BIDS_REL = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # fraction of bid_max
    NUM_KEYWORDS = 10
    EPISODES = 2
    MAX_DAYS = 60
    # Constraint targets used only to compute the c_k vector — the
    # *physical* reachability is independent of them, but we keep the
    # margin_formula so the recorded margin matches the post-2026-05
    # convention.
    TARGET_UTILIZATION = 0.8
    TARGET_CTR = 0.001
    TARGET_MARGIN = 0.70
    MARGIN_FORMULA = "revenue_share"

    out_dir = _REPO_ROOT / "runs" / "pilot_adcraft_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    combos = [
        (b, bm, br) for b in BUDGETS for bm in BID_MAXES for br in BIDS_REL
    ]
    for budget, bid_max, bid_rel in tqdm(combos, desc="calibrate"):
        bid = bid_max * bid_rel
        rows.extend(
            run_constant_bid(
                num_keywords=NUM_KEYWORDS,
                budget=budget,
                bid_max=bid_max,
                bid=bid,
                target_utilization=TARGET_UTILIZATION,
                target_ctr=TARGET_CTR,
                target_margin=TARGET_MARGIN,
                margin_formula=MARGIN_FORMULA,
                max_days=MAX_DAYS,
                episodes=EPISODES,
            )
        )

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    # Aggregate (budget, bid) -> mean over episodes
    keys = sorted({(r["budget"], r["bid"]) for r in rows})
    print("\nAggregated (per budget × bid, mean over episodes):")
    print(
        f"  {'budget':>8} {'bid':>6} | {'util':>7} {'ctr':>7} {'margin':>8}"
    )
    for budget, bid in keys:
        rs = [r for r in rows if r["budget"] == budget and r["bid"] == bid]
        u = np.mean([r["util_mean"] for r in rs])
        c = np.mean([r["ctr_mean"] for r in rs])
        m = np.mean([r["margin_mean"] for r in rs])
        flag = " <- target_util reachable" if u >= 0.7 else ""
        print(
            f"  {budget:>8.0f} {bid:>6.2f} | {u:>7.3f} {c:>7.3f} {m:>8.2f}{flag}"
        )


if __name__ == "__main__":
    main()

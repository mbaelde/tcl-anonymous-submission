"""Heterogeneous-policy calibration sweep for MultiConstraintAdCraft.

Companion to ``calibrate.py``. The uniform sweep at budget=150 from
2026-05-17 found that no constant per-keyword bid jointly satisfies
(target_util>=0.7, target_margin>=0.0 in revenue_share) — but a
constrained-RL setting only makes sense if *some* policy does and a
naive one doesn't. This script tests whether a *heterogeneous* policy
(bid b_max on the top-r% keywords ranked by an oracle score, 0 on the
rest) opens a feasibility region the uniform sweep cannot reach.

Two oracle scores are swept side-by-side:

* ``"rev"``    — score_i = E[V_i] * ctr_buy_i * ctr_sell_i * E[reward_i],
                 i.e. expected daily revenue per keyword. Maximizes
                 CTR and revenue but not necessarily margin.
* ``"margin"`` — score_i = E[reward_i] * ctr_sell_i / E[cpc_i(b_max)],
                 expected revenue-per-dollar at bid_max. Maximizes
                 margin but may starve utilization.

Output: ``runs/pilot_adcraft_hetero/summary.csv``. Run::

    uv run python -m experiments.pilot_adcraft.calibrate_hetero
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


def _score_keywords(env: MultiConstraintAdCraft, bid_max: float, score: str, n_mc: int = 200) -> np.ndarray:
    """Return per-keyword scores after env.reset(); MC-estimated where needed.

    ``rev``    : E[V] * ctr_buy * ctr_sell * E[reward]
    ``margin`` : E[reward] * ctr_sell / E[cpc(b_max)]
    """
    kws = env._base.keywords
    scores = np.empty(len(kws), dtype=np.float64)
    for i, kw in enumerate(kws):
        vol = float(np.mean(kw.sample_volume(n_mc)))
        reward = float(np.mean(kw.sample_reward(n_mc)))
        if score == "rev":
            scores[i] = vol * kw.buyside_ctr * kw.sellside_paid_ctr * reward
        elif score == "margin":
            # CPC mean ≈ √b/4 + b/2 from rust.cost_create; cf. lib.rs:54-67.
            cpc_mean = math.sqrt(bid_max) / 4.0 + bid_max / 2.0
            scores[i] = reward * kw.sellside_paid_ctr / max(cpc_mean, 1e-6)
        else:
            raise ValueError(f"unknown score {score!r}")
    return scores


def run_hetero(
    *,
    num_keywords: int,
    budget: float,
    bid_max: float,
    r: float,
    score: str,
    target_utilization: float,
    target_ctr: float,
    target_margin: float,
    margin_formula: str,
    max_days: int,
    episodes: int,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    n_top = max(1, int(round(r * num_keywords))) if r > 0 else 0
    for ep in range(episodes):
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
        env.reset(seed=ep)
        scores = _score_keywords(env, bid_max=bid_max, score=score)
        top_idx = np.argsort(-scores)[:n_top]
        action = np.zeros(num_keywords, dtype=np.float32)
        action[top_idx] = bid_max

        util_acc, ctr_acc, mgn_acc = [], [], []
        cost_acc, rev_acc, imp_acc, clk_acc = [], [], [], []
        for _ in range(env.max_days):
            _, _, term, trunc, info = env.step(action)
            c = info["costs"]
            util_acc.append(env.target_utilization - float(c[0]))
            ctr_acc.append(env.target_ctr - float(c[1]))
            mgn_acc.append(env.target_margin - float(c[2]))
            cost_acc.append(float(np.sum(info.get("cost", [0.0]))))
            rev_acc.append(float(np.sum(info.get("revenue", [0.0]))))
            imp_acc.append(float(np.sum(info.get("impressions", [0]))))
            clk_acc.append(float(np.sum(info.get("buyside_clicks", [0]))))
            if term or trunc:
                break
        rows.append({
            "budget": budget,
            "bid_max": bid_max,
            "score": score,
            "r": r,
            "n_top": n_top,
            "episode": ep,
            "util_mean": float(np.mean(util_acc)),
            "ctr_mean": float(np.mean(ctr_acc)),
            "margin_mean": float(np.mean(mgn_acc)),
            "cost_day_mean": float(np.mean(cost_acc)),
            "revenue_day_mean": float(np.mean(rev_acc)),
            "imp_day_mean": float(np.mean(imp_acc)),
            "clk_day_mean": float(np.mean(clk_acc)),
        })
    return rows


def main() -> None:
    # --- Grid ---------------------------------------------------------
    BUDGETS = [100.0, 150.0, 200.0]
    BID_MAXES = [10.0]
    R_VALUES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    SCORES = ["rev", "margin"]
    NUM_KEYWORDS = 10
    EPISODES = 10
    MAX_DAYS = 60
    TARGET_UTILIZATION = 0.7
    TARGET_CTR = 0.001
    TARGET_MARGIN = 0.0  # revenue_share neutral threshold; we read raw margin
    MARGIN_FORMULA = "revenue_share"

    out_dir = _REPO_ROOT / "runs" / "pilot_adcraft_hetero"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    combos = [
        (b, bm, r, s)
        for b in BUDGETS
        for bm in BID_MAXES
        for r in R_VALUES
        for s in SCORES
    ]
    for budget, bid_max, r, score in tqdm(combos, desc="hetero"):
        rows.extend(run_hetero(
            num_keywords=NUM_KEYWORDS,
            budget=budget,
            bid_max=bid_max,
            r=r,
            score=score,
            target_utilization=TARGET_UTILIZATION,
            target_ctr=TARGET_CTR,
            target_margin=TARGET_MARGIN,
            margin_formula=MARGIN_FORMULA,
            max_days=MAX_DAYS,
            episodes=EPISODES,
        ))

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    # Aggregate (score, budget, r) -> mean ± std over episodes.
    # NB: util_mean/ctr_mean/margin_mean already store the *realized*
    # quantity (cf. `target - c_k = realized` in `_costs_vector`).
    keys = sorted({(x["score"], x["budget"], x["r"]) for x in rows})
    print("\nAggregated (per score × budget × r, mean ± std over episodes):")
    print(f"  {'score':>7} {'budget':>7} {'r':>5} {'n_top':>5} | "
          f"{'util_m':>7} ± {'util_s':>6}  "
          f"{'ctr_m':>8} ± {'ctr_s':>7}  "
          f"{'mgn_m':>7} ± {'mgn_s':>6}")
    for score, budget, r in keys:
        rs = [
            x for x in rows
            if x["score"] == score and x["budget"] == budget and x["r"] == r
        ]
        u = np.array([x["util_mean"] for x in rs])     # realized util
        c = np.array([x["ctr_mean"] for x in rs])      # realized ctr
        m = np.array([x["margin_mean"] for x in rs])   # realized margin
        n_top = int(rs[0]["n_top"])
        # heuristic: util >= 0.6 AND margin >= 0.0 AND ctr >= 0.001
        # (joint feasibility absent from the uniform-policy sweep).
        flag = ""
        if u.mean() >= 0.6 and m.mean() >= 0.0 and c.mean() >= 0.001:
            flag = "  <- jointly feasible"
        print(
            f"  {score:>7} {budget:>7.0f} {r:>5.2f} {n_top:>5d} | "
            f"{u.mean():>7.3f} ± {u.std():>6.3f}  "
            f"{c.mean():>8.5f} ± {c.std():>7.5f}  "
            f"{m.mean():>7.3f} ± {m.std():>6.3f}{flag}"
        )


if __name__ == "__main__":
    main()

"""Paired Wilcoxon signed-rank tests on per-seed CSR / return (paper Item 5).

Runs the Wilcoxon signed-rank test between a reference agent (default
``tcl_standalone``) and every other agent present, on the steady-state
constraint-satisfaction rate (CSR) of each cost and on the episode return.

The CSR / return extraction is a faithful copy of the canonical analysis in
``experiments.pilot_adcraft.analyze_phase5_a1_standalone`` (steady = last 20 %
of episodes, ``episode_cost_k <= 0`` per episode, ``episode_cost_k{k}`` tag for
``lag_multi`` and ``episode_cost_{k}`` otherwise) so the per-seed numbers match
the published tables exactly.

Seeds are *paired* by value across agents: only seeds present in both the
reference and the comparison agent enter the test. With 10 seeds the exact
signed-rank null is used (scipy default ``mode="auto"``).

Usage (from repo root)::

    # Main bench, formulation A vs B (merge two run dirs, K=3 AdCraft):
    uv run python -m experiments.wilcoxon_stats \
        --run-dirs runs_vm/phase5_a1_standalone_ll runs_vm/phase5_a1_v2 \
        --ref-agent tcl_standalone --k-costs 3 \
        --out runs_vm/wilcoxon_phase5_a1.csv

    # beta*-scan (single dir, K=1):
    uv run python -m experiments.wilcoxon_stats \
        --run-dirs runs_vm/beta_star_scan --ref-agent tcl --k-costs 1 \
        --out runs_vm/wilcoxon_beta_star.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# --- Canonical extraction config (mirror analyze_phase5_a1_standalone) -------

COST_TAG_TEMPLATES: dict[str, str] = {"lag_multi": "rollout/episode_cost_k{k}"}
_DEFAULT_COST_TEMPLATE = "rollout/episode_cost_{k}"
STEADY_FRACTION = 0.2

AGENT_LABELS = {
    "tcl_standalone": "TCL-standalone (A)",
    "tcl": "TCL shaped (B)",
    "fixed": "Fixed weights",
    "lag_multi": "Lagrangian",
    "pid_lagrangian": "PID-Lagrangian",
    "hprs": "HPRS",
}


def discover_cells(run_dir: Path) -> list[tuple[str, int, Path]]:
    """Yield (agent, seed, tb_run_dir) for every ``<agent>/seed=<n>/tb`` cell."""
    cells: list[tuple[str, int, Path]] = []
    for cell in sorted(run_dir.glob("*/seed=*/tb")):
        m = re.fullmatch(r"seed=(\d+)", cell.parent.name)
        if not m:
            continue
        seed = int(m.group(1))
        agent = cell.parent.parent.name
        run_dirs = [p for p in cell.iterdir() if p.is_dir()]
        if not run_dirs:
            continue
        tb_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
        cells.append((agent, seed, tb_run))
    return cells


def _load_scalars(tb_run: Path, tags: list[str]) -> dict[str, np.ndarray]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(tb_run), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out: dict[str, np.ndarray] = {}
    for tag in tags:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        out[tag] = np.array([e.value for e in events], dtype=np.float64)
    return out


def extract_metrics(agent: str, tb_run: Path, k_costs: int) -> dict[str, float]:
    """Steady-state CSR per cost and steady-state mean return for one cell."""
    cost_tmpl = COST_TAG_TEMPLATES.get(agent, _DEFAULT_COST_TEMPLATE)
    alt_tmpl = (
        "rollout/episode_cost_k{k}"
        if not cost_tmpl.endswith("k{k}")
        else _DEFAULT_COST_TEMPLATE
    )
    cost_tags = [cost_tmpl.format(k=k) for k in range(k_costs)]
    alt_tags = [alt_tmpl.format(k=k) for k in range(k_costs)]

    scalars = _load_scalars(
        tb_run,
        [*cost_tags, *alt_tags, "rollout/episode_steps", "rollout/episode_return"],
    )

    metrics: dict[str, float] = {}

    ret = scalars.get("rollout/episode_return")
    if ret is not None and ret.size:
        start = int(ret.size * (1.0 - STEADY_FRACTION))
        metrics["return"] = float(ret[start:].mean())

    ep_steps = scalars.get("rollout/episode_steps")
    for k in range(k_costs):
        cost_v = scalars.get(cost_tags[k])
        if cost_v is None:
            cost_v = scalars.get(alt_tags[k])
        if cost_v is None or cost_v.size == 0:
            continue
        if ep_steps is not None and ep_steps.size:
            n = min(cost_v.size, ep_steps.size)
            safe = np.where(ep_steps[:n] > 0, ep_steps[:n], 1)
            csr = (cost_v[:n] / safe <= 0.0).astype(np.float64)
        else:
            n = cost_v.size
            csr = (cost_v[:n] <= 0.0).astype(np.float64)
        start = int(n * (1.0 - STEADY_FRACTION))
        steady = csr[start:]
        if steady.size:
            metrics[f"csr_c{k}"] = float(steady.mean())
    return metrics


def collect(run_dirs: list[Path], k_costs: int) -> dict[str, dict[str, dict[int, float]]]:
    """data[agent][metric][seed] = value, merged across all run dirs."""
    data: dict[str, dict[str, dict[int, float]]] = {}
    for run_dir in run_dirs:
        cells = discover_cells(run_dir)
        if not cells:
            print(f"[warn] no cells found in {run_dir}")
        for agent, seed, tb_run in cells:
            m = extract_metrics(agent, tb_run, k_costs)
            for metric, value in m.items():
                data.setdefault(agent, {}).setdefault(metric, {})[seed] = value
    return data


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (preserve input order)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def run_tests(
    data: dict[str, dict[str, dict[int, float]]],
    ref_agent: str,
    metrics: list[str],
    alternative: str,
) -> list[dict]:
    if ref_agent not in data:
        raise SystemExit(
            f"reference agent '{ref_agent}' not found. Present: {sorted(data)}"
        )
    others = [a for a in data if a != ref_agent]
    rows: list[dict] = []
    for metric in metrics:
        ref_by_seed = data[ref_agent].get(metric, {})
        for agent in others:
            cmp_by_seed = data[agent].get(metric, {})
            seeds = sorted(set(ref_by_seed) & set(cmp_by_seed))
            if len(seeds) < 1:
                continue
            x = np.array([ref_by_seed[s] for s in seeds])
            y = np.array([cmp_by_seed[s] for s in seeds])
            diff = x - y
            mean_diff = float(diff.mean())
            median_diff = float(np.median(diff))
            n = len(seeds)
            if np.allclose(diff, 0.0):
                # All seeds identical (e.g. both CSR=1.000): no signal, p=1.
                stat, p = float("nan"), 1.0
                note = "all-equal"
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = wilcoxon(
                        x, y, alternative=alternative, zero_method="wilcox"
                    )
                stat, p = float(res.statistic), float(res.pvalue)
                note = ""
            rows.append(
                {
                    "metric": metric,
                    "ref": ref_agent,
                    "vs": agent,
                    "n": n,
                    "ref_mean": float(x.mean()),
                    "vs_mean": float(y.mean()),
                    "mean_diff": mean_diff,
                    "median_diff": median_diff,
                    "W": stat,
                    "p_raw": p,
                    "note": note,
                }
            )
    # Holm correction across the whole family of comparisons.
    if rows:
        adj = _holm([r["p_raw"] for r in rows])
        for r, pa in zip(rows, adj, strict=True):
            r["p_holm"] = pa
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs", nargs="+", type=Path, required=True,
        help="One or more run dirs (agents merged across all of them).",
    )
    parser.add_argument("--ref-agent", default="tcl_standalone")
    parser.add_argument("--k-costs", type=int, default=3)
    parser.add_argument(
        "--alternative", default="two-sided",
        choices=["two-sided", "greater", "less"],
        help="'greater' tests ref > vs (CSR dominance); default two-sided.",
    )
    parser.add_argument("--out", type=Path, default=None, help="CSV output path.")
    args = parser.parse_args()

    data = collect(args.run_dirs, args.k_costs)
    if not data:
        raise SystemExit("No cells found in any run dir.")

    metrics = [f"csr_c{k}" for k in range(args.k_costs)] + ["return"]
    rows = run_tests(data, args.ref_agent, metrics, args.alternative)
    if not rows:
        raise SystemExit("No comparisons could be formed (check ref agent / seeds).")

    ref_lab = AGENT_LABELS.get(args.ref_agent, args.ref_agent)
    print(f"\n=== Wilcoxon signed-rank ({args.alternative}) — ref = {ref_lab} ===")
    print(
        f"{'metric':>9} {'vs':>16} {'n':>3} {'ref_mean':>9} {'vs_mean':>9} "
        f"{'Δmean':>8} {'W':>6} {'p_raw':>9} {'p_holm':>9}"
    )
    for r in rows:
        vs_lab = AGENT_LABELS.get(r["vs"], r["vs"])
        flag = "*" if r["p_holm"] < 0.05 else " "
        w = "   nan" if np.isnan(r["W"]) else f"{r['W']:6.1f}"
        print(
            f"{r['metric']:>9} {vs_lab:>16} {r['n']:>3} {r['ref_mean']:>9.4f} "
            f"{r['vs_mean']:>9.4f} {r['mean_diff']:>+8.4f} {w} "
            f"{r['p_raw']:>9.4f} {r['p_holm']:>8.4f}{flag}"
        )
    n_seeds = max((len(v) for a in data.values() for v in a.values()), default=0)
    print(f"\n* = Holm-adjusted p < 0.05.  max seeds/cell = {n_seeds}.")
    if n_seeds < 6:
        print("[warn] < 6 seeds: signed-rank power is very low; results indicative only.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "metric", "ref", "vs", "n", "ref_mean", "vs_mean",
            "mean_diff", "median_diff", "W", "p_raw", "p_holm", "note",
        ]
        with args.out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

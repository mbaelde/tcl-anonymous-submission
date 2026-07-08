"""Head-to-head comparison: standalone (A) last_layer vs shaped (B) baselines.

Reads TensorBoard events from TWO run directories:
  - runs/phase5_a1_standalone_ll/tcl_standalone/seed=*/  (Formulation A)
  - runs/phase5_a1_v2/{tcl,fixed,lag_multi,hprs}/seed=*/  (Formulation B baselines)

Produces under figures/phase5_a1_standalone_ll/:
  1. summary.csv  — steady-state CSR + return per (agent, seed)
  2. csr_vs_steps.png  — CSR_k(t) per constraint, 3 subplots
  3. return_vs_steps.png  — episode return trajectories
  4. summary_bar.png  — bar chart: steady-state CSR (mean±std) per agent per constraint

Usage (from repo root)::

    uv run python -m experiments.pilot_adcraft.analyze_phase5_a1_standalone

    # or with explicit dirs:
    uv run python -m experiments.pilot_adcraft.analyze_phase5_a1_standalone \\
        --standalone-dir runs/phase5_a1_standalone_ll \\
        --shaped-dir runs/phase5_a1_v2 \\
        --out-dir figures/phase5_a1_standalone_ll
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex":                 False,
    "mathtext.fontset":            "cm",
    "font.family":                 "serif",
    "font.serif":                  ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus":          False,
    "font.size":                   10,
    "axes.titlesize":              10,
    "axes.labelsize":              9,
    "xtick.labelsize":             8,
    "ytick.labelsize":             8,
    "legend.fontsize":             8,
    "figure.dpi":                  150,
    "savefig.bbox":                "tight",
    "savefig.pad_inches":          0.05,
})
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Visual config
# ---------------------------------------------------------------------------

AGENT_COLORS = {
    "tcl_standalone": "#e377c2",   # pink — Formulation A
    "tcl":            "#1f77b4",   # blue
    "fixed":          "#2ca02c",   # green
    "lag_multi":      "#d62728",   # red
    "hprs":           "#ff7f0e",   # orange
}

AGENT_LABELS = {
    "tcl_standalone": "TCL-standalone (A)",
    "tcl":            "TCL shaped (B)",
    "fixed":          "Fixed weights",
    "lag_multi":      "Lagrangian",
    "hprs":           "HPRS",
}

COST_TAG_TEMPLATES: dict[str, str] = {
    "lag_multi": "rollout/episode_cost_k{k}",
}
_DEFAULT_COST_TEMPLATE = "rollout/episode_cost_{k}"

K_COSTS = 3
SMOOTH_WINDOW = 20
STEADY_FRACTION = 0.2


# ---------------------------------------------------------------------------
# Helpers (mirrors analyze.py — kept local to avoid circular imports)
# ---------------------------------------------------------------------------

def discover_cells(run_dir: Path) -> list[tuple[str, int, Path]]:
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


def load_scalars(tb_run: Path, tags: list[str]) -> dict[str, np.ndarray]:
    ea = EventAccumulator(str(tb_run), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out: dict[str, np.ndarray] = {}
    for tag in tags:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        out[f"{tag}/steps"] = np.array([e.step for e in events], dtype=np.int64)
        out[f"{tag}/values"] = np.array([e.value for e in events], dtype=np.float64)
    return out


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x.astype(np.float64)
    cs = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.size):
        lo = max(0, i + 1 - window)
        out[i] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
    return out


def per_episode_csr(cost_sum: np.ndarray, ep_steps: np.ndarray) -> np.ndarray:
    safe = np.where(ep_steps > 0, ep_steps, 1)
    return (cost_sum / safe <= 0.0).astype(np.float64)


def steps_to_csr09(steps: np.ndarray, smoothed: np.ndarray) -> int | None:
    mask = smoothed >= 0.9
    return int(steps[np.argmax(mask)]) if mask.any() else None


def analyze_cell(agent: str, tb_run: Path) -> dict:
    cost_tmpl = COST_TAG_TEMPLATES.get(agent, _DEFAULT_COST_TEMPLATE)
    alt_tmpl = "rollout/episode_cost_k{k}" if not cost_tmpl.endswith("k{k}") else _DEFAULT_COST_TEMPLATE
    cost_tags = [cost_tmpl.format(k=k) for k in range(K_COSTS)]
    alt_tags = [alt_tmpl.format(k=k) for k in range(K_COSTS)]

    scalars = load_scalars(tb_run, [
        *cost_tags, *alt_tags,
        "rollout/episode_steps",
        "rollout/episode_return",
    ])

    if "rollout/episode_steps/values" not in scalars:
        return {"tb_run": str(tb_run), "error": "no episode_steps"}

    ep_steps_axis = scalars["rollout/episode_steps/steps"]
    ep_steps_vals = scalars["rollout/episode_steps/values"]

    metrics: dict = {"tb_run": str(tb_run)}

    # Return
    if "rollout/episode_return/values" in scalars:
        ret_v = scalars["rollout/episode_return/values"]
        ret_s = scalars["rollout/episode_return/steps"]
        steady_start = int(ret_v.size * (1.0 - STEADY_FRACTION))
        metrics["return_steady_mean"] = float(ret_v[steady_start:].mean()) if ret_v.size else float("nan")
        metrics["return_steady_std"] = float(ret_v[steady_start:].std()) if ret_v.size else float("nan")
        metrics["_return"] = (ret_s, ret_v)
    else:
        metrics["return_steady_mean"] = float("nan")
        metrics["return_steady_std"] = float("nan")
        metrics["_return"] = (np.array([]), np.array([]))

    csr_per_k: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(K_COSTS):
        primary, alt = cost_tags[k], alt_tags[k]
        if f"{primary}/values" in scalars:
            cost_v = scalars[f"{primary}/values"]
        elif f"{alt}/values" in scalars:
            cost_v = scalars[f"{alt}/values"]
        else:
            continue

        n = min(cost_v.size, ep_steps_vals.size)
        csr = per_episode_csr(cost_v[:n], ep_steps_vals[:n])
        csr_sm = smooth(csr, SMOOTH_WINDOW)
        csr_per_k[k] = (ep_steps_axis[:n], csr_sm)

        steady_start = int(n * (1.0 - STEADY_FRACTION))
        steady = csr[steady_start:]
        metrics[f"k{k}/steps_to_csr_0.9"] = steps_to_csr09(ep_steps_axis[:n], csr_sm)
        metrics[f"k{k}/csr_steady_mean"] = float(steady.mean()) if steady.size else float("nan")
        metrics[f"k{k}/csr_steady_std"] = float(steady.std()) if steady.size else float("nan")

    metrics["_csr_per_k"] = csr_per_k
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _dedup_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lab in zip(handles, labels):
        key = lab  # full label already deduplicated by caller
        if key not in seen:
            seen[key] = h
    if seen:
        ax.legend(list(seen.values()), list(seen.keys()), fontsize=8, loc="best")


def plot_csr(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, K_COSTS, figsize=(4.5 * K_COSTS, 3.8), sharey=True)
    constraint_labels = [r"$c_1$: util $\geq$ 0.80", r"$c_2$: CTR $\geq$ 0.15", r"$c_3$: margin $\geq -4$"]
    for k, ax in enumerate(axes):
        for r in rows:
            csr_per_k = r.get("_csr_per_k", {})
            if k not in csr_per_k:
                continue
            steps, csr = csr_per_k[k]
            agent = str(r["agent"])
            color = AGENT_COLORS.get(agent, "gray")
            lw = 2.0 if agent == "tcl_standalone" else 1.0
            ls = "-" if agent == "tcl_standalone" else "--"
            ax.plot(steps, csr, color=color, alpha=0.65, lw=lw, ls=ls,
                    label=AGENT_LABELS.get(agent, agent))
        ax.axhline(0.9, color="black", ls=":", alpha=0.4, lw=0.8, label="_nolegend_")
        ax.set_xlabel("env step")
        ax.set_title(f"CSR: {constraint_labels[k]}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, ls=":", alpha=0.4)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("smoothed CSR (window=20 ep)")
    _dedup_legend(axes[-1])
    fig.suptitle(
        "Phase 5-A1: Standalone (A) [rb_mode=ignore] vs. Shaped (B) baselines",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_return(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for r in rows:
        steps, vals = r.get("_return", (np.array([]), np.array([])))
        if steps.size == 0:
            continue
        agent = str(r["agent"])
        color = AGENT_COLORS.get(agent, "gray")
        lw = 2.0 if agent == "tcl_standalone" else 1.0
        ls = "-" if agent == "tcl_standalone" else "--"
        ax.plot(steps, vals, color=color, alpha=0.55, lw=lw, ls=ls,
                label=f"{AGENT_LABELS.get(agent, agent)} s{r['seed']}")
    ax.set_xlabel("env step")
    ax.set_ylabel(r"episode return ($r_b$)")
    ax.set_title("Episode return: Standalone (A) vs. Shaped (B)")
    ax.grid(True, ls=":", alpha=0.4)
    _dedup_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_summary_bar(rows: list[dict], out_path: Path) -> None:
    """Bar chart: steady-state CSR mean ± std, grouped by constraint, one bar per agent."""
    agents_ordered = ["tcl_standalone", "tcl", "fixed", "lag_multi", "hprs"]
    agents_present = [a for a in agents_ordered if any(r["agent"] == a for r in rows)]

    # Aggregate across seeds per agent
    agg: dict[str, dict[int, list[float]]] = {a: {k: [] for k in range(K_COSTS)} for a in agents_present}
    for r in rows:
        a = str(r["agent"])
        if a not in agg:
            continue
        for k in range(K_COSTS):
            v = r.get(f"k{k}/csr_steady_mean", float("nan"))
            if not np.isnan(v):
                agg[a][k].append(v)

    fig, axes = plt.subplots(1, K_COSTS, figsize=(2.23 * K_COSTS, 4.2), sharey=True)
    constraint_labels = [r"$c_1$: util $\geq$ 0.80", r"$c_2$: CTR $\geq$ 0.15", r"$c_3$: margin $\geq -4$"]
    x = np.arange(len(agents_present))
    width = 0.65

    for k, ax in enumerate(axes):
        means = [np.mean(agg[a][k]) if agg[a][k] else 0.0 for a in agents_present]
        stds = [np.std(agg[a][k]) if len(agg[a][k]) > 1 else 0.0 for a in agents_present]
        colors = [AGENT_COLORS.get(a, "gray") for a in agents_present]
        bars = ax.bar(x, means, width, color=colors, alpha=0.80, yerr=stds,
                      capsize=4, error_kw={"elinewidth": 1.2})
        # Highlight standalone
        for i, a in enumerate(agents_present):
            if a == "tcl_standalone":
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(1.8)
        ax.axhline(0.9, color="black", ls=":", alpha=0.5, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [AGENT_LABELS.get(a, a) for a in agents_present],
            rotation=45, ha="right",
        )
        ax.set_title(f"CSR: {constraint_labels[k]}")
        ax.set_ylim(0, 1.09)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        ax.tick_params(axis="y", labelsize=8)

    axes[0].set_ylabel(r"steady-state CSR (mean $\pm$ std, last 20%)")
    fig.suptitle(
        "Phase 5-A1: Steady-state CSR, Standalone (A) vs. Shaped (B)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = ["agent", "seed", "return_steady_mean", "return_steady_std"]
    for k in range(K_COSTS):
        fieldnames += [
            f"k{k}/steps_to_csr_0.9",
            f"k{k}/csr_steady_mean",
            f"k{k}/csr_steady_std",
        ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--standalone-dir", type=Path,
        default=Path("runs/phase5_a1_standalone_ll"),
        help="Run dir for tcl_standalone (Formulation A, last_layer).",
    )
    parser.add_argument(
        "--shaped-dir", type=Path,
        default=Path("runs/phase5_a1_v2"),
        help="Run dir for shaped baselines (Formulation B).",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("figures/phase5_a1_standalone_ll"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    # Load standalone cells
    standalone_cells = discover_cells(args.standalone_dir)
    if not standalone_cells:
        print(f"[warn] no cells in {args.standalone_dir} — run the bench first:")
        print(f"  uv run python -m experiments.pilot_adcraft.run "
              f"--config experiments/pilot_adcraft/config.phase5_a1_standalone_ll.yaml")
    for agent, seed, tb_run in standalone_cells:
        print(f"  loading {agent} seed={seed} (standalone)...", end=" ", flush=True)
        m = analyze_cell(agent, tb_run)
        m["agent"] = agent
        m["seed"] = seed
        rows.append(m)
        print("done")

    # Load shaped cells
    shaped_cells = discover_cells(args.shaped_dir)
    if not shaped_cells:
        print(f"[warn] no cells in {args.shaped_dir} — shaped baselines missing.")
    for agent, seed, tb_run in shaped_cells:
        print(f"  loading {agent} seed={seed} (shaped)...", end=" ", flush=True)
        m = analyze_cell(agent, tb_run)
        m["agent"] = agent
        m["seed"] = seed
        rows.append(m)
        print("done")

    if not rows:
        raise SystemExit("No data found. Run benches before analysis.")

    write_summary_csv(rows, args.out_dir / "summary.csv")
    plot_csr(rows, args.out_dir / "csr_vs_steps.png")
    plot_return(rows, args.out_dir / "return_vs_steps.png")
    plot_summary_bar(rows, args.out_dir / "summary_bar.pdf")
    print(f"\nAll outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()

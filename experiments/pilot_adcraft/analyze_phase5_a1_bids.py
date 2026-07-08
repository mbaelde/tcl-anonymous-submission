"""
analyze_phase5_a1_bids.py — Bid distribution analysis for phase5_a1_v2.

Confirms the "low-bid drift → util violation" mechanism by plotting
rollout/action_mean over training for each agent × seed.

Expected outcome (based on phase5_a1 results):
  - Failure seeds (lag_multi s1, tcl s3): action_mean → ~0.25–0.32 (underspend)
  - Success seeds: action_mean → ~0.40–0.51 (util ≥ 0.80 feasible zone)
  - lag_multi all seeds: same bid range as failure (λ≡0 → unconstrained SAC)
  - fixed: stable around the cost-weighted bid level

Run from repo root:
    uv run python experiments/pilot_adcraft/analyze_phase5_a1_bids.py
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs" / "phase5_a1_v2"
FIGURES_DIR = REPO_ROOT / "figures" / "phase5_a1_v2"
OUT_FILE = FIGURES_DIR / "bid_distributions.png"

AGENTS = ["lag_multi", "fixed", "tcl", "hprs"]
SEEDS = [1, 2, 3]
SMOOTH_WINDOW = 15

# Empirical feasibility window (from calibration sweep)
BID_FEASIBLE_LO = 0.33   # bid≈0.33–0.36 → util just above 0.80
BID_FEASIBLE_HI = 0.51   # bid≈0.51 → util≈0.99

# phase5_a1 (v1) CSR_c0 results — used to annotate failure vs success
PHASE5_A1_CSR_C0 = {
    "lag_multi": {1: 0.750, 2: 1.000, 3: 1.000},
    "fixed":     {1: 1.000, 2: 1.000, 3: 1.000},
    "tcl":       {1: 1.000, 2: 1.000, 3: 0.780},
    "hprs":      {1: 1.000, 2: 0.960, 3: 1.000},
}

COLORS = {
    "lag_multi": "#1f77b4",
    "fixed":     "#2ca02c",
    "tcl":       "#d62728",
    "hprs":      "#ff7f0e",
}

SEED_LINE_STYLES = {1: "-", 2: "--", 3: ":"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_tb_run(agent: str, seed: int) -> str:
    pattern = str(RUNS_DIR / agent / f"seed={seed}" / "tb" / "*/")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No TB run found for agent={agent} seed={seed} (pattern={pattern})"
        )
    return sorted(matches)[-1]  # most recent if multiple


def load_scalar(ea: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=float)
    vals = np.array([e.value for e in events], dtype=float)
    return steps, vals


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def load_action_mean(agent: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load rollout/action_mean for a single (agent, seed) cell."""
    tb_dir = find_tb_run(agent, seed)
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    steps, vals = load_scalar(ea, "rollout/action_mean")
    return steps, vals


def final_bid(steps: np.ndarray, vals: np.ndarray, last_n: int = 10) -> float:
    """Mean bid over the last `last_n` episode points."""
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals[-last_n:]))


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading TensorBoard action_mean data from v2 runs...")
    data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    missing: list[str] = []

    for agent in AGENTS:
        data[agent] = {}
        for seed in SEEDS:
            try:
                steps, vals = load_action_mean(agent, seed)
                data[agent][seed] = (steps, vals)
                print(f"  {agent} seed={seed}: {len(vals)} episodes, "
                      f"final_bid={final_bid(steps, vals):.3f}")
            except (FileNotFoundError, KeyError) as e:
                missing.append(f"{agent}/seed={seed}: {e}")
                data[agent][seed] = (np.array([]), np.array([]))

    if missing:
        print("\n[WARNING] Missing runs:")
        for m in missing:
            print(f"  {m}")

    # ------------------------------------------------------------------
    # Figure layout: 4 subplots (one per agent) + 1 summary bar chart
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 10), facecolor="white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    agent_axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    ax_summary = fig.add_subplot(gs[:, 2])

    # Feasibility band
    for ax in agent_axes:
        ax.axhspan(BID_FEASIBLE_LO, BID_FEASIBLE_HI, color="green", alpha=0.08,
                   label="feasible zone (util≥0.80)")
        ax.axhline(BID_FEASIBLE_LO, color="green", linewidth=0.8, linestyle="--")

    # Per-agent trajectory plots
    final_bids: dict[str, dict[int, float]] = {}
    for ax, agent in zip(agent_axes, AGENTS):
        final_bids[agent] = {}
        color = COLORS[agent]
        for seed in SEEDS:
            steps, vals = data[agent][seed]
            if len(vals) == 0:
                continue
            vals_sm = smooth(vals, SMOOTH_WINDOW)
            fb = final_bid(steps, vals_sm)
            final_bids[agent][seed] = fb
            csr = PHASE5_A1_CSR_C0[agent][seed]
            is_failure = csr < 0.95
            lw = 2.2 if is_failure else 1.4
            ls = SEED_LINE_STYLES[seed]
            marker = ""
            label = f"s{seed} | CSR_c0={csr:.3f}"
            if is_failure:
                label += " ← FAIL"
            ax.plot(steps, vals_sm, color=color, linewidth=lw, linestyle=ls,
                    label=label)

        ax.set_title(agent, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Training Step", fontsize=8)
        ax.set_ylabel("Mean Bid (action_mean)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("white")
        ax.set_ylim(0.0, 1.0)

    # Summary: final bid per (agent, seed) as grouped bar chart
    bar_x = np.arange(len(SEEDS))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(AGENTS))
    for i, agent in enumerate(AGENTS):
        heights = [final_bids[agent].get(s, float("nan")) for s in SEEDS]
        bars = ax_summary.bar(
            bar_x + offsets[i], heights, width,
            color=COLORS[agent], alpha=0.8, label=agent
        )
        # Annotate with CSR_c0
        for j, (bar, s) in enumerate(zip(bars, SEEDS)):
            csr = PHASE5_A1_CSR_C0[agent][s]
            if csr < 0.95:
                ax_summary.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    "✗", ha="center", va="bottom", fontsize=9, color="red"
                )

    ax_summary.axhspan(BID_FEASIBLE_LO, BID_FEASIBLE_HI, color="green", alpha=0.08,
                       label="feasible zone")
    ax_summary.axhline(BID_FEASIBLE_LO, color="green", linewidth=0.8, linestyle="--")
    ax_summary.set_title("Final Bid (last 10 eps)\nby Agent × Seed\n(✗ = CSR_c0 < 0.95)",
                         fontsize=10, fontweight="bold")
    ax_summary.set_xticks(bar_x)
    ax_summary.set_xticklabels([f"seed={s}" for s in SEEDS], fontsize=9)
    ax_summary.set_ylabel("Mean Bid (action_mean)", fontsize=8)
    ax_summary.set_ylim(0.0, 0.8)
    ax_summary.legend(fontsize=8, loc="upper right")
    ax_summary.grid(True, alpha=0.3, axis="y")
    ax_summary.set_facecolor("white")

    fig.suptitle(
        "Phase 5-A1 v2 — Bid Distributions: Low-Bid Drift → Util Violation Mechanism\n"
        f"(green band = feasible zone bid≥{BID_FEASIBLE_LO:.2f}; "
        f"✗ = CSR_c0 < 0.95 from v1 results)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved: {OUT_FILE}")
    print(f"File size: {os.path.getsize(OUT_FILE) / 1024:.1f} KB")

    # --- Print summary table ---
    print("\n=== Final Bid Summary (mean of last 10 episodes) ===")
    print(f"{'Agent':<12} {'Seed':>6} {'FinalBid':>10} {'CSR_c0':>8} {'Status':>10}")
    print("-" * 50)
    for agent in AGENTS:
        for seed in SEEDS:
            fb = final_bids[agent].get(seed, float("nan"))
            csr = PHASE5_A1_CSR_C0[agent][seed]
            status = "FAIL" if csr < 0.95 else "ok"
            print(f"{agent:<12} {seed:>6} {fb:>10.3f} {csr:>8.3f} {status:>10}")


if __name__ == "__main__":
    main()

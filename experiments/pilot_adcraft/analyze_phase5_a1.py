"""
analyze_phase5_a1.py — Mechanism diagnostic for phase5_a1 experiment.

Produces figures/phase5_a1/mechanism_diagnostic.png with 3 subplots:
  1. Episode return (all agents, mean ± 1std across seeds, smoothed)
  2. Episode cost c0 normalised by ep_steps (all agents, mean ± 1std, smoothed)
  3. dual/cost_batch_mean_k0 for lag_multi only (per-seed, not averaged)

Paths are hardcoded relative to the repo root (runs/, figures/).
Run from repo root: uv run python experiments/pilot_adcraft/analyze_phase5_a1.py
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # tcl-code/
RUNS_DIR = REPO_ROOT / "runs" / "phase5_a1"
FIGURES_DIR = REPO_ROOT / "figures" / "phase5_a1"
OUT_FILE = FIGURES_DIR / "mechanism_diagnostic.png"

AGENTS = ["lag_multi", "fixed", "tcl", "hprs"]
SEEDS = [1, 2, 3]
SMOOTH_WINDOW = 20

COLORS = {
    "lag_multi": "#1f77b4",   # blue
    "fixed":     "#2ca02c",   # green
    "tcl":       "#d62728",   # red
    "hprs":      "#ff7f0e",   # orange
}

SEED_COLORS = ["#1f77b4", "#9467bd", "#17becf"]  # for the 3 seeds of lag_multi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_tb_run(agent: str, seed: int) -> str:
    """Return the path to the TensorBoard event directory."""
    pattern = str(RUNS_DIR / agent / f"seed={seed}" / "tb" / "*/")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No TB run found for agent={agent} seed={seed} (pattern={pattern})")
    return matches[0]


def load_scalar(ea: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays for a scalar tag."""
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=float)
    vals = np.array([e.value for e in events], dtype=float)
    return steps, vals


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Uniform moving average with edge padding."""
    if window <= 1 or len(values) == 0:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def interp_to_common_steps(
    series_list: list[tuple[np.ndarray, np.ndarray]],
    n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate multiple (steps, vals) series onto a common step grid.
    Returns (common_steps, mean, std).
    """
    # Common range = intersection of all series
    x_min = max(s[0][0] for s in series_list)
    x_max = min(s[0][-1] for s in series_list)
    if x_min >= x_max:
        # Fallback: union range
        x_min = min(s[0][0] for s in series_list)
        x_max = max(s[0][-1] for s in series_list)
    common = np.linspace(x_min, x_max, n_points)
    interped = np.stack([np.interp(common, s[0], s[1]) for s in series_list], axis=0)
    return common, interped.mean(axis=0), interped.std(axis=0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def cost_tag_for_agent(agent: str) -> str:
    """Different agents use different tag names for c0 cost."""
    if agent == "lag_multi":
        return "rollout/episode_cost_k0"
    return "rollout/episode_cost_0"


def load_agent_return_cost(
    agent: str,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """
    For a given agent, load return and c0 cost for all seeds.
    Returns {
        'return': [(steps, smoothed_vals), ...],   # per seed
        'cost_norm': [(steps, smoothed_vals), ...],   # per seed, normalised
    }
    """
    returns = []
    costs = []
    cost_tag = cost_tag_for_agent(agent)

    for seed in SEEDS:
        tb_dir = find_tb_run(agent, seed)
        ea = EventAccumulator(tb_dir)
        ea.Reload()

        # Return
        steps_r, vals_r = load_scalar(ea, "rollout/episode_return")
        vals_r_sm = smooth(vals_r, SMOOTH_WINDOW)
        returns.append((steps_r, vals_r_sm))

        # Cost c0 — normalise by episode steps
        steps_c, vals_c = load_scalar(ea, cost_tag)
        steps_ep, vals_ep = load_scalar(ea, "rollout/episode_steps")
        # Align episode_steps to cost steps (they should be identical, but interpolate to be safe)
        ep_steps_aligned = np.interp(steps_c, steps_ep, vals_ep)
        ep_steps_aligned = np.where(ep_steps_aligned > 0, ep_steps_aligned, 1.0)
        vals_c_norm = vals_c / ep_steps_aligned
        vals_c_norm_sm = smooth(vals_c_norm, SMOOTH_WINDOW)
        costs.append((steps_c, vals_c_norm_sm))

    return {"return": returns, "cost_norm": costs}


def load_lag_multi_dual() -> list[tuple[np.ndarray, np.ndarray]]:
    """Load dual/cost_batch_mean_k0 for lag_multi, one series per seed."""
    series = []
    for seed in SEEDS:
        tb_dir = find_tb_run("lag_multi", seed)
        ea = EventAccumulator(tb_dir)
        ea.Reload()
        steps, vals = load_scalar(ea, "dual/cost_batch_mean_k0")
        series.append((steps, vals))
    return series


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_band(
    ax: plt.Axes,
    common_steps: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    label: str,
) -> None:
    ax.plot(common_steps, mean, color=color, label=label, linewidth=1.5)
    ax.fill_between(common_steps, mean - std, mean + std, color=color, alpha=0.20)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data for all agents ---
    print("Loading TensorBoard data...")
    agent_data: dict[str, dict] = {}
    for agent in AGENTS:
        print(f"  {agent}...", end=" ", flush=True)
        agent_data[agent] = load_agent_return_cost(agent)
        print("done")

    lag_dual = load_lag_multi_dual()
    print("  lag_multi dual/cost_batch_mean_k0... done")

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white")
    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------------ #
    # Subplot 1 — Episode return
    # ------------------------------------------------------------------ #
    ax1 = axes[0]
    for agent in AGENTS:
        series = agent_data[agent]["return"]
        common, mean, std = interp_to_common_steps(series)
        plot_band(ax1, common, mean, std, COLORS[agent], label=agent)

    ax1.set_title("Episode Return", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("white")

    # ------------------------------------------------------------------ #
    # Subplot 2 — Normalised episode cost c0
    # ------------------------------------------------------------------ #
    ax2 = axes[1]
    for agent in AGENTS:
        series = agent_data[agent]["cost_norm"]
        common, mean, std = interp_to_common_steps(series)
        plot_band(ax2, common, mean, std, COLORS[agent], label=agent)

    ax2.axhline(0.0, color="black", linewidth=1.0, linestyle="--", label="CSR_c0 = 0")
    ax2.set_title("Episode Cost c0 (normalised by ep_steps)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("mean cost_k0 / ep_steps")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("white")

    # ------------------------------------------------------------------ #
    # Subplot 3 — lag_multi dual/cost_batch_mean_k0 per seed
    # ------------------------------------------------------------------ #
    ax3 = axes[2]
    for i, (seed, (steps, vals)) in enumerate(zip(SEEDS, lag_dual)):
        ax3.plot(steps, vals, color=SEED_COLORS[i], label=f"seed={seed}", linewidth=1.2)

    ax3.axhline(0.0, color="black", linewidth=1.5, linestyle="--", label="y = 0 (λ threshold)")
    ax3.set_title("lag_multi — Batch Cost Mean c0\n(dual/cost_batch_mean_k0 per seed)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Batch mean cost k0")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor("white")

    # ------------------------------------------------------------------ #
    # Final layout
    # ------------------------------------------------------------------ #
    fig.suptitle(
        "Phase 5-A1 — Mechanism Diagnostic: Why does lag_multi have λ≡0 yet violate c0?",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved: {OUT_FILE}")
    print(f"File size: {os.path.getsize(OUT_FILE) / 1024:.1f} KB")


if __name__ == "__main__":
    main()

"""Analyzer for the beta-schedule comparison experiment.

Reads TensorBoard event files under `--run-dir` (one per cell), computes
per-episode Constraint Satisfaction Rate (CSR) per constraint, and produces

  1. `summary.csv` — one row per cell with steps-to-CSR>=0.9, steady-state
     CSR (last 20% of training), and residual CSR variance, per constraint.
  2. `csr_vs_steps.png` — smoothed CSR_k(t) curves, K subplots, one curve
     per (schedule, seed).
  3. `beta_traces.png` — beta_k(t) per schedule (sanity that the configured
     schedules look as expected).

CSR per episode: 1 if the *average* per-step cost over the episode is
<= threshold, 0 otherwise. We rely on `rollout/episode_cost_{k}` (summed
over the episode) and `rollout/episode_steps` to recover the average.
Threshold defaults to 0.0 (canonical: c_k > 0 = violation).

Usage::

    uv run python -m experiments.beta_schedule.analyze \\
        --run-dir runs/beta_schedule --out-dir figures/beta_schedule
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCHEDULE_COLORS = {
    "increasing": "#1f77b4",
    "decreasing": "#d62728",
    "fixed": "#2ca02c",
}


def discover_cells(run_dir: Path) -> list[tuple[str, int, Path]]:
    """Yield (schedule_name, seed, tb_run_dir) tuples.

    Layout expected: ``<run_dir>/<schedule>/seed=<seed>/tb/<run_name>/``
    where ``<run_name>`` is the timestamped directory created by
    ``sac_tcl.train``.
    """
    cells: list[tuple[str, int, Path]] = []
    for cell in sorted(run_dir.glob("*/seed=*/tb")):
        m = re.fullmatch(r"seed=(\d+)", cell.parent.name)
        if not m:
            continue
        seed = int(m.group(1))
        schedule = cell.parent.parent.name
        # tb/<run_name>/events.out.tfevents.*
        run_dirs = [p for p in cell.iterdir() if p.is_dir()]
        if not run_dirs:
            continue
        # Pick the most recent run directory (largest mtime).
        tb_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
        cells.append((schedule, seed, tb_run))
    return cells


def load_scalars(tb_run: Path, tags: list[str]) -> dict[str, np.ndarray]:
    """Read the requested scalar tags from a TB run directory.

    Returns a dict ``tag -> (steps, values)`` shaped as two 1-D arrays.
    Missing tags are silently omitted from the returned dict.
    """
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


def per_episode_csr(
    cost_sum: np.ndarray,
    ep_steps: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """CSR = 1 iff mean per-step cost over the episode is <= threshold."""
    safe_steps = np.where(ep_steps > 0, ep_steps, 1)
    mean_cost = cost_sum / safe_steps
    return (mean_cost <= threshold).astype(np.float64)


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average (length == len(x))."""
    if window <= 1 or x.size == 0:
        return x.astype(np.float64)
    cs = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.size):
        lo = max(0, i + 1 - window)
        out[i] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
    return out


def steps_to_threshold(
    steps: np.ndarray,
    smoothed: np.ndarray,
    threshold: float,
) -> int | None:
    """First env step at which the smoothed CSR is >= threshold."""
    mask = smoothed >= threshold
    if not mask.any():
        return None
    return int(steps[np.argmax(mask)])


def analyze_cell(
    tb_run: Path,
    k_costs: int,
    threshold: float,
    smooth_window: int,
    steady_fraction: float,
) -> dict[str, object]:
    """Return per-cell summary metrics + raw arrays for plotting."""
    cost_tags = [f"rollout/episode_cost_{k}" for k in range(k_costs)]
    beta_tags = [f"train/beta_{k}" for k in range(k_costs)]
    tags = [*cost_tags, "rollout/episode_steps", "rollout/episode_return", *beta_tags]
    scalars = load_scalars(tb_run, tags)

    # Episode boundaries are determined by the `rollout/episode_steps` tag —
    # all rollout/* tags are emitted at the same step.
    if "rollout/episode_steps/values" not in scalars:
        return {"tb_run": str(tb_run), "error": "no episode_steps tag"}

    ep_steps_axis = scalars["rollout/episode_steps/steps"]
    ep_steps_vals = scalars["rollout/episode_steps/values"]

    metrics: dict[str, object] = {"tb_run": str(tb_run)}
    csr_per_k: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for k in range(k_costs):
        tag = f"rollout/episode_cost_{k}"
        if f"{tag}/values" not in scalars:
            metrics[f"k{k}/error"] = f"missing {tag}"
            continue
        cost_sum = scalars[f"{tag}/values"]
        # Defensive: align lengths (TB writes are interleaved but co-emitted).
        n = min(cost_sum.shape[0], ep_steps_vals.shape[0])
        csr = per_episode_csr(cost_sum[:n], ep_steps_vals[:n], threshold=threshold)
        csr_smoothed = smooth(csr, smooth_window)
        csr_per_k[k] = (ep_steps_axis[:n], csr_smoothed)

        steady_start = int(n * (1.0 - steady_fraction))
        steady = csr[steady_start:]
        metrics[f"k{k}/steps_to_csr_0.9"] = steps_to_threshold(
            ep_steps_axis[:n], csr_smoothed, 0.9
        )
        metrics[f"k{k}/csr_steady_mean"] = float(steady.mean()) if steady.size else float("nan")
        metrics[f"k{k}/csr_steady_var"] = float(steady.var()) if steady.size else float("nan")

    beta_per_k: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(k_costs):
        if f"train/beta_{k}/values" not in scalars:
            continue
        beta_per_k[k] = (
            scalars[f"train/beta_{k}/steps"],
            scalars[f"train/beta_{k}/values"],
        )

    metrics["_csr_per_k"] = csr_per_k
    metrics["_beta_per_k"] = beta_per_k
    return metrics


def write_summary_csv(rows: list[dict[str, object]], out_path: Path, k_costs: int) -> None:
    fieldnames = ["schedule", "seed", "tb_run"]
    for k in range(k_costs):
        fieldnames += [
            f"k{k}/steps_to_csr_0.9",
            f"k{k}/csr_steady_mean",
            f"k{k}/csr_steady_var",
        ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")


def plot_csr(rows: list[dict[str, object]], k_costs: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, k_costs, figsize=(4.5 * k_costs, 3.5), sharey=True)
    if k_costs == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        for r in rows:
            csr_per_k = r.get("_csr_per_k", {})
            if k not in csr_per_k:
                continue
            steps, csr = csr_per_k[k]
            color = SCHEDULE_COLORS.get(str(r["schedule"]), "gray")
            ax.plot(steps, csr, color=color, alpha=0.6, lw=1.0,
                    label=f"{r['schedule']} s{r['seed']}")
        ax.axhline(0.9, color="black", ls=":", alpha=0.4, lw=0.8)
        ax.set_xlabel("env step")
        ax.set_title(f"CSR_{k}(t)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, ls=":", alpha=0.4)
    axes[0].set_ylabel("smoothed CSR")
    # Dedup legend (one entry per schedule).
    handles, labels = axes[0].get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lab in zip(handles, labels, strict=False):
        key = lab.split()[0]
        if key not in seen:
            seen[key] = h
    if seen:
        axes[-1].legend(list(seen.values()), list(seen.keys()), fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_beta_traces(rows: list[dict[str, object]], k_costs: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, k_costs, figsize=(4.5 * k_costs, 3.0), sharey=True)
    if k_costs == 1:
        axes = [axes]
    # One trace per (schedule, seed); seeds overlap so they should coincide.
    seen_per_ax: list[dict[str, bool]] = [defaultdict(bool) for _ in range(k_costs)]
    for k, ax in enumerate(axes):
        for r in rows:
            beta_per_k = r.get("_beta_per_k", {})
            if k not in beta_per_k:
                continue
            steps, betas = beta_per_k[k]
            schedule = str(r["schedule"])
            color = SCHEDULE_COLORS.get(schedule, "gray")
            label = schedule if not seen_per_ax[k][schedule] else None
            ax.plot(steps, betas, color=color, alpha=0.7, lw=1.0, label=label)
            seen_per_ax[k][schedule] = True
        ax.set_xlabel("env step")
        ax.set_title(rf"$\beta_{k}(t)$")
        ax.grid(True, ls=":", alpha=0.4)
    axes[0].set_ylabel(r"$\beta$")
    axes[-1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def analyze(
    run_dir: Path,
    out_dir: Path,
    k_costs: int,
    threshold: float,
    smooth_window: int,
    steady_fraction: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = discover_cells(run_dir)
    if not cells:
        raise SystemExit(f"No TB cells found under {run_dir}")

    rows: list[dict[str, object]] = []
    for schedule, seed, tb_run in cells:
        metrics = analyze_cell(
            tb_run,
            k_costs=k_costs,
            threshold=threshold,
            smooth_window=smooth_window,
            steady_fraction=steady_fraction,
        )
        metrics["schedule"] = schedule
        metrics["seed"] = seed
        rows.append(metrics)

    write_summary_csv(rows, out_dir / "summary.csv", k_costs=k_costs)
    plot_csr(rows, k_costs=k_costs, out_path=out_dir / "csr_vs_steps.png")
    plot_beta_traces(rows, k_costs=k_costs, out_path=out_dir / "beta_traces.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/beta_schedule"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/beta_schedule"))
    parser.add_argument("--k-costs", type=int, default=3,
                        help="Number of constraints (K). Default 3 for AdCraft K=3.")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Per-step cost threshold; CSR = 1 iff mean_ep_cost <= threshold.")
    parser.add_argument("--smooth-window", type=int, default=20,
                        help="Episode-window for the moving average of CSR.")
    parser.add_argument("--steady-fraction", type=float, default=0.2,
                        help="Fraction of episodes (from the end) used for steady-state stats.")
    args = parser.parse_args()
    analyze(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        k_costs=args.k_costs,
        threshold=args.threshold,
        smooth_window=args.smooth_window,
        steady_fraction=args.steady_fraction,
    )


if __name__ == "__main__":
    main()

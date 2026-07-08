"""Analyzer for the section 7.1 pilot: 4 agents on MultiConstraintAdCraft (K=3).

Reads TensorBoard event files under `--run-dir` (one per cell) and produces

  1. `summary.csv` — one row per (agent, seed) cell with steps-to-CSR>=0.9,
     steady-state CSR mean/var (last 20% of training), and steady-state
     episode return, per constraint.
  2. `return_vs_steps.png` — episode-return trajectories, one curve per
     (agent, seed); per-agent color.
  3. `csr_vs_steps.png` — smoothed CSR_k(t), one curve per (agent, seed),
     K subplots.
  4. `dual_traces.png` — lambda_k(t) for the lag_multi cells only (sanity).

Tag-naming caveat: `sac_lagrangian_multi` writes cost scalars under
``rollout/episode_cost_k{k}`` (k-prefix) while `sac_tcl` / `sac_fixed` /
`sac_hprs` write ``rollout/episode_cost_{k}`` (no prefix). The analyzer
tries both names per agent and uses whichever is present.

Usage::

    uv run python -m experiments.pilot_adcraft.analyze \\
        --run-dir runs/pilot_adcraft --out-dir figures/pilot_adcraft
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

AGENT_COLORS = {
    "tcl": "#1f77b4",
    "lag_multi": "#d62728",
    "fixed": "#2ca02c",
    "hprs": "#9467bd",
}

# Per-agent cost-tag template. ``{k}`` is substituted with the constraint
# index. `sac_lagrangian_multi` uses the k-prefix, the others do not.
COST_TAG_TEMPLATES = {
    "lag_multi": "rollout/episode_cost_k{k}",
    "fixed": "rollout/episode_cost_{k}",
    "tcl": "rollout/episode_cost_{k}",
    "hprs": "rollout/episode_cost_{k}",
}


def discover_cells(run_dir: Path) -> list[tuple[str, int, Path]]:
    """Yield (agent_name, seed, tb_run_dir) tuples.

    Layout expected: ``<run_dir>/<agent>/seed=<seed>/tb/<run_name>/``
    where ``<run_name>`` is the timestamped directory created by the
    agent's ``train`` entry point.
    """
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
    """Read the requested scalar tags from a TB run directory.

    Returns a dict ``"<tag>/steps" -> steps_array`` and
    ``"<tag>/values" -> values_array``. Missing tags are silently omitted.
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
    agent: str,
    tb_run: Path,
    k_costs: int,
    threshold: float,
    smooth_window: int,
    steady_fraction: float,
) -> dict[str, object]:
    """Return per-cell summary metrics + raw arrays for plotting."""
    cost_template = COST_TAG_TEMPLATES.get(agent, "rollout/episode_cost_{k}")
    cost_tags = [cost_template.format(k=k) for k in range(k_costs)]
    # Fallback set — try the other naming if the primary is absent (defensive).
    alt_template = (
        "rollout/episode_cost_{k}"
        if cost_template.endswith("_k{k}")
        else "rollout/episode_cost_k{k}"
    )
    alt_tags = [alt_template.format(k=k) for k in range(k_costs)]

    tags: list[str] = [
        *cost_tags,
        *alt_tags,
        "rollout/episode_steps",
        "rollout/episode_return",
    ]
    if agent == "tcl":
        tags += [f"train/beta_{k}" for k in range(k_costs)]
    if agent == "lag_multi":
        tags += [f"dual/lambda_k{k}" for k in range(k_costs)]

    scalars = load_scalars(tb_run, tags)

    if "rollout/episode_steps/values" not in scalars:
        return {"tb_run": str(tb_run), "error": "no episode_steps tag"}

    ep_steps_axis = scalars["rollout/episode_steps/steps"]
    ep_steps_vals = scalars["rollout/episode_steps/values"]

    metrics: dict[str, object] = {"tb_run": str(tb_run)}

    # Episode return — recorded on the same step axis as costs.
    if "rollout/episode_return/values" in scalars:
        ret_vals = scalars["rollout/episode_return/values"]
        ret_steps = scalars["rollout/episode_return/steps"]
        n_ret = ret_vals.size
        steady_start = int(n_ret * (1.0 - steady_fraction))
        steady_ret = ret_vals[steady_start:]
        metrics["return_steady_mean"] = (
            float(steady_ret.mean()) if steady_ret.size else float("nan")
        )
        metrics["_return"] = (ret_steps, ret_vals)
    else:
        metrics["return_steady_mean"] = float("nan")
        metrics["_return"] = (np.array([]), np.array([]))

    csr_per_k: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(k_costs):
        primary = cost_tags[k]
        alt = alt_tags[k]
        if f"{primary}/values" in scalars:
            cost_sum = scalars[f"{primary}/values"]
        elif f"{alt}/values" in scalars:
            cost_sum = scalars[f"{alt}/values"]
        else:
            metrics[f"k{k}/error"] = f"missing cost tag (tried {primary} and {alt})"
            continue
        n = min(cost_sum.shape[0], ep_steps_vals.shape[0])
        csr = per_episode_csr(cost_sum[:n], ep_steps_vals[:n], threshold=threshold)
        csr_smoothed = smooth(csr, smooth_window)
        csr_per_k[k] = (ep_steps_axis[:n], csr_smoothed)

        steady_start = int(n * (1.0 - steady_fraction))
        steady = csr[steady_start:]
        metrics[f"k{k}/steps_to_csr_0.9"] = steps_to_threshold(
            ep_steps_axis[:n], csr_smoothed, 0.9
        )
        metrics[f"k{k}/csr_steady_mean"] = (
            float(steady.mean()) if steady.size else float("nan")
        )
        metrics[f"k{k}/csr_steady_var"] = (
            float(steady.var()) if steady.size else float("nan")
        )
    metrics["_csr_per_k"] = csr_per_k

    # Agent-specific extras: lambda traces (lag_multi), beta traces (tcl).
    lambda_per_k: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if agent == "lag_multi":
        for k in range(k_costs):
            tag = f"dual/lambda_k{k}"
            if f"{tag}/values" not in scalars:
                continue
            lambda_per_k[k] = (scalars[f"{tag}/steps"], scalars[f"{tag}/values"])
    metrics["_lambda_per_k"] = lambda_per_k

    return metrics


def write_summary_csv(
    rows: list[dict[str, object]], out_path: Path, k_costs: int
) -> None:
    fieldnames = ["agent", "seed", "tb_run", "return_steady_mean"]
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


def _dedup_legend(ax: plt.Axes) -> None:
    """Keep one legend handle per agent (label is always 'agent sN')."""
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lab in zip(handles, labels, strict=False):
        key = lab.split()[0]
        if key not in seen:
            seen[key] = h
    if seen:
        ax.legend(list(seen.values()), list(seen.keys()), fontsize=8, loc="best")


def plot_return(rows: list[dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for r in rows:
        steps, vals = r.get("_return", (np.array([]), np.array([])))
        if steps.size == 0:
            continue
        agent = str(r["agent"])
        color = AGENT_COLORS.get(agent, "gray")
        ax.plot(steps, vals, color=color, alpha=0.55, lw=1.0,
                label=f"{agent} s{r['seed']}")
    ax.set_xlabel("env step")
    ax.set_ylabel("episode return")
    ax.set_title("Episode return (pilot AdCraft, K=3)")
    ax.grid(True, ls=":", alpha=0.4)
    _dedup_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_csr(
    rows: list[dict[str, object]], k_costs: int, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, k_costs, figsize=(4.5 * k_costs, 3.5), sharey=True)
    if k_costs == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        for r in rows:
            csr_per_k = r.get("_csr_per_k", {})
            if k not in csr_per_k:
                continue
            steps, csr = csr_per_k[k]
            agent = str(r["agent"])
            color = AGENT_COLORS.get(agent, "gray")
            ax.plot(steps, csr, color=color, alpha=0.55, lw=1.0,
                    label=f"{agent} s{r['seed']}")
        ax.axhline(0.9, color="black", ls=":", alpha=0.4, lw=0.8)
        ax.set_xlabel("env step")
        ax.set_title(f"CSR_{k}(t)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, ls=":", alpha=0.4)
    axes[0].set_ylabel("smoothed CSR")
    _dedup_legend(axes[-1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_dual_traces(
    rows: list[dict[str, object]], k_costs: int, out_path: Path
) -> None:
    lag_rows = [r for r in rows if str(r["agent"]) == "lag_multi"]
    if not lag_rows:
        print(f"skip {out_path} (no lag_multi cells)")
        return
    fig, axes = plt.subplots(1, k_costs, figsize=(4.5 * k_costs, 3.0), sharey=True)
    if k_costs == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        for r in lag_rows:
            lambda_per_k = r.get("_lambda_per_k", {})
            if k not in lambda_per_k:
                continue
            steps, lam = lambda_per_k[k]
            ax.plot(steps, lam, alpha=0.7, lw=1.0, label=f"s{r['seed']}")
        ax.set_xlabel("env step")
        ax.set_title(rf"$\lambda_{k}(t)$")
        ax.grid(True, ls=":", alpha=0.4)
    axes[0].set_ylabel(r"$\lambda$")
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
    for agent, seed, tb_run in cells:
        metrics = analyze_cell(
            agent=agent,
            tb_run=tb_run,
            k_costs=k_costs,
            threshold=threshold,
            smooth_window=smooth_window,
            steady_fraction=steady_fraction,
        )
        metrics["agent"] = agent
        metrics["seed"] = seed
        rows.append(metrics)

    write_summary_csv(rows, out_dir / "summary.csv", k_costs=k_costs)
    plot_return(rows, out_path=out_dir / "return_vs_steps.png")
    plot_csr(rows, k_costs=k_costs, out_path=out_dir / "csr_vs_steps.png")
    plot_dual_traces(rows, k_costs=k_costs, out_path=out_dir / "dual_traces.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/pilot_adcraft"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/pilot_adcraft"))
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

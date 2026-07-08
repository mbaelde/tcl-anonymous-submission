"""Reward-shift validation analyzer — Proposition 4 fix experiment.

Reads TensorBoard data from runs/phase5_reward_shift (TCL with C=150),
and compares against A1 baseline (TCL no shift, from runs/phase5_a1).

Produces:
  figures/phase5_reward_shift/reward_shift_csr_curves.png
      — CSR_c0 over training steps: TCL (no shift) vs TCL (C=150), 3 seeds each.
  figures/phase5_reward_shift/summary.csv
      — per-seed CSR_c0, CSR_c2, Return_ss for both conditions + Fixed-SAC (A1).

Usage (from repo root)::

    uv run python experiments/pilot_adcraft/analyze_reward_shift.py
"""

from __future__ import annotations

import csv
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_A1   = REPO_ROOT / "runs" / "phase5_a1"
RUNS_RS   = REPO_ROOT / "runs" / "phase5_reward_shift"
FIGS_DIR  = REPO_ROOT / "figures" / "phase5_reward_shift"

SEEDS         = [1, 2, 3]
STEADY_FRAC   = 0.20   # last 20% of episodes
SMOOTH_WINDOW = 20
K_COSTS       = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_tb_run(runs_root: Path, agent: str, seed: int) -> Path:
    pattern = str(runs_root / agent / f"seed={seed}" / "tb" / "*/")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No TB run for agent={agent} seed={seed} under {runs_root}"
        )
    return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))


def load_scalars(tb_run: Path, tags: list[str]) -> dict[str, np.ndarray]:
    ea = EventAccumulator(str(tb_run), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out: dict[str, np.ndarray] = {}
    for tag in tags:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        out[f"{tag}/steps"]  = np.array([e.step  for e in events], dtype=np.int64)
        out[f"{tag}/values"] = np.array([e.value for e in events], dtype=np.float64)
    return out


def per_episode_csr(cost_sum: np.ndarray, ep_steps: np.ndarray) -> np.ndarray:
    """1 iff mean per-step cost <= 0 (canonical: c_k > 0 = violation)."""
    safe = np.where(ep_steps > 0, ep_steps, 1.0)
    return (cost_sum / safe <= 0.0).astype(np.float64)


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size == 0:
        return x.astype(np.float64)
    cs = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.size):
        lo = max(0, i + 1 - w)
        out[i] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
    return out


def cell_metrics(
    runs_root: Path, agent: str, seed: int
) -> dict[str, float | np.ndarray]:
    tb_run = find_tb_run(runs_root, agent, seed)
    tags = [
        "rollout/episode_return",
        "rollout/episode_steps",
        *[f"rollout/episode_cost_{k}" for k in range(K_COSTS)],
    ]
    sc = load_scalars(tb_run, tags)

    ep_steps = sc.get("rollout/episode_steps/values", np.array([]))
    ret_vals  = sc.get("rollout/episode_return/values", np.array([]))
    ret_steps = sc.get("rollout/episode_return/steps",  np.array([]))

    n = min(ep_steps.size, ret_vals.size)
    steady_start = int(n * (1.0 - STEADY_FRAC))
    return_ss = float(ret_vals[steady_start:].mean()) if n > 0 else float("nan")

    csr_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    csr_ss: dict[int, float] = {}
    for k in range(K_COSTS):
        key = f"rollout/episode_cost_{k}/values"
        if key not in sc:
            continue
        cost_vals = sc[key]
        m = min(cost_vals.size, ep_steps.size)
        csr = per_episode_csr(cost_vals[:m], ep_steps[:m])
        csr_sm = smooth(csr, SMOOTH_WINDOW)
        ep_steps_key = sc.get("rollout/episode_steps/steps", np.arange(m))
        step_axis = sc.get(f"rollout/episode_return/steps", np.arange(m))
        csr_curves[k] = (ep_steps_key[:m] if ep_steps_key.size >= m else np.arange(m), csr_sm)
        csr_ss[k] = float(csr[steady_start:m].mean()) if m > steady_start else float("nan")

    return {
        "return_ss": return_ss,
        "csr_ss": csr_ss,
        "csr_curves": csr_curves,
        "ret_steps": ret_steps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    conditions = [
        ("TCL (no shift, A1)",  RUNS_A1,  "tcl"),
        ("TCL (C=150)",         RUNS_RS,  "tcl"),
        ("Fixed-SAC (A1 ref)",  RUNS_A1,  "fixed"),
    ]

    # ---- collect per-seed metrics ----------------------------------------
    rows: list[dict] = []
    curve_data: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    for label, runs_root, agent in conditions:
        c0_curves = []
        for seed in SEEDS:
            try:
                m = cell_metrics(runs_root, agent, seed)
            except FileNotFoundError as e:
                print(f"  WARN: {e}")
                continue
            rows.append({
                "condition": label,
                "seed": seed,
                "CSR_c0": round(m["csr_ss"].get(0, float("nan")), 4),
                "CSR_c2": round(m["csr_ss"].get(2, float("nan")), 4),
                "Return_ss": round(m["return_ss"], 1),
            })
            if 0 in m["csr_curves"]:
                c0_curves.append(m["csr_curves"][0])
        curve_data[label] = c0_curves

    # ---- CSV summary -------------------------------------------------------
    csv_path = FIGS_DIR / "summary.csv"
    fieldnames = ["condition", "seed", "CSR_c0", "CSR_c2", "Return_ss"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written: {csv_path}")

    # ---- aggregate stats ---------------------------------------------------
    print("\n--- Reward-shift validation (mean ± std over 3 seeds) ---")
    from collections import defaultdict
    cond_rows: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        cond_rows[r["condition"]].append(r)

    for label, rs in cond_rows.items():
        c0 = [r["CSR_c0"] for r in rs if not np.isnan(r["CSR_c0"])]
        c2 = [r["CSR_c2"] for r in rs if not np.isnan(r["CSR_c2"])]
        ret = [r["Return_ss"] for r in rs if not np.isnan(r["Return_ss"])]
        print(
            f"  {label:30s}  "
            f"CSR_c0={np.mean(c0):.3f}±{np.std(c0):.3f}  "
            f"CSR_c2={np.mean(c2):.3f}±{np.std(c2):.3f}  "
            f"Return={np.mean(ret):.0f}±{np.std(ret):.0f}"
        )

    # ---- CSR_c0 curves figure ----------------------------------------------
    colors = {
        "TCL (no shift, A1)":  "#d62728",   # red
        "TCL (C=150)":         "#1f77b4",   # blue
        "Fixed-SAC (A1 ref)":  "#2ca02c",   # green
    }
    linestyles = {
        "TCL (no shift, A1)":  "--",
        "TCL (C=150)":         "-",
        "Fixed-SAC (A1 ref)":  ":",
    }

    fig, ax = plt.subplots(figsize=(4.9, 3.5), facecolor="white")
    ax.set_facecolor("white")

    for label, curves in curve_data.items():
        if not curves:
            continue
        color = colors.get(label, "gray")
        ls    = linestyles.get(label, "-")
        x_min = max(c[0][0]  for c in curves)
        x_max = min(c[0][-1] for c in curves)
        common = np.linspace(x_min, x_max, 500)
        interped = np.stack([np.interp(common, c[0], c[1]) for c in curves])
        mean, std = interped.mean(0), interped.std(0)
        ax.plot(common, mean, color=color, linestyle=ls, linewidth=1.8, label=label)
        ax.fill_between(common, mean - std, mean + std, color=color, alpha=0.18)

    ax.axhline(0.9, color="black", linewidth=0.8, linestyle=":", label="CSR = 0.9")
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"CSR$_{c_1}$ (utilization)")
    ax.set_title(
        r"Reward-shift fix (Prop. 4): CSR$_{c_1}$ over training"
        "\n"
        r"A1 env (target\_util=0.80, drift=0.03, $\beta$=10)",
    )
    ax.legend(loc="lower right")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_path = FIGS_DIR / "reward_shift_csr_curves.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    fig_pdf = FIGS_DIR / "reward_shift_csr_curves.pdf"
    fig.savefig(fig_pdf, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved: {fig_path}")
    print(f"Figure saved: {fig_pdf}")


if __name__ == "__main__":
    main()

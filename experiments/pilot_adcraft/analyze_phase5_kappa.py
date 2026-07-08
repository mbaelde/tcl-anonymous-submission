"""Analyze Phase 5 κ-calibration: Gaussian gate (empirical vs formula κ) vs B1 linear.

Produces:
  figures/phase5_kappa/
    csr_comparison.png  — CSR_c0/c2 over steps: 3 conditions × 3 seeds
    summary.txt         — steady-state CSR and return for all conditions

Usage::

    uv run python -m experiments.pilot_adcraft.analyze_phase5_kappa
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
KAPPA_DIR = _REPO_ROOT / "runs" / "phase5_kappa"
B1_TCL_DIR = _REPO_ROOT / "runs" / "phase5_b1" / "tcl"
OUT_DIR = _REPO_ROOT / "figures" / "phase5_kappa"

SEEDS = [1, 2, 3]
K_COSTS = 3
STEADY_FRACTION = 0.20
SMOOTH_WINDOW = 20

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_tb_run(seed_dir: Path) -> Path | None:
    tb = seed_dir / "tb"
    if not tb.exists():
        return None
    runs = sorted(tb.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


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
    safe_steps = np.where(ep_steps > 0, ep_steps, 1)
    mean_cost = cost_sum / safe_steps
    return (mean_cost <= 0.0).astype(np.float64)


def load_cell(seed_dir: Path) -> dict | None:
    tb_run = find_tb_run(seed_dir)
    if tb_run is None:
        return None
    tags = [
        "rollout/episode_return",
        "rollout/episode_steps",
        *(f"rollout/episode_cost_{k}" for k in range(K_COSTS)),
    ]
    s = load_scalars(tb_run, tags)
    if "rollout/episode_steps/values" not in s:
        return None

    ep_steps = s["rollout/episode_steps/values"]
    ep_steps_axis = s["rollout/episode_steps/steps"]

    csr_per_k: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(K_COSTS):
        tag = f"rollout/episode_cost_{k}"
        if f"{tag}/values" not in s:
            continue
        cost = s[f"{tag}/values"]
        n = min(cost.shape[0], ep_steps.shape[0])
        csr = per_episode_csr(cost[:n], ep_steps[:n])
        csr_per_k[k] = (ep_steps_axis[:n], smooth(csr, SMOOTH_WINDOW))

    ret_vals = s.get("rollout/episode_return/values", np.array([]))

    ss_csr: dict[int, float] = {}
    for k, (steps, vals) in csr_per_k.items():
        n = len(vals)
        ss_start = int(n * (1.0 - STEADY_FRACTION))
        ss_csr[k] = float(vals[ss_start:].mean()) if ss_start < n else float("nan")

    n_ret = len(ret_vals)
    ss_ret_start = int(n_ret * (1.0 - STEADY_FRACTION))
    ss_return = float(ret_vals[ss_ret_start:].mean()) if ss_ret_start < n_ret else float("nan")

    return {"csr_per_k": csr_per_k, "ss_csr": ss_csr, "ss_return": ss_return}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conditions: dict[str, dict[int, dict]] = {}

    # B1 linear baseline (existing data)
    b1_cells: dict[int, dict] = {}
    for seed in SEEDS:
        cell = load_cell(B1_TCL_DIR / f"seed={seed}")
        if cell:
            b1_cells[seed] = cell
    if b1_cells:
        conditions["B1 linear (β=10)"] = b1_cells

    # Gaussian conditions
    for agent in ("tcl_gaussian_empirical", "tcl_gaussian_formula"):
        cells: dict[int, dict] = {}
        for seed in SEEDS:
            cell = load_cell(KAPPA_DIR / agent / f"seed={seed}")
            if cell:
                cells[seed] = cell
        if cells:
            label = "κ empirical (0.5/10/10)" if "empirical" in agent else "κ formula (0.308/17.33/17.33)"
            conditions[label] = cells

    if len(conditions) <= 1:
        print("Gaussian runs not found or not yet complete. Re-run after training finishes.")
        return

    # ── Figure: CSR_c0 and CSR_c2 comparison ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    cmap = plt.get_cmap("tab10")
    colors = {name: cmap(i) for i, name in enumerate(conditions)}
    seed_alphas = {1: 1.0, 2: 0.65, 3: 0.35}

    for ax, k, ylabel in zip(axes, [0, 2], ["CSR$_{c_0}$ (utilization)", "CSR$_{c_2}$ (margin)"]):
        for label, cells in conditions.items():
            for seed, cell in cells.items():
                if k not in cell["csr_per_k"]:
                    continue
                steps, vals = cell["csr_per_k"][k]
                ss_val = cell["ss_csr"].get(k, float("nan"))
                ax.plot(steps, vals, color=colors[label], alpha=seed_alphas[seed],
                        linewidth=1.2 if seed > 1 else 1.8,
                        label=f"{label} seed={seed} (ss={ss_val:.3f})" if seed == 1 else None)
        ax.axhline(0.9, linestyle="--", color="k", alpha=0.4, linewidth=0.8)
        ax.set_title(ylabel)
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7, ncol=1)

    fig.suptitle("κ-calibration ablation: Gaussian gate vs linear gate (B1 env, drift=0.01)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "csr_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'csr_comparison.png'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    lines = ["Phase 5 κ-calibration — Gaussian gate vs B1 linear\n",
             "=" * 65, ""]
    lines.append(f"{'':>35} {'CSR_c0':>8} {'CSR_c1':>8} {'CSR_c2':>8} {'Return_ss':>10}")
    lines.append("-" * 65)

    for label, cells in conditions.items():
        if not cells:
            lines.append(f"{label:>35}  (no data)")
            continue
        for seed, cell in sorted(cells.items()):
            csrs = " ".join(f"{cell['ss_csr'].get(k, float('nan')):8.3f}"
                            for k in range(K_COSTS))
            lines.append(f"{label:>35} seed={seed}  {csrs} {cell['ss_return']:10.1f}")
        c0_vals = [c["ss_csr"].get(0, float("nan")) for c in cells.values()]
        c2_vals = [c["ss_csr"].get(2, float("nan")) for c in cells.values()]
        ret_vals_list = [c["ss_return"] for c in cells.values()]
        lines.append(
            f"{'MEAN':>35}{'':7}"
            f"{np.nanmean(c0_vals):8.3f}"
            f"{'':8}"
            f"{np.nanmean(c2_vals):8.3f}"
            f"{np.nanmean(ret_vals_list):10.1f}"
        )
        lines.append("")

    summary_path = OUT_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n{'=' * 65}")
    print("\n".join(lines))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()

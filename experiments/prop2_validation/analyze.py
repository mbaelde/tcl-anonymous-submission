"""Analyzer for Proposition 2 validation.

Reads all `traj.npz` files under `--run-dir`, measures the steady-state
peak-to-peak amplitude of lambda(t), and compares against the
theoretical prediction

    A_lambda(alpha, omega) = 2 * alpha * A / sqrt(alpha^2 + omega^2),

which is O(alpha / omega) for omega >> alpha. Produces:

  1. `summary.csv` — one row per cell with (alpha, omega, seed, A_emp,
     A_theory, ratio).
  2. `prop2_amplitude_vs_omega.png` — empirical vs theoretical amplitude
     vs omega, one curve per alpha (log-log).
  3. `prop2_lambda_traces.png` — sample lambda(t) traces for visual
     inspection of the oscillation regime.

Steady-state window: by default the last 40% of the trajectory.

Usage:
    uv run python -m experiments.prop2_validation.analyze \\
        --run-dir runs/prop2 --out-dir figures/prop2
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def amplitude_peak_to_peak(x: np.ndarray, *, q_low: float = 0.02, q_high: float = 0.98) -> float:
    """Robust peak-to-peak amplitude via quantile clipping.

    Using quantiles instead of min/max avoids spikes from the
    first few transient steps after the steady state is reached.
    """
    return float(np.quantile(x, q_high) - np.quantile(x, q_low))


def steady_state_window(lam: np.ndarray, fraction: float = 0.4) -> np.ndarray:
    n = lam.shape[0]
    start = int(n * (1.0 - fraction))
    return lam[start:]


def theoretical_amplitude(alpha: float, omega: float, A: float) -> float:
    return 2.0 * alpha * A / math.sqrt(alpha * alpha + omega * omega)


def discover_cells(run_dir: Path) -> list[Path]:
    return sorted(run_dir.rglob("traj.npz"))


def parse_cell_dir(traj_path: Path) -> tuple[float | None, float | None, int | None]:
    """Recover (alpha, omega_ppe, seed) from the path components."""
    alpha = omega_ppe = seed = None
    for part in traj_path.parts:
        m = re.fullmatch(r"alpha=([\deE+\-.]+)", part)
        if m:
            alpha = float(m.group(1))
            continue
        m = re.fullmatch(r"omega_ppe=([\deE+\-.]+)", part)
        if m:
            omega_ppe = float(m.group(1))
            continue
        m = re.fullmatch(r"seed=(\d+)", part)
        if m:
            seed = int(m.group(1))
    return alpha, omega_ppe, seed


def analyze(run_dir: Path, out_dir: Path, *, steady_fraction: float = 0.4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = discover_cells(run_dir)
    if not cells:
        raise SystemExit(f"No traj.npz found under {run_dir}")

    rows: list[dict[str, float | int | str]] = []
    for traj_path in cells:
        d = np.load(traj_path)
        lam = d["lam"].astype(np.float64)
        omega = float(d["omega"])
        amplitude_env = float(d["amplitude"])
        # Prefer the path-derived alpha (lambda_lr) — saved in the npz too.
        alpha_path, omega_ppe_path, seed_path = parse_cell_dir(traj_path)
        alpha = float(d["lambda_lr"]) if alpha_path is None else alpha_path
        seed = int(d["seed"]) if seed_path is None else seed_path

        lam_ss = steady_state_window(lam, fraction=steady_fraction)
        A_emp = amplitude_peak_to_peak(lam_ss)
        A_theory = theoretical_amplitude(alpha=alpha, omega=omega, A=amplitude_env)
        ratio = A_emp / A_theory if A_theory > 0 else float("nan")

        rows.append({
            "alpha": alpha,
            "omega": omega,
            "omega_ppe": omega_ppe_path if omega_ppe_path is not None else float("nan"),
            "seed": seed,
            "A_emp": A_emp,
            "A_theory": A_theory,
            "ratio": ratio,
            "lambda_mean_ss": float(lam_ss.mean()),
            "path": str(traj_path),
        })

    # ---- summary.csv ----
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    # ---- amplitude vs omega, by alpha ----
    by_alpha: dict[float, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_alpha[r["alpha"]][r["omega"]].append(r["A_emp"])

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    alphas_sorted = sorted(by_alpha.keys())
    A_env = amplitude_env  # last loaded; assumed constant across grid
    for i, alpha in enumerate(alphas_sorted):
        omegas = sorted(by_alpha[alpha].keys())
        emp_mean = [np.mean(by_alpha[alpha][w]) for w in omegas]
        emp_std = [np.std(by_alpha[alpha][w]) for w in omegas]
        color = cmap(i / max(len(alphas_sorted) - 1, 1))
        ax.errorbar(omegas, emp_mean, yerr=emp_std, fmt="o-",
                    color=color, label=f"emp. α={alpha:.0e}")
        theory = [theoretical_amplitude(alpha, w, A_env) for w in omegas]
        ax.plot(omegas, theory, "--", color=color, alpha=0.7,
                label=f"theory α={alpha:.0e}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\omega$ (rad/step)")
    ax.set_ylabel(r"peak-to-peak amplitude of $\lambda(t)$")
    ax.set_title(r"Prop 2: dual oscillation amplitude vs $\omega$")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig_path = out_dir / "prop2_amplitude_vs_omega.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"wrote {fig_path}")

    # ---- sample traces: one row per alpha, columns over omega ----
    omegas_all = sorted({r["omega"] for r in rows})
    n_rows = len(alphas_sorted)
    n_cols = len(omegas_all)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.0 * n_rows),
                             squeeze=False, sharex=True)
    for i, alpha in enumerate(alphas_sorted):
        for j, omega in enumerate(omegas_all):
            ax = axes[i][j]
            cands = [r for r in rows if r["alpha"] == alpha and r["omega"] == omega]
            if not cands:
                ax.set_axis_off()
                continue
            r0 = min(cands, key=lambda r: r["seed"])
            d = np.load(r0["path"])
            lam = d["lam"]
            ax.plot(lam, lw=0.7)
            ax.set_title(f"α={alpha:.0e}, ω={omega:.3f}", fontsize=8)
            ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("step", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\lambda(t)$", fontsize=8)
    fig.tight_layout()
    fig_path = out_dir / "prop2_lambda_traces.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"wrote {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/prop2"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/prop2"))
    parser.add_argument("--steady-fraction", type=float, default=0.4,
                        help="Fraction of the trajectory (from the end) to use as steady state.")
    args = parser.parse_args()
    analyze(args.run_dir, args.out_dir, steady_fraction=args.steady_fraction)


if __name__ == "__main__":
    main()

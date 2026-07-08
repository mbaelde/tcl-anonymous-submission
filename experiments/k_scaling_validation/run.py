"""K-scaling validation of Theorem 2: Hessian scales as O(K^3 beta^2).

Extends toy_validation/run.py to K in {2, 3, 5, 8, 10}, confirming that the
dominant term 2K^3 beta^2 M_1^2 in Theorem 2 captures the K-dependence.

Expected slopes (log-log):
  - log(Hess_max) ~ 2 * log(beta) + const  at fixed K  (slope ≈ 2)
  - log(Hess_max) ~ 3 * log(K)   + const  at fixed beta (slope ≈ 3)

Usage:
    py -3.14 -m uv run python -m experiments.k_scaling_validation.run \\
        --output-dir runs/k_scaling_validation
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.toy_validation.run import (
    hessian_max_on_grid,
    thm2_bound_paper,
    write_csv,
)

K_GRID: list[int] = [2, 3, 5, 8, 10]
BETA_GRID: np.ndarray = np.array([10.0, 20.0, 50.0, 100.0, 200.0])


def run_k_scaling(A: float, omega: float) -> list[dict]:
    M1 = A * omega
    M2 = A * omega**2
    rows: list[dict] = []
    total = len(K_GRID) * len(BETA_GRID)
    done = 0
    for K in K_GRID:
        for beta in BETA_GRID.tolist():
            done += 1
            print(f"  [{done}/{total}] K={K}, beta={beta:.0f}...", flush=True)
            stats = hessian_max_on_grid(beta=float(beta), K=K, A=A, omega=omega)
            bound = thm2_bound_paper(float(beta), K, M1, M2)
            dominant = 2.0 * (K**3) * (beta**2) * (M1**2)
            rows.append(
                {
                    "K": K,
                    "beta": float(beta),
                    "hess_max": stats["hess_max"],
                    "a_argmax": stats["a_argmax"],
                    "bound_paper": bound,
                    "bound_dominant": dominant,
                    "ratio_empirical_to_dominant": stats["hess_max"] / dominant if dominant > 0 else float("nan"),
                    "log_K": math.log(K),
                    "log_beta": math.log(float(beta)),
                    "log_hess": math.log(stats["hess_max"]) if stats["hess_max"] > 0 else float("-inf"),
                    "log_bound": math.log(bound) if bound > 0 else float("-inf"),
                }
            )
    return rows


def fit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    p = np.polyfit(x[mask], y[mask], 1)
    return float(p[0]), float(p[1])


def compute_fits(rows: list[dict]) -> dict:
    fits: dict = {}

    # Slope in log(beta) at fixed K  — expected ≈ 2
    fits["slope_vs_beta_fixed_K"] = {}
    for K in K_GRID:
        sub = [r for r in rows if r["K"] == K]
        log_b = np.array([r["log_beta"] for r in sub])
        log_h = np.array([r["log_hess"] for r in sub])
        slope, intercept = fit_slope(log_b, log_h)
        fits["slope_vs_beta_fixed_K"][f"K={K}"] = {
            "slope": slope,
            "intercept": intercept,
            "predicted": 2.0,
        }

    # Slope in log(K) at fixed beta  — expected ≈ 3
    fits["slope_vs_K_fixed_beta"] = {}
    for beta in BETA_GRID.tolist():
        sub = [r for r in rows if r["beta"] == beta]
        log_k = np.array([r["log_K"] for r in sub])
        log_h = np.array([r["log_hess"] for r in sub])
        slope, intercept = fit_slope(log_k, log_h)
        fits["slope_vs_K_fixed_beta"][f"beta={beta:.0f}"] = {
            "slope": slope,
            "intercept": intercept,
            "predicted": 3.0,
        }

    return fits


def make_figures(rows: list[dict], out: Path, A: float, omega: float) -> None:
    try:
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
            "font.size":                   9,
            "axes.titlesize":              9,
            "axes.labelsize":              8,
            "xtick.labelsize":             7,
            "ytick.labelsize":             7,
            "legend.fontsize":             7,
            "figure.dpi":                  150,
            "savefig.bbox":                "tight",
            "savefig.pad_inches":          0.05,
        })
    except ImportError:
        print("  matplotlib not available -- skipping figures")
        return

    M1 = A * omega
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.5))

    # Panel 1: Hess_max vs beta (log-log) per K
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(K_GRID)))
    for i, K in enumerate(K_GRID):
        sub = [r for r in rows if r["K"] == K]
        betas = np.array([r["beta"] for r in sub])
        hess = np.array([r["hess_max"] for r in sub])
        bound = np.array([r["bound_paper"] for r in sub])
        ax.loglog(betas, hess, "o-", color=colors[i], label=f"K={K}")
        ax.loglog(betas, bound, "--", color=colors[i], alpha=0.45)
    # Reference slope-2 line
    beta_ref = np.array([10.0, 200.0])
    ref_val = rows[0]["hess_max"] * (beta_ref / 10.0) ** 2
    ax.loglog(beta_ref, ref_val, "k:", lw=1, label="slope 2 ref")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\max|\partial^2_a R_\mathrm{TCL}|$")
    ax.set_title(r"Hessian vs $\beta$ (log-log), slope $\approx 2$")
    ax.legend()
    ax.grid(True, which="both", ls=":")

    # Panel 2: Hess_max vs K (log-log) at each fixed beta
    ax = axes[1]
    beta_colors = plt.cm.plasma(np.linspace(0.0, 0.85, len(BETA_GRID)))
    K_arr = np.array(K_GRID, dtype=float)
    for j, beta in enumerate(BETA_GRID.tolist()):
        sub = [r for r in rows if r["beta"] == beta]
        Ks = np.array([r["K"] for r in sub], dtype=float)
        hess = np.array([r["hess_max"] for r in sub])
        dominant = 2.0 * Ks**3 * beta**2 * M1**2
        ax.loglog(Ks, hess, "o-", color=beta_colors[j], label=rf"$\beta$={beta:.0f}")
        ax.loglog(Ks, dominant, "--", color=beta_colors[j], alpha=0.45)
    # Reference slope-3 line anchored at K=2, beta=10
    K_ref = np.array([2.0, 10.0])
    anchor = [r for r in rows if r["K"] == 2 and r["beta"] == 10.0][0]["hess_max"]
    ref_val3 = anchor * (K_ref / 2.0) ** 3
    ax.loglog(K_ref, ref_val3, "k:", lw=1, label="slope 3 ref")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$\max|\partial^2_a R_\mathrm{TCL}|$")
    ax.set_title(r"Hessian vs $K$ (log-log), slope $\approx 3$")
    ax.legend()
    ax.grid(True, which="both", ls=":")

    plt.tight_layout()
    fig_path = out / "k_scaling.pdf"
    plt.savefig(str(fig_path), bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--A", type=float, default=0.3)
    parser.add_argument("--omega", type=float, default=1.0)
    cli = parser.parse_args()

    out = cli.output_dir
    out.mkdir(parents=True, exist_ok=True)

    M1 = cli.A * cli.omega
    print(f"K-scaling validation  K={K_GRID}  beta={BETA_GRID.tolist()}")
    print(f"A={cli.A}  omega={cli.omega}  M1={M1:.3f}  dominant=2*K^3*beta^2*{M1**2:.4f}")
    print()

    rows = run_k_scaling(A=cli.A, omega=cli.omega)
    write_csv(rows, out / "k_scaling.csv")

    fits = compute_fits(rows)
    metadata = {
        "K_grid": K_GRID,
        "beta_grid": BETA_GRID.tolist(),
        "A": cli.A,
        "omega": cli.omega,
        "M1": M1,
        "M2": cli.A * cli.omega**2,
        "fits": fits,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Console summary
    print()
    print("=== Slope in log(beta) at fixed K  (predicted ~2.0) ===")
    print(f"{'':6} {'slope':>8}  {'predicted':>9}")
    for label, info in fits["slope_vs_beta_fixed_K"].items():
        print(f"{label:<6} {info['slope']:>8.4f}  {info['predicted']:>9.1f}")

    print()
    print("=== Slope in log(K) at fixed beta  (predicted ~3.0) ===")
    print(f"{'':12} {'slope':>8}  {'predicted':>9}")
    for label, info in fits["slope_vs_K_fixed_beta"].items():
        print(f"{label:<12} {info['slope']:>8.4f}  {info['predicted']:>9.1f}")

    print()
    make_figures(rows, out, cli.A, cli.omega)

    print(f"\nOutputs: {out}/k_scaling.csv  metadata.json  k_scaling.pdf")


if __name__ == "__main__":
    main()

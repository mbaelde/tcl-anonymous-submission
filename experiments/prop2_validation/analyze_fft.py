"""FFT-based analyzer for Proposition 2 validation.

Measures A_emp via complex demodulation at the forcing frequency omega,
which isolates the steady-state oscillation and rejects transient noise
(e.g. from random_phase_at_reset). For a steady-state lambda(t) ~
B + (Asin) * sin(omega t + phi), the demodulation amplitude

    L_hat(omega) = (2/N) * sum_t lambda(t) * exp(-i omega t)

satisfies |L_hat| -> Asin in the steady state. The peak-to-peak amplitude
equivalent is 2 * |L_hat|, comparable to A_theory = 2 alpha A / sqrt(alpha^2 + omega^2).

Usage:
    py -3.14 -m uv run python -m experiments.prop2_validation.analyze_fft \
        --run-dir runs/prop2_analytic --out-dir figures/prop2_analytic_fft
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np


def amplitude_demodulation(lam: np.ndarray, omega: float) -> float:
    """Peak-to-peak amplitude of the omega-component via complex demodulation."""
    n = lam.shape[0]
    t = np.arange(n, dtype=np.float64)
    z = lam * np.exp(-1j * omega * t)
    coeff = (2.0 / n) * z.sum()
    return float(2.0 * abs(coeff))


def parse_cell_dir(traj_path: Path) -> tuple[float | None, float | None, int | None]:
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


def theoretical_amplitude(alpha: float, omega: float, A: float) -> float:
    return 2.0 * alpha * A / math.sqrt(alpha * alpha + omega * omega)


def analyze(run_dir: Path, out_dir: Path, steady_fraction: float = 0.4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = sorted(run_dir.rglob("traj.npz"))
    if not cells:
        raise SystemExit(f"No traj.npz under {run_dir}")

    rows: list[dict] = []
    for traj_path in cells:
        d = np.load(traj_path)
        lam = d["lam"].astype(np.float64)
        omega = float(d["omega"])
        A = float(d["amplitude"])
        alpha_path, omega_ppe_path, seed_path = parse_cell_dir(traj_path)
        alpha = float(d["lambda_lr"]) if alpha_path is None else alpha_path
        seed = int(d["seed"]) if seed_path is None else seed_path

        n = lam.shape[0]
        ss = lam[int(n * (1.0 - steady_fraction)):]
        A_emp = amplitude_demodulation(ss, omega)
        A_th = theoretical_amplitude(alpha, omega, A)
        ratio = A_emp / A_th if A_th > 0 else float("nan")

        rows.append({
            "alpha": alpha,
            "omega": omega,
            "omega_ppe": omega_ppe_path,
            "seed": seed,
            "A_emp_fft": A_emp,
            "A_theory": A_th,
            "ratio": ratio,
            "lambda_mean_ss": float(ss.mean()),
            "path": str(traj_path),
        })

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("runs/prop2"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures/prop2_fft"))
    ap.add_argument("--steady-fraction", type=float, default=0.4)
    args = ap.parse_args()
    analyze(args.run_dir, args.out_dir, steady_fraction=args.steady_fraction)


if __name__ == "__main__":
    main()

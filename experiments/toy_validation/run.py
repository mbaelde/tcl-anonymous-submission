"""Toy-environment validation of Proposition 1 and Theorem 2.

Proposition 1 (uniform approximation):
    |J(pi; beta) - J_inf(pi)| <= K^2 / (1-gamma) * exp(-beta * Delta_pi)
    where Delta_pi = min_j ess_inf |R_j - 1/2|.

    Setup (pointwise): a single-state bandit with deterministic policy pi(s_0) = a*
    reduces the constraint margin to Delta = min_j |R_j(a*) - 1/2|. Since the support
    of d^pi is a single point (s_0, a*), we can fix R_j values directly (no need to
    solve for a*) and measure |R_TCL(beta) - R_inf| pointwise. With gamma in [0,1),
    J = R / (1-gamma), so the discount factor only rescales by 1/(1-gamma).

Theorem 2 (Hessian bound):
    ||nabla^2_a R_TCL||_inf <= 2 K^3 beta^2 M_1^2 + K^2 beta (2 M_1^2 + M_2) + K M_2

    Setup: parametric environment R_j(a) = 1/2 + A sin(omega a + phi_j) with
    phi_j = j * pi / K, A = 0.3, omega = 1, so M_1 = A omega = 0.3 and
    M_2 = A omega^2 = 0.3. Hessian measured by centered finite differences on a
    grid of a in [-1, 1], excluding a small window around the constraint thresholds
    where the cascade gating is in its current-frontier regime (Cor. 5.3(b)).

Outputs CSV (prop1.csv, thm2.csv) + JSON metadata + two log-plots.

Usage:
    py -3.14 -m uv run python -m experiments.toy_validation.run \\
        --output-dir runs/toy_validation
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


# --------------------------------------------------------------------------------------
# Core TCL primitives
# --------------------------------------------------------------------------------------


def sigma_beta(x: np.ndarray | float, beta: float) -> np.ndarray | float:
    """Numerically stable sigmoid sigma_beta(x) = 1 / (1 + exp(-beta x))."""
    z = beta * np.asarray(x, dtype=np.float64)
    out = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    return float(out) if np.isscalar(x) else out


def r_tcl_from_values(R: np.ndarray, beta: float) -> float:
    """Compute R_TCL = sum_k w_k R_k where w_k = prod_{j<k} sigma_beta(R_j - 1/2).

    R : shape (K,) array of constraint satisfactions R_j in [0, 1].
    """
    K = R.shape[0]
    sigmas = sigma_beta(R - 0.5, beta)  # shape (K,)
    # w_1 = 1; w_{k+1} = w_k * sigma_k. We use cumulative product shifted by one.
    weights = np.ones(K, dtype=np.float64)
    if K > 1:
        weights[1:] = np.cumprod(sigmas[:-1])
    return float(np.sum(weights * R))


def r_inf_from_values(R: np.ndarray) -> float:
    """Cumulative-lex limit R_infty(s,a) = sum_k 1{forall j<k, R_j >= 1/2} R_k."""
    K = R.shape[0]
    total = 0.0
    gate_open = True
    for k in range(K):
        if gate_open:
            total += float(R[k])
        if R[k] < 0.5:
            gate_open = False
    return total


# --------------------------------------------------------------------------------------
# Parametric environment for Theorem 2: R_j(a) = 1/2 + A sin(omega a + phi_j)
# --------------------------------------------------------------------------------------


def r_j_env(a: np.ndarray, K: int, A: float = 0.3, omega: float = 1.0) -> np.ndarray:
    """Return array of shape (K, len(a)) with R_j(a) = 1/2 + A sin(omega a + j pi / K)."""
    a = np.asarray(a, dtype=np.float64)
    phi = np.array([(j + 1) * math.pi / K for j in range(K)])  # j=1..K
    arg = omega * a[None, :] + phi[:, None]
    return 0.5 + A * np.sin(arg)


def r_tcl_grid(a_grid: np.ndarray, beta: float, K: int, A: float, omega: float) -> np.ndarray:
    """Compute R_TCL(a; beta) on a 1D grid of actions."""
    R = r_j_env(a_grid, K, A=A, omega=omega)  # (K, N)
    sigmas = sigma_beta(R - 0.5, beta)  # (K, N)
    N = a_grid.shape[0]
    weights = np.ones((K, N), dtype=np.float64)
    if K > 1:
        weights[1:] = np.cumprod(sigmas[:-1], axis=0)
    return np.sum(weights * R, axis=0)


# --------------------------------------------------------------------------------------
# Proposition 1 experiment
# --------------------------------------------------------------------------------------


def prop1_cases() -> list[dict]:
    """Define a small set of pointwise validation cases with controlled margin Delta.

    Each case fixes (K, R_values) so that Delta = min_j |R_j - 1/2| is known.
    """
    return [
        {"name": "K2_sat_d0.05", "K": 2, "R": [0.55, 0.55], "Delta": 0.05},
        {"name": "K2_sat_d0.10", "K": 2, "R": [0.60, 0.60], "Delta": 0.10},
        {"name": "K2_sat_d0.20", "K": 2, "R": [0.70, 0.70], "Delta": 0.20},
        {"name": "K2_mix_d0.10", "K": 2, "R": [0.40, 0.60], "Delta": 0.10},
        {"name": "K3_sat_d0.10", "K": 3, "R": [0.60, 0.60, 0.60], "Delta": 0.10},
        {"name": "K3_sat_d0.20", "K": 3, "R": [0.70, 0.70, 0.70], "Delta": 0.20},
        {"name": "K3_mix_d0.10", "K": 3, "R": [0.60, 0.40, 0.60], "Delta": 0.10},
    ]


def run_prop1(beta_grid: np.ndarray, gamma: float) -> list[dict]:
    rows: list[dict] = []
    for case in prop1_cases():
        R = np.array(case["R"], dtype=np.float64)
        K = case["K"]
        Delta = case["Delta"]
        R_inf = r_inf_from_values(R)
        # Theoretical bound at this point: K^2 / (1-gamma) * exp(-beta * Delta).
        # Pointwise it's just K^2 * exp(-beta*Delta) (gamma rescales linearly).
        for beta in beta_grid:
            R_tcl = r_tcl_from_values(R, float(beta))
            err = abs(R_tcl - R_inf)
            # Per-step bound (drop the 1/(1-gamma) factor since we measure pointwise R).
            bound_pointwise = K**2 * math.exp(-beta * Delta)
            rows.append(
                {
                    "case": case["name"],
                    "K": K,
                    "Delta": Delta,
                    "beta": float(beta),
                    "R_tcl": R_tcl,
                    "R_inf": R_inf,
                    "err_pointwise": err,
                    "bound_pointwise": bound_pointwise,
                    "log_err": math.log(err) if err > 0 else float("-inf"),
                    "log_bound": math.log(bound_pointwise),
                }
            )
    return rows


def fit_pente(beta: np.ndarray, log_err: np.ndarray) -> tuple[float, float]:
    """Fit log(err) ~ pente * beta + intercept; returns (pente, intercept)."""
    mask = np.isfinite(log_err)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    p = np.polyfit(beta[mask], log_err[mask], 1)
    return float(p[0]), float(p[1])


# --------------------------------------------------------------------------------------
# Theorem 2 experiment
# --------------------------------------------------------------------------------------


def hessian_max_on_grid(
    beta: float,
    K: int,
    A: float = 0.3,
    omega: float = 1.0,
    a_min: float = -math.pi,
    a_max: float = math.pi,
    n_points: int = 20001,
    h: float = 1e-4,
) -> dict:
    """Estimate sup_a |partial^2_a R_TCL(a; beta)| by centered finite differences.

    We sample a on a fine grid covering a full period of the sin parameterization,
    so the supremum is attained in the interior. The Hessian peak is in a window
    |R_j - 1/2| ~ 1/beta around each threshold crossing (current-frontier regime,
    Cor. 5.3(b)), which shrinks with beta; n_points is chosen to keep at least
    ~5 grid points inside that window up to beta = 50.
    """
    a = np.linspace(a_min, a_max, n_points)
    f_minus = r_tcl_grid(a - h, beta, K, A, omega)
    f_zero = r_tcl_grid(a, beta, K, A, omega)
    f_plus = r_tcl_grid(a + h, beta, K, A, omega)
    hess = (f_plus - 2.0 * f_zero + f_minus) / (h * h)
    hess_max = float(np.max(np.abs(hess)))
    a_argmax = float(a[int(np.argmax(np.abs(hess)))])
    return {
        "hess_max": hess_max,
        "a_argmax": a_argmax,
        "n_total": n_points,
    }


def thm2_bound_paper(beta: float, K: int, M1: float, M2: float) -> float:
    """Loose Hessian bound from Theorem 2(ii)."""
    return 2.0 * (K**3) * (beta**2) * (M1**2) + (K**2) * beta * (2 * M1**2 + M2) + K * M2


def thm2_bound_fine_K2(beta: float, M1: float, M2: float) -> float:
    """Tighter K=2 Hessian bound from Remark A.4 (factor ~166 improvement)."""
    return (
        (beta**2) / (6.0 * math.sqrt(3.0)) * (M1**2)
        + (beta / 2.0) * (M1**2)
        + (2.0 + beta / 4.0) * M2
    )


def run_thm2(beta_grid: np.ndarray, K_grid: list[int], A: float, omega: float) -> list[dict]:
    M1 = A * omega
    M2 = A * (omega**2)
    rows: list[dict] = []
    for K in K_grid:
        for beta in beta_grid:
            stats = hessian_max_on_grid(beta=float(beta), K=K, A=A, omega=omega)
            bound_paper = thm2_bound_paper(float(beta), K, M1, M2)
            bound_fine = thm2_bound_fine_K2(float(beta), M1, M2) if K == 2 else float("nan")
            rows.append(
                {
                    "K": K,
                    "beta": float(beta),
                    "hess_max": stats["hess_max"],
                    "a_argmax": stats["a_argmax"],
                    "bound_paper": bound_paper,
                    "bound_fine_K2": bound_fine,
                    "log_beta": math.log(float(beta)),
                    "log_hess": math.log(stats["hess_max"]) if stats["hess_max"] > 0 else float("-inf"),
                    "n_total": stats["n_total"],
                }
            )
    return rows


# --------------------------------------------------------------------------------------
# Output: CSV, JSON metadata
# --------------------------------------------------------------------------------------


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.10g}")
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--A", type=float, default=0.3)
    parser.add_argument("--omega", type=float, default=1.0)
    cli = parser.parse_args()

    out = cli.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Beta grid: decade [1, 100] with log spacing.
    beta_prop1 = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    beta_thm2 = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])

    print("Running Proposition 1 (pointwise approximation)...")
    rows_p1 = run_prop1(beta_prop1, gamma=cli.gamma)
    write_csv(rows_p1, out / "prop1.csv")

    # Per-case linear fit on log(err) vs beta. We fit the asymptotic regime
    # beta * Delta >= 2 (i.e. the leading exp term dominates the leading constant)
    # and report the all-points fit too for transparency.
    fits_p1: dict[str, dict[str, float]] = {}
    for case in prop1_cases():
        name = case["name"]
        sub = [r for r in rows_p1 if r["case"] == name]
        betas = np.array([r["beta"] for r in sub])
        log_errs = np.array([r["log_err"] for r in sub])
        pente_all, intercept_all = fit_pente(betas, log_errs)
        asymp_mask = betas * case["Delta"] >= 2.0
        if asymp_mask.sum() >= 2:
            pente_asymp, intercept_asymp = fit_pente(betas[asymp_mask], log_errs[asymp_mask])
        else:
            pente_asymp, intercept_asymp = float("nan"), float("nan")
        fits_p1[name] = {
            "pente_all": pente_all,
            "pente_asymp": pente_asymp,
            "pente_predite": -float(case["Delta"]),
            "intercept_all": intercept_all,
            "intercept_asymp": intercept_asymp,
            "n_asymp": int(asymp_mask.sum()),
            "K": int(case["K"]),
            "Delta": float(case["Delta"]),
        }

    print("Running Theorem 2 (Hessian scaling)...")
    rows_t2 = run_thm2(beta_thm2, K_grid=[2, 3], A=cli.A, omega=cli.omega)
    write_csv(rows_t2, out / "thm2.csv")

    # Log-log fit on Hessian vs beta (per K). Asymptotic regime beta >= 10 where
    # beta^2 term dominates the beta term: crossover ~ (2 M_1^2 + M_2) / (2 K M_1^2)
    # which is O(1) for our parameters; we use beta >= 10 to be safe.
    fits_t2: dict[str, dict[str, float]] = {}
    for K in [2, 3]:
        sub = [r for r in rows_t2 if r["K"] == K]
        log_b = np.array([r["log_beta"] for r in sub])
        log_h = np.array([r["log_hess"] for r in sub])
        betas_arr = np.array([r["beta"] for r in sub])
        pente_all, intercept_all = fit_pente(log_b, log_h)
        mask10 = betas_arr >= 10.0
        mask50 = betas_arr >= 50.0
        if mask10.sum() >= 2:
            pente_b10, intercept_b10 = fit_pente(log_b[mask10], log_h[mask10])
        else:
            pente_b10, intercept_b10 = float("nan"), float("nan")
        if mask50.sum() >= 2:
            pente_b50, intercept_b50 = fit_pente(log_b[mask50], log_h[mask50])
        else:
            pente_b50, intercept_b50 = float("nan"), float("nan")
        fits_t2[f"K={K}"] = {
            "pente_log_log_all": pente_all,
            "pente_log_log_asymp_b10": pente_b10,
            "pente_log_log_asymp_b50": pente_b50,
            "pente_predite_asymptotique": 2.0,
            "intercept_all": intercept_all,
            "intercept_asymp_b10": intercept_b10,
            "intercept_asymp_b50": intercept_b50,
            "n_asymp_b10": int(mask10.sum()),
            "n_asymp_b50": int(mask50.sum()),
        }

    metadata = {
        "gamma": cli.gamma,
        "A": cli.A,
        "omega": cli.omega,
        "M1": cli.A * cli.omega,
        "M2": cli.A * (cli.omega**2),
        "beta_grid_prop1": beta_prop1.tolist(),
        "beta_grid_thm2": beta_thm2.tolist(),
        "fits_prop1": fits_p1,
        "fits_thm2": fits_t2,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # ----------------------------------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------------------------------
    print()
    print("=== Proposition 1: log(err) ~ pente * beta + intercept (asymp: beta*Delta >= 2) ===")
    print(f"{'case':<18} {'Delta':>7} {'pente_all':>10} {'pente_asymp':>12} {'pred':>10} {'rel_asymp':>10}")
    for name, info in fits_p1.items():
        rel = abs(info["pente_asymp"] - info["pente_predite"]) / abs(info["pente_predite"])
        print(
            f"{name:<18} {info['Delta']:>7.3f} {info['pente_all']:>10.5f} "
            f"{info['pente_asymp']:>12.5f} {info['pente_predite']:>10.5f} {rel:>10.4f}"
        )

    print()
    print("=== Theorem 2: log(hess_max) ~ pente * log(beta) + intercept ===")
    print(f"{'K':<4} {'pente_all':>10} {'pente_b>=10':>12} {'pente_b>=50':>12} {'predite':>10}")
    for label, info in fits_t2.items():
        print(
            f"{label:<4} {info['pente_log_log_all']:>10.5f} "
            f"{info['pente_log_log_asymp_b10']:>12.5f} "
            f"{info['pente_log_log_asymp_b50']:>12.5f} "
            f"{info['pente_predite_asymptotique']:>10.3f}"
        )

    print()
    print(f"CSVs written to: {out}/prop1.csv, {out}/thm2.csv")
    print(f"Metadata: {out}/metadata.json")


if __name__ == "__main__":
    main()

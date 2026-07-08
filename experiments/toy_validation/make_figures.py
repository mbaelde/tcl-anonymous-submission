"""Generate the two validation figures for Proposition 1 and Theorem 2.

Reads:
    runs/toy_validation/prop1.csv
    runs/toy_validation/thm2.csv
    runs/toy_validation/metadata.json

Writes:
    figures/generated/prop1_loglinear.pdf
    figures/generated/thm2_loglog.pdf

Usage:
    py -3.14 -m uv run python -m experiments.toy_validation.make_figures \\
        --input-dir runs/toy_validation \\
        --output-dir figures/generated
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

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
import numpy as np


def read_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (TypeError, ValueError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def figure_prop1(rows: list[dict], meta: dict, out_path: Path) -> None:
    cases_order = [
        "K2_sat_d0.05",
        "K2_sat_d0.10",
        "K2_sat_d0.20",
        "K2_mix_d0.10",
        "K3_sat_d0.10",
        "K3_sat_d0.20",
        "K3_mix_d0.10",
    ]
    cmap = plt.get_cmap("viridis")
    colors = {name: cmap(i / max(1, len(cases_order) - 1)) for i, name in enumerate(cases_order)}

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in cases_order:
        sub = [r for r in rows if r["case"] == name]
        if not sub:
            continue
        betas = np.array([r["beta"] for r in sub])
        errs = np.array([r["err_pointwise"] for r in sub])
        K = int(sub[0]["K"])
        Delta = float(sub[0]["Delta"])
        label = f"$K={K}$, $\\Delta={Delta:g}$ ({'mix' if 'mix' in name else 'sat'})"
        ax.semilogy(betas, errs, "o-", color=colors[name], label=label, linewidth=1.2, markersize=4)
        # Theoretical bound K^2 exp(-beta Delta) as dashed line, same color.
        bounds = (K**2) * np.exp(-betas * Delta)
        ax.semilogy(betas, bounds, "--", color=colors[name], alpha=0.4, linewidth=1.0)

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$|R_{\mathrm{TCL}}(\beta) - R_\infty|$  (log scale)")
    ax.set_title(r"Proposition 1 validation: pointwise approximation error vs.\ $\beta$")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    ax.text(
        0.98,
        0.97,
        "Dashed: $K^2 e^{-\\beta\\Delta}$ bound\nSolid: measured",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_thm2(rows: list[dict], meta: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    K_colors = {2: "tab:blue", 3: "tab:orange"}
    M1 = float(meta["M1"])
    M2 = float(meta["M2"])
    for K in [2, 3]:
        sub = [r for r in rows if int(r["K"]) == K]
        if not sub:
            continue
        betas = np.array([r["beta"] for r in sub])
        hess = np.array([r["hess_max"] for r in sub])
        ax.loglog(
            betas,
            hess,
            "o-",
            color=K_colors[K],
            label=rf"$K={K}$ measured $\sup_a \|\nabla^2_a R_{{\mathrm{{TCL}}}}\|_\infty$",
            linewidth=1.4,
            markersize=5,
        )
        bound_paper = 2 * (K**3) * (betas**2) * (M1**2) + (K**2) * betas * (2 * M1**2 + M2) + K * M2
        ax.loglog(
            betas,
            bound_paper,
            "--",
            color=K_colors[K],
            alpha=0.45,
            linewidth=1.0,
            label=rf"$K={K}$ paper bound $2K^3\beta^2 M_1^2 + \ldots$",
        )
        if K == 2:
            bound_fine = (betas**2) / (6.0 * math.sqrt(3.0)) * (M1**2) + (betas / 2.0) * (M1**2) + (2.0 + betas / 4.0) * M2
            ax.loglog(
                betas,
                bound_fine,
                ":",
                color=K_colors[K],
                alpha=0.7,
                linewidth=1.2,
                label=rf"$K=2$ fine bound (Remark A.4)",
            )

    # Pure beta^2 reference line for visual slope check.
    betas_ref = np.array([10.0, 200.0])
    c_ref = 0.01
    ax.loglog(betas_ref, c_ref * betas_ref**2, "k-", alpha=0.4, linewidth=0.8)
    ax.text(220.0, c_ref * 200**2 * 0.6, r"slope $=2$", fontsize=9, color="0.3")

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\|\nabla^2_a R_{\mathrm{TCL}}\|_\infty$  (log scale)")
    ax.set_title(r"Theorem 2 validation: Hessian sup-norm vs.\ $\beta$ (log-log)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    cli = parser.parse_args()

    meta = json.loads((cli.input_dir / "metadata.json").read_text(encoding="utf-8"))
    rows_p1 = read_csv(cli.input_dir / "prop1.csv")
    rows_t2 = read_csv(cli.input_dir / "thm2.csv")

    figure_prop1(rows_p1, meta, cli.output_dir / "prop1_loglinear.pdf")
    figure_thm2(rows_t2, meta, cli.output_dir / "thm2_loglog.pdf")
    figure_prop1(rows_p1, meta, cli.output_dir / "prop1_loglinear.png")
    figure_thm2(rows_t2, meta, cli.output_dir / "thm2_loglog.png")
    print(f"Figures written to {cli.output_dir}/")


if __name__ == "__main__":
    main()

"""Plot Appendix B.4 calibration: CSR_c0 vs rb_weight for last_layer mode."""
import argparse
import pathlib

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
import pandas as pd

VARIANTS = [
    ("ignore",     None,   "D:/repos/tcl-code/figures/phase5_a1_standalone"),
    ("last_layer", 0.01,   "D:/repos/tcl-code/figures/phase5_a1_standalone_ll_w001"),
    ("last_layer", 0.1,    "D:/repos/tcl-code/figures/phase5_a1_standalone_ll_w01"),
    ("last_layer", 1.0,    "D:/repos/tcl-code/figures/phase5_a1_standalone_ll"),
]

# TCL shaped (B) reference from phase5_a1_v2 (Session 8 results)
SHAPED_CSR = {"mean": 0.35, "std": 0.07}
FIXED_CSR  = {"mean": 0.94, "std": 0.11}


def load_csr(path: str) -> tuple[float, float]:
    df = pd.read_csv(pathlib.Path(path) / "summary.csv")
    vals = df["k0/csr_steady_mean"].values
    return float(np.mean(vals)), float(np.std(vals))


def main() -> None:
    records = []
    for mode, weight, path in VARIANTS:
        mean, std = load_csr(path)
        label = r"Standalone (A)" if mode == "ignore" else rf"$\lambda={weight}$"
        records.append({"label": label, "weight": weight if weight is not None else 0.0, "mean": mean, "std": std, "mode": mode})

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 4.0))

    # --- Left: CSR_c0 bar chart ---
    ax = axes[0]
    x = np.arange(len(records))
    colors = ["#4daf4a", "#4daf4a", "#4daf4a", "#e77c22"]  # green for OK, orange for degraded
    bars = ax.bar(x, [r["mean"] for r in records], yerr=[r["std"] for r in records],
                  color=colors, capsize=5, alpha=0.85, width=0.55)
    # shaped (B) reference lines
    ax.axhline(SHAPED_CSR["mean"], color="#e377c2", ls="--", lw=1.5, label=f'TCL shaped (B): {SHAPED_CSR["mean"]:.2f}$\\pm${SHAPED_CSR["std"]:.2f}')
    ax.axhline(FIXED_CSR["mean"], color="#1f77b4", ls=":", lw=1.2, label=f'fixed: {FIXED_CSR["mean"]:.2f}$\\pm${FIXED_CSR["std"]:.2f}')
    ax.axhline(1.0, color="gray", ls="-", lw=0.7, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in records], rotation=45, ha="right")
    ax.set_ylabel(r"CSR$_{c_1}$ (util $\geq$ 0.80)")
    ax.set_ylim(0, 1.15)
    ax.set_title(r"(a) CSR$_{c_1}$ vs $\lambda$")
    ax.legend(fontsize=7, loc="lower right")
    for bar, r in zip(bars, records):
        ax.text(bar.get_x() + bar.get_width()/2, r["mean"] + r["std"] + 0.02,
                f'{r["mean"]:.3f}', ha="center", va="bottom", fontsize=8, fontweight="bold")

    # --- Right: steady-state return ---
    returns = {
        "Standalone (A)":  (-3739.0, 9.0),
        "$\\lambda=0.01$": (-3739.0, 9.0),
        "$\\lambda=0.1$":  (-3723.0, 23.0),
        "$\\lambda=1.0$":  (-3077.0, 508.0),
    }
    ax2 = axes[1]
    labels2 = [r["label"] for r in records]
    means2  = [returns[l][0] for l in labels2]
    stds2   = [returns[l][1] for l in labels2]
    colors2 = ["#4daf4a", "#4daf4a", "#4daf4a", "#e77c22"]
    ax2.bar(x, means2, yerr=stds2, color=colors2, capsize=5, alpha=0.85, width=0.55)
    ax2.axhline(-2393, color="#e377c2", ls="--", lw=1.5, label="TCL shaped (B): -2393")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels2, rotation=45, ha="right")
    ax2.set_ylabel("Steady-state return (mean over seeds)")
    ax2.set_title(r"(b) Episode return vs $\lambda$")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.38)
    out = pathlib.Path("D:/repos/tcl-code/figures/b4_calibration")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "b4_csr_return_vs_lambda.pdf")
    print(f"saved {out / 'b4_csr_return_vs_lambda.png'}")
    plt.close()


if __name__ == "__main__":
    main()

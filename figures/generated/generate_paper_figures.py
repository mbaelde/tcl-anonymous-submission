"""
Generate the 4 paper figures for TCL_paper_draft.md §7.

Usage (from tcl-code root):
    python figures/generated/generate_paper_figures.py

Output: figures/generated/{dual_collapse,b1_csr_curves,loss_budget_inversion,scenario_heatmap}.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = "figures/generated"
os.makedirs(OUT, exist_ok=True)

# ── colour palette (consistent across figures) ─────────────────────────────
COLORS = {
    "lag_multi": "#E07B39",   # orange
    "fixed":     "#4878CF",   # blue
    "tcl":       "#3A9E6D",   # green
    "hprs":      "#CC3333",   # red
}
LABELS = {
    "lag_multi": "Lag-SAC",
    "fixed":     "Fixed-SAC",
    "tcl":       "TCL-SAC",
    "hprs":      "HPRS-SAC",
}
AGENTS = ["lag_multi", "fixed", "tcl", "hprs"]
SEEDS  = [1, 2, 3]

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

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_tb_dir(run_root: str) -> str:
    """Return the single TensorBoard event directory inside run_root."""
    tb = os.path.join(run_root, "tb")
    subdirs = [d for d in os.listdir(tb) if os.path.isdir(os.path.join(tb, d))]
    return os.path.join(tb, subdirs[0])


def load_scalar(run_root: str, tag: str):
    """Return (steps[], values[]) numpy arrays for a scalar tag."""
    ea = EventAccumulator(_find_tb_dir(run_root), size_guidance={"scalars": 0})
    ea.Reload()
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def csr_from_costs(steps, costs, window: int = 1):
    """Rolling CSR: fraction of last `window` episodes with cost < 0."""
    satisfied = (costs < 0).astype(float)
    if window == 1:
        return steps, satisfied
    # rolling mean
    cumsum = np.cumsum(np.insert(satisfied, 0, 0))
    rolling = (cumsum[window:] - cumsum[:-window]) / window
    return steps[window - 1:], rolling


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Dual collapse (Lag-SAC λ_k ≡ 0)
# ─────────────────────────────────────────────────────────────────────────────

def fig1_dual_collapse():
    """2-panel figure: batch mean (buffer) vs online episode cost for Lag-SAC.
    Left: dual/cost_batch_mean_k0 — stays negative → λ update never fires.
    Right: rollout/episode_cost_k0 — occasionally positive → real violations exist.
    Together they prove the replay-buffer dilution mechanism.
    """
    base = "runs/phase5_b1/lag_multi"
    seed_colors = ["#444444", "#888888", "#BBBBBB"]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), sharey=False)
    ax_batch, ax_online = axes

    for si, seed in enumerate(SEEDS):
        run_root = f"{base}/seed={seed}"
        lw, alpha = (1.8, 1.0) if si == 0 else (1.1, 0.7)
        label = f"seed {seed}"

        # left: batch mean from replay buffer
        try:
            steps, vals = load_scalar(run_root, "dual/cost_batch_mean_k0")
            ax_batch.plot(steps / 1000, vals,
                          color=seed_colors[si], linewidth=lw, alpha=alpha, label=label)
        except Exception:
            pass

        # right: real episode cost
        try:
            steps, vals = load_scalar(run_root, "rollout/episode_cost_k0")
            ax_online.plot(steps / 1000, vals,
                           color=seed_colors[si], linewidth=lw, alpha=alpha, label=label)
        except Exception:
            pass

    for ax in axes:
        ax.axhline(0, color="#CC3333", linewidth=1.0, linestyle="--", zorder=3)
        ax.set_xlabel(r"steps ($\times 10^3$)")

    ax_batch.set_ylabel("cost $c_1$ (utilization)")
    ax_batch.set_title("Replay buffer batch mean\n$\\bar{c}_1^{\\mathrm{buf}}$ (drives $\\lambda$ update)",
                        fontsize=9)
    ax_online.set_title("Online episode cost\n$c_1^{\\mathrm{ep}}$ (true constraint signal)",
                        fontsize=9)

    # annotations
    ax_batch.text(0.5, 0.12, r"$\bar{c}_1^{\mathrm{buf}} < 0$ always" "\n"
                  r"$\Rightarrow\;\lambda \leftarrow \max(0,\,\lambda + \alpha\bar{c}_1) \equiv 0$",
                  transform=ax_batch.transAxes, ha="center", fontsize=8,
                  color="#CC3333",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CC3333", alpha=0.85))

    ax_online.text(0.5, 0.88, "real violations exist\n(seed-dependent)",
                   transform=ax_online.transAxes, ha="center", fontsize=8,
                   color="#444444",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888888", alpha=0.85))

    handles, labels = ax_batch.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=8)
    fig.suptitle("Lag-SAC dual collapse: buffer dilution mechanism (slow-drift, $K=3$)", fontsize=10)
    fig.tight_layout()
    out = f"{OUT}/dual_collapse.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[✓] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — CSR_c1 training curves (slow-drift, 2×2 grid)
# ─────────────────────────────────────────────────────────────────────────────

def _cost_tag(run_root: str, k: int) -> str:
    """Return the correct cost tag for this agent (k0 vs k, naming varies)."""
    ea = EventAccumulator(_find_tb_dir(run_root), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    for candidate in [f"rollout/episode_cost_k{k}", f"rollout/episode_cost_{k}"]:
        if candidate in tags:
            return candidate
    raise KeyError(f"No cost tag for k={k} in {run_root}. Available: {tags}")


def fig2_b1_csr_curves():
    base    = "runs/phase5_b1"
    ROLLING = 20   # rolling window in episodes

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.0), sharex=True, sharey=True)
    axes_flat = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]

    for ax, agent in zip(axes_flat, AGENTS):
        color = COLORS[agent]
        all_vals = []

        for seed in SEEDS:
            run_root = f"{base}/{agent}/seed={seed}"
            try:
                tag = _cost_tag(run_root, 0)
                steps, costs = load_scalar(run_root, tag)
                s, csr = csr_from_costs(steps, costs, window=ROLLING)
                ax.plot(s / 1000, csr, color=color, linewidth=0.9,
                        alpha=0.45)
                all_vals.append((s, csr))
            except Exception:
                pass

        # mean across seeds
        if all_vals:
            min_len = min(len(v) for _, v in all_vals)
            mean_csr = np.mean([v[:min_len] for _, v in all_vals], axis=0)
            mean_steps = all_vals[0][0][:min_len]
            ax.plot(mean_steps / 1000, mean_csr, color=color,
                    linewidth=2.2, label="mean")

        ax.axhline(0.9, color="black", linewidth=0.8, linestyle=":",
                   label="CSR = 0.9")
        ax.set_title(LABELS[agent], color=color, fontweight="bold")
        ax.set_ylim(-0.05, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    for ax in axes[1]:
        ax.set_xlabel(r"steps ($\times 10^3$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("CSR$_{c_1}$ (utilization)")

    # shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray",   linewidth=0.9, alpha=0.45, label="individual seed"),
        Line2D([0], [0], color="gray",   linewidth=2.2,              label="mean over seeds"),
        Line2D([0], [0], color="black",  linewidth=0.8, linestyle=":", label="CSR = 0.9"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle(r"Utilization CSR$_{c_1}$ over training, slow-drift scenario (drift$=0.01$)",
                 fontsize=10)
    fig.tight_layout()
    out = f"{OUT}/b1_csr_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[✓] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Loss-budget reward inversion (analytical)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_loss_budget_inversion():
    r_base = -10.0
    c1     = np.linspace(-0.6, 0.6, 1000)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    for beta, style, label in [(10, "-",  r"$\beta = 10$"),
                                (50, "--", r"$\beta = 50$")]:
        r_tcl = r_base * sigmoid(-beta * c1)
        ax.plot(c1, r_tcl, linewidth=2.0, linestyle=style, label=label)

    # feasible asymptote
    ax.axhline(r_base, color="black", linewidth=0.8, linestyle=":",
               label=r"$r_{\mathrm{base}} = -10$ (feasible asymptote)")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

    # feasibility boundary
    ax.axvline(0, color="#999999", linewidth=1.0, linestyle="--")
    ax.text(-0.55, 1.0, r"$\leftarrow$ feasible", va="top", fontsize=8, color="#555555")
    ax.text( 0.03, 1.0, r"infeasible $\rightarrow$", va="top", fontsize=8, color="#555555")

    # shade infeasible region
    ax.axvspan(0, 0.6, alpha=0.07, color="#CC3333")

    # annotation arrow
    ax.annotate("infeasible region\npreferred ($r_{TCL} > r_{base}$)",
                xy=(0.25, r_base * sigmoid(-50 * 0.25)),
                xytext=(0.38, -4.5),
                arrowprops=dict(arrowstyle="->", color="#CC3333", lw=1.2),
                fontsize=8, color="#CC3333")

    ax.set_xlabel(r"Utilization violation $c_1$  (positive $\Rightarrow$ infeasible)")
    ax.set_ylabel(r"$r_{\mathrm{TCL}} = r_{\mathrm{base}} \cdot \sigma(-\beta c_1)$")
    ax.set_title("Loss-budget reward inversion under TCL")
    ax.legend(loc="lower left", frameon=True)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(r_base - 0.5, 1.5)

    fig.tight_layout()
    out = f"{OUT}/loss_budget_inversion.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[✓] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Scenario heatmap (CSR_c1 mean × agent × scenario)
# ─────────────────────────────────────────────────────────────────────────────

def fig4_scenario_heatmap():
    # Data from experimental results (means over 3 seeds)
    scenarios = ["Easily-feasible\n(drift=0.03,\nutil=0.40)",
                 "Tight target\n(util=0.80,\ndrift=0.03, $\\beta=10$)",
                 "Slow drift\n(util=0.80,\ndrift=0.01, $\\beta=10$)",
                 "Hierarchical $\\beta$\n(util=0.80,\ndrift=0.03, $\\beta=50/10/5$)"]

    data = np.array([
        # Lag-SAC
        [0.983, 0.917, 0.642, 0.775],
        # Fixed-SAC
        [0.977, 1.000, 0.400, 1.000],
        # TCL-SAC
        [0.978, 0.927, 1.000, 0.600],
        # HPRS-SAC
        [0.958, 0.987, 0.197, 0.510],
    ])

    agent_labels = [LABELS[a] for a in AGENTS]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # annotate cells
    for i in range(len(AGENTS)):
        for j in range(len(scenarios)):
            val = data[i, j]
            text_color = "black" if 0.25 < val < 0.80 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=7)
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels(agent_labels, fontsize=9)

    # color bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("CSR$_{c_1}$ (utilization)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Utilization satisfaction rate across scenarios and agents",
                 fontsize=10, pad=8)

    # highlight TCL row
    for j in range(len(scenarios)):
        ax.add_patch(plt.Rectangle((j - 0.5, 1.5), 1, 1,
                                   fill=False, edgecolor=COLORS["tcl"],
                                   linewidth=1.8))

    fig.tight_layout()
    out = f"{OUT}/scenario_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[✓] {out}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating paper figures…")
    fig1_dual_collapse()
    fig2_b1_csr_curves()
    fig3_loss_budget_inversion()
    fig4_scenario_heatmap()
    print("Done — all figures in", OUT)

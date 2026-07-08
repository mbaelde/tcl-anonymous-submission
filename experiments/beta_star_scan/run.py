"""Beta*-scan: Proposition 4 validation on AdCraft A1 (loss-budget regime).

Sweeps the TCL shaped gain beta in {1, 2, 5, 7, 10, 15, 20} on the A1
environment (target_util=0.80, drift=0.03). Below the critical gain beta* the
shaped TCL reward correctly incentivises constraint satisfaction; above it the
reward inverts and CSR_c0 collapses (Proposition 4).

Usage:
    py -3.14 -m uv run python -m experiments.beta_star_scan.run \\
        --config experiments/beta_star_scan/config.yaml

    # To just regenerate figures from existing results:
    py -3.14 -m uv run python -m experiments.beta_star_scan.run \\
        --config experiments/beta_star_scan/config.yaml --analyze-only
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym
import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents import sac_tcl  # noqa: E402
from tcl.envs.adcraft_laplacian import MultiConstraintAdCraftLaplacian  # noqa: E402
from tcl.envs.adcraft_multiconstraint import MultiConstraintAdCraft  # noqa: E402


def make_env_factory(env_cfg: dict):
    env_kind = str(env_cfg.get("env_kind", "legacy"))

    def factory(args) -> gym.Env:  # type: ignore[no-untyped-def]
        common = dict(
            num_keywords=int(env_cfg["num_keywords"]),
            budget=float(env_cfg["budget"]),
            bid_max=float(env_cfg["bid_max"]),
            max_days=int(env_cfg["max_days"]),
            target_utilization=float(env_cfg["target_utilization"]),
            target_ctr=float(env_cfg["target_ctr"]),
            target_margin=float(env_cfg["target_margin"]),
            margin_formula=str(env_cfg.get("margin_formula", "cost_markup")),
        )
        if "updater_params" in env_cfg:
            common["updater_params"] = list(env_cfg["updater_params"])
        elif "drift_rate" in env_cfg:
            dr = float(env_cfg["drift_rate"])
            common["updater_params"] = [["vol", dr], ["ctr", dr], ["cvr", dr]]
        if env_kind == "laplacian":
            return MultiConstraintAdCraftLaplacian(**common)
        return MultiConstraintAdCraft(**common)

    return factory


def beta_label(b: float) -> str:
    return f"beta={b:.0f}" if b == int(b) else f"beta={b}"


def build_args(
    sac_cfg: dict,
    tcl_base: dict,
    beta: float,
    seed: int,
    log_dir: Path,
) -> sac_tcl.Args:
    b_str = f"{beta},{beta},{beta}"
    return sac_tcl.Args(
        exp_name=f"beta_star_{beta:.0f}",
        seed=seed,
        torch_deterministic=True,
        cuda=bool(sac_cfg.get("cuda", False)),
        log_dir=str(log_dir),
        total_timesteps=int(sac_cfg["total_timesteps"]),
        buffer_size=int(sac_cfg["buffer_size"]),
        batch_size=int(sac_cfg["batch_size"]),
        learning_starts=int(sac_cfg["learning_starts"]),
        gamma=float(sac_cfg["gamma"]),
        tau=float(sac_cfg["tau"]),
        policy_lr=float(sac_cfg["policy_lr"]),
        q_lr=float(sac_cfg["q_lr"]),
        policy_frequency=int(sac_cfg["policy_frequency"]),
        target_network_frequency=int(sac_cfg["target_network_frequency"]),
        autotune_alpha=bool(sac_cfg["autotune_alpha"]),
        alpha_init=float(sac_cfg["alpha_init"]),
        thresholds=str(tcl_base.get("thresholds", "0.0,0.0,0.0")),
        betas_init=b_str,
        betas_final=str(tcl_base.get("betas_final", "")),
        beta_schedule=str(tcl_base.get("beta_schedule", "linear")),
        beta_anneal_steps=int(tcl_base.get("beta_anneal_steps", 0)),
        reward_shift=float(tcl_base.get("reward_shift", 0.0)),
    )


def run_cell(
    beta: float,
    cfg: dict,
    seed: int,
    cell_dir: Path,
) -> dict[str, float]:
    args = build_args(
        sac_cfg=cfg["sac"],
        tcl_base=cfg["tcl_base"],
        beta=beta,
        seed=seed,
        log_dir=cell_dir / "tb",
    )
    env_factory = make_env_factory(cfg["env"])
    result = sac_tcl.train(args, env_factory=env_factory)

    with (cell_dir / "result.txt").open("w", encoding="utf-8") as f:
        f.write(f"beta: {beta}\n")
        f.write(f"seed: {seed}\n")
        for k, v in result.items():
            f.write(f"{k}: {v}\n")
    return result


def _worker_run_cell(
    payload: tuple[float, dict, int, str],
) -> tuple[float, int, str | None]:
    import torch

    torch.set_num_threads(1)
    beta, cfg, seed, cell_dir_str = payload
    try:
        run_cell(beta, cfg, seed=seed, cell_dir=Path(cell_dir_str))
    except Exception as e:
        return beta, seed, f"{type(e).__name__}: {e}"
    return beta, seed, None


def build_jobs(cfg: dict, skip_existing: bool = True) -> list[tuple]:
    """Enumerate ``(worker, payload)`` pairs for every pending cell.

    Mirrors ``main``'s ``betas x seeds`` grid so ``experiments.run_all_flat`` can
    merge this experiment's cells into one global pool.
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    betas = [float(b) for b in cfg["beta_sweep"]]
    seeds = [int(s) for s in cfg["seeds"]]
    jobs: list[tuple] = []
    for beta in betas:
        for seed in seeds:
            cell_dir = output_dir / beta_label(beta) / f"seed={seed}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            if skip_existing and (cell_dir / "result.txt").exists():
                continue
            jobs.append((_worker_run_cell, (beta, cfg, seed, str(cell_dir))))
    return jobs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def parse_result(path: Path) -> dict:
    result: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip()] = v.strip()
    return result


def load_tb_scalars(tb_dir: Path, tag: str) -> tuple[list[int], list[float]]:
    """Load scalar values from a TensorBoard event file. Returns (steps, values)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        event_files = list(tb_dir.glob("**/*.tfevents.*"))
        if not event_files:
            return [], []
        ea = EventAccumulator(str(event_files[0].parent))
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            return [], []
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except Exception:
        return [], []


def compute_csr_from_tb(tb_dir: Path, k: int, steady_frac: float = 0.2) -> float | None:
    """Compute CSR_k = fraction of steady-state episodes where cost_k <= 0."""
    _, values = load_tb_scalars(tb_dir, f"rollout/episode_cost_{k}")
    if not values:
        return None
    n = len(values)
    cutoff = int(n * (1.0 - steady_frac))
    steady = values[cutoff:]
    if not steady:
        return None
    return float(sum(1 for v in steady if v <= 0) / len(steady))


def compute_ep_return_from_tb(tb_dir: Path, steady_frac: float = 0.2) -> float | None:
    """Compute mean steady-state episode return."""
    _, values = load_tb_scalars(tb_dir, "rollout/episode_return")
    if not values:
        return None
    n = len(values)
    cutoff = int(n * (1.0 - steady_frac))
    steady = values[cutoff:]
    return float(sum(steady) / len(steady)) if steady else None


def collect_results(output_dir: Path, betas: list[float], seeds: list[int]) -> list[dict]:
    rows: list[dict] = []
    for beta in betas:
        for seed in seeds:
            cell_dir = output_dir / beta_label(beta) / f"seed={seed}"
            result_file = cell_dir / "result.txt"
            if not result_file.exists():
                continue
            row = {"beta": beta, "seed": seed}
            # Read log_path from result.txt to locate TensorBoard directory
            r = parse_result(result_file)
            log_path = r.get("log_path", "")
            if log_path:
                tb_dir = Path(log_path)
                if not tb_dir.is_absolute():
                    tb_dir = (Path(__file__).resolve().parents[2] / tb_dir)
            else:
                tb_dir = cell_dir / "tb"
            # Compute CSR and return from TensorBoard
            for k in range(3):
                csr = compute_csr_from_tb(tb_dir, k)
                if csr is not None:
                    row[f"csr_c{k}"] = csr
            ep_ret = compute_ep_return_from_tb(tb_dir)
            if ep_ret is not None:
                row["ep_return"] = ep_ret
            rows.append(row)
    return rows


def make_analysis(rows: list[dict], betas: list[float], output_dir: Path) -> None:
    import math

    import numpy as np

    # Group by beta
    stats: dict[float, dict] = {}
    for beta in betas:
        sub = [r for r in rows if r["beta"] == beta]
        csr_c0 = [r["csr_c0"] for r in sub if "csr_c0" in r]
        if not csr_c0:
            continue
        stats[beta] = {
            "mean": float(np.mean(csr_c0)),
            "std": float(np.std(csr_c0)),
            "n": len(csr_c0),
            "values": csr_c0,
        }

    # Write summary CSV
    import csv

    csv_path = output_dir / "beta_star_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["beta", "csr_c0_mean", "csr_c0_std", "n_seeds"])
        for beta in sorted(stats):
            s = stats[beta]
            w.writerow([beta, f"{s['mean']:.4f}", f"{s['std']:.4f}", s["n"]])

    # Console table
    print("\n=== Beta*-scan: CSR_c0 results ===")
    print(f"{'beta':>8}  {'CSR_c0 mean':>12}  {'std':>8}  {'seeds':>6}")
    for beta in sorted(stats):
        s = stats[beta]
        print(f"{beta:>8.1f}  {s['mean']:>12.4f}  {s['std']:>8.4f}  {s['n']:>6}")

    # Find empirical beta* as the beta where CSR_c0 first drops below 0.5
    beta_star_empirical = None
    sorted_betas = sorted(stats.keys())
    for i, beta in enumerate(sorted_betas):
        if stats[beta]["mean"] < 0.5:
            beta_star_empirical = beta
            break
    if beta_star_empirical is not None:
        print(f"\nEmpirical beta* (first beta with CSR_c0 < 0.5): {beta_star_empirical}")
    else:
        print("\nNo failure detected in the scanned range.")

    # Generate figure
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

        fig, ax = plt.subplots(figsize=(5.25, 3.8))
        betas_plot = sorted(stats.keys())
        means = [stats[b]["mean"] for b in betas_plot]
        stds = [stats[b]["std"] for b in betas_plot]
        ax.errorbar(
            betas_plot,
            means,
            yerr=stds,
            fmt="o-",
            capsize=4,
            color="tab:blue",
            label=r"TCL shaped (mean $\pm$ std, 3 seeds)",
        )
        # Individual seeds
        for b in betas_plot:
            for v in stats[b]["values"]:
                ax.scatter(b, v, color="tab:blue", alpha=0.3, s=20)
        ax.axhline(0.5, ls="--", color="gray", lw=1, label=r"CSR$_{c_1}$ = 0.5")
        if beta_star_empirical is not None:
            ax.axvline(
                beta_star_empirical,
                ls=":",
                color="red",
                lw=1.5,
                label=rf"empirical $\beta^* \approx {beta_star_empirical}$",
            )
        ax.set_xlabel(r"$\beta$ (fixed gain)")
        ax.set_ylabel(r"CSR$_{c_1}$ (primary constraint satisfaction rate)")
        ax.set_title(
            r"Proposition 4 validation: CSR$_{c_1}$ vs $\beta$" + "\n"
            r"A1 env (target\_util=0.80, drift=0.03), TCL shaped, no reward shift",
        )
        ax.set_ylim(-0.05, 1.10)
        ax.legend()
        ax.grid(True, ls=":")

        fig_path = output_dir / "beta_star_scan.pdf"
        plt.savefig(str(fig_path), bbox_inches="tight")
        plt.close(fig)
        print(f"Figure: {fig_path}")
    except ImportError:
        print("matplotlib not available - skipping figure")

    print(f"CSV: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only-beta", type=float, default=None)
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Cells to run concurrently. Default=1 (sequential).",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip training; just collect results and regenerate figures.",
    )
    cli = parser.parse_args()

    with cli.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    betas: list[float] = [float(b) for b in cfg["beta_sweep"]]
    seeds: list[int] = [int(s) for s in cfg["seeds"]]

    if cli.only_beta is not None:
        betas = [b for b in betas if b == cli.only_beta]
    if cli.only_seed is not None:
        seeds = [s for s in seeds if s == cli.only_seed]

    if not cli.analyze_only:
        pending: list[tuple[float, int, Path]] = []
        for beta in betas:
            for seed in seeds:
                cell_dir = output_dir / beta_label(beta) / f"seed={seed}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                if cli.skip_existing and (cell_dir / "result.txt").exists():
                    continue
                pending.append((beta, seed, cell_dir))

        print(
            f"Beta*-scan: {len(betas)} betas x {len(seeds)} seeds = "
            f"{len(betas)*len(seeds)} total, {len(pending)} to run"
        )

        if cli.parallel <= 1:
            for beta, seed, cell_dir in tqdm(pending, desc="beta_star_scan"):
                run_cell(beta, cfg, seed=seed, cell_dir=cell_dir)
        else:
            payloads = [(b, cfg, s, str(d)) for b, s, d in pending]
            failures: list[tuple[float, int, str]] = []
            with ProcessPoolExecutor(max_workers=cli.parallel) as ex:
                futures = [ex.submit(_worker_run_cell, p) for p in payloads]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="beta_star_scan"):
                    beta, seed, err = fut.result()
                    if err is not None:
                        failures.append((beta, seed, err))
                        tqdm.write(f"[FAIL] beta={beta} seed={seed}: {err}")
            if failures:
                raise SystemExit(f"{len(failures)} cell(s) failed")

    # Analysis
    all_betas = [float(b) for b in cfg["beta_sweep"]]
    all_seeds = [int(s) for s in cfg["seeds"]]
    rows = collect_results(output_dir, all_betas, all_seeds)
    make_analysis(rows, all_betas, output_dir)


if __name__ == "__main__":
    main()

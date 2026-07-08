"""Prop. 5 validation: asymptotic bridge TCL-SAC(A) vs TCL-SAC(B) as beta -> inf.

Sweeps beta in {10, 30, 100, 300} on the B1 environment (drift=0.01, easily
feasible). Trains both TCL-SAC(B) (shaped) and TCL-SAC(A) (standalone, rb_mode=ignore)
at each beta and compares episode return and CSR_c0. Prop. 5 predicts the gap
|return_A - return_B| -> 0 as beta -> inf on Pi_det.

Usage:
    PYTHONUTF8=1 py -3.14 -m uv run python -m experiments.prop5_validation.run \
        --config experiments/prop5_validation/config.yaml --parallel 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym
import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents import sac_tcl, sac_tcl_standalone  # noqa: E402
from tcl.envs.adcraft_laplacian import MultiConstraintAdCraftLaplacian  # noqa: E402
from tcl.envs.adcraft_multiconstraint import MultiConstraintAdCraft  # noqa: E402


AGENTS = ["shaped_B", "standalone_A"]


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
        if "drift_rate" in env_cfg:
            dr = float(env_cfg["drift_rate"])
            common["updater_params"] = [["vol", dr], ["ctr", dr], ["cvr", dr]]
        if env_kind == "laplacian":
            return MultiConstraintAdCraftLaplacian(**common)
        return MultiConstraintAdCraft(**common)

    return factory


def cell_label(agent: str, beta: float, seed: int) -> str:
    b = f"{beta:.0f}" if beta == int(beta) else str(beta)
    return f"{agent}/beta={b}/seed={seed}"


def build_args_shaped(
    sac_cfg: dict, tcl_base: dict, beta: float, seed: int, log_dir: Path
) -> sac_tcl.Args:
    b_str = f"{beta},{beta},{beta}"
    return sac_tcl.Args(
        exp_name=f"prop5_B_beta{beta:.0f}",
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


def build_args_standalone(
    sac_cfg: dict, tcl_base: dict, beta: float, seed: int, log_dir: Path
) -> sac_tcl_standalone.Args:
    b_str = f"{beta},{beta},{beta}"
    return sac_tcl_standalone.Args(
        exp_name=f"prop5_A_beta{beta:.0f}",
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
        rb_mode="ignore",
        rb_weight=1.0,
    )


def run_cell(agent: str, beta: float, cfg: dict, seed: int, cell_dir: Path) -> None:
    env_factory = make_env_factory(cfg["env"])
    tb_dir = cell_dir / "tb"
    if agent == "shaped_B":
        args = build_args_shaped(cfg["sac"], cfg["tcl_base"], beta, seed, tb_dir)
        result = sac_tcl.train(args, env_factory=env_factory)
    else:
        args = build_args_standalone(cfg["sac"], cfg["tcl_base"], beta, seed, tb_dir)
        result = sac_tcl_standalone.train(args, env_factory=env_factory)

    with (cell_dir / "result.txt").open("w", encoding="utf-8") as f:
        f.write(f"agent: {agent}\nbeta: {beta}\nseed: {seed}\n")
        for k, v in result.items():
            f.write(f"{k}: {v}\n")


def _worker(payload: tuple) -> tuple[str, float, int, str | None]:
    import torch

    torch.set_num_threads(1)
    agent, beta, cfg, seed, cell_dir_str = payload
    try:
        run_cell(agent, beta, cfg, seed, Path(cell_dir_str))
    except Exception as e:
        return agent, beta, seed, f"{type(e).__name__}: {e}"
    return agent, beta, seed, None


def build_jobs(cfg: dict, skip_existing: bool = True) -> list[tuple]:
    """Enumerate ``(worker, payload)`` pairs for every pending cell.

    Mirrors ``main``'s ``AGENTS x betas x seeds`` grid so ``experiments.run_all_flat``
    can merge this experiment's cells into one global pool.
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    betas = [float(b) for b in cfg["beta_sweep"]]
    seeds = [int(s) for s in cfg["seeds"]]
    jobs: list[tuple] = []
    for agent in AGENTS:
        for beta in betas:
            for seed in seeds:
                b_str = f"{beta:.0f}" if beta == int(beta) else str(beta)
                cell_dir = output_dir / agent / f"beta={b_str}" / f"seed={seed}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                if skip_existing and (cell_dir / "result.txt").exists():
                    continue
                jobs.append((_worker, (agent, beta, cfg, seed, str(cell_dir))))
    return jobs


# ---------------------------------------------------------------------------
# Analysis helpers (reuse beta_star_scan patterns)
# ---------------------------------------------------------------------------


def load_tb_scalars(tb_dir: Path, tag: str) -> tuple[list[int], list[float]]:
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
        return [e.step for e in events], [e.value for e in events]
    except Exception:
        return [], []


def steady_mean(tb_dir: Path, tag: str, steady_frac: float = 0.2) -> float | None:
    _, values = load_tb_scalars(tb_dir, tag)
    if not values:
        return None
    cutoff = int(len(values) * (1.0 - steady_frac))
    tail = values[cutoff:]
    return float(sum(tail) / len(tail)) if tail else None


def compute_csr(tb_dir: Path, k: int, steady_frac: float = 0.2) -> float | None:
    _, values = load_tb_scalars(tb_dir, f"rollout/episode_cost_{k}")
    if not values:
        return None
    cutoff = int(len(values) * (1.0 - steady_frac))
    tail = values[cutoff:]
    return float(sum(1 for v in tail if v <= 0) / len(tail)) if tail else None


def collect_results(
    output_dir: Path, betas: list[float], seeds: list[int]
) -> list[dict]:
    rows = []
    for agent in AGENTS:
        for beta in betas:
            for seed in seeds:
                b_str = f"{beta:.0f}" if beta == int(beta) else str(beta)
                cell_dir = output_dir / agent / f"beta={b_str}" / f"seed={seed}"
                if not (cell_dir / "result.txt").exists():
                    continue
                tb_dir = cell_dir / "tb"
                row: dict = {"agent": agent, "beta": beta, "seed": seed}
                ep_ret = steady_mean(tb_dir, "rollout/episode_return")
                if ep_ret is not None:
                    row["ep_return"] = ep_ret
                csr = compute_csr(tb_dir, 0)
                if csr is not None:
                    row["csr_c0"] = csr
                rows.append(row)
    return rows


def make_analysis(rows: list[dict], betas: list[float], output_dir: Path) -> None:
    import numpy as np

    # Aggregate by (agent, beta)
    stats: dict[tuple[str, float], dict] = {}
    for agent in AGENTS:
        for beta in betas:
            sub = [r for r in rows if r["agent"] == agent and r["beta"] == beta]
            rets = [r["ep_return"] for r in sub if "ep_return" in r]
            csrs = [r["csr_c0"] for r in sub if "csr_c0" in r]
            if rets:
                stats[(agent, beta)] = {
                    "ret_mean": float(np.mean(rets)),
                    "ret_std": float(np.std(rets)),
                    "csr_mean": float(np.mean(csrs)) if csrs else float("nan"),
                    "n": len(rets),
                }

    # CSV
    csv_path = output_dir / "prop5.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent", "beta", "ep_return_mean", "ep_return_std", "csr_c0_mean", "n"])
        for (agent, beta) in sorted(stats):
            s = stats[(agent, beta)]
            w.writerow([agent, beta, f"{s['ret_mean']:.1f}", f"{s['ret_std']:.1f}",
                        f"{s['csr_mean']:.4f}", s["n"]])

    # Console
    print("\n=== Prop. 5 validation: episode return and CSR_c0 ===")
    print(f"{'agent':>15} {'beta':>6} {'return':>10} {'std':>8} {'csr_c0':>8} {'n':>4}")
    for (agent, beta) in sorted(stats):
        s = stats[(agent, beta)]
        print(f"{agent:>15} {beta:>6.0f} {s['ret_mean']:>10.1f} "
              f"{s['ret_std']:>8.1f} {s['csr_mean']:>8.4f} {s['n']:>4}")

    # Gap analysis
    print("\n=== Return gap |A - B| vs beta ===")
    print(f"{'beta':>6} {'gap':>10}")
    for beta in sorted(betas):
        sA = stats.get(("standalone_A", beta))
        sB = stats.get(("shaped_B", beta))
        if sA and sB:
            gap = abs(sA["ret_mean"] - sB["ret_mean"])
            print(f"{beta:>6.0f} {gap:>10.1f}")

    # Figure
    try:
        import matplotlib
        matplotlib.use("pdf")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "text.usetex":      True,
            "font.family":      "serif",
            "font.size":        10,
            "axes.titlesize":   10,
            "axes.labelsize":   9,
            "xtick.labelsize":  8,
            "ytick.labelsize":  8,
            "legend.fontsize":  8,
            "figure.dpi":       150,
            "savefig.bbox":     "tight",
            "savefig.pad_inches": 0.05,
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.85, 4.0))
        colors = {"shaped_B": "tab:blue", "standalone_A": "tab:orange"}
        labels = {"shaped_B": "TCL-SAC(B) shaped", "standalone_A": "TCL-SAC(A) standalone"}

        for agent in AGENTS:
            betas_plot = sorted(b for (a, b) in stats if a == agent)
            means = [stats[(agent, b)]["ret_mean"] for b in betas_plot]
            stds = [stats[(agent, b)]["ret_std"] for b in betas_plot]
            csrs = [stats[(agent, b)]["csr_mean"] for b in betas_plot]
            ax1.errorbar(betas_plot, means, yerr=stds, fmt="o-", capsize=4,
                         color=colors[agent], label=labels[agent])
            ax2.plot(betas_plot, csrs, "o-", color=colors[agent], label=labels[agent])

        ax1.set_xlabel(r"$\beta$")
        ax1.set_ylabel("Episode return (steady-state mean)")
        ax1.set_title(r"Prop. 5: return convergence (A) vs (B) as $\beta \to \infty$")
        ax1.legend()
        ax1.grid(True, ls=":")
        ax1.set_xscale("log")

        ax2.set_xlabel(r"$\beta$")
        ax2.set_ylabel(r"CSR$_{c_1}$")
        ax2.set_title(r"Prop. 5: CSR$_{c_1}$ convergence")
        ax2.set_ylim(-0.05, 1.10)
        ax2.legend()
        ax2.grid(True, ls=":")
        ax2.set_xscale("log")

        fig.tight_layout()
        fig_path = output_dir / "prop5.pdf"
        plt.savefig(str(fig_path), bbox_inches="tight")
        plt.close(fig)
        print(f"\nFigure: {fig_path}")
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
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--analyze-only", action="store_true")
    cli = parser.parse_args()

    with cli.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    betas: list[float] = [float(b) for b in cfg["beta_sweep"]]
    seeds: list[int] = [int(s) for s in cfg["seeds"]]

    if not cli.analyze_only:
        pending = []
        for agent in AGENTS:
            for beta in betas:
                for seed in seeds:
                    b_str = f"{beta:.0f}" if beta == int(beta) else str(beta)
                    cell_dir = output_dir / agent / f"beta={b_str}" / f"seed={seed}"
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    if cli.skip_existing and (cell_dir / "result.txt").exists():
                        continue
                    pending.append((agent, beta, seed, cell_dir))

        total = len(AGENTS) * len(betas) * len(seeds)
        print(f"Prop5 validation: {len(AGENTS)} agents x {len(betas)} betas x "
              f"{len(seeds)} seeds = {total} total, {len(pending)} to run")

        if cli.parallel <= 1:
            for agent, beta, seed, cell_dir in tqdm(pending, desc="prop5"):
                run_cell(agent, beta, cfg, seed=seed, cell_dir=cell_dir)
        else:
            payloads = [(a, b, cfg, s, str(d)) for a, b, s, d in pending]
            failures = []
            with ProcessPoolExecutor(max_workers=cli.parallel) as ex:
                futures = [ex.submit(_worker, p) for p in payloads]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="prop5"):
                    agent, beta, seed, err = fut.result()
                    if err:
                        failures.append((agent, beta, seed, err))
                        tqdm.write(f"[FAIL] {agent} beta={beta} seed={seed}: {err}")
            if failures:
                raise SystemExit(f"{len(failures)} cell(s) failed")

    rows = collect_results(output_dir, betas, seeds)
    make_analysis(rows, betas, output_dir)


if __name__ == "__main__":
    main()

"""Tau-threshold sensitivity: TCL-SAC(B) vs Lag-SAC robustness to tau_util miscalibration.

Sweeps target_utilization in {0.65, 0.70, 0.75, 0.80, 0.85, 0.90} (nominal=0.80)
on the A1 loss-budget env (drift=0.03). Compares TCL-SAC(B) and Lag-SAC(multi).
Hypothesis: TCL gate is local in (s,a) => more robust than the Lagrangian integrator
which drifts without bound under persistent violations.

Usage:
    PYTHONUTF8=1 py -3.14 -m uv run python -m experiments.tau_sensitivity.run \
        --config experiments/tau_sensitivity/config.yaml --parallel 6
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

from agents import sac_lagrangian_multi, sac_tcl  # noqa: E402
from tcl.envs.adcraft_laplacian import MultiConstraintAdCraftLaplacian  # noqa: E402
from tcl.envs.adcraft_multiconstraint import MultiConstraintAdCraft  # noqa: E402


def make_env_factory(env_cfg: dict, tau_util: float):
    env_kind = str(env_cfg.get("env_kind", "legacy"))

    def factory(args) -> gym.Env:  # type: ignore[no-untyped-def]
        common = dict(
            num_keywords=int(env_cfg["num_keywords"]),
            budget=float(env_cfg["budget"]),
            bid_max=float(env_cfg["bid_max"]),
            max_days=int(env_cfg["max_days"]),
            target_utilization=tau_util,
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


def tau_label(tau: float) -> str:
    return f"tau={tau:.2f}"


def build_tcl_args(
    sac_cfg: dict, tcl_cfg: dict, tau_util: float, seed: int, log_dir: Path
) -> sac_tcl.Args:
    return sac_tcl.Args(
        exp_name=f"tau_tcl_{tau_util:.2f}",
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
        thresholds=str(tcl_cfg.get("thresholds", "0.0,0.0,0.0")),
        betas_init=str(tcl_cfg.get("betas_init", "10,10,10")),
        betas_final=str(tcl_cfg.get("betas_final", "")),
        beta_schedule=str(tcl_cfg.get("beta_schedule", "linear")),
        beta_anneal_steps=int(tcl_cfg.get("beta_anneal_steps", 0)),
        reward_shift=float(tcl_cfg.get("reward_shift", 0.0)),
    )


def build_lag_args(
    sac_cfg: dict, lag_cfg: dict, tau_util: float, seed: int, log_dir: Path
) -> sac_lagrangian_multi.Args:
    return sac_lagrangian_multi.Args(
        exp_name=f"tau_lag_{tau_util:.2f}",
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
        lambda_init=float(lag_cfg.get("lambda_init", 0.0)),
        lambda_lr=float(lag_cfg.get("lambda_lr", 1e-3)),
    )


def run_cell(
    agent: str, tau_util: float, cfg: dict, seed: int, cell_dir: Path
) -> None:
    env_factory = make_env_factory(cfg["env_base"], tau_util)
    tb_dir = cell_dir / "tb"
    if agent == "tcl":
        args = build_tcl_args(cfg["sac"], cfg["tcl_cfg"], tau_util, seed, tb_dir)
        result = sac_tcl.train(args, env_factory=env_factory)
    else:
        args = build_lag_args(cfg["sac"], cfg["lag_cfg"], tau_util, seed, tb_dir)
        result = sac_lagrangian_multi.train(args, env_factory=env_factory)

    with (cell_dir / "result.txt").open("w", encoding="utf-8") as f:
        f.write(f"agent: {agent}\ntau_util: {tau_util}\nseed: {seed}\n")
        for k, v in result.items():
            f.write(f"{k}: {v}\n")


def _worker(payload: tuple) -> tuple[str, float, int, str | None]:
    import torch

    torch.set_num_threads(1)
    agent, tau_util, cfg, seed, cell_dir_str = payload
    try:
        run_cell(agent, tau_util, cfg, seed, Path(cell_dir_str))
    except Exception as e:
        return agent, tau_util, seed, f"{type(e).__name__}: {e}"
    return agent, tau_util, seed, None


def build_jobs(cfg: dict, skip_existing: bool = True) -> list[tuple]:
    """Enumerate ``(worker, payload)`` pairs for every pending cell.

    Mirrors ``main``'s ``agents x taus x seeds`` grid so ``experiments.run_all_flat``
    can merge this experiment's cells into one global pool.
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tau_sweep = [float(t) for t in cfg["tau_util_sweep"]]
    agents = list(cfg["agents"])
    seeds = [int(s) for s in cfg["seeds"]]
    jobs: list[tuple] = []
    for agent in agents:
        for tau in tau_sweep:
            for seed in seeds:
                cell_dir = output_dir / agent / tau_label(tau) / f"seed={seed}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                if skip_existing and (cell_dir / "result.txt").exists():
                    continue
                jobs.append((_worker, (agent, tau, cfg, seed, str(cell_dir))))
    return jobs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def load_tb_scalars(tb_dir: Path, tag: str) -> list[float]:
    """Parse TF event files for a single tag without full protobuf reload (~100x faster).

    Supports both simple_value (wire5) and tensor/float_val (wire2) encoding.
    Falls back to EventFileLoader if the fast path fails.
    """
    import struct as _struct

    def _fast_read(path: Path, tag: str) -> list[float]:
        tag_b = tag.encode()
        assert len(tag_b) < 128, "tag too long for single-byte varint"
        # Proto encoding: Value.tag = field 1, wire LEN → 0x0a + varint(len) + bytes
        marker = bytes([0x0A, len(tag_b)]) + tag_b
        values: list[float] = []
        with path.open("rb") as f:
            while True:
                hdr = f.read(12)  # uint64 dlen + uint32 crc_len
                if len(hdr) < 12:
                    break
                (dlen,) = _struct.unpack_from("<Q", hdr)
                data = f.read(dlen)
                f.read(4)  # crc_data
                if len(data) < dlen:
                    break
                if marker not in data:
                    continue
                idx = data.find(marker)
                rest = data[idx + len(marker) :]
                if not rest:
                    continue
                b = rest[0]
                if b == 0x15 and len(rest) >= 5:
                    # simple_value: field 2, wire5 (fixed32)
                    values.append(_struct.unpack_from("<f", rest, 1)[0])
                elif b == 0x42:
                    # tensor: field 8, wire2 — look for float_val packed: 0x22 0x04 + 4 bytes
                    fi = rest.find(b"\x22\x04", 1, 120)
                    if fi != -1 and len(rest) >= fi + 6:
                        values.append(_struct.unpack_from("<f", rest, fi + 2)[0])
        return values

    event_files = sorted(tb_dir.glob("**/*.tfevents.*"))
    if not event_files:
        return []
    try:
        values: list[float] = []
        for ef in event_files:
            values.extend(_fast_read(ef, tag))
        return values
    except Exception:
        try:
            from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

            values = []
            for ef in event_files:
                for event in EventFileLoader(str(ef)).Load():
                    if event.HasField("summary"):
                        for v in event.summary.value:
                            if v.tag != tag:
                                continue
                            if v.HasField("tensor") and v.tensor.float_val:
                                values.append(v.tensor.float_val[0])
            return values
        except Exception:
            return []


def steady_mean(tb_dir: Path, tag: str, steady_frac: float = 0.2) -> float | None:
    values = load_tb_scalars(tb_dir, tag)
    if not values:
        return None
    cutoff = int(len(values) * (1.0 - steady_frac))
    tail = values[cutoff:]
    return float(sum(tail) / len(tail)) if tail else None


def compute_csr(tb_dir: Path, k: int, steady_frac: float = 0.2) -> float | None:
    # Some agents log episode_cost_0, others episode_cost_k0 — try both.
    values = load_tb_scalars(tb_dir, f"rollout/episode_cost_{k}")
    if not values:
        values = load_tb_scalars(tb_dir, f"rollout/episode_cost_k{k}")
    if not values:
        return None
    cutoff = int(len(values) * (1.0 - steady_frac))
    tail = values[cutoff:]
    return float(sum(1 for v in tail if v <= 0) / len(tail)) if tail else None


def collect_results(
    output_dir: Path, tau_sweep: list[float], agents: list[str], seeds: list[int]
) -> list[dict]:
    rows = []
    for agent in agents:
        for tau in tau_sweep:
            for seed in seeds:
                cell_dir = output_dir / agent / tau_label(tau) / f"seed={seed}"
                if not (cell_dir / "result.txt").exists():
                    continue
                tb_dir = cell_dir / "tb"
                row: dict = {"agent": agent, "tau_util": tau, "seed": seed}
                ep_ret = steady_mean(tb_dir, "rollout/episode_return")
                if ep_ret is not None:
                    row["ep_return"] = ep_ret
                csr = compute_csr(tb_dir, 0)
                if csr is not None:
                    row["csr_c0"] = csr
                rows.append(row)
    return rows


def make_analysis(
    rows: list[dict], tau_sweep: list[float], agents: list[str], output_dir: Path
) -> None:
    import numpy as np

    stats: dict[tuple[str, float], dict] = {}
    for agent in agents:
        for tau in tau_sweep:
            sub = [r for r in rows if r["agent"] == agent and r["tau_util"] == tau]
            csrs = [r["csr_c0"] for r in sub if "csr_c0" in r]
            rets = [r["ep_return"] for r in sub if "ep_return" in r]
            if csrs:
                stats[(agent, tau)] = {
                    "csr_mean": float(np.mean(csrs)),
                    "csr_std": float(np.std(csrs)),
                    "ret_mean": float(np.mean(rets)) if rets else float("nan"),
                    "n": len(csrs),
                }

    # CSV
    csv_path = output_dir / "tau_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent", "tau_util", "csr_c0_mean", "csr_c0_std", "ep_return_mean", "n"])
        for (agent, tau) in sorted(stats):
            s = stats[(agent, tau)]
            w.writerow([agent, tau, f"{s['csr_mean']:.4f}", f"{s['csr_std']:.4f}",
                        f"{s['ret_mean']:.1f}", s["n"]])

    # Console
    print("\n=== Tau sensitivity: CSR_c0 vs tau_util ===")
    for agent in agents:
        print(f"\n  Agent: {agent}")
        print(f"  {'tau_util':>10} {'csr_c0':>8} {'std':>7} {'n':>4}")
        for tau in sorted(tau_sweep):
            s = stats.get((agent, tau))
            if s:
                marker = " <-- nominal" if abs(tau - 0.80) < 1e-6 else ""
                print(f"  {tau:>10.2f} {s['csr_mean']:>8.4f} {s['csr_std']:>7.4f} "
                      f"{s['n']:>4}{marker}")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {"tcl": "tab:blue", "lagrangian": "tab:orange"}
        labels = {"tcl": "TCL-SAC(B)", "lagrangian": "Lag-SAC"}

        for agent in agents:
            taus_plot = sorted(tau for (a, _) in stats if a == agent
                               for tau in [_] if (a, tau) in stats)
            taus_plot = sorted(set(taus_plot))
            means = [stats[(agent, t)]["csr_mean"] for t in taus_plot]
            stds = [stats[(agent, t)]["csr_std"] for t in taus_plot]
            ax.errorbar(taus_plot, means, yerr=stds, fmt="o-", capsize=4,
                        color=colors.get(agent, "gray"), label=labels.get(agent, agent))

        ax.axvline(0.80, ls="--", color="gray", lw=1, label="nominal tau_util=0.80")
        ax.axhline(0.5, ls=":", color="lightgray", lw=1)
        ax.set_xlabel("target_utilization (tau_util)", fontsize=12)
        ax.set_ylabel("CSR_c0 (steady-state, last 20%)", fontsize=11)
        ax.set_title(
            "Threshold sensitivity: TCL-SAC(B) vs Lag-SAC\n"
            "A1 env (drift=0.03, loss-budget), beta=10, 3 seeds",
            fontsize=11,
        )
        ax.set_ylim(-0.05, 1.10)
        ax.legend(fontsize=10)
        ax.grid(True, ls=":")

        fig_path = output_dir / "tau_sensitivity.pdf"
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

    tau_sweep: list[float] = [float(t) for t in cfg["tau_util_sweep"]]
    agents: list[str] = list(cfg["agents"])
    seeds: list[int] = [int(s) for s in cfg["seeds"]]

    if not cli.analyze_only:
        pending = []
        for agent in agents:
            for tau in tau_sweep:
                for seed in seeds:
                    cell_dir = output_dir / agent / tau_label(tau) / f"seed={seed}"
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    if cli.skip_existing and (cell_dir / "result.txt").exists():
                        continue
                    pending.append((agent, tau, seed, cell_dir))

        total = len(agents) * len(tau_sweep) * len(seeds)
        print(f"Tau sensitivity: {len(agents)} agents x {len(tau_sweep)} taus x "
              f"{len(seeds)} seeds = {total} total, {len(pending)} to run")

        if cli.parallel <= 1:
            for agent, tau, seed, cell_dir in tqdm(pending, desc="tau_sensitivity"):
                run_cell(agent, tau, cfg, seed=seed, cell_dir=cell_dir)
        else:
            payloads = [(a, t, cfg, s, str(d)) for a, t, s, d in pending]
            failures = []
            with ProcessPoolExecutor(max_workers=cli.parallel) as ex:
                futures = [ex.submit(_worker, p) for p in payloads]
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc="tau_sensitivity"):
                    agent, tau, seed, err = fut.result()
                    if err:
                        failures.append((agent, tau, seed, err))
                        tqdm.write(f"[FAIL] {agent} tau={tau} seed={seed}: {err}")
            if failures:
                raise SystemExit(f"{len(failures)} cell(s) failed")

    rows = collect_results(output_dir, tau_sweep, agents, seeds)
    make_analysis(rows, tau_sweep, agents, output_dir)


if __name__ == "__main__":
    main()

"""PID-Lagrangian benchmark: SAC+PID-Lag (Stooke 2020) vs TCL-SAC(B) vs Lag-SAC on A1.

Ref: Stooke, Achiam, Abbeel (2020). "Responsive Safety in RL by PID Lagrangian Methods."

Usage:
    PYTHONUTF8=1 py -3.14 -m uv run python -m experiments.pid_lagrangian_bench.run \
        --config experiments/pid_lagrangian_bench/config.yaml --parallel 9
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

from agents import sac_lagrangian_multi, sac_pid_lagrangian, sac_tcl  # noqa: E402
from tcl.envs.adcraft_laplacian import MultiConstraintAdCraftLaplacian  # noqa: E402
from tcl.envs.adcraft_multiconstraint import MultiConstraintAdCraft  # noqa: E402

N_COSTS = 3  # AdCraft K=3


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


def run_cell(agent: str, cfg: dict, seed: int, cell_dir: Path) -> None:
    env_factory = make_env_factory(cfg["env"])
    tb_dir = cell_dir / "tb"
    sac = cfg["sac"]

    common = dict(
        seed=seed, torch_deterministic=True,
        cuda=bool(sac.get("cuda", False)),
        log_dir=str(tb_dir),
        total_timesteps=int(sac["total_timesteps"]),
        buffer_size=int(sac["buffer_size"]),
        batch_size=int(sac["batch_size"]),
        learning_starts=int(sac["learning_starts"]),
        gamma=float(sac["gamma"]), tau=float(sac["tau"]),
        policy_lr=float(sac["policy_lr"]), q_lr=float(sac["q_lr"]),
        policy_frequency=int(sac["policy_frequency"]),
        target_network_frequency=int(sac["target_network_frequency"]),
        autotune_alpha=bool(sac["autotune_alpha"]),
        alpha_init=float(sac["alpha_init"]),
    )

    if agent == "tcl":
        tcl = cfg["tcl_cfg"]
        args = sac_tcl.Args(
            exp_name=f"pid_bench_tcl",
            thresholds=str(tcl.get("thresholds", "0.0,0.0,0.0")),
            betas_init=str(tcl.get("betas_init", "10,10,10")),
            betas_final=str(tcl.get("betas_final", "")),
            beta_schedule=str(tcl.get("beta_schedule", "linear")),
            beta_anneal_steps=int(tcl.get("beta_anneal_steps", 0)),
            reward_shift=float(tcl.get("reward_shift", 0.0)),
            **common,
        )
        result = sac_tcl.train(args, env_factory=env_factory)

    elif agent == "pid_lagrangian":
        pid = cfg["pid_cfg"]
        args = sac_pid_lagrangian.Args(
            exp_name="pid_bench_pid",
            lambda_init=float(pid.get("lambda_init", 0.0)),
            pid_kp=float(pid.get("pid_kp", 1e-3)),
            pid_ki=float(pid.get("pid_ki", 1e-4)),
            pid_kd=float(pid.get("pid_kd", 0.0)),
            **common,
        )
        result = sac_pid_lagrangian.train(args, env_factory=env_factory)

    else:  # lagrangian
        lag = cfg["lag_cfg"]
        args = sac_lagrangian_multi.Args(
            exp_name="pid_bench_lag",
            lambda_init=float(lag.get("lambda_init", 0.0)),
            lambda_lr=float(lag.get("lambda_lr", 1e-3)),
            **common,
        )
        result = sac_lagrangian_multi.train(args, env_factory=env_factory)

    with (cell_dir / "result.txt").open("w", encoding="utf-8") as f:
        f.write(f"agent: {agent}\nseed: {seed}\n")
        for k, v in result.items():
            f.write(f"{k}: {v}\n")


def _worker(payload: tuple) -> tuple[str, int, str | None]:
    import torch
    torch.set_num_threads(1)
    agent, cfg, seed, cell_dir_str = payload
    try:
        run_cell(agent, cfg, seed, Path(cell_dir_str))
    except Exception as e:
        return agent, seed, f"{type(e).__name__}: {e}"
    return agent, seed, None


def build_jobs(cfg: dict, skip_existing: bool = True) -> list[tuple]:
    """Enumerate ``(worker, payload)`` pairs for every pending cell.

    Mirrors ``main``'s ``agents x seeds`` grid so ``experiments.run_all_flat`` can
    merge this experiment's cells into one global pool.
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    agents = list(cfg["agents"])
    seeds = [int(s) for s in cfg["seeds"]]
    jobs: list[tuple] = []
    for agent in agents:
        for seed in seeds:
            cell_dir = output_dir / agent / f"seed={seed}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            if skip_existing and (cell_dir / "result.txt").exists():
                continue
            jobs.append((_worker, (agent, cfg, seed, str(cell_dir))))
    return jobs


def load_tb_values(tb_dir: Path, tag: str) -> list[float]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        files = list(tb_dir.glob("**/*.tfevents.*"))
        if not files:
            return []
        ea = EventAccumulator(str(files[0].parent))
        ea.Reload()
        # Try both tag formats (lagrangian_multi uses k{k}, others use _{k})
        for t in [tag, tag.replace("_cost_", "_cost_k").replace("/_", "/k")]:
            if t in ea.Tags().get("scalars", []):
                return [e.value for e in ea.Scalars(t)]
        return []
    except Exception:
        return []


def steady_stat(tb_dir: Path, tag: str, fn, steady_frac: float = 0.2):
    values = load_tb_values(tb_dir, tag)
    if not values:
        return None
    tail = values[int(len(values) * (1 - steady_frac)):]
    return fn(tail) if tail else None


def collect_results(output_dir: Path, agents: list[str], seeds: list[int]) -> list[dict]:
    rows = []
    for agent in agents:
        for seed in seeds:
            cell_dir = output_dir / agent / f"seed={seed}"
            if not (cell_dir / "result.txt").exists():
                continue
            tb_dir = cell_dir / "tb"
            row: dict = {"agent": agent, "seed": seed}
            ep_ret = steady_stat(tb_dir, "rollout/episode_return", lambda v: sum(v) / len(v))
            if ep_ret is not None:
                row["ep_return"] = ep_ret
            for k in range(N_COSTS):
                values = load_tb_values(tb_dir, f"rollout/episode_cost_{k}")
                if not values:
                    # fallback: lagrangian_multi tag format
                    values = load_tb_values(tb_dir, f"rollout/episode_cost_k{k}")
                if values:
                    tail = values[int(len(values) * 0.8):]
                    row[f"csr_c{k}"] = float(sum(1 for v in tail if v <= 0) / len(tail))
            rows.append(row)
    return rows


def make_analysis(rows: list[dict], agents: list[str], output_dir: Path) -> None:
    import numpy as np

    stats: dict[str, dict] = {}
    for agent in agents:
        sub = [r for r in rows if r["agent"] == agent]
        if not sub:
            continue
        rets = [r["ep_return"] for r in sub if "ep_return" in r]
        csr0 = [r["csr_c0"] for r in sub if "csr_c0" in r]
        stats[agent] = {
            "ret_mean": float(np.mean(rets)) if rets else float("nan"),
            "ret_std": float(np.std(rets)) if rets else float("nan"),
            "csr_c0_mean": float(np.mean(csr0)) if csr0 else float("nan"),
            "csr_c0_std": float(np.std(csr0)) if csr0 else float("nan"),
            "n": len(sub),
        }

    csv_path = output_dir / "pid_bench.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent", "ep_return_mean", "ep_return_std", "csr_c0_mean", "csr_c0_std", "n"])
        for agent in agents:
            s = stats.get(agent, {})
            w.writerow([agent, f"{s.get('ret_mean', float('nan')):.1f}",
                        f"{s.get('ret_std', float('nan')):.1f}",
                        f"{s.get('csr_c0_mean', float('nan')):.4f}",
                        f"{s.get('csr_c0_std', float('nan')):.4f}",
                        s.get("n", 0)])

    print("\n=== PID-Lagrangian benchmark results ===")
    print(f"{'Agent':>18} {'Return':>10} {'std':>8} {'CSR_c0':>8} {'std':>7} {'n':>4}")
    for agent in agents:
        s = stats.get(agent, {})
        print(f"{agent:>18} {s.get('ret_mean', float('nan')):>10.1f} "
              f"{s.get('ret_std', float('nan')):>8.1f} "
              f"{s.get('csr_c0_mean', float('nan')):>8.4f} "
              f"{s.get('csr_c0_std', float('nan')):>7.4f} {s.get('n', 0):>4}")
    print(f"CSV: {csv_path}")


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
    agents: list[str] = list(cfg["agents"])
    seeds: list[int] = [int(s) for s in cfg["seeds"]]

    if not cli.analyze_only:
        pending = []
        for agent in agents:
            for seed in seeds:
                cell_dir = output_dir / agent / f"seed={seed}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                if cli.skip_existing and (cell_dir / "result.txt").exists():
                    continue
                pending.append((agent, seed, cell_dir))

        print(f"PID bench: {len(agents)} agents x {len(seeds)} seeds = "
              f"{len(agents)*len(seeds)} total, {len(pending)} to run")

        if cli.parallel <= 1:
            for agent, seed, cell_dir in tqdm(pending, desc="pid_bench"):
                run_cell(agent, cfg, seed=seed, cell_dir=cell_dir)
        else:
            payloads = [(a, cfg, s, str(d)) for a, s, d in pending]
            failures = []
            with ProcessPoolExecutor(max_workers=cli.parallel) as ex:
                futures = [ex.submit(_worker, p) for p in payloads]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="pid_bench"):
                    agent, seed, err = fut.result()
                    if err:
                        failures.append((agent, seed, err))
                        tqdm.write(f"[FAIL] {agent} seed={seed}: {err}")
            if failures:
                raise SystemExit(f"{len(failures)} cell(s) failed")

    rows = collect_results(output_dir, agents, seeds)
    make_analysis(rows, agents, output_dir)


if __name__ == "__main__":
    main()

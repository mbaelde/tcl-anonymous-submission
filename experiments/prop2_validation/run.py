"""Grid runner for Proposition 2 validation.

Sweeps Lag-SAC over (alpha, omega, seed). For each cell, calls
`agents.sac_lagrangian.train` with the matching hyperparameters and
dumps the lambda(t) trajectory to

    <output_dir>/alpha={alpha}/omega={omega:.4f}/seed={seed}/traj.npz

Usage:
    uv run python -m experiments.prop2_validation.run \\
        --config experiments/prop2_validation/config.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

# Make sibling `agents/` importable when invoked as a module.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.sac_lagrangian import Args, train  # noqa: E402


def build_args(cfg: dict, alpha: float, omega: float, seed: int, dump_path: Path) -> Args:
    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
    return Args(
        exp_name="prop2",
        seed=seed,
        torch_deterministic=True,
        cuda=False,  # CPU is faster for this 2D obs + 1D action env.
        log_dir=str(dump_path.parent / "tb"),
        # env
        horizon=int(env_cfg["horizon"]),
        b0=float(env_cfg["b0"]),
        amplitude=float(env_cfg["amplitude"]),
        omega=float(omega),
        random_phase_at_reset=bool(env_cfg["random_phase_at_reset"]),
        # agent
        total_timesteps=int(agent_cfg["total_timesteps"]),
        buffer_size=int(agent_cfg.get("buffer_size", 200_000)),
        gamma=float(agent_cfg["gamma"]),
        tau=float(agent_cfg["tau"]),
        batch_size=int(agent_cfg["batch_size"]),
        learning_starts=int(agent_cfg["learning_starts"]),
        policy_lr=float(agent_cfg["policy_lr"]),
        q_lr=float(agent_cfg["q_lr"]),
        autotune_alpha=bool(agent_cfg["autotune_alpha"]),
        alpha_init=float(agent_cfg["alpha_init"]),
        # Lagrangian
        lambda_init=0.0,
        lambda_lr=float(alpha),
        lambda_update_frequency=1,
        cost_baseline=0.0,
        dump_trajectory_path=str(dump_path),
    )


def run_cell(
    cfg: dict,
    alpha: float,
    omega_ppe: float,
    omega: float,
    seed: int,
    cell_dir: Path,
) -> dict[str, float]:
    """Train one (alpha, omega, seed) cell and persist the result summary."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    dump_path = cell_dir / "traj.npz"
    cell_args = build_args(
        cfg, alpha=alpha, omega=omega, seed=seed, dump_path=dump_path
    )
    result = train(cell_args)
    with (cell_dir / "result.txt").open("w", encoding="utf-8") as f:
        f.write(f"alpha: {alpha}\n")
        f.write(f"omega_ppe: {omega_ppe}\n")
        f.write(f"omega: {omega}\n")
        f.write(f"seed: {seed}\n")
        for k, v in result.items():
            f.write(f"{k}: {v}\n")
    return result


def _worker_run_cell(
    payload: tuple[dict, float, float, float, int, str],
) -> tuple[float, float, int, str | None]:
    """ProcessPoolExecutor worker: must be top-level (picklable on Windows).

    Pins PyTorch to a single thread to avoid OpenMP oversubscription when
    multiple cells run in parallel.
    """
    import torch

    torch.set_num_threads(1)
    cfg, alpha, omega_ppe, omega, seed, cell_dir_str = payload
    try:
        run_cell(
            cfg=cfg,
            alpha=alpha,
            omega_ppe=omega_ppe,
            omega=omega,
            seed=seed,
            cell_dir=Path(cell_dir_str),
        )
    except Exception as e:
        return alpha, omega_ppe, seed, f"{type(e).__name__}: {e}"
    return alpha, omega_ppe, seed, None


def build_jobs(cfg: dict, skip_existing: bool = True) -> list[tuple]:
    """Enumerate ``(worker, payload)`` pairs for every pending cell.

    Mirrors ``main``'s grid enumeration so ``experiments.run_all_flat`` can merge
    this experiment's cells into one global pool. No ``--only-*`` filtering; the
    payload matches what ``_worker_run_cell`` unpacks.
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(a) for a in cfg["grid"]["alphas"]]
    omegas_ppe = [float(o) for o in cfg["grid"]["omegas_periods_per_episode"]]
    seeds = [int(s) for s in cfg["grid"]["seeds"]]
    horizon = int(cfg["env"]["horizon"])
    jobs: list[tuple] = []
    for alpha in alphas:
        for omega_ppe in omegas_ppe:
            omega = 2.0 * math.pi * omega_ppe / horizon
            for seed in seeds:
                cell_dir = (
                    output_dir
                    / f"alpha={alpha:.0e}"
                    / f"omega_ppe={omega_ppe:g}"
                    / f"seed={seed}"
                )
                cell_dir.mkdir(parents=True, exist_ok=True)
                if skip_existing and (cell_dir / "traj.npz").exists():
                    continue
                jobs.append(
                    (_worker_run_cell, (cfg, alpha, omega_ppe, omega, seed, str(cell_dir)))
                )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip cells whose traj.npz already exists.")
    parser.add_argument("--only-alpha", type=float, default=None)
    parser.add_argument("--only-omega-ppe", type=float, default=None,
                        help="Filter on omega in periods-per-episode units.")
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of cells to run concurrently (ProcessPoolExecutor). "
        "Default 1 = sequential. Each worker pins torch to 1 thread.",
    )
    cli = parser.parse_args()

    with cli.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = [float(a) for a in cfg["grid"]["alphas"]]
    omegas_ppe = [float(o) for o in cfg["grid"]["omegas_periods_per_episode"]]
    seeds = [int(s) for s in cfg["grid"]["seeds"]]
    horizon = int(cfg["env"]["horizon"])

    cells = []
    for alpha in alphas:
        if cli.only_alpha is not None and not math.isclose(alpha, cli.only_alpha):
            continue
        for omega_ppe in omegas_ppe:
            if (cli.only_omega_ppe is not None
                    and not math.isclose(omega_ppe, cli.only_omega_ppe)):
                continue
            omega = 2.0 * math.pi * omega_ppe / horizon
            for seed in seeds:
                if cli.only_seed is not None and seed != cli.only_seed:
                    continue
                cells.append((alpha, omega_ppe, omega, seed))

    print(
        f"Running {len(cells)} cells "
        f"({len(alphas)} alphas x {len(omegas_ppe)} omegas x {len(seeds)} seeds), "
        f"parallel={cli.parallel}"
    )

    pending: list[tuple[float, float, float, int, Path]] = []
    for alpha, omega_ppe, omega, seed in cells:
        cell_dir = (
            output_dir
            / f"alpha={alpha:.0e}"
            / f"omega_ppe={omega_ppe:g}"
            / f"seed={seed}"
        )
        cell_dir.mkdir(parents=True, exist_ok=True)
        if cli.skip_existing and (cell_dir / "traj.npz").exists():
            continue
        pending.append((alpha, omega_ppe, omega, seed, cell_dir))

    if cli.parallel <= 1:
        for alpha, omega_ppe, omega, seed, cell_dir in tqdm(pending, desc="prop2"):
            run_cell(
                cfg=cfg,
                alpha=alpha,
                omega_ppe=omega_ppe,
                omega=omega,
                seed=seed,
                cell_dir=cell_dir,
            )
        return

    payloads = [(cfg, a, op, om, s, str(d)) for a, op, om, s, d in pending]
    failures: list[tuple[float, float, int, str]] = []
    with ProcessPoolExecutor(max_workers=cli.parallel) as ex:
        futures = [ex.submit(_worker_run_cell, p) for p in payloads]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="prop2"):
            alpha, omega_ppe, seed, err = fut.result()
            if err is not None:
                failures.append((alpha, omega_ppe, seed, err))
                tqdm.write(
                    f"[FAIL] alpha={alpha:.0e} omega_ppe={omega_ppe:g} "
                    f"seed={seed}: {err}"
                )
    if failures:
        raise SystemExit(f"{len(failures)} cell(s) failed; see [FAIL] lines above.")


if __name__ == "__main__":
    main()

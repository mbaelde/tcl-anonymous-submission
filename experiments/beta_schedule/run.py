"""Beta-schedule runner for section 4.3 / 6.2: 3 regimes on MultiConstraintAdCraft.

For each (schedule, seed) cell, calls ``sac_tcl.train`` with a
``MultiConstraintAdCraft`` factory injected via ``env_factory``. Each cell
writes its own TensorBoard logs under ``<output_dir>/<schedule>/seed=<seed>/tb``
and a one-line result summary alongside.

Usage::

    uv run python -m experiments.beta_schedule.run \\
        --config experiments/beta_schedule/config.yaml

CLI filters mirror ``experiments.pilot_adcraft.run`` to keep the workflow
familiar (``--only-schedule``, ``--only-seed``, ``--skip-existing``).

NB: ``tcl.envs.MultiConstraintAdCraft`` cannot be seeded from Python because
the underlying AdCraft Rust sim uses ``thread_rng()``. Cross-seed variation
comes from PyTorch / NumPy stochasticity only.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym
import yaml
from tqdm import tqdm

# Make sibling `agents/` importable when invoked as a module.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents import sac_tcl  # noqa: E402
from tcl.envs.adcraft_multiconstraint import MultiConstraintAdCraft  # noqa: E402
from tcl.envs.adcraft_laplacian import MultiConstraintAdCraftLaplacian  # noqa: E402


def make_env_factory(env_cfg: dict):
    """Return an env_factory for the configured env kind.

    env_cfg["env_kind"] selects the environment:
    - "laplacian" (default for new runs): pure-Python Laplacian sim, §G.1-faithful.
    - "legacy": original Rust-backed AdCraft sim (retained for reproducibility).
    """
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
        return MultiConstraintAdCraft(
            num_keywords=common["num_keywords"],
            budget=common["budget"],
            bid_max=common["bid_max"],
            max_days=common["max_days"],
            target_utilization=common["target_utilization"],
            target_ctr=common["target_ctr"],
            target_margin=common["target_margin"],
            margin_formula=common["margin_formula"],
        )

    return factory


def build_args(
    sac_cfg: dict,
    tcl_base: dict,
    schedule_cfg: dict,
    seed: int,
    schedule_name: str,
    log_dir: Path,
) -> sac_tcl.Args:
    """Build the sac_tcl.Args dataclass for a (schedule, seed) cell."""
    return sac_tcl.Args(
        exp_name=f"beta_{schedule_name}",
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
        thresholds=str(tcl_base["thresholds"]),
        betas_init=str(schedule_cfg["betas_init"]),
        betas_final=str(schedule_cfg.get("betas_final", "")),
        beta_schedule=str(schedule_cfg["beta_schedule"]),
        beta_anneal_steps=int(schedule_cfg["beta_anneal_steps"]),
    )


def run_cell(
    schedule_name: str,
    cfg: dict,
    seed: int,
    cell_dir: Path,
) -> dict[str, float]:
    """Train one (schedule, seed) cell and persist the result summary."""
    args = build_args(
        sac_cfg=cfg["sac"],
        tcl_base=cfg["tcl_base"],
        schedule_cfg=cfg["schedules"][schedule_name],
        seed=seed,
        schedule_name=schedule_name,
        log_dir=cell_dir / "tb",
    )
    env_factory = make_env_factory(cfg["env"])
    result = sac_tcl.train(args, env_factory=env_factory)

    with (cell_dir / "result.txt").open("w", encoding="utf-8") as f:
        f.write(f"schedule: {schedule_name}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"betas_init: {args.betas_init}\n")
        f.write(f"betas_final: {args.betas_final}\n")
        f.write(f"beta_schedule: {args.beta_schedule}\n")
        f.write(f"beta_anneal_steps: {args.beta_anneal_steps}\n")
        for k, v in result.items():
            f.write(f"{k}: {v}\n")
    return result


def _worker_run_cell(
    payload: tuple[str, dict, int, str],
) -> tuple[str, int, str | None]:
    """ProcessPoolExecutor worker: must be top-level (picklable on Windows).

    Pins PyTorch to a single thread to avoid OpenMP oversubscription when
    multiple cells run in parallel. Returns ``(schedule, seed, error_or_None)``
    so the parent can surface failures without killing the whole pool.
    """
    import torch

    torch.set_num_threads(1)
    schedule_name, cfg, seed, cell_dir_str = payload
    try:
        run_cell(schedule_name, cfg, seed=seed, cell_dir=Path(cell_dir_str))
    except Exception as e:
        return schedule_name, seed, f"{type(e).__name__}: {e}"
    return schedule_name, seed, None


def build_jobs(cfg: dict, skip_existing: bool = True) -> list[tuple]:
    """Enumerate ``(worker, payload)`` pairs for every pending cell.

    Mirrors ``main``'s ``schedules x seeds`` grid so ``experiments.run_all_flat``
    can merge this experiment's cells into one global pool.
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    schedules = list(cfg["schedules"].keys())
    seeds = [int(s) for s in cfg["seeds"]]
    jobs: list[tuple] = []
    for schedule_name in schedules:
        for seed in seeds:
            cell_dir = output_dir / schedule_name / f"seed={seed}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            if skip_existing and (cell_dir / "result.txt").exists():
                continue
            jobs.append((_worker_run_cell, (schedule_name, cfg, seed, str(cell_dir))))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cells whose result.txt already exists.",
    )
    parser.add_argument(
        "--only-schedule",
        type=str,
        default=None,
        help="Run only this schedule name (e.g. increasing, decreasing, fixed).",
    )
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

    schedules = list(cfg["schedules"].keys())
    if cli.only_schedule is not None:
        if cli.only_schedule not in schedules:
            raise SystemExit(
                f"--only-schedule {cli.only_schedule!r} not in {schedules}"
            )
        schedules = [cli.only_schedule]

    seeds = [int(s) for s in cfg["seeds"]]
    if cli.only_seed is not None:
        seeds = [s for s in seeds if s == cli.only_seed]

    cells = [(sch, s) for sch in schedules for s in seeds]
    print(
        f"Running {len(cells)} cells "
        f"({len(schedules)} schedules x {len(seeds)} seeds), "
        f"parallel={cli.parallel}"
    )

    pending: list[tuple[str, int, Path]] = []
    for schedule_name, seed in cells:
        cell_dir = output_dir / schedule_name / f"seed={seed}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        if cli.skip_existing and (cell_dir / "result.txt").exists():
            continue
        pending.append((schedule_name, seed, cell_dir))

    if cli.parallel <= 1:
        for schedule_name, seed, cell_dir in tqdm(pending, desc="beta_schedule"):
            run_cell(schedule_name, cfg, seed=seed, cell_dir=cell_dir)
        return

    payloads = [(s, cfg, sd, str(d)) for s, sd, d in pending]
    failures: list[tuple[str, int, str]] = []
    with ProcessPoolExecutor(max_workers=cli.parallel) as ex:
        futures = [ex.submit(_worker_run_cell, p) for p in payloads]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="beta_schedule"
        ):
            schedule_name, seed, err = fut.result()
            if err is not None:
                failures.append((schedule_name, seed, err))
                tqdm.write(f"[FAIL] {schedule_name} seed={seed}: {err}")
    if failures:
        raise SystemExit(f"{len(failures)} cell(s) failed; see [FAIL] lines above.")


if __name__ == "__main__":
    main()

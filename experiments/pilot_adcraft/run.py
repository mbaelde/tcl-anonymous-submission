"""Pilot runner for section 7.1: four baselines on MultiConstraintAdCraft (K=3).

For each (agent, seed) cell, calls the corresponding ``train`` entry point
with a ``MultiConstraintAdCraft`` factory injected via ``env_factory``. Each
agent writes its own TensorBoard logs under ``<output_dir>/<agent>/<seed>/tb``
and a one-line result summary alongside.

Usage::

    uv run python -m experiments.pilot_adcraft.run \\
        --config experiments/pilot_adcraft/config.yaml

CLI filters mirror ``experiments.prop2_validation.run`` to keep the workflow
familiar.

NB: ``tcl.envs.MultiConstraintAdCraft`` cannot be seeded from Python because
the underlying AdCraft Rust sim uses ``thread_rng()``. Cross-seed variation
comes from PyTorch / NumPy stochasticity only; see
``tests/test_adcraft_multiconstraint.py::test_seed_determinism`` (xfail).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml
from tqdm import tqdm

# Make sibling `agents/` importable when invoked as a module.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents import (  # noqa: E402
    sac_fixed,
    sac_hprs,
    sac_lagrangian_multi,
    sac_tcl,
    sac_tcl_standalone,
)
from tcl.envs.adcraft_multiconstraint import MultiConstraintAdCraft  # noqa: E402
from tcl.envs.adcraft_laplacian import MultiConstraintAdCraftLaplacian  # noqa: E402

AGENT_REGISTRY: dict[str, Any] = {
    "lag_multi": sac_lagrangian_multi,
    "fixed": sac_fixed,
    "tcl": sac_tcl,
    "hprs": sac_hprs,
    # Standalone (A) formulation; rb_mode chosen per cell via cfg["agents"][name].
    "tcl_standalone": sac_tcl_standalone,
    # Aliases for κ-calibration ablation (phase5_kappa): both use sac_tcl with kappas set
    "tcl_gaussian_empirical": sac_tcl,
    "tcl_gaussian_formula": sac_tcl,
}


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
        # Optional non-stationarity spec: updater_params overrides env defaults.
        # Accepts either a raw list (e.g. [["vol",0.01],["ctr",0.01],["cvr",0.01]])
        # or a drift_rate float shorthand (applies the same rate to vol/ctr/cvr).
        if "updater_params" in env_cfg:
            common["updater_params"] = list(env_cfg["updater_params"])
        elif "drift_rate" in env_cfg:
            dr = float(env_cfg["drift_rate"])
            common["updater_params"] = [["vol", dr], ["ctr", dr], ["cvr", dr]]
        if env_kind == "laplacian":
            return MultiConstraintAdCraftLaplacian(**common)
        return MultiConstraintAdCraft(**common)

    return factory



def build_common_kwargs(sac_cfg: dict, seed: int, log_dir: Path) -> dict[str, Any]:
    """Hyperparameters shared by all four agent Args dataclasses."""
    return {
        "seed": seed,
        "torch_deterministic": True,
        "cuda": bool(sac_cfg.get("cuda", False)),
        "log_dir": str(log_dir),
        "total_timesteps": int(sac_cfg["total_timesteps"]),
        "buffer_size": int(sac_cfg["buffer_size"]),
        "batch_size": int(sac_cfg["batch_size"]),
        "learning_starts": int(sac_cfg["learning_starts"]),
        "gamma": float(sac_cfg["gamma"]),
        "tau": float(sac_cfg["tau"]),
        "policy_lr": float(sac_cfg["policy_lr"]),
        "q_lr": float(sac_cfg["q_lr"]),
        "policy_frequency": int(sac_cfg["policy_frequency"]),
        "target_network_frequency": int(sac_cfg["target_network_frequency"]),
        "autotune_alpha": bool(sac_cfg["autotune_alpha"]),
        "alpha_init": float(sac_cfg["alpha_init"]),
    }


def build_args_for_agent(agent_name: str, base: dict[str, Any], extras: dict[str, Any]):
    """Instantiate the agent-specific Args dataclass with merged kwargs."""
    module = AGENT_REGISTRY[agent_name]
    kwargs = {**base, **extras, "exp_name": f"pilot_{agent_name}"}
    return module.Args(**kwargs)


def run_cell(
    agent_name: str,
    cfg: dict,
    seed: int,
    cell_dir: Path,
) -> dict[str, float]:
    """Train one (agent, seed) cell and persist the result summary."""
    base = build_common_kwargs(cfg["sac"], seed=seed, log_dir=cell_dir / "tb")
    extras = dict(cfg["agents"][agent_name])
    agent_args = build_args_for_agent(agent_name, base, extras)
    env_factory = make_env_factory(cfg["env"])
    result = AGENT_REGISTRY[agent_name].train(agent_args, env_factory=env_factory)

    with (cell_dir / "result.txt").open("w", encoding="utf-8") as f:
        f.write(f"agent: {agent_name}\n")
        f.write(f"seed: {seed}\n")
        for k, v in result.items():
            f.write(f"{k}: {v}\n")
    return result


def _worker_run_cell(
    payload: tuple[str, dict, int, str],
) -> tuple[str, int, str | None]:
    """ProcessPoolExecutor worker: must be top-level (picklable on Windows).

    Pins PyTorch to a single thread to avoid OpenMP oversubscription when
    multiple cells run in parallel. Returns ``(agent, seed, error_or_None)``
    so the parent can surface failures without killing the whole pool.
    """
    import torch

    torch.set_num_threads(1)
    agent_name, cfg, seed, cell_dir_str = payload
    try:
        run_cell(agent_name, cfg, seed=seed, cell_dir=Path(cell_dir_str))
    except Exception as e:
        return agent_name, seed, f"{type(e).__name__}: {e}"
    return agent_name, seed, None


def build_jobs(cfg: dict, skip_existing: bool = True) -> list[tuple]:
    """Enumerate ``(worker, payload)`` pairs for every pending cell.

    Mirrors ``main``'s ``agents x seeds`` grid so ``experiments.run_all_flat`` can
    merge this experiment's cells into one global pool.
    """
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    agents_to_run = list(cfg["agents"].keys())
    seeds = [int(s) for s in cfg["seeds"]]
    jobs: list[tuple] = []
    for agent_name in agents_to_run:
        for seed in seeds:
            cell_dir = output_dir / agent_name / f"seed={seed}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            if skip_existing and (cell_dir / "result.txt").exists():
                continue
            jobs.append((_worker_run_cell, (agent_name, cfg, seed, str(cell_dir))))
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
        "--only-agent",
        type=str,
        default=None,
        choices=sorted(AGENT_REGISTRY.keys()),
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

    agents_to_run = (
        [cli.only_agent] if cli.only_agent is not None else list(cfg["agents"].keys())
    )
    seeds = [int(s) for s in cfg["seeds"]]
    if cli.only_seed is not None:
        seeds = [s for s in seeds if s == cli.only_seed]

    cells = [(a, s) for a in agents_to_run for s in seeds]
    print(
        f"Running {len(cells)} cells "
        f"({len(agents_to_run)} agents x {len(seeds)} seeds), "
        f"parallel={cli.parallel}"
    )

    # Resolve every cell dir up-front so the skip-existing filter applies
    # uniformly to both code paths.
    pending: list[tuple[str, int, Path]] = []
    for agent_name, seed in cells:
        cell_dir = output_dir / agent_name / f"seed={seed}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        if cli.skip_existing and (cell_dir / "result.txt").exists():
            continue
        pending.append((agent_name, seed, cell_dir))

    if cli.parallel <= 1:
        for agent_name, seed, cell_dir in tqdm(pending, desc="pilot_adcraft"):
            run_cell(agent_name, cfg, seed=seed, cell_dir=cell_dir)
        return

    payloads = [(a, cfg, s, str(d)) for a, s, d in pending]
    failures: list[tuple[str, int, str]] = []
    with ProcessPoolExecutor(max_workers=cli.parallel) as ex:
        futures = [ex.submit(_worker_run_cell, p) for p in payloads]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="pilot_adcraft"
        ):
            agent_name, seed, err = fut.result()
            if err is not None:
                failures.append((agent_name, seed, err))
                tqdm.write(f"[FAIL] {agent_name} seed={seed}: {err}")
    if failures:
        raise SystemExit(f"{len(failures)} cell(s) failed; see [FAIL] lines above.")


if __name__ == "__main__":
    main()

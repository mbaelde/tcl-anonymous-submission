"""Diagnostic runner for pilot_adcraft.

Drives the 4-cell mini-grid defined in `config.diagnostic.yaml`:
  - Cells A/B/C: lag_multi SAC variants with different (total_timesteps,
    env_overrides). Re-uses the same SAC core hyperparameters as the
    full pilot but with the corrected constraint targets.
  - Cell D: constant-bid baseline. Sweeps a fixed bid value across all
    keywords for `episodes_per_bid` episodes, records per-step
    utilization / CTR / margin. No SAC. Gives the achievable physics
    floor against which the SAC cells are compared.

Usage::

    py -3.14 -m uv run python -m experiments.pilot_adcraft.diagnostic \\
        --config experiments/pilot_adcraft/config.diagnostic.yaml \\
        --parallel 4

The cells are independent — parallel=4 runs them concurrently. The
constant-bid cell is cheap (<1 min total) so parallelism mainly helps
the three 60k–250k SAC cells.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents import sac_lagrangian_multi  # noqa: E402
from tcl.envs.adcraft_multiconstraint import MultiConstraintAdCraft  # noqa: E402


def make_env(env_cfg: dict[str, Any]) -> MultiConstraintAdCraft:
    return MultiConstraintAdCraft(
        num_keywords=int(env_cfg["num_keywords"]),
        budget=float(env_cfg["budget"]),
        bid_max=float(env_cfg["bid_max"]),
        max_days=int(env_cfg["max_days"]),
        target_utilization=float(env_cfg["target_utilization"]),
        target_ctr=float(env_cfg["target_ctr"]),
        target_margin=float(env_cfg["target_margin"]),
        margin_formula=str(env_cfg.get("margin_formula", "cost_markup")),
    )


def run_sac_cell(
    cell_name: str,
    cfg: dict[str, Any],
    cell_spec: dict[str, Any],
    seed: int,
    cell_dir: Path,
) -> dict[str, Any]:
    """Train one lag_multi cell with the corrected constraint targets."""
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg.update(cell_spec.get("env_overrides") or {})

    sac_cfg = cfg["sac"]
    args = sac_lagrangian_multi.Args(
        exp_name=f"diag_{cell_name}",
        seed=seed,
        torch_deterministic=True,
        cuda=bool(sac_cfg.get("cuda", False)),
        log_dir=str(cell_dir / "tb"),
        total_timesteps=int(cell_spec["total_timesteps"]),
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
        **cfg["lag_multi"],
    )

    def env_factory(_args) -> gym.Env:  # type: ignore[no-untyped-def]
        return make_env(env_cfg)

    result = sac_lagrangian_multi.train(args, env_factory=env_factory)

    with (cell_dir / "result.txt").open("w", encoding="utf-8") as f:
        f.write(f"cell: {cell_name}\n")
        f.write(f"kind: sac\n")
        f.write(f"seed: {seed}\n")
        f.write(f"total_timesteps: {cell_spec['total_timesteps']}\n")
        f.write(f"env_overrides: {json.dumps(cell_spec.get('env_overrides') or {})}\n")
        f.write(f"target_ctr: {env_cfg['target_ctr']}\n")
        f.write(f"target_margin: {env_cfg['target_margin']}\n")
        f.write(f"margin_formula: {env_cfg.get('margin_formula', 'cost_markup')}\n")
        for k, v in result.items():
            f.write(f"{k}: {v}\n")
    return {"cell": cell_name, "seed": seed, "kind": "sac", **result}


def run_constant_bid_cell(
    cell_name: str,
    cfg: dict[str, Any],
    cell_spec: dict[str, Any],
    cell_dir: Path,
) -> dict[str, Any]:
    """Sweep constant per-keyword bids; record realized util/CTR/margin."""
    env = make_env(cfg["env"])
    rows: list[dict[str, float]] = []
    bids: list[float] = [float(b) for b in cell_spec["bids"]]
    episodes_per_bid = int(cell_spec["episodes_per_bid"])

    for bid in bids:
        action = np.full(env.num_keywords, bid, dtype=np.float32)
        for ep in range(episodes_per_bid):
            obs, info = env.reset(seed=ep)  # AdCraft uses thread_rng, see wrapper note
            util_acc: list[float] = []
            ctr_acc: list[float] = []
            mgn_acc: list[float] = []
            ep_return = 0.0
            for _step in range(env.max_days):
                obs, r, term, trunc, info = env.step(action)
                c = info["costs"]
                util_acc.append(env.target_utilization - float(c[0]))
                ctr_acc.append(env.target_ctr - float(c[1]))
                mgn_acc.append(env.target_margin - float(c[2]))
                ep_return += r
                if term or trunc:
                    break
            rows.append({
                "bid": bid,
                "episode": ep,
                "util_mean": float(np.mean(util_acc)),
                "ctr_mean": float(np.mean(ctr_acc)),
                "margin_mean": float(np.mean(mgn_acc)),
                "ep_return": ep_return,
                "csr_util": float(np.mean(np.array(util_acc) >= env.target_utilization)),
                "csr_ctr": float(np.mean(np.array(ctr_acc) >= env.target_ctr)),
                "csr_margin": float(np.mean(np.array(mgn_acc) >= env.target_margin)),
            })

    csv_path = cell_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_bid: dict[float, list[dict[str, float]]] = {b: [] for b in bids}
    for row in rows:
        by_bid[row["bid"]].append(row)
    aggregated = {
        bid: {
            "util_mean": float(np.mean([r["util_mean"] for r in rs])),
            "ctr_mean": float(np.mean([r["ctr_mean"] for r in rs])),
            "margin_mean": float(np.mean([r["margin_mean"] for r in rs])),
            "csr_util": float(np.mean([r["csr_util"] for r in rs])),
            "csr_ctr": float(np.mean([r["csr_ctr"] for r in rs])),
            "csr_margin": float(np.mean([r["csr_margin"] for r in rs])),
            "ep_return": float(np.mean([r["ep_return"] for r in rs])),
        }
        for bid, rs in by_bid.items()
    }
    with (cell_dir / "result.txt").open("w", encoding="utf-8") as f:
        f.write(f"cell: {cell_name}\n")
        f.write(f"kind: constant_bid\n")
        f.write(f"bids: {bids}\n")
        f.write(f"episodes_per_bid: {episodes_per_bid}\n")
        f.write(f"target_ctr: {cfg['env']['target_ctr']}\n")
        f.write(f"target_margin: {cfg['env']['target_margin']}\n")
        f.write(f"margin_formula: {cfg['env'].get('margin_formula', 'cost_markup')}\n")
        f.write("per_bid_aggregates:\n")
        for bid, ag in aggregated.items():
            f.write(f"  bid={bid}: {json.dumps(ag)}\n")

    return {"cell": cell_name, "kind": "constant_bid", "aggregated": aggregated}


def _worker_run(payload: tuple[str, str, dict, int | None, str]) -> tuple[str, int | None, str | None]:
    """Top-level worker function for ProcessPoolExecutor (picklable on Windows)."""
    import torch
    torch.set_num_threads(1)
    kind, cell_name, cfg, seed, cell_dir_str = payload
    cell_dir = Path(cell_dir_str)
    cell_spec = cfg["cells"][cell_name]
    try:
        if kind == "sac":
            assert seed is not None
            run_sac_cell(cell_name, cfg, cell_spec, seed=seed, cell_dir=cell_dir)
        else:
            run_constant_bid_cell(cell_name, cfg, cell_spec, cell_dir=cell_dir)
    except Exception as e:
        return cell_name, seed, f"{type(e).__name__}: {e}"
    return cell_name, seed, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--only-cell", type=str, default=None,
        help="Run only this cell name (e.g. A_canonical_short).",
    )
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=1)
    cli = parser.parse_args()

    with cli.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cells_to_run = (
        [cli.only_cell] if cli.only_cell is not None else list(cfg["cells"].keys())
    )
    seeds = [int(s) for s in cfg["seeds"]]
    if cli.only_seed is not None:
        seeds = [s for s in seeds if s == cli.only_seed]

    payloads: list[tuple[str, str, dict, int | None, str]] = []
    for cell_name in cells_to_run:
        spec = cfg["cells"][cell_name]
        kind = spec["kind"]
        if kind == "constant_bid":
            cell_dir = output_dir / cell_name
            cell_dir.mkdir(parents=True, exist_ok=True)
            if cli.skip_existing and (cell_dir / "result.txt").exists():
                continue
            payloads.append((kind, cell_name, cfg, None, str(cell_dir)))
        else:
            for seed in seeds:
                cell_dir = output_dir / cell_name / f"seed={seed}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                if cli.skip_existing and (cell_dir / "result.txt").exists():
                    continue
                payloads.append((kind, cell_name, cfg, seed, str(cell_dir)))

    print(f"Running {len(payloads)} diagnostic cells, parallel={cli.parallel}")

    if cli.parallel <= 1:
        for p in tqdm(payloads, desc="diagnostic"):
            kind, cell_name, _, seed, cell_dir_str = p
            cell_dir = Path(cell_dir_str)
            spec = cfg["cells"][cell_name]
            if kind == "sac":
                assert seed is not None
                run_sac_cell(cell_name, cfg, spec, seed=seed, cell_dir=cell_dir)
            else:
                run_constant_bid_cell(cell_name, cfg, spec, cell_dir=cell_dir)
        return

    failures: list[tuple[str, int | None, str]] = []
    with ProcessPoolExecutor(max_workers=cli.parallel) as ex:
        futures = [ex.submit(_worker_run, p) for p in payloads]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="diagnostic"):
            cell_name, seed, err = fut.result()
            if err is not None:
                failures.append((cell_name, seed, err))
                tqdm.write(f"[FAIL] {cell_name} seed={seed}: {err}")
    if failures:
        raise SystemExit(f"{len(failures)} cell(s) failed; see [FAIL] lines above.")


if __name__ == "__main__":
    main()

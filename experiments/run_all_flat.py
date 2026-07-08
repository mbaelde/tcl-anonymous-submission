"""Flat global-pool runner: merge ALL experiments' cells into ONE process pool.

``run_all_vm.sh`` runs the seed-based experiments **sequentially**, each spinning up
its own ``ProcessPoolExecutor``. That wastes cores twice over: (1) the tail of every
experiment drains the pool from N workers down to 0 before the next one starts, and
(2) small experiments (e.g. ``pid_lagrangian_bench`` = 30 cells) never fill a 44-core
box even at their peak.

This driver instead asks every experiment module for its list of pending cells (via
the ``build_jobs(cfg, skip_existing)`` hook each ``run.py`` exposes) and feeds the
*union* of all cells to a single pool of ``--workers`` processes. Cores stay saturated
until the very last cell, regardless of how the cells are distributed across
experiments. Per-cell work, configs and output layout are identical to the sequential
path — only the scheduling changes — so ``--skip-existing`` lets you switch a partially
finished sequential run straight onto this driver with no recomputation.

After training, the experiments that embed an analysis step (prop5, beta_star_scan,
tau_sensitivity, pid/pendulum benches) are re-invoked with ``--analyze-only``, and the
two deterministic analytical experiments are run last — mirroring ``run_all_vm.sh``.

Usage (from repo root)::

    uv run python -m experiments.run_all_flat                 # all, workers = os.cpu_count()
    uv run python -m experiments.run_all_flat --workers 44
    uv run python -m experiments.run_all_flat --dry-run       # just print cell counts
    uv run python -m experiments.run_all_flat --only prop2_validation tau_sensitivity
    uv run python -m experiments.run_all_flat --no-skip-existing   # force recompute
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SEEDS = list(range(1, 11))  # seeds 1..10, matching run_all_vm.sh


@dataclass(frozen=True)
class Exp:
    name: str          # run name -> runs_<root>/<name>
    module: str        # importable module exposing build_jobs() + main()
    config: str        # source config (repo-relative)
    analyze: bool      # whether main() embeds an analysis pass (re-run --analyze-only)


# Mirrors the seed-based experiments of run_all_vm.sh, in the same order.
EXPERIMENTS: list[Exp] = [
    Exp("prop2_validation", "experiments.prop2_validation.run",
        "experiments/prop2_validation/config.yaml", analyze=False),
    Exp("prop5_validation", "experiments.prop5_validation.run",
        "experiments/prop5_validation/config.yaml", analyze=True),
    Exp("prop5_validation_b2", "experiments.prop5_validation_b2.run",
        "experiments/prop5_validation_b2/config.yaml", analyze=True),
    Exp("phase5_a1_v2", "experiments.pilot_adcraft.run",
        "experiments/pilot_adcraft/config.phase5_a1_v2.yaml", analyze=False),
    Exp("phase5_b1", "experiments.pilot_adcraft.run",
        "experiments/pilot_adcraft/config.phase5_b1.yaml", analyze=False),
    Exp("phase5_reward_shift", "experiments.pilot_adcraft.run",
        "experiments/pilot_adcraft/config.phase5_reward_shift.yaml", analyze=False),
    Exp("beta_star_scan", "experiments.beta_star_scan.run",
        "experiments/beta_star_scan/config.yaml", analyze=True),
    Exp("beta_schedule", "experiments.beta_schedule.run",
        "experiments/beta_schedule/config.yaml", analyze=False),
    Exp("phase5_kappa", "experiments.pilot_adcraft.run",
        "experiments/pilot_adcraft/config.phase5_kappa.yaml", analyze=False),
    Exp("phase5_a1_standalone", "experiments.pilot_adcraft.run",
        "experiments/pilot_adcraft/config.phase5_a1_standalone.yaml", analyze=False),
    Exp("phase5_a1_standalone_ll_w001", "experiments.pilot_adcraft.run",
        "experiments/pilot_adcraft/config.phase5_a1_standalone_ll_w001.yaml", analyze=False),
    Exp("phase5_a1_standalone_ll_w01", "experiments.pilot_adcraft.run",
        "experiments/pilot_adcraft/config.phase5_a1_standalone_ll_w01.yaml", analyze=False),
    Exp("phase5_a1_standalone_ll", "experiments.pilot_adcraft.run",
        "experiments/pilot_adcraft/config.phase5_a1_standalone_ll.yaml", analyze=False),
    Exp("tau_sensitivity", "experiments.tau_sensitivity.run",
        "experiments/tau_sensitivity/config.yaml", analyze=True),
    Exp("pid_lagrangian_bench", "experiments.pid_lagrangian_bench.run",
        "experiments/pid_lagrangian_bench/config.yaml", analyze=True),
    Exp("constrained_pendulum_bench", "experiments.constrained_pendulum_bench.run",
        "experiments/constrained_pendulum_bench/config.yaml", analyze=True),
]

# Deterministic, no-seed analytical experiments (run last, cheap).
ANALYTIC_CMDS: list[tuple[str, list[str]]] = [
    ("prop2_analytic", ["-m", "experiments.prop2_analytic.run"]),
    ("k_scaling_validation",
     ["-m", "experiments.k_scaling_validation.run",
      "--output-dir", "{runs_root}/k_scaling_validation"]),
]


def patch_config(exp: Exp, runs_root: str, cfg_out_dir: Path) -> tuple[dict, Path]:
    """Load the source config, override seeds (1..10) and output_dir, persist a copy.

    The persisted copy is what the post-hoc ``--analyze-only`` pass reads, so it must
    carry the same output_dir as the training run.
    """
    with open(_REPO_ROOT / exp.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["seeds"] = list(SEEDS)
    cfg["output_dir"] = f"{runs_root}/{exp.name}"
    cfg_out_dir.mkdir(parents=True, exist_ok=True)
    patched_path = cfg_out_dir / f"{exp.name}.yaml"
    with open(patched_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return cfg, patched_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Pool size (default: all cores).")
    parser.add_argument("--runs-root", default="runs_vm",
                        help="Output root, e.g. runs_vm (default).")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only these experiment names (default: all).")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Recompute every cell even if its result already exists.")
    parser.add_argument("--no-analyze", action="store_true",
                        help="Skip the post-training analysis + analytical experiments.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only enumerate and print per-experiment cell counts.")
    args = parser.parse_args()

    skip_existing = not args.no_skip_existing
    selected = EXPERIMENTS
    if args.only:
        wanted = set(args.only)
        selected = [e for e in EXPERIMENTS if e.name in wanted]
        missing = wanted - {e.name for e in selected}
        if missing:
            raise SystemExit(f"Unknown experiment name(s): {sorted(missing)}")

    cfg_out_dir = Path(args.runs_root) / "_flat_cfgs"

    # ---- enumerate every pending cell across all selected experiments ----
    all_jobs: list[tuple] = []          # (worker, payload, exp_name)
    patched: dict[str, Path] = {}       # exp_name -> patched config path
    per_exp_counts: list[tuple[str, int]] = []
    for exp in selected:
        cfg, patched_path = patch_config(exp, args.runs_root, cfg_out_dir)
        patched[exp.name] = patched_path
        mod = importlib.import_module(exp.module)
        jobs = mod.build_jobs(cfg, skip_existing=skip_existing)
        per_exp_counts.append((exp.name, len(jobs)))
        for worker, payload in jobs:
            all_jobs.append((worker, payload, exp.name))

    print("=" * 64)
    print(f"Flat runner — {len(selected)} experiments, workers={args.workers}, "
          f"skip_existing={skip_existing}")
    print("-" * 64)
    for name, n in per_exp_counts:
        print(f"  {name:<32} {n:>5} cell(s) pending")
    print("-" * 64)
    print(f"  TOTAL pending cells: {len(all_jobs)}")
    print("=" * 64)

    if args.dry_run:
        return
    if not all_jobs:
        print("Nothing to run (all cells already complete?).")
    else:
        failures: list[tuple[str, tuple, str]] = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut_to_name = {
                ex.submit(worker, payload): name
                for worker, payload, name in all_jobs
            }
            for fut in tqdm(as_completed(fut_to_name), total=len(fut_to_name),
                            desc="flat", smoothing=0.05):
                name = fut_to_name[fut]
                result = fut.result()       # all workers return (..., err_or_None)
                err = result[-1]
                if err is not None:
                    failures.append((name, result[:-1], err))
                    tqdm.write(f"[FAIL][{name}] {result[:-1]}: {err}")
        if failures:
            print(f"\n{len(failures)} cell(s) failed across experiments:")
            for name, ident, err in failures:
                print(f"  [{name}] {ident}: {err}")
            raise SystemExit(1)
        print("\nAll training cells completed.")

    # ---- post-training analysis (mirrors run_all_vm.sh) ----
    if args.no_analyze:
        return

    for exp in selected:
        if not exp.analyze:
            continue
        print(f"\n[analyze] {exp.name}")
        subprocess.run(
            [sys.executable, "-m", exp.module,
             "--config", str(patched[exp.name]), "--analyze-only"],
            cwd=str(_REPO_ROOT), check=True,
        )

    # analytical experiments only make sense on a full (non-subset) run
    if args.only is None:
        for name, cmd in ANALYTIC_CMDS:
            print(f"\n[analytic] {name}")
            resolved = [c.format(runs_root=args.runs_root) for c in cmd]
            subprocess.run([sys.executable, *resolved], cwd=str(_REPO_ROOT), check=True)

    print("\nFlat run complete. Results under", f"{args.runs_root}/")


if __name__ == "__main__":
    main()

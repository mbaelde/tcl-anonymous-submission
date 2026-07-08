"""Progress monitor for a (possibly running) VM sweep.

For every ``<agent>/seed=<n>`` cell under the given run dir(s), reports the
latest training step reached (read from the TensorBoard events) and, if a target
``--total`` step count is given, the completion percentage. Cells whose
``result.txt`` already exists are counted as finished (100 %).

Usage (from repo root)::

    # quick check, one experiment dir, known total_timesteps
    uv run python -m experiments.progress --run-dirs runs_vm/phase5_a1_v2 --total 59000

    # several dirs, just step counts (no %)
    uv run python -m experiments.progress --run-dirs runs_vm/*/

    # read total_timesteps from the experiment config instead of --total
    uv run python -m experiments.progress --run-dirs runs_vm/beta_star_scan \
        --total-from-config experiments/beta_star_scan/config.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_STEP_TAGS = [
    "rollout/episode_return",
    "rollout/episode_steps",
    "charts/SPS",
    "losses/qf1_loss",
]


def find_tfevents_dir(tb_dir: Path) -> Path | None:
    """tfevents live either directly in tb/ or in tb/<run_name>/ — pick newest."""
    if not tb_dir.exists():
        return None
    candidates = [p for p in tb_dir.iterdir() if p.is_dir()]
    if not candidates:
        # events may sit directly under tb/
        return tb_dir if any(tb_dir.glob("*.tfevents.*")) else None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_step(tb_dir: Path) -> int:
    ev = find_tfevents_dir(tb_dir)
    if ev is None:
        return 0
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        ea = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        ea.Reload()
        available = set(ea.Tags().get("scalars", []))
        best = 0
        for tag in _STEP_TAGS:
            if tag in available:
                events = ea.Scalars(tag)
                if events:
                    best = max(best, int(events[-1].step))
        return best
    except Exception:
        return 0


def discover_cells(run_dir: Path) -> list[tuple[str, int, Path]]:
    """Find every ``seed=<n>`` leaf at any depth under ``run_dir``.

    Handles both the 2-level layout (``<agent>/seed=<n>``) used by most
    experiments and the deeper layouts such as prop2_validation's
    ``alpha=<a>/omega_ppe=<o>/seed=<n>``. The group label is the path from
    ``run_dir`` down to the seed's parent, so cells stay distinguishable.
    """
    cells: list[tuple[str, int, Path]] = []
    for seed_dir in sorted(run_dir.rglob("seed=*")):
        m = re.fullmatch(r"seed=(\d+)", seed_dir.name)
        if not m or not seed_dir.is_dir():
            continue
        try:
            group = str(seed_dir.parent.relative_to(run_dir))
        except ValueError:
            group = seed_dir.parent.name
        cells.append((group, int(m.group(1)), seed_dir))
    return cells


def is_done(cell_dir: Path) -> bool:
    """A cell is finished once it has written its result or trajectory.

    Most experiments drop a ``result.txt``; prop2_validation writes a
    ``traj.npz`` (and a ``result.txt``) but no ``tb/`` subdir.
    """
    return (cell_dir / "result.txt").exists() or (cell_dir / "traj.npz").exists()


def total_from_config(path: Path) -> int | None:
    import yaml

    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sac = cfg.get("sac", cfg)
    val = sac.get("total_timesteps")
    return int(val) if val is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--total", type=int, default=None,
                        help="Target total_timesteps per cell (enables percentages).")
    parser.add_argument("--total-from-config", type=Path, default=None,
                        help="Read total_timesteps from a config.yaml instead of --total.")
    parser.add_argument("--quiet", action="store_true",
                        help="Print only the global summary, not per-cell rows.")
    args = parser.parse_args()

    total = args.total
    if total is None and args.total_from_config is not None:
        total = total_from_config(args.total_from_config)

    all_cells: list[tuple[str, int, Path]] = []
    for run_dir in args.run_dirs:
        cells = discover_cells(run_dir)
        if not cells:
            print(f"[warn] no cells under {run_dir}")
        all_cells.extend(cells)

    if not all_cells:
        raise SystemExit("No cells found.")

    rows = []
    n_done = 0
    steps_done = 0
    for agent, seed, cell_dir in all_cells:
        done = is_done(cell_dir)
        if done:
            step = total if total else latest_step(cell_dir / "tb")
            n_done += 1
        else:
            step = latest_step(cell_dir / "tb")
        steps_done += step if step else 0
        pct = (100.0 * step / total) if (total and step) else None
        rows.append((agent, seed, step, done, pct))

    if not args.quiet:
        print(f"\n{'agent':>18} {'seed':>5} {'step':>10} {'done':>5} {'pct':>7}")
        for agent, seed, step, done, pct in rows:
            pct_s = f"{pct:6.1f}%" if pct is not None else "    --"
            print(f"{agent:>18} {seed:>5} {step:>10} {'✓' if done else ' ':>5} {pct_s:>7}")

    n = len(rows)
    print(f"\nCells: {n_done}/{n} finished ({100.0 * n_done / n:.1f}%).")
    if total:
        target = n * total
        print(f"Steps: {steps_done:,} / {target:,} ({100.0 * steps_done / target:.1f}% overall).")
    else:
        print(f"Steps done (sum): {steps_done:,}  (pass --total for a %).")


if __name__ == "__main__":
    main()

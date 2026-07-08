"""Wall-time benchmark: training steps/sec for TCL-SAC, Lag-SAC, Fixed-SAC, HPRS-SAC.

Measures the wall-clock time per training step for each agent at K=3 (AdCraft A1).
Validates that TCL (which adds K sigmoid multiplications per step) has negligible
overhead vs simpler baselines.

Usage:
    PYTHONUTF8=1 py -3.14 -m uv run python -m experiments.wall_time_bench.run
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents import sac_fixed, sac_hprs, sac_lagrangian_multi, sac_tcl  # noqa: E402
from tcl.envs.adcraft_laplacian import MultiConstraintAdCraftLaplacian  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BENCH_STEPS = 10_000      # steps per agent (enough for stable measurement)
LEARNING_STARTS = 1_000   # must be < BENCH_STEPS
SEED = 42


def make_env_factory():
    def factory(args) -> gym.Env:  # type: ignore[no-untyped-def]
        return MultiConstraintAdCraftLaplacian(
            num_keywords=100,
            budget=100.0,
            bid_max=3.0,
            max_days=60,
            target_utilization=0.80,
            target_ctr=0.15,
            target_margin=-4.0,
            margin_formula="revenue_share",
            updater_params=[["vol", 0.03], ["ctr", 0.03], ["cvr", 0.03]],
        )
    return factory


@dataclass
class BenchResult:
    agent: str
    total_steps: int
    wall_sec: float
    steps_per_sec: float
    overhead_vs_fixed: float | None = None


def run_agent_bench(name: str, train_fn, args) -> BenchResult:
    env_factory = make_env_factory()
    t0 = time.perf_counter()
    train_fn(args, env_factory=env_factory)
    elapsed = time.perf_counter() - t0
    sps = BENCH_STEPS / elapsed
    print(f"  {name}: {elapsed:.1f}s total, {sps:.0f} steps/sec")
    return BenchResult(agent=name, total_steps=BENCH_STEPS, wall_sec=elapsed,
                       steps_per_sec=sps)


def common_sac_kwargs():
    return dict(
        seed=SEED,
        torch_deterministic=True,
        cuda=False,
        log_dir=str(Path("runs/wall_time_bench/tb")),
        total_timesteps=BENCH_STEPS,
        buffer_size=BENCH_STEPS + 1000,
        batch_size=256,
        learning_starts=LEARNING_STARTS,
        gamma=0.99,
        tau=0.005,
        policy_lr=3e-4,
        q_lr=3e-4,
        policy_frequency=2,
        target_network_frequency=1,
        autotune_alpha=True,
        alpha_init=0.2,
    )


def main() -> None:
    output_dir = Path("runs/wall_time_bench")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[BenchResult] = []
    kw = common_sac_kwargs()

    print(f"\nWall-time benchmark: {BENCH_STEPS} steps per agent, K=3 (AdCraft A1)")
    print("=" * 60)

    # Fixed-SAC (reference baseline — no reward shaping)
    print("\nFixed-SAC:")
    args_fixed = sac_fixed.Args(
        exp_name="wt_fixed",
        log_dir=str(output_dir / "tb_fixed"),
        **{k: v for k, v in kw.items() if k not in ("log_dir",)},
        cost_weights="1.0,1.0,1.0",
    )
    results.append(run_agent_bench("Fixed-SAC", sac_fixed.train, args_fixed))

    # Lag-SAC (multi)
    print("\nLag-SAC (multi):")
    args_lag = sac_lagrangian_multi.Args(
        exp_name="wt_lag",
        log_dir=str(output_dir / "tb_lag"),
        **{k: v for k, v in kw.items() if k not in ("log_dir",)},
        lambda_init=0.0,
        lambda_lr=1e-3,
    )
    results.append(run_agent_bench("Lag-SAC", sac_lagrangian_multi.train, args_lag))

    # TCL-SAC(B)
    print("\nTCL-SAC(B):")
    args_tcl = sac_tcl.Args(
        exp_name="wt_tcl",
        log_dir=str(output_dir / "tb_tcl"),
        **{k: v for k, v in kw.items() if k not in ("log_dir",)},
        thresholds="0.0,0.0,0.0",
        betas_init="10,10,10",
        betas_final="",
        beta_schedule="linear",
        beta_anneal_steps=0,
        reward_shift=0.0,
    )
    results.append(run_agent_bench("TCL-SAC(B)", sac_tcl.train, args_tcl))

    # HPRS-SAC
    print("\nHPRS-SAC:")
    args_hprs = sac_hprs.Args(
        exp_name="wt_hprs",
        log_dir=str(output_dir / "tb_hprs"),
        **{k: v for k, v in kw.items() if k not in ("log_dir",)},
        thresholds="0.0,0.0,0.0",
        base_weight=1.0,
        weight_decay=1.0,
    )
    results.append(run_agent_bench("HPRS-SAC", sac_hprs.train, args_hprs))

    # Compute overhead vs Fixed-SAC
    fixed_sps = next(r.steps_per_sec for r in results if r.agent == "Fixed-SAC")
    for r in results:
        r.overhead_vs_fixed = (fixed_sps - r.steps_per_sec) / fixed_sps * 100.0

    # Console table
    print("\n=== Wall-time benchmark results ===")
    print(f"{'Agent':>14} {'steps/sec':>10} {'overhead vs Fixed':>18}")
    for r in results:
        ovh = f"+{r.overhead_vs_fixed:.1f}% slower" if r.overhead_vs_fixed and r.overhead_vs_fixed > 0 else "reference"
        print(f"{r.agent:>14} {r.steps_per_sec:>10.0f} {ovh:>18}")

    # CSV
    csv_path = output_dir / "wall_time.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent", "total_steps", "wall_sec", "steps_per_sec", "overhead_pct_vs_fixed"])
        for r in results:
            w.writerow([r.agent, r.total_steps, f"{r.wall_sec:.2f}",
                        f"{r.steps_per_sec:.1f}", f"{r.overhead_vs_fixed:.2f}"])
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()

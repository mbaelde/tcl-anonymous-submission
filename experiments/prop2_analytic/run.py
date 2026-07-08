"""Analytic validation of Proposition 2.

Replaces SAC with the closed-form inner-loop optimal policy a*(lambda) = 1 - lambda
(derivative of the reward r = a - 0.5 a^2 set to zero given the Lagrangian
penalty -lambda * cost, with cost = a - b(phi)). This matches exactly the
assumption used in the proof of Prop 2: perfect timescale separation between
a fast inner-loop policy and a slow outer-loop dual ascent.

For each (alpha, omega, seed) cell, the script integrates the discrete system

    a_t      = clip(1 - lambda_t, 0, 1)
    cost_t   = a_t - (b0 + A sin(phi_t))
    lambda_{t+1} = max(0, lambda_t + alpha * cost_t)

for total_timesteps and dumps the lambda(t) trajectory in the same npz layout
as agents.sac_lagrangian, so experiments.prop2_validation.analyze can be used
as-is on the resulting runs/ tree.

Usage:
    py -3.14 -m uv run python -m experiments.prop2_analytic.run \\
        --config experiments/prop2_validation/config.yaml \\
        --output-dir runs/prop2_analytic
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def run_cell(
    *,
    alpha: float,
    omega: float,
    seed: int,
    horizon: int,
    b0: float,
    amplitude: float,
    total_timesteps: int,
    random_phase_at_reset: bool,
    dump_path: Path,
) -> None:
    rng = np.random.default_rng(seed)

    steps = np.empty(total_timesteps, dtype=np.int64)
    traj_lambda = np.empty(total_timesteps, dtype=np.float64)
    traj_cost = np.empty(total_timesteps, dtype=np.float64)

    lam = 0.0
    phase0 = float(rng.uniform(0.0, 2.0 * math.pi)) if random_phase_at_reset else 0.0
    step_in_episode = 0

    for t in range(total_timesteps):
        phi = (phase0 + omega * step_in_episode) % (2.0 * math.pi)
        a = max(0.0, min(1.0, 1.0 - lam))
        cap = b0 + amplitude * math.sin(phi)
        cost = a - cap

        steps[t] = t
        traj_lambda[t] = lam
        traj_cost[t] = cost

        lam = max(0.0, lam + alpha * cost)

        step_in_episode += 1
        if step_in_episode >= horizon:
            step_in_episode = 0
            if random_phase_at_reset:
                phase0 = float(rng.uniform(0.0, 2.0 * math.pi))

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dump_path,
        steps=steps,
        lam=traj_lambda,
        cost=traj_cost,
        omega=np.array(omega),
        lambda_lr=np.array(alpha),
        amplitude=np.array(amplitude),
        b0=np.array(b0),
        horizon=np.array(horizon),
        seed=np.array(seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    cli = parser.parse_args()

    with cli.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
    horizon = int(env_cfg["horizon"])
    b0 = float(env_cfg["b0"])
    amplitude = float(env_cfg["amplitude"])
    random_phase_at_reset = bool(env_cfg["random_phase_at_reset"])
    total_timesteps = int(agent_cfg["total_timesteps"])

    alphas = [float(a) for a in cfg["grid"]["alphas"]]
    omegas_ppe = [float(o) for o in cfg["grid"]["omegas_periods_per_episode"]]
    seeds = [int(s) for s in cfg["grid"]["seeds"]]

    cells: list[tuple[float, float, float, int, Path]] = []
    for alpha in alphas:
        for omega_ppe in omegas_ppe:
            omega = 2.0 * math.pi * omega_ppe / horizon
            for seed in seeds:
                cell_dir = (
                    cli.output_dir
                    / f"alpha={alpha:.0e}"
                    / f"omega_ppe={omega_ppe:g}"
                    / f"seed={seed}"
                )
                cells.append((alpha, omega_ppe, omega, seed, cell_dir))

    print(f"Running {len(cells)} cells (analytic policy, single process)")

    for alpha, omega_ppe, omega, seed, cell_dir in tqdm(cells, desc="prop2_analytic"):
        cell_dir.mkdir(parents=True, exist_ok=True)
        dump_path = cell_dir / "traj.npz"
        if cli.skip_existing and dump_path.exists():
            continue
        run_cell(
            alpha=alpha,
            omega=omega,
            seed=seed,
            horizon=horizon,
            b0=b0,
            amplitude=amplitude,
            total_timesteps=total_timesteps,
            random_phase_at_reset=random_phase_at_reset,
            dump_path=dump_path,
        )


if __name__ == "__main__":
    main()

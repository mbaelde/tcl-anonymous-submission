# tcl-code

Reference implementation and experiments for the paper

> **Threshold-Cascaded Lexicographic Rewards for Multi-Constrained RL in Real-Time Bidding**
> Anonymous Authors — Anonymous Institution

## Overview

This repository contains:

- CleanRL-style single-file SAC agent implementations for constrained RL.
- Shared library `tcl/` for environments, reward modules, and utilities.
- `experiments/` with reproducible configs and runners for every empirical claim in the paper.

## Agents

| File | Description |
|---|---|
| `agents/sac_lagrangian.py` | RCPO-style scalar Lagrangian (K=1, §4.3 / Prop 2) |
| `agents/sac_lagrangian_multi.py` | Multi-constraint Lagrangian, K-generic (§7.1 baseline) |
| `agents/sac_tcl.py` | TCL Formulation (B): shaped reward `r·Π σ(β(τ−g))` |
| `agents/sac_tcl_standalone.py` | TCL Formulation (A): standalone reward `Σ w·R_k` |
| `agents/sac_fixed.py` | Fixed weighted linear combination baseline |
| `agents/sac_hprs.py` | Hierarchical Potential-based Reward Shaping (Berducci et al., 2024) |
| `agents/sac_pid_lagrangian.py` | PID-Lagrangian dual update (Stooke et al., 2020) |

## Environments

### `SinusoidalCMDP`

Toy single-constraint non-stationary CMDP used to validate Proposition 2
(dual-oscillation amplitude $\mathcal{O}(\alpha/\omega)$).

### `MultiConstraintAdCraftLaplacian` — §7.1 target environment

Pure-Python reimplementation of the AdCraft SEM benchmark faithful to
§G.1 + Table 1 of Gomrokchi et al. (arXiv:2306.11971). Replaces the
upstream `Mikata-Project/adcraft` package (abandoned Aug 2023, deviates
from the paper — see [`docs/adcraft_laplacian_rationale.md`](docs/adcraft_laplacian_rationale.md)).

```python
from tcl.envs import MultiConstraintAdCraftLaplacian

env = MultiConstraintAdCraftLaplacian(
    num_keywords=100,
    budget=100.0,
    bid_max=3.0,
    pricing_mode="second",  # "second" = paper default, "first" = Anonymous Institution prod
)
```

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `num_keywords` | 100 | Number of keywords (§G.1: 100) |
| `budget` | 100.0 | Per-step spend cap |
| `bid_max` | 3.0 | Max bid (§G.1 grid: [0.01, 3.00]) |
| `pricing_mode` | `"second"` | `"second"` (pays c) or `"first"` (pays b) |
| `target_utilization` | 0.5 | Budget-utilization floor for c₁ |
| `target_ctr` | 0.05 | CTR floor for c₂ |
| `target_margin` | -0.5 | Margin floor for c₃ (revenue_share) |
| `updater_params` | vol/ctr/cvr 3%/day | Non-stationarity drift spec |

Action space: `Box(0, bid_max, shape=(num_keywords,))`.
Observation space: `Box(-inf, inf, shape=(5·num_keywords + 2,))`.
`info["costs"]`: shape `(3,)`, convention `c_k > 0` ⟺ constraint k violated.

### `ConstrainedPendulum`

`Pendulum-v1` augmented with a single angle constraint `g_1 = |θ| − θ_max`.
Used to validate TCL generalisation beyond the AdCraft simulator.

```python
from tcl.envs import ConstrainedPendulum

env = ConstrainedPendulum(theta_max=0.5)  # ~28 degrees
```

### `MultiConstraintAdCraft` (legacy)

Wrapper around the upstream Rust-backed `adcraft` package. Retained for
backward compatibility. **Not recommended for new experiments.**

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

On Windows, prefix all `uv run` commands with `PYTHONUTF8=1 py -3.14 -m`:

```bash
PYTHONUTF8=1 py -3.14 -m uv run python -m experiments.<name>.run ...
```

## Experiments

All runners accept `--parallel N` (fan out across N worker processes, each
pinned to one PyTorch thread) and `--skip-existing` to resume interrupted sweeps.

| Experiment | Paper claim | Cells | Command |
|---|---|---|---|
| `prop2_validation` | Prop 2: dual-oscillation ∝ α/ω | 60 | see below |
| `prop2_analytic` | Prop 2: closed-form validation | — | see below |
| `prop5_validation` | Prop 5: Formulation (A) vs (B) at β→∞ | 24 | see below |
| `pilot_adcraft` | §7.1: four baselines on AdCraft K=3 | 12 | see below |
| `k_scaling_validation` | §7.1: K-scaling (K=1,2,3) | 36 | see below |
| `beta_schedule` | §7.2: β-annealing schedule comparison | 9 | see below |
| `beta_star_scan` | §7.1.4 Prop 4: empirical β* localization | 21 | see below |
| `tau_sensitivity` | §7.1: threshold τ sensitivity | 36 | see below |
| `wall_time_bench` | Computational overhead vs baselines | 4 | see below |
| `pid_lagrangian_bench` | Stooke 2020 vs TCL vs Lag on AdCraft | 9 | see below |
| `constrained_pendulum_bench` | TCL generalisation on Pendulum (non-AdCraft) | 9 | see below |

```bash
# Proposition 2 — dual-oscillation amplitude vs (alpha, omega)
uv run python -m experiments.prop2_validation.run \
    --config experiments/prop2_validation/config.yaml --parallel 4

# Proposition 2 — closed-form analytic check
uv run python -m experiments.prop2_analytic.run

# Proposition 5 — Formulation A vs B at high beta (B1 env, drift=0.01)
uv run python -m experiments.prop5_validation.run \
    --config experiments/prop5_validation/config.yaml --parallel 8

# Section 7.1 — four baselines on AdCraft K=3
uv run python -m experiments.pilot_adcraft.run \
    --config experiments/pilot_adcraft/config.yaml --parallel 4

# Section 7.1 — K-scaling (K=1, K=2, K=3)
uv run python -m experiments.k_scaling_validation.run \
    --config experiments/k_scaling_validation/config.yaml --parallel 6

# Section 7.2 — beta-annealing schedule comparison
uv run python -m experiments.beta_schedule.run \
    --config experiments/beta_schedule/config.yaml --parallel 3

# Section 7.1.4 — empirical beta* localization (loss-budget regime)
uv run python -m experiments.beta_star_scan.run \
    --config experiments/beta_star_scan/config.yaml --parallel 7

# Section 7.1 — threshold tau sensitivity (TCL vs Lag, 6 tau values)
uv run python -m experiments.tau_sensitivity.run \
    --config experiments/tau_sensitivity/config.yaml --parallel 6

# Computational overhead — steps/sec for all agents on AdCraft K=3
uv run python -m experiments.wall_time_bench.run

# PID-Lagrangian baseline (Stooke 2020) vs TCL vs Lag on AdCraft A1
uv run python -m experiments.pid_lagrangian_bench.run \
    --config experiments/pid_lagrangian_bench/config.yaml --parallel 9

# Generalisation — TCL vs PID-Lag vs Lag on ConstrainedPendulum (K=1)
uv run python -m experiments.constrained_pendulum_bench.run \
    --config experiments/constrained_pendulum_bench/config.yaml --parallel 9
```

## Tests

```bash
uv run pytest
```

## Layout

```
tcl-code/
├── tcl/                    # Library: envs, rewards, utils
│   ├── envs/               # AdCraftLaplacian, ConstrainedPendulum, SinusoidalCMDP, ...
│   ├── rewards/
│   └── utils/
├── agents/                 # SAC agent implementations
│   ├── common/             # Shared modules (networks, buffer, utils)
│   ├── sac_tcl.py          # TCL Formulation (B) — main agent
│   ├── sac_tcl_standalone.py  # TCL Formulation (A)
│   ├── sac_lagrangian.py      # Scalar Lagrangian K=1 (Prop 2)
│   ├── sac_lagrangian_multi.py  # Multi-constraint Lagrangian (K-generic)
│   ├── sac_pid_lagrangian.py    # PID-Lagrangian (Stooke 2020)
│   ├── sac_fixed.py
│   └── sac_hprs.py
├── experiments/            # Reproducible experiment configs and runners
│   └── ROADMAP.md          # Opus review items and execution plan
├── tests/
└── figures/                # Generated figures (not checked in)
```

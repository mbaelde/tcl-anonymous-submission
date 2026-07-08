# Pilot AdCraft — section 7.1 baselines

## What this resolves

Section 7.1 of `TCL_paper_draft.md`: validates that the four shaped-reward
agents (Lagrangian-multi, Fixed-weights, **TCL**, HPRS) all *learn* on the
K=3 MultiConstraintAdCraft env, and gives the headline comparison numbers
that anchor the paper's empirical narrative. The pilot precedes the
schedule-sweep (§4.3, `experiments/beta_schedule/`) and the prop-2
validation (§A.2, `experiments/prop2_validation/`).

## Claim under test (informal)

On the same K=3 constrained MDP (utilization / CTR / margin shortfalls,
canonical $c_k > 0$ = violation, $\tau_k = 0$), with an otherwise identical
SAC core, the four shaping schemes should rank predictably:

- **TCL** and **Lag-multi**: reach CSR ≥ 0.9 on all $K$ constraints,
  with comparable steady-state return.
- **Fixed-weights**: prone to under- or over-pressuring some constraint
  (constant cost weights cannot adapt to the env), so the worst-case
  $k$ should lag.
- **HPRS** (potential-based shaping, no constraint pressure): exists to
  separate "shaped reward improves credit assignment" from "shaping
  enforces constraints". Expected to track return well but underperform
  on CSR.

The pilot **does not** assert which constraint-aware scheme wins — that's
what §7.2/§7.3 are for. The pilot is here to (i) confirm wiring, (ii)
collect the baseline numbers, (iii) rule out env-side pathologies.

## Setup

- Env: `MultiConstraintAdCraft` (K=3), identical to the one used by
  `experiments/beta_schedule/` and `experiments/prop2_validation/`.
- Agents: `agents/sac_lagrangian_multi.py`, `agents/sac_fixed.py`,
  `agents/sac_tcl.py`, `agents/sac_hprs.py`. Shared SAC core.
- Per-cell artefacts: TensorBoard logs at
  `<output_dir>/<agent>/seed=<seed>/tb/` + a `result.txt` summary.

## Sweep grid (`config.yaml`)

| Knob               | Values                                       |
|--------------------|----------------------------------------------|
| agent              | `{lag_multi, fixed, tcl, hprs}`              |
| seed               | `{1, 2, 3}`                                  |
| total steps / cell | $60{,}000$                                   |

Total: **12 cells**. CPU-bound (Rust AdCraft sim + K=3 wrapper, `cuda: false`).
Per-cell wall-clock ≈ 25–35 min on the dev box → **≈ 5–7 h** sequentially,
**≈ 1.5–2 h** with `--parallel 4` on a 4+ physical-core machine. Each
worker pins `torch.set_num_threads(1)` to avoid OpenMP oversubscription.

## Suggested staging

1. **Smoke** (`config.smoke.yaml`, 4 agents × 1 seed × 1500 steps,
   $\le 3$ min) to verify the wiring and the TB scalar tags.
2. **Full grid** (4 agents × 3 seeds × 60k steps) once the smoke is
   convincing.

The intermediate "1 seed, full steps" pilot is skipped here: the smoke is
enough to catch wiring bugs and the full grid is cheap enough on a single
machine.

## Analyzer (`analyze.py`)

For each cell, the analyzer reads the TB event file and extracts:

- `rollout/episode_return` — task reward trajectory.
- `rollout/episode_cost_{k}` (TCL, Fixed, HPRS) **or** `rollout/episode_cost_k{k}`
  (Lag-multi — see note below) + `rollout/episode_steps`, used to compute
  per-episode **CSR**:
  $\text{CSR}_k(\text{ep}) = \mathbb{1}[\overline{c_k}(\text{ep}) \le \tau_k]$.
  Averaged over a sliding window for the plots.
- `dual/lambda_k{k}` (lag_multi only) — Lagrange multiplier trace.
- `train/beta_{k}` (tcl only) — sanity check that no annealing kicked in.

> **Tag naming note.** `sac_lagrangian_multi` historically emits its cost
> scalars under `rollout/episode_cost_k{k}` (k-prefix), while `sac_tcl`,
> `sac_fixed`, and `sac_hprs` emit `rollout/episode_cost_{k}` (no prefix).
> The analyzer transparently tries both names — do not "fix" this in the
> agent code without first re-running prior cells, since the TB tags are
> the only persistent record of past sweeps.

Outputs (under `<out-dir>/`):

- `summary.csv` — one row per cell with: steps-to-CSR≥0.9 per constraint,
  steady-state CSR (last 20 % of training), steady-state return.
- `return_vs_steps.png` — episode-return curves, one per (agent, seed).
- `csr_vs_steps.png` — per-constraint CSR$_k(t)$ smoothed, one curve per
  (agent, seed), $K$ subplots.
- `dual_traces.png` — $\lambda_k(t)$ for the lag_multi cells (sanity).

## Acceptance criteria

- All four agents complete the 60k-step budget without crashing on any
  seed (smoke catches wiring; full run catches numerical blow-ups).
- At least TCL **and** Lag-multi reach CSR ≥ 0.9 on all $K=3$ constraints
  in steady state, on a majority of seeds. If neither does, the env
  needs re-tuning before the §4.3 sweep is meaningful.
- The ranking on **steady-state return** and on **steady-state CSR** is
  consistent enough across seeds to support the §7.1 narrative claim
  (otherwise the seed budget is too small and we should bump to 5).

## Failure modes to watch for

- **Lag-multi** $\lambda_k$ saturates near zero on all $k$ → either the
  policy is already feasible (loosen $\tau_k$ or pick a harder env) or
  $\lambda_{lr}$ is too small (cost gradient never catches up). The
  `dual_traces.png` is the diagnostic.
- **Fixed-weights** wildly outperforms the adaptive baselines → the
  hand-picked `cost_weights` happen to match the env; rerun with a worse
  weight vector to confirm the gap is real.
- **HPRS** CSR matches TCL → the potential is leaking constraint info;
  audit the `phi(s)` definition.
- All four agents trace identical CSR curves → the constraint is trivially
  satisfied by random policies; tighten the env targets.

## Diagnostic results — 2026-05-17

The initial 12-cell pilot **and** the 9-cell `beta_schedule` run both
showed `CSR_k0 = 0` (utilization) across every (agent, seed). The 4-cell
diagnostic (`diagnostic.py`, `config.diagnostic.yaml`, ~3h10 wall-clock,
parallel=8) isolates the cause:

| Cell                  | timesteps | constraints active | λ_k0 final | λ_k1 final | λ_k2 final |
|-----------------------|----------:|--------------------|----------:|----------:|----------:|
| A_canonical_short     | 60k       | all                | ≈46       | 0         | 47–95     |
| **B_canonical_long**  | **250k**  | **all**            | **≈198**  | **≈0**    | **202–275** |
| C_pacing_only_long    | 250k      | pacing only        | ≈197      | 0         | 0         |
| D_constant_bid_sweep  | —         | —                  | util_max=14% at bid=10 | csr_ctr=100% | margin always negative |

**Verdict.**
- **H1 (horizon too short) — invalidated.** Going from 60k → 250k *raises*
  λ_k0 (46 → 198) instead of resorbing it. SAC learns the dual pressure;
  it cannot satisfy it.
- **H2 (target spec wrong) — partially valid.** `target_ctr=0.001` is met
  trivially (λ_k1 = 0 across all cells). But `target_utilization=0.8`
  and `target_margin=0.70 (revenue_share)` are infeasible under the
  current env physics.
- **H3 (env physics) — confirmed.** Cell C strips ctr/margin and runs
  pacing alone for 250k steps; λ_k0 still saturates at ≈197. Cell D
  shows that no constant bid (0.5 → 10) reaches `util > 14%` or
  `margin > 0` — the constrained MDP is **infeasible** at the canonical
  `(budget=1000, bid_max=10, num_keywords=10)` config.

**Consequence.** The original pilot's `CSR_k0 = 0` is a property of the
env, not of the agents. Before re-running the pilot, the env needs to
be re-calibrated to a regime where at least the `B` cell's `λ_k0`
saturation is broken — e.g. lower `target_utilization` to ~0.10–0.12
(matches D's reachable floor), or scale `budget` / `bid_max` up so 0.8
is physically attainable. The choice will be made after a quick audit
of the wrapper's bid → impression → revenue chain.

Data: `runs/pilot_adcraft_diagnostic/` (10 `result.txt` + TB events).
Commits: `d014617` (prop2_analytic), `168b77e` (wrapper margin_formula
parameter + diagnostic plumbing).

## Calibrated re-run — 2026-05-17 (end of day)

After the diagnostic, a constant-bid sweep (`calibrate.py`, 49 combos × 2
episodes, 42 s) identified `budget=150, bid_max=10` as the sweet spot:
util reachable at `bid≥5` (~0.74–0.78), margin reachable at `bid≤1`
(~−0.84 to −2.4), no constant bid satisfies both → heterogeneous
per-keyword bidding required. Pilot re-run on `config.calibrated.yaml`
(targets `util=0.7 / ctr=0.001 / margin=−3.0 (revenue_share)`, 4 agents
× 3 seeds × 60k steps, parallel=8, 1h08 wall-clock):

| Constraint  | CSR steady mean (12 cells) | Verdict                              |
|-------------|---------------------------:|--------------------------------------|
| k0 (util)   | 0.0                        | binding, never reached               |
| k1 (CTR)    | ≈ 1.0                      | trivially met (as expected)          |
| k2 (margin) | ≈ 1.0                      | easily met (target −3.0 is generous) |

`λ_k0` for `lag_multi` rises **linearly** from 0 to 35–38 across 60k
steps (vs 198 saturated in the pre-calibration diagnostic) → the dual
pressure is healthy, but the policy does not move enough in 60k steps to
push utilization up. All 4 agents settle on the same "economic"
equilibrium (low bids → margin satisfied, util sacrificed).

**Verdict.** Env is now physically sound, but the §7.1 narrative
(TCL / Lag-multi reach CSR ≥ 0.9 on all $K$) is **not yet validated**.
Three hypotheses to iterate on:

1. `target_utilization=0.7` still too ambitious (sweep showed util max
   ~0.78 at `bid=10` uniformly — narrow margin).
2. `lambda_lr=1e-3` too weak (λ_k0 only reaches 35 at 60k → not enough
   pressure to dominate the task reward).
3. `total_timesteps=60k` too short for the heterogeneous-bid tradeoff.

Data: `runs/pilot_adcraft_calibrated/` + `figures/pilot_adcraft_calibrated/`
(`csr_vs_steps.png`, `dual_traces.png`, `return_vs_steps.png`,
`summary.csv`).

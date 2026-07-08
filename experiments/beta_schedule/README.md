# Beta-schedule comparison — section 4.3 / 6.2

## What this resolves

The `%%NON RESOLU%%` block in `TCL_paper_draft.md` §4.3: the paper describes an
**increasing** $\beta$ schedule (soft → strict cascade) while the production
RTB code (`reward.py`) implements a **decreasing** one (graceful relaxation of
the hierarchy during fine-tuning). This experiment compares both regimes on the
same K=3 constrained MDP used by the §7.1 pilot, plus a fixed-$\beta$ baseline.

## Claim under test (informal)

For a multi-constraint TCL shaped reward

$$
r_\text{TCL}(s, a) \;=\; r(s, a) \cdot \prod_{k=1}^{K}
\sigma\!\bigl(-\beta_k(t) \, (c_k(s,a) - \tau_k)\bigr),
$$

with $\beta_k(t)$ interpolated linearly over the first
`beta_anneal_steps` of training, the three regimes

- **increasing** $\beta: 2 \to 10$  (paper, cascade tightening),
- **decreasing** $\beta: 10 \to 2$  (RTB-prod, hierarchy relaxation),
- **fixed**      $\beta = 10$       (baseline, no annealing),

should differ in (a) constraint-satisfaction speed and (b) residual variance
of the constraint-satisfaction rate (CSR) at steady state. The paper's
prediction is that **increasing** dominates (soft early gradient + strict
late enforcement); the prod-code design is bet on **decreasing** (strict
prior helps the policy learn the hierarchy, then relaxation gives slack for
fine-tuning). The fixed baseline isolates the contribution of annealing
itself.

## Setup

- Env: `MultiConstraintAdCraft` (K=3, $\tau_k = 0$, canonical $c_k > 0$ =
  violation), identical to the §7.1 pilot. Constraints: utilization
  shortfall, CTR target shortfall, margin target shortfall.
- Agent: `agents/sac_tcl.py` (TCL only). Shared SAC core across all cells.
- Per-cell artefacts: TensorBoard logs at `<output_dir>/<schedule>/seed=<seed>/tb/`
  + a `result.txt` summary.

## Sweep grid (`config.yaml`)

| Knob               | Values                                       |
|--------------------|----------------------------------------------|
| schedule           | `{increasing, decreasing, fixed}`            |
| seed               | `{1, 2, 3}`                                  |
| total steps / cell | $60{,}000$                                   |
| anneal window      | $40{,}000$ (first 2/3 of training)           |

Total: **9 cells**. CPU-bound (Rust AdCraft sim + K=3 wrapper, `cuda: false`).
Per-cell wall-clock should be in the same ballpark as the §7.1 pilot TCL cell
(~25–35 min on the dev box) → **≈ 4 h** sequentially, **≈ 1–1.5 h** with
`--parallel 3` on a 3+ physical-core machine (one worker per schedule).

## Suggested staging

1. **Smoke** (`config.smoke.yaml`, 3 schedules × 1 seed × 1500 steps,
   $\le 2$ min) to verify the wiring and the TB scalar tags.
2. **Pilot** (3 schedules × 1 seed × full 60k steps, ~1 h) to confirm the
   regime separation is visible before paying for seed replicates.
3. **Full grid** (3 schedules × 3 seeds × 60k steps) once the pilot is
   convincing.

## Analyzer (`analyze.py`)

For each cell, the analyzer reads the TB event file and extracts:

- `rollout/episode_return` — task reward trajectory.
- `rollout/episode_cost_{k}`, $k \in \{0, 1, 2\}$ — per-constraint episodic
  cost, used to compute per-episode **CSR**:
  $\text{CSR}_k(\text{ep}) = \mathbb{1}[\overline{c_k}(\text{ep}) \le \tau_k]$.
  Averaged over a sliding window for the plots.
- `train/beta_{k}` — confirms the schedule shape.

Outputs (under `<output_dir>/analysis/`):

- `summary.csv` — one row per cell with: steps-to-CSR≥0.9 per constraint,
  steady-state CSR (last 20 % of training), residual CSR variance.
- `csr_vs_steps.png` — per-constraint CSR$_k(t)$ smoothed, one curve per
  schedule × seed.
- `beta_traces.png` — $\beta_k(t)$ across the three regimes (sanity check
  that the schedules look as configured).

## Acceptance criteria

- The three $\beta_k(t)$ traces visually match their config (sanity).
- All three regimes reach CSR ≥ 0.9 on all $K=3$ constraints within the
  60k-step budget (else the env is broken, not the schedule).
- The ranking on **steady-state CSR variance** is consistent across seeds
  (i.e. the choice of regime has a *replicable* effect, even if small).

The experiment **does not** assert which regime wins — the verdict feeds
back into the paper §4.3 wording.

## Failure modes to watch for

- All three regimes give identical trajectories → either annealing is
  inactive (check `train/beta_k` traces) or the constraint is already
  satisfied at $\beta=2$ (loosen $\tau$ or pick a harder env).
- Decreasing regime collapses CSR late in training: indicates the strict
  prior was load-bearing — argues for the paper's increasing schedule
  even though prod uses the opposite.
- Increasing regime never reaches steady state: anneal window too long
  relative to `total_timesteps`; shorten `beta_anneal_steps`.

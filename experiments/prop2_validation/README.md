# Proposition 2 — SAC sweep (negative result, kept as ablation)

## Status (2026-05-17)

**This module does NOT validate Proposition 2.** The canonical validation
of Prop 2 lives in [`prop2_analytic/`](../prop2_analytic/README.md) and
achieves ratio = 1.00 ± 0.02. This module is preserved as an ablation
showing what happens when the timescale-separation hypothesis (H4) is
violated.

## Claim under test

In a single-constraint CMDP with sinusoidally non-stationary budget cap
$b(\phi_t) = b_0 + A \sin(\omega t)$, the Lagrangian dual variable
$\lambda(t)$ trained by primal-dual gradient ascent at rate $\alpha$
oscillates in steady state with peak-to-peak amplitude

$$
\Delta\lambda(\alpha, \omega) \;=\; \frac{2 \alpha A}{\sqrt{\alpha^2 + \omega^2}}
\;\approx\; \frac{2 \alpha A}{\omega} \qquad (\omega \gg \alpha).
$$

## Why this protocol fails

Three assumptions of the proof are broken by SAC:

1. **No timescale separation.** $\lambda$ is updated at every gradient
   step, same cadence as the actor/critic. The proof assumes the inner
   policy reaches $a^\star(\lambda) = \arg\max_a (r - \lambda c)$
   instantly between dual updates.
2. **Stochastic policy.** SAC adds an entropy bonus, so the realized
   action distribution is not $\delta_{a^\star(\lambda)}$ even at
   convergence. The expected cost gradient has an extra entropy term
   not present in the linearization.
3. **ReLU projection.** $\lambda \gets \max(0, \lambda + \alpha \,
   \mathrm{cost})$ is non-smooth; near $\lambda = 0$ the linearization
   used in the proof breaks.

Empirically we observed amplitude ratios in $[0.3, 2.5]$ across the
60-cell grid (4 $\alpha$ × 5 $\omega$ × 3 seeds), with no clean collapse
onto the theoretical curve.

## Setup (preserved as-is)

- Env: `SinusoidalCMDP(horizon=144, b0=0.5, amplitude=0.2,
  random_phase_at_reset=True)`.
- Agent: `agents/sac_lagrangian.py` with default SAC core, learnable
  dual variable $\lambda$ updated at every gradient step.
- Per-cell artefact: `traj.npz` containing `lam[t], cost[t]` for the
  full trajectory plus the cell's hyperparameters.
- Grid: $\alpha \in \{10^{-3}, 3\cdot10^{-3}, 10^{-2}, 3\cdot10^{-2}\}$,
  $\omega_\mathrm{ppe} \in \{0.5, 1, 2, 4, 8\}$, 3 seeds, 60k steps.

## Analyzer (`analyze.py`)

Quantile peak-to-peak $q_{0.98} - q_{0.02}$ on the last 40 % of
$\lambda(t)$, vs theoretical $2\alpha A / \sqrt{\alpha^2 + \omega^2}$.
Reused by `prop2_analytic/` (same data layout).

## Ablations preserved here

- `config.long_horizon.yaml`: same grid, horizon=10000,
  `random_phase_at_reset=true`. Demonstrates that even with the SAC
  agent removed (via the analytic protocol), random phase resets inject
  transients of duration $\sim 1/\alpha$ that inflate the peak-to-peak
  estimator unless $\mathrm{horizon} \gg 1/\alpha$.
- `analyze_fft.py`: FFT-demodulation analyzer that isolates the
  $\omega$-component. Works only on phase-coherent trajectories;
  destroyed by phase resets.

## Where Prop 2 is actually validated

See [`experiments/prop2_analytic/`](../prop2_analytic/). The closed-form
policy $a^\star(\lambda) = \mathrm{clip}(1 - \lambda, 0, 1)$ realizes the
H4 timescale-separation assumption exactly, and the empirical amplitude
matches theory to numerical precision (ratio 1.00 ± 0.02 on all
$\omega_\mathrm{ppe} \geq 1$ cells, 60 cells in ~6 s).

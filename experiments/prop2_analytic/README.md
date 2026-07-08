# Proposition 2 — analytic validation (Option A)

## Claim under test

Same as `experiments/prop2_validation/`: in a single-constraint CMDP with
sinusoidally non-stationary budget cap $b(\phi_t) = b_0 + A \sin(\omega t)$,
the Lagrangian dual variable $\lambda(t)$ trained by primal-dual gradient
ascent at rate $\alpha$ oscillates in steady state with peak-to-peak
amplitude

$$
\Delta\lambda(\alpha, \omega) \;=\; \frac{2 \alpha A}{\sqrt{\alpha^2 + \omega^2}}
\;\approx\; \frac{2 \alpha A}{\omega} \qquad (\omega \gg \alpha).
$$

## Why a separate "analytic" module

The full SAC version (`prop2_validation/`) does **not** validate the
proposition: SAC violates the timescale-separation hypothesis (H4) used
in the proof. The policy is stochastic (entropy bonus), $\lambda$ moves
on the same timescale as the actor, and the ReLU projection on $\lambda$
breaks the linearization. Empirically we observed amplitude ratios in
$[0.3, 2.5]$ that did not collapse onto a clean curve.

This module replaces SAC with the closed-form inner-loop optimal policy

$$
a^\star(\lambda) \;=\; \mathrm{clip}(1 - \lambda, 0, 1)
$$

(derivative of $r - \lambda c$ set to zero, with $r = a - \tfrac{1}{2}a^2$
and $c = a - b(\phi)$). This realizes exactly the timescale-separation
assumption used in the proof, so the empirical dynamics should match the
theory to numerical precision. 60 cells run in ~6 s on a laptop.

## Canonical setup

Defined in `config.yaml`:

- Env: `SinusoidalCMDP(horizon=144, b0=0.5, amplitude=0.2,
  random_phase_at_reset=false)` — **deterministic phase** (see "Why
  fixed phase" below).
- Dynamics: $\lambda_{t+1} = \max\!\left(0,\; \lambda_t + \alpha \,(a^\star(\lambda_t) - b(\phi_t))\right)$.
- 60 cells: $\alpha \in \{10^{-3}, 3{\cdot}10^{-3}, 10^{-2}, 3{\cdot}10^{-2}\}$,
  $\omega_\mathrm{ppe} \in \{0.5, 1, 2, 4, 8\}$, 3 seeds, 60k steps each.

```
py -3.14 -m uv run python -m experiments.prop2_analytic.run \
    --config experiments/prop2_analytic/config.yaml \
    --output-dir runs/prop2_analytic
py -3.14 -m uv run python -m experiments.prop2_validation.analyze \
    --run-dir runs/prop2_analytic \
    --out-dir figures/prop2_analytic
```

## Why fixed phase (`random_phase_at_reset=false`)

With `random_phase_at_reset=true`, every episode boundary (every 144
steps) reinjects the forcing phase uniformly in $[0, 2\pi)$ while
preserving $\lambda$. Each reset triggers a relaxation transient of
duration $\sim 1/\alpha$ in the dual trajectory. The robust peak-to-peak
amplitude estimator $q_{0.98} - q_{0.02}$ picks up those transients and
reports values **inflated** by a factor that depends on
$\alpha \cdot \mathrm{horizon}$:

| $\alpha$ | $1/\alpha$ | ratio with phase resets |
|---|---|---|
| $10^{-3}$ | 1000 | $\approx 1.86$ (transients dominate) |
| $3\cdot10^{-3}$ | 333 | $\approx 1.81$ |
| $10^{-2}$ | 100 | $\approx 1.63$ (relaxes just before reset) |
| $3\cdot10^{-2}$ | 33 | $\approx 1.30$ (near-stationary) |

This is not a violation of Prop 2 — the **scaling** $\propto
1/\sqrt{\alpha^2 + \omega^2}$ is respected — but the multiplicative
constant is polluted. Two equivalent fixes:

1. **Fixed phase** (canonical here): turn off the phase reset.
2. **Long horizon**: make horizon $\gg 1/\alpha$ so transients decay
   before the next reset. Tested with horizon $= 10{,}000$ in
   `prop2_validation/config.long_horizon.yaml`; ratio $\to 1.00$ for
   $\alpha \geq 3\cdot10^{-3}$, residual $\sim 1.20$ for $\alpha = 10^{-3}$
   ($1/\alpha = 0.1 \times$ horizon, transients not fully decayed).

We choose (1) because it is cheaper to read in the paper and removes any
free parameter.

## Acceptance criteria (achieved 2026-05-17)

- Ratio $\hat{\Delta\lambda} / \Delta\lambda_\mathrm{theory}$
  $\in [0.99, 1.02]$ on $\omega_\mathrm{ppe} \geq 1$ for every
  $(\alpha, \omega, \mathrm{seed})$. Std across seeds is numerically 0
  (the policy is deterministic and the forcing is identical).
- $\omega_\mathrm{ppe} = 0.5$ remains at ratio $\sim 0.21$–$0.30$:
  expected, the horizon captures only half a period, so peak-to-peak
  measures only half the swing. Kept in the figure for completeness;
  excluded from the acceptance band.

## Sample row (`figures/prop2_analytic/summary.csv`)

```
alpha,omega_ppe,seed,A_emp,A_theory,ratio
0.001,1.0,1,0.00916,0.00916,1.0000
0.001,8.0,1,0.001135,0.001146,0.9908
0.03,2.0,1,0.131667,0.130040,1.0125
```

## Ablations preserved

- `prop2_validation/config.long_horizon.yaml`: same grid, horizon=10000,
  `random_phase_at_reset=true`. Used to show that long-horizon also
  recovers ratio = 1 (alternative to fixed phase).
- `prop2_validation/analyze_fft.py`: FFT-demodulation analyzer at the
  forcing frequency $\omega$. Works only on phase-coherent data; recovers
  ratio $\sim 1$ on this run, $\sim 0.03$–$0.38$ on the original
  random-phase data (the omega-component is destroyed by phase resets).

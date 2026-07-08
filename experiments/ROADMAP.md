# TCL Paper — Experiments Roadmap

Hiérarchie Opus (session 2026-05-25). Items critiques TMLR en premier.

---

## Statuts

| Item | Titre | Criticité | Statut | Script |
|------|-------|-----------|--------|--------|
| 1 | β*-scan autour de β* | Critique | **en cours** | `beta_star_scan/` |
| 2 | PID-Lagrangian (Stooke 2020) | Critique | À faire | `pid_lagrangian/` (design ci-dessous) |
| 3 | Env hors AdCraft (Safety Gym) | Critique | À faire | `safety_gym_bench/` (design ci-dessous) |
| 4 | Prop. 5 validation (β→∞, gap A↔B→0) | Forte | À faire | `prop5_validation/` |
| 5 | 5–10 seeds + Wilcoxon | Forte | À faire | Modifier configs existantes |
| 6 | K-scaling (K∈{2,3,5,8,10}) | Forte | ✅ Terminé | `k_scaling_validation/` |
| 7 | Sensibilité τ_k (±10%, ±20%) | Utile | À faire | `tau_sensitivity/` |
| 8 | Trace prod Anonymous Institution (A-B test) | Utile | N/A (données prod) | — |
| 9 | Wall-time / FLOPs overhead | Utile | À faire | `wall_time_bench/` |

---

## Item 1 — β*-scan (en cours)

**Commande :**
```bash
PYTHONUTF8=1 py -3.14 -m uv run python -m experiments.beta_star_scan.run \
    --config experiments/beta_star_scan/config.yaml --parallel 7
```

**Analyse (après fin) :**
```bash
PYTHONUTF8=1 py -3.14 -m uv run python -m experiments.beta_star_scan.run \
    --config experiments/beta_star_scan/config.yaml --analyze-only
```

**Livrables :** `runs/beta_star_scan/beta_star_scan.pdf`, `beta_star_scan.csv`
**Paper :** §7.1.4 après ligne ~699 + update β* ∈ [3.5,10] → valeur empirique

---

## Item 4 — Prop. 5 validation (β→∞)

**Hypothèse :** Prop. 5 prédit que les politiques optimales (A) et (B) coïncident sur Π_det quand β→∞.
**Protocole :** Entraîner TCL-SAC(B) et TCL-SAC(A) à β ∈ {10, 30, 100, 300} sur B1 (drift=0.01,
env facilement faisable). Tracer return et CSR_c0 vs β pour les deux formulations. Montrer que
le gap |return_A − return_B| → 0.

**Commande :**
```bash
PYTHONUTF8=1 py -3.14 -m uv run python -m experiments.prop5_validation.run \
    --config experiments/prop5_validation/config.yaml --parallel 8
```

**Livrables :** `runs/prop5_validation/prop5.pdf`, `prop5.csv`
**Paper :** §7.1.x ou Appendice A.5 — validation empirique Prop. 5

---

## Item 7 — Sensibilité τ_k

**Hypothèse :** La gate cascade locale en (s,a) devrait être plus robuste à une mauvaise
calibration des seuils que le Lagrangian (dont l'intégrateur dérive sans borne).
**Protocole :** Multiplier τ_util ∈ {0.65, 0.70, 0.75, 0.80, 0.85, 0.90} sur A1 (drift=0.03).
Comparer TCL-SAC(B) vs Lag-SAC : CSR_c0 et return. Config nominale = τ_util=0.80.

**Commande :**
```bash
PYTHONUTF8=1 py -3.14 -m uv run python -m experiments.tau_sensitivity.run \
    --config experiments/tau_sensitivity/config.yaml --parallel 6
```

**Livrables :** `runs/tau_sensitivity/tau_sensitivity.pdf`, `tau_sensitivity.csv`
**Paper :** §7.1.x (robustesse) ou Appendice expérimental

---

## Item 9 — Wall-time benchmark

**Protocole :** Mesurer le wall-time par step pour chaque agent (TCL, Lag, Fixed, HPRS) sur
10 000 steps, K=3 (AdCraft A1). Reporter steps/sec et overhead relatif vs Fixed-SAC.

**Commande :**
```bash
PYTHONUTF8=1 py -3.14 -m uv run python -m experiments.wall_time_bench.run
```

**Livrables :** `runs/wall_time_bench/wall_time.csv`, sortie console
**Paper :** §4.1 ou tableau inline — "production-ready overhead"

---

## Item 2 — PID-Lagrangian (Stooke 2020) [design]

**Référence :** Stooke et al. (2020) "Responsive Safety in Reinforcement Learning by PID
Lagrangian Methods". arXiv:2007.03964.

**Principe :** Remplace le gradient dual standard Δλ = α·c̄ par un contrôleur PI :
```
λ_k(t+1) = max(0, λ_k(t) + K_P·c_k(t) + K_I·∫c_k·dt + K_D·Δc_k(t))
```
Évite les oscillations de Prop. 2 grâce à l'amortissement intégral.

**Fichier à créer :** `agents/sac_pid_lagrangian.py`
**Base :** Copier `sac_lagrangian.py`, remplacer la mise à jour des multiplicateurs (lignes ~300-320)
par le contrôleur PID. Ajouter Args : `pid_kp: float = 0.1`, `pid_ki: float = 0.01`,
`pid_kd: float = 0.0`. Logger λ_k, intégrale, dérivée dans TensorBoard.

**Config expérience :** `experiments/pid_lagrangian_bench/config.yaml`
Même setup que Phase 5 A1 (drift=0.03) + B1 (drift=0.01). 3 seeds, 60k steps.

---

## Item 3 — Env hors AdCraft [design]

**Option A (recommandée) — Constrained Pendulum synthétique :**
```python
# gym wrapper: Pendulum-v1 + contrainte angle |θ| ≤ θ_max
class ConstrainedPendulum(gym.Wrapper):
    def __init__(self, env, theta_max=0.5, cost_threshold=0.0):
        ...
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        theta = np.arctan2(obs[1], obs[0])
        cost = float(abs(theta) - self.theta_max)  # >0 if violated
        info["costs"] = [cost]
        return obs, reward, done, info
```
K=1 contrainte, env standard gym, implémentable en <50 lignes.

**Option B — Safety Gym (PointGoal1) :**
```bash
pip install safety-gymnasium
# Env: SafetyPointGoal1-v0 (K=1 hazard constraint)
```
Plus crédible pour TMLR mais dépendance externe lourde (MuJoCo).

**Fichier à créer :** `tcl/envs/constrained_pendulum.py` (Option A préférée)
**Config expérience :** `experiments/constrained_pendulum_bench/config.yaml`
Comparer TCL-SAC(B), Lag-SAC, Fixed-SAC sur 100k steps, 5 seeds.

---

## Item 5 — Plus de seeds

Réexécuter Phase 5 A1 (tableau principal §7.1) avec seeds ∈ {1..10}.
Tests Wilcoxon pairés TCL-SAC vs chaque baseline sur CSR_c0.

```bash
# Modifier config.phase5_a1_v2.yaml : seeds: [1,2,3,4,5,6,7,8,9,10]
PYTHONUTF8=1 py -3.14 -m uv run python -m experiments.pilot_adcraft.run_multi \
    --config experiments/pilot_adcraft/config.phase5_a1_v2_10seeds.yaml --parallel 10
```

---

## Ordre d'exécution recommandé

```
Maintenant (pendant β*-scan) :
  → écrire sac_pid_lagrangian.py (Item 2)
  → écrire constrained_pendulum.py (Item 3)

Après β*-scan terminé :
  1. prop5_validation  (2h, 8 workers, cheap — toy B1 env)
  2. tau_sensitivity   (3h, 6 workers)
  3. wall_time_bench   (10 min, séquentiel)

Décision priorité TMLR (discussion) :
  4. pid_lagrangian_bench (si Item 2 implémenté)
  5. constrained_pendulum_bench (si Item 3 implémenté)
  6. phase5_a1_10seeds (Item 5, ~4h)
```

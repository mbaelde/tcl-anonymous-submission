# VM Runbook — TCL Paper Full Reproduction

Reproduit toutes les expériences du papier avec 10 seeds sur une VM 44 cœurs.
Résultats dans `runs_vm/`, séparés des runs locaux (`runs/`).

---

## 1. Prérequis VM (Ubuntu 22.04+)

```bash
# Python 3.12
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3.12-dev git

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # ou relancer le shell

# Vérifier
uv --version
python3.12 --version
```

---

## 2. Cloner le repo (privé)

Deux options :

**Option A — SSH key** (recommandé) :
```bash
# Sur ton PC local, copier ta clé publique sur la VM
ssh-copy-id user@<vm-ip>

# Sur la VM, ajouter la clé GitHub (si pas déjà fait)
# → https://github.com/settings/keys

git clone git@github.com:anonymous/tcl-code.git
cd tcl-code
```

**Option B — Token HTTPS** :
```bash
git clone https://<token>@github.com/anonymous/tcl-code.git
cd tcl-code
```

---

## 3. Installer les dépendances

```bash
cd tcl-code
uv sync
```

Vérifie que les imports critiques fonctionnent :
```bash
uv run python -c "from tcl.envs import MultiConstraintAdCraftLaplacian; print('OK')"
uv run python -c "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; print('OK')"
```

---

## 4. Lancer tous les runs

```bash
# Depuis la racine du repo
bash experiments/run_all_vm.sh
```

Options disponibles :
```bash
PARALLEL=44 bash experiments/run_all_vm.sh      # utiliser 44 workers (tous les cœurs)
SKIP_DONE=0 bash experiments/run_all_vm.sh      # forcer le rerun même si result.txt existe
```

Le script tourne en **séquentiel par expérience**, chaque expérience en **parallèle interne** (`--parallel 40`).
Durée estimée totale : **6-10 h** selon la VM (10 seeds × ~15 expériences, certaines longues).

Pour lancer en background avec log :
```bash
nohup bash experiments/run_all_vm.sh > vm_run.log 2>&1 &
tail -f vm_run.log   # suivre en direct
```

Vérifier la progression :
```bash
# Compter les result.txt dans runs_vm/
find runs_vm -name "result.txt" | wc -l

# Voir les dernières lignes du log
tail -50 vm_run.log
```

---

## 5. Lancer une seule expérience

Si tu veux rejouer uniquement une expérience spécifique avec 10 seeds :

```bash
# Exemple : beta_star_scan avec 10 seeds
python3 -c "
import yaml
with open('experiments/beta_star_scan/config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['seeds'] = list(range(1, 11))
cfg['output_dir'] = 'runs_vm/beta_star_scan'
with open('/tmp/vm_beta_star_scan.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
PYTHONUTF8=1 uv run python -m experiments.beta_star_scan.run \
    --config /tmp/vm_beta_star_scan.yaml \
    --parallel 40 --skip-existing
```

---

## 6. Récupérer les résultats

Depuis ton **PC local**, dans le dossier parent de `tcl-code/` :

```bash
# Synchroniser runs_vm/ du serveur → local (delta seulement)
rsync -avz --progress \
    user@<vm-ip>:/path/to/tcl-code/runs_vm/ \
    D:/repos/tcl-code/runs_vm/

# Récupérer aussi le log
scp user@<vm-ip>:/path/to/tcl-code/vm_run.log D:/repos/tcl-code/
```

Structure récupérée :
```
runs_vm/
├── prop2_validation/
├── prop5_validation/
├── phase5_a1_v2/
├── phase5_b1/
├── phase5_reward_shift/
├── phase5_kappa/
├── phase5_a1_standalone/
├── phase5_a1_standalone_ll/
├── phase5_a1_standalone_ll_w001/
├── phase5_a1_standalone_ll_w01/
├── beta_schedule/
├── beta_star_scan/
├── tau_sensitivity/
├── pid_lagrangian_bench/
├── constrained_pendulum_bench/
└── k_scaling_validation/
```

---

## 7. Post-traitement et figures

Une fois les résultats récupérés en local, relancer les scripts d'analyse en pointant vers `runs_vm/` :

```bash
# Exemple : analyse beta_star_scan VM
PYTHONUTF8=1 py -3.14 -m uv run python -m experiments.beta_star_scan.run \
    --config experiments/beta_star_scan/config.yaml \
    --analyze-only
# (modifier output_dir dans config ou passer --output-dir runs_vm/beta_star_scan)

# Exemple : analyse pid_lagrangian_bench VM
PYTHONUTF8=1 py -3.14 -m uv run python -m experiments.pid_lagrangian_bench.run \
    --config experiments/pid_lagrangian_bench/config.yaml \
    --analyze-only
```

Pour les expériences pilot_adcraft, utiliser les scripts `analyze_*.py` du dossier :
```bash
PYTHONUTF8=1 py -3.14 -m uv run python experiments/pilot_adcraft/analyze.py \
    --input runs_vm/phase5_a1_v2 ...
```

---

## 8. Notes importantes

- **Seeds** : les runs VM utilisent seeds 1..10 (les seeds 1-3 sont identiques aux runs locaux — les résultats sont directement comparables).
- **output_dir** : tous les résultats VM vont dans `runs_vm/<name>/` pour ne pas écraser les runs locaux dans `runs/`.
- **--skip-existing** : actif par défaut — si un `result.txt` existe déjà, la cellule est sautée. Utile pour reprendre un run interrompu.
- **TensorBoard** : les events `.tfevents.*` sont inclus dans `runs_vm/` — les figures d'analyse les lisent automatiquement.
- **Figures** : les scripts d'analyse écrivent dans `figures/` (local) — chemin à ajuster si tu veux séparer les figures VM.
- **Wilcoxon / stats** : avec 10 seeds, les tests Wilcoxon signed-rank (papier item 5) sont faisables. Script dédié `experiments/wilcoxon_stats.py` (paired signed-rank ref vs chaque baseline, par CSR/cost + return, correction Holm). Ajoute `scipy` aux deps → relancer `uv sync` après pull.

---

## 9. Tests Wilcoxon (Item 5)

Une fois `runs_vm/` rempli, depuis la racine du repo :

```bash
# Bench principal §7.1 — formulation A vs B (merge 2 dirs, K=3 AdCraft)
uv run python -m experiments.wilcoxon_stats \
    --run-dirs runs_vm/phase5_a1_standalone_ll runs_vm/phase5_a1_v2 \
    --ref-agent tcl_standalone --k-costs 3 --alternative greater \
    --out runs_vm/wilcoxon_phase5_a1.csv

# β*-scan — bimodalité suspectée β≥10 (K=1, ref=tcl shaped)
uv run python -m experiments.wilcoxon_stats \
    --run-dirs runs_vm/beta_star_scan --ref-agent tcl --k-costs 1 \
    --out runs_vm/wilcoxon_beta_star.csv

# PID-Lagrangian bench (K=3, ref=tcl shaped vs pid/lag)
uv run python -m experiments.wilcoxon_stats \
    --run-dirs runs_vm/pid_lagrangian_bench --ref-agent tcl --k-costs 3 \
    --alternative greater --out runs_vm/wilcoxon_pid.csv
```

`--alternative greater` teste *ref > baseline* (dominance CSR) ; pour le return ou
une comparaison non-directionnelle, omettre (défaut `two-sided`). Le CSR par seed
est extrait avec la même convention que les figures publiées (steady = 20 % final,
`episode_cost_k ≤ 0`). Les cellules où ref et baseline sont identiques (ex. CSR=1.000
des deux côtés) sont marquées `all-equal` (p=1, pas de signal). Colonne `p_holm` =
p-value corrigée Holm-Bonferroni sur toute la famille de comparaisons.

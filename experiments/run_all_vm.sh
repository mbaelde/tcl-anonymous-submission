#!/usr/bin/env bash
# Full reproduction sweep for the TCL paper on a 44-core VM.
#
# Seeds 1..10, --parallel 40, --skip-existing throughout.
# Each experiment writes to runs_vm/<name>/ (separate from local runs/).
#
# Usage:
#   bash experiments/run_all_vm.sh              # all experiments
#   PARALLEL=44 bash experiments/run_all_vm.sh  # override parallelism
#   SKIP_DONE=0 bash experiments/run_all_vm.sh  # force rerun all cells

set -euo pipefail

PARALLEL=${PARALLEL:-40}
SKIP_FLAG=${SKIP_DONE:-1}
SKIP=""
if [ "$SKIP_FLAG" = "1" ]; then SKIP="--skip-existing"; fi

export PYTHONUTF8=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Patch a config: override seeds (1..10) and output_dir (runs_vm/<name>)
mk_cfg() {
    local src="$1" name="$2"
    local dst="/tmp/vm_cfg_${name}.yaml"
    python3 - "$src" "$dst" "$name" << 'PYEOF'
import sys, yaml, pathlib
src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg['seeds'] = list(range(1, 11))
cfg['output_dir'] = f"runs_vm/{name}"
with open(dst, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
PYEOF
    echo "$dst"
}

run_exp() {
    local module="$1" cfg="$2"
    log "START $module  [cfg: $cfg]"
    uv run python -m "$module" --config "$cfg" --parallel "$PARALLEL" $SKIP
    log "DONE  $module"
    echo ""
}

log "========================================================"
log "TCL Paper — full VM reproduction run"
log "Seeds: 1..10 | Workers: $PARALLEL | Output: runs_vm/"
log "========================================================"
echo ""

# ---------------------------------------------------------------------------
# §5.2 — Proposition 2 validation (dual-oscillation amplitude vs alpha/omega)
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/prop2_validation/config.yaml prop2_validation)
run_exp experiments.prop2_validation.run "$cfg"

# ---------------------------------------------------------------------------
# §5.4 — Proposition 5 validation (asymptotic bridge Formulation A vs B)
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/prop5_validation/config.yaml prop5_validation)
run_exp experiments.prop5_validation.run "$cfg"

# ---------------------------------------------------------------------------
# §7.1.1+7.1.2 — Baseline comparison: easily-feasible + tight util (A1 env)
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/pilot_adcraft/config.phase5_a1_v2.yaml phase5_a1_v2)
run_exp experiments.pilot_adcraft.run "$cfg"

# ---------------------------------------------------------------------------
# §7.1.3 — Slow-drift regime (B1 env, drift=0.01)
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/pilot_adcraft/config.phase5_b1.yaml phase5_b1)
run_exp experiments.pilot_adcraft.run "$cfg"

# ---------------------------------------------------------------------------
# §7.1.4 — Reward-shift fix validation
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/pilot_adcraft/config.phase5_reward_shift.yaml phase5_reward_shift)
run_exp experiments.pilot_adcraft.run "$cfg"

# ---------------------------------------------------------------------------
# §7.1.4 — Empirical beta*-scan (Proposition 4 localization)
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/beta_star_scan/config.yaml beta_star_scan)
run_exp experiments.beta_star_scan.run "$cfg"

# ---------------------------------------------------------------------------
# §7.1.5 — Beta-annealing schedule comparison (B1 + A1 loss-budget)
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/beta_schedule/config.yaml beta_schedule)
run_exp experiments.beta_schedule.run "$cfg"

# ---------------------------------------------------------------------------
# §7.1.6 — Kappa-calibration ablation (Gaussian gate, B1 env)
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/pilot_adcraft/config.phase5_kappa.yaml phase5_kappa)
run_exp experiments.pilot_adcraft.run "$cfg"

# ---------------------------------------------------------------------------
# §7.1.8 — Formulation (A) standalone vs all baselines (A1 loss-budget)
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/pilot_adcraft/config.phase5_a1_standalone.yaml phase5_a1_standalone)
run_exp experiments.pilot_adcraft.run "$cfg"

# ---------------------------------------------------------------------------
# Appendix B.4 — Last-layer hybrid calibration (lambda=0.01)
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/pilot_adcraft/config.phase5_a1_standalone_ll_w001.yaml phase5_a1_standalone_ll_w001)
run_exp experiments.pilot_adcraft.run "$cfg"

# ---------------------------------------------------------------------------
# Appendix B.4 — Last-layer hybrid calibration (lambda=0.10)
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/pilot_adcraft/config.phase5_a1_standalone_ll_w01.yaml phase5_a1_standalone_ll_w01)
run_exp experiments.pilot_adcraft.run "$cfg"

# ---------------------------------------------------------------------------
# Appendix B.4 — Last-layer hybrid calibration (lambda=1.0)
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/pilot_adcraft/config.phase5_a1_standalone_ll.yaml phase5_a1_standalone_ll)
run_exp experiments.pilot_adcraft.run "$cfg"

# ---------------------------------------------------------------------------
# §7.1 — Tau sensitivity (TCL vs Lag, 6 tau values)
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/tau_sensitivity/config.yaml tau_sensitivity)
run_exp experiments.tau_sensitivity.run "$cfg"

# ---------------------------------------------------------------------------
# PID-Lagrangian benchmark (Stooke 2020) vs TCL vs Lag on AdCraft A1
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/pid_lagrangian_bench/config.yaml pid_lagrangian_bench)
run_exp experiments.pid_lagrangian_bench.run "$cfg"

# ---------------------------------------------------------------------------
# Constrained Pendulum — TCL(A) vs TCL(B) vs PID-Lag vs Lag (K=1, non-AdCraft)
# ---------------------------------------------------------------------------
cfg=$(mk_cfg experiments/constrained_pendulum_bench/config.yaml constrained_pendulum_bench)
run_exp experiments.constrained_pendulum_bench.run "$cfg"

# ---------------------------------------------------------------------------
# Analytical experiments (no seeds, deterministic)
# ---------------------------------------------------------------------------
log "START prop2_analytic"
uv run python -m experiments.prop2_analytic.run
log "DONE  prop2_analytic"
echo ""

log "START k_scaling_validation"
uv run python -m experiments.k_scaling_validation.run \
    --output-dir runs_vm/k_scaling_validation
log "DONE  k_scaling_validation"
echo ""

log "========================================================"
log "ALL EXPERIMENTS COMPLETED"
log "Results in: $(pwd)/runs_vm/"
log "========================================================"

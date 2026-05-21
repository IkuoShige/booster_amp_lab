#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/booster_amp_lab}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/workspace/Isaac_uv_template/.venv/bin/activate}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/logs/force100_compare_${TIMESTAMP}}"
POLICIES="${POLICIES:-v9_4999 v10b_750 v11_6500 v12b_8050 v13b_7400}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NUM_ENVS="${NUM_ENVS:-32}"
export FORCES="${FORCES:-100 150}"
export STEPS="${STEPS:-320}"
export SETTLE_STEPS="${SETTLE_STEPS:-100}"
export COMMANDS="${COMMANDS:-0.0,0.0,0.0;0.2,0.0,0.0;0.5,0.0,0.0;1.0,0.0,0.0;1.5,0.0,0.0;2.0,0.0,0.0;0.0,0.4,0.0;0.0,1.0,0.0;0.0,0.0,1.5}"
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_DURATION_MIN_S="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_DURATION_MIN_S:-0.10}"
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_DURATION_MAX_S="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_DURATION_MAX_S:-0.10}"
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_INTERVAL_MIN_S="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_INTERVAL_MIN_S:-3.0}"
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_INTERVAL_MAX_S="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_INTERVAL_MAX_S:-3.0}"
export BOOSTER_TRACK_ADAPTER_BASE_CONTACT_DURATION_S="${BOOSTER_TRACK_ADAPTER_BASE_CONTACT_DURATION_S:-0.0}"

declare -A CHECKPOINTS
CHECKPOINTS[v9_4999]="${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter/2026-05-13_16-38-08_track_adapter_81999_stage3_recovery_v9_layerwise_20260513_163804_ppo/model_4999.pt"
CHECKPOINTS[v10b_750]="${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter/2026-05-14_11-50-36_track_adapter_81999_stage3_recovery_v10b_force_kick_warm_20260514_115031_ppo_force_kick/model_750.pt"
CHECKPOINTS[v11_6500]="${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter/2026-05-14_15-16-50_track_adapter_81999_stage3_recovery_v11_adaptive_force_kick_20260514_151645_ppo_adaptive_force_kick/model_6500.pt"
CHECKPOINTS[v12b_8050]="${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter/2026-05-15_03-07-15_track_adapter_81999_stage3_recovery_v12b_ramped_force_kick_20260515_030711_ppo_ramped_force_kick/model_8050.pt"
CHECKPOINTS[v13b_7400]="${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter/2026-05-15_06-23-19_track_adapter_81999_stage3_recovery_v13b_single_kick_recovery_20260515_062314_ppo_single_kick_recovery/model_7400.pt"

mkdir -p "$OUT_ROOT"

echo "[force100_compare] timestamp=${TIMESTAMP}"
echo "[force100_compare] out_root=${OUT_ROOT}"
echo "[force100_compare] policies=${POLICIES}"
echo "[force100_compare] forces=${FORCES}"
echo "[force100_compare] num_envs=${NUM_ENVS} steps=${STEPS} settle=${SETTLE_STEPS}"
echo "[force100_compare] commands=${COMMANDS}"

for name in $POLICIES; do
  checkpoint="${CHECKPOINTS[$name]}"
  if [[ ! -f "$checkpoint" ]]; then
    echo "[force100_compare] missing checkpoint ${name}: ${checkpoint}" >&2
    exit 1
  fi
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    complete=1
    for force_n in $FORCES; do
      if [[ ! -f "${OUT_ROOT}/${name}/F${force_n}/velocity_force_kick.json" ]]; then
        complete=0
      fi
    done
    if [[ "$complete" == "1" ]]; then
      echo "[force100_compare] skip existing ${name}"
      continue
    fi
  fi
  echo "[force100_compare] start ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  CHECKPOINT="$checkpoint" \
    NAME="${name}_force100_150" \
    TIMESTAMP="$TIMESTAMP" \
    OUT_BASE="${OUT_ROOT}/${name}" \
    scripts/rsl_rl/eval_track_adapter_pure_horizontal_force_sweep.sh
  echo "[force100_compare] done ${name} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

echo "[force100_compare] all done ${OUT_ROOT}"

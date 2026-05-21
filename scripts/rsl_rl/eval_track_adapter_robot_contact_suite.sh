#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/booster_amp_lab}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/workspace/Isaac_uv_template/.venv/bin/activate}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
NAME="${NAME:-track_adapter_robot_contact}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/logs/robot_contact_eval_${NAME}_${TIMESTAMP}}"

: "${CHECKPOINT:?Set CHECKPOINT to the Track Adapter checkpoint to evaluate.}"

TASK="${TASK:-Booster-Run-AMP-TrackAdapter-v0}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-32}"
SEED="${SEED:-321}"
SUSTAINED_FORCES="${SUSTAINED_FORCES:-${FORCES:-10 20 30 50}}"
BUMP_FORCES="${BUMP_FORCES:-50 80 120 160}"

STILLNESS_STEPS="${STILLNESS_STEPS:-320}"
STILLNESS_SETTLE_STEPS="${STILLNESS_SETTLE_STEPS:-100}"

SUSTAINED_MODES="${SUSTAINED_MODES:-none event}"
SUSTAINED_PUSH_START_STEP="${SUSTAINED_PUSH_START_STEP:-80}"
SUSTAINED_PUSH_DURATION_STEPS="${SUSTAINED_PUSH_DURATION_STEPS:-125}"
SUSTAINED_POST_PUSH_STEPS="${SUSTAINED_POST_PUSH_STEPS:-100}"
SUSTAINED_RECOVERY_WINDOW_S="${SUSTAINED_RECOVERY_WINDOW_S:-4.0}"

BUMP_STEPS="${BUMP_STEPS:-320}"
BUMP_SETTLE_STEPS="${BUMP_SETTLE_STEPS:-100}"
BUMP_DURATION_S="${BUMP_DURATION_S:-0.10}"
BUMP_INTERVAL_S="${BUMP_INTERVAL_S:-1.0}"

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT"

log() {
  echo "[robot_contact_suite] $*" | tee -a "$OUT_ROOT/eval_all.log"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    log "missing required eval helper: $path"
    exit 1
  fi
}

require_file "scripts/rsl_rl/eval_velocity_command_grid.py"
require_file "scripts/rsl_rl/eval_track_adapter_sustained_push_single.sh"
require_file "scripts/rsl_rl/eval_track_adapter_pure_horizontal_force_sweep.sh"

source "$VENV_ACTIVATE"

export PYTHONPATH="${REPO_ROOT}/rsl_rl:${REPO_ROOT}/source/booster_rl_tasks:${REPO_ROOT}/booster_assets/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export BOOSTER_TRACK_ADAPTER_FULL_TRAINING="${BOOSTER_TRACK_ADAPTER_FULL_TRAINING:-1}"
export BOOSTER_TRACK_ADAPTER_SMOKE_PASSED="${BOOSTER_TRACK_ADAPTER_SMOKE_PASSED:-1}"
export BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING="${BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING:-1}"
export BOOSTER_TRACK_ADAPTER_SMOKE_MODE="${BOOSTER_TRACK_ADAPTER_SMOKE_MODE:-0}"
export BOOSTER_TRACK_ADAPTER_BASE_CHECKPOINT="${BOOSTER_TRACK_ADAPTER_BASE_CHECKPOINT:-${REPO_ROOT}/logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt}"
export BOOSTER_TRACK_ADAPTER_MODE="${BOOSTER_TRACK_ADAPTER_MODE:-layerwise}"
export BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS="${BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS:-128}"
export BOOSTER_TRACK_ADAPTER_LAYERWISE_OUTPUT_INIT_SCALE="${BOOSTER_TRACK_ADAPTER_LAYERWISE_OUTPUT_INIT_SCALE:-0.0}"
export BOOSTER_TRACK_ADAPTER_RESIDUAL_SCALE="${BOOSTER_TRACK_ADAPTER_RESIDUAL_SCALE:-0.165}"
export BOOSTER_TRACK_ADAPTER_BASE_CONTACT_DURATION_S="${BOOSTER_TRACK_ADAPTER_BASE_CONTACT_DURATION_S:-0.0}"

log "timestamp=${TIMESTAMP}"
log "checkpoint=${CHECKPOINT}"
log "out_root=${OUT_ROOT}"
log "sustained_forces=${SUSTAINED_FORCES}"
log "bump_forces=${BUMP_FORCES}"
log "num_envs=${NUM_ENVS} seed=${SEED}"

log "start stillness: zero command, no push"
BOOSTER_TRACK_ADAPTER_PUSH=0 \
BOOSTER_TRACK_ADAPTER_FORCE_PUSH=0 \
python scripts/rsl_rl/eval_velocity_command_grid.py \
  --task "$TASK" \
  --headless \
  --device "$DEVICE" \
  --num_envs "$NUM_ENVS" \
  --seed "$SEED" \
  --checkpoint "$CHECKPOINT" \
  --disable_push \
  --steps "$STILLNESS_STEPS" \
  --settle_steps "$STILLNESS_SETTLE_STEPS" \
  --commands "0.0,0.0,0.0" \
  --output_json "$OUT_ROOT/stillness_zero_command_no_push.json" \
  --output_csv "$OUT_ROOT/stillness_zero_command_no_push.csv" \
  2>&1 | tee "$OUT_ROOT/stillness_zero_command_no_push.log"
log "done stillness"

log "start sustained body-frame pushes"
CHECKPOINT="$CHECKPOINT" \
NAME="${NAME}_sustained" \
OUT_ROOT="$OUT_ROOT/sustained_body_frame_push" \
FORCES="$SUSTAINED_FORCES" \
MODES="$SUSTAINED_MODES" \
NUM_ENVS="$NUM_ENVS" \
SEED="$SEED" \
PUSH_START_STEP="$SUSTAINED_PUSH_START_STEP" \
PUSH_DURATION_STEPS="$SUSTAINED_PUSH_DURATION_STEPS" \
POST_PUSH_STEPS="$SUSTAINED_POST_PUSH_STEPS" \
RECOVERY_WINDOW_S="$SUSTAINED_RECOVERY_WINDOW_S" \
FORCE_BODY_X=1.0 \
FORCE_BODY_Y=0.0 \
FORCE_BODY_Z=0.0 \
scripts/rsl_rl/eval_track_adapter_sustained_push_single.sh
log "done sustained body-frame pushes"

log "start repeated random horizontal bump sweep"
CHECKPOINT="$CHECKPOINT" \
NAME="${NAME}_random_bump" \
OUT_BASE="$OUT_ROOT/random_horizontal_bump_sweep" \
FORCES="$BUMP_FORCES" \
NUM_ENVS="$NUM_ENVS" \
SEED="$SEED" \
COMMANDS="0.0,0.0,0.0" \
STEPS="$BUMP_STEPS" \
SETTLE_STEPS="$BUMP_SETTLE_STEPS" \
BOOSTER_TRACK_ADAPTER_FORCE_PUSH_DURATION_MIN_S="$BUMP_DURATION_S" \
BOOSTER_TRACK_ADAPTER_FORCE_PUSH_DURATION_MAX_S="$BUMP_DURATION_S" \
BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_DURATION_MIN_S="$BUMP_DURATION_S" \
BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_DURATION_MAX_S="$BUMP_DURATION_S" \
BOOSTER_TRACK_ADAPTER_FORCE_PUSH_INTERVAL_MIN_S="$BUMP_INTERVAL_S" \
BOOSTER_TRACK_ADAPTER_FORCE_PUSH_INTERVAL_MAX_S="$BUMP_INTERVAL_S" \
BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_INTERVAL_MIN_S="$BUMP_INTERVAL_S" \
BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_INTERVAL_MAX_S="$BUMP_INTERVAL_S" \
BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FAILURE_MINING=0 \
BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FIXED_DIRECTION_PROB=0 \
BOOSTER_TRACK_ADAPTER_FORCE_PUSH_GLOBAL_FRAME=0 \
scripts/rsl_rl/eval_track_adapter_pure_horizontal_force_sweep.sh
log "done repeated random horizontal bump sweep"

log "complete summary candidates:"
log "  stillness=${OUT_ROOT}/stillness_zero_command_no_push.json"
log "  sustained=${OUT_ROOT}/sustained_body_frame_push/summary.md"
log "  bumps=${OUT_ROOT}/random_horizontal_bump_sweep"

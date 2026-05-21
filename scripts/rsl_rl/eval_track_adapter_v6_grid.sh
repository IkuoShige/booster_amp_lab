#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/booster_amp_lab}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/workspace/Isaac_uv_template/.venv/bin/activate}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-128}"
SEED="${SEED:-42}"
TASK="${TASK:-Booster-Run-AMP-TrackAdapter-v0}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
NAME="${NAME:-v4_model600}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter/2026-05-11_08-28-29_track_adapter_gated_recovery_v4_20260511_082805_ppo/model_600.pt}"
OUT="${OUT:-${REPO_ROOT}/logs/fixed_eval_${NAME}_v6_grid_${TIMESTAMP}}"

COMMANDS="${COMMANDS:-0.0,0.0,0.0;0.2,0.0,0.0;0.5,0.0,0.0;1.0,0.0,0.0;1.5,0.0,0.0;2.0,0.0,0.0;0.0,0.2,0.0;0.0,0.4,0.0;0.0,1.0,0.0}"
STEPS="${STEPS:-400}"
SETTLE_STEPS="${SETTLE_STEPS:-120}"
PUSH_GRID="${PUSH_GRID:-1}"
PUSH_XY="${PUSH_XY:-0.60}"
PUSH_YAW="${PUSH_YAW:-0.10}"
PUSH_INTERVAL_MIN="${PUSH_INTERVAL_MIN:-1.0}"
PUSH_INTERVAL_MAX="${PUSH_INTERVAL_MAX:-1.0}"

cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

export PYTHONPATH="${REPO_ROOT}/rsl_rl:${REPO_ROOT}/source/booster_rl_tasks:${REPO_ROOT}/booster_assets/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export BOOSTER_TRACK_ADAPTER_FULL_TRAINING="${BOOSTER_TRACK_ADAPTER_FULL_TRAINING:-1}"
export BOOSTER_TRACK_ADAPTER_SMOKE_PASSED="${BOOSTER_TRACK_ADAPTER_SMOKE_PASSED:-1}"
export BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING="${BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING:-1}"
export BOOSTER_TRACK_ADAPTER_SMOKE_MODE="${BOOSTER_TRACK_ADAPTER_SMOKE_MODE:-0}"

mkdir -p "$OUT"

echo "[adapter_eval] timestamp=${TIMESTAMP}"
echo "[adapter_eval] name=${NAME}"
echo "[adapter_eval] task=${TASK}"
echo "[adapter_eval] checkpoint=${CHECKPOINT}"
echo "[adapter_eval] output=${OUT}"
echo "[adapter_eval] commands=${COMMANDS}"

python scripts/rsl_rl/eval_velocity_command_grid.py \
  --task "$TASK" \
  --headless \
  --device "$DEVICE" \
  --num_envs "$NUM_ENVS" \
  --seed "$SEED" \
  --checkpoint "$CHECKPOINT" \
  --disable_push \
  --steps "$STEPS" \
  --settle_steps "$SETTLE_STEPS" \
  --commands "$COMMANDS" \
  --output_json "$OUT/velocity_no_push.json" \
  --output_csv "$OUT/velocity_no_push.csv" \
  2>&1 | tee "$OUT/eval.log"

if [[ "$PUSH_GRID" == "1" ]]; then
  python scripts/rsl_rl/eval_velocity_command_grid.py \
    --task "$TASK" \
    --headless \
    --device "$DEVICE" \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --checkpoint "$CHECKPOINT" \
    --push_xy "$PUSH_XY" \
    --push_yaw "$PUSH_YAW" \
    --push_interval_min "$PUSH_INTERVAL_MIN" \
    --push_interval_max "$PUSH_INTERVAL_MAX" \
    --steps "$STEPS" \
    --settle_steps "$SETTLE_STEPS" \
    --commands "$COMMANDS" \
    --output_json "$OUT/velocity_interval_push.json" \
    --output_csv "$OUT/velocity_interval_push.csv" \
    2>&1 | tee -a "$OUT/eval.log"
fi

python scripts/rsl_rl/eval_track_adapter_recovery.py \
  --task "$TASK" \
  --headless \
  --device "$DEVICE" \
  --num_envs "$NUM_ENVS" \
  --seed "$SEED" \
  --checkpoint "$CHECKPOINT" \
  --disable_push \
  --scenario_set high_speed_jerk \
  --push_modes both \
  --warmup_steps 80 \
  --move_steps 120 \
  --stop_steps 160 \
  --eval_push_speed 0.60 \
  --eval_push_yaw 0.10 \
  --eval_push_step 0 \
  --output_json "$OUT/high_speed_jerk.json" \
  --output_csv "$OUT/high_speed_jerk.csv" \
  2>&1 | tee -a "$OUT/eval.log"

echo "[adapter_eval] done output=${OUT}"

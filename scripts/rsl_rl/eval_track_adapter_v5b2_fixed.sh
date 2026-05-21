#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/booster_amp_lab}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/workspace/Isaac_uv_template/.venv/bin/activate}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-128}"
SEED="${SEED:-42}"
TASK="${TASK:-Booster-Run-AMP-TrackAdapter-v0}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"

cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

export PYTHONPATH="${REPO_ROOT}/rsl_rl:${REPO_ROOT}/source/booster_rl_tasks:${REPO_ROOT}/booster_assets/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export BOOSTER_TRACK_ADAPTER_FULL_TRAINING="${BOOSTER_TRACK_ADAPTER_FULL_TRAINING:-1}"
export BOOSTER_TRACK_ADAPTER_SMOKE_PASSED="${BOOSTER_TRACK_ADAPTER_SMOKE_PASSED:-1}"
export BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING="${BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING:-1}"
export BOOSTER_TRACK_ADAPTER_SMOKE_MODE="${BOOSTER_TRACK_ADAPTER_SMOKE_MODE:-0}"

COMMANDS="${COMMANDS:-0.0,0.0,0.0;0.2,0.0,0.0;0.4,0.0,0.0;0.8,0.0,0.0;1.0,0.0,0.0;2.0,0.0,0.0;0.0,1.0,0.0;1.0,1.0,0.0;0.0,0.0,0.3}"
V4_600="${V4_600:-${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter/2026-05-11_08-28-29_track_adapter_gated_recovery_v4_20260511_082805_ppo/model_600.pt}"
V5B2_DIR="${V5B2_DIR:-${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter/2026-05-11_11-21-18_track_adapter_pushdiag_v5b2_resume600_20260511_112114_ppo}"
V5B2_625="${V5B2_625:-${V5B2_DIR}/model_625.pt}"
V5B2_FINAL="${V5B2_FINAL:-$(find "$V5B2_DIR" -maxdepth 1 -type f -name 'model_*.pt' ! -name '*aborted*' | sort -V | tail -n 1)}"

run_fixed_eval() {
  local name="$1"
  local checkpoint="$2"
  local out="${REPO_ROOT}/logs/fixed_eval_${name}_${TIMESTAMP}"
  mkdir -p "$out"

  echo "[eval] name=${name}"
  echo "[eval] checkpoint=${checkpoint}"
  echo "[eval] output=${out}"

  python scripts/rsl_rl/eval_velocity_command_grid.py \
    --task "$TASK" \
    --headless \
    --device "$DEVICE" \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --checkpoint "$checkpoint" \
    --disable_push \
    --steps 350 \
    --settle_steps 100 \
    --commands "$COMMANDS" \
    --output_json "$out/velocity_no_push.json" \
    --output_csv "$out/velocity_no_push.csv" \
    2>&1 | tee "$out/eval.log"

  python scripts/rsl_rl/eval_track_adapter_recovery.py \
    --task "$TASK" \
    --headless \
    --device "$DEVICE" \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --checkpoint "$checkpoint" \
    --disable_push \
    --scenario_set high_speed_jerk \
    --push_modes both \
    --warmup_steps 80 \
    --move_steps 120 \
    --stop_steps 160 \
    --eval_push_speed 0.60 \
    --eval_push_yaw 0.10 \
    --eval_push_step 0 \
    --output_json "$out/high_speed_jerk.json" \
    --output_csv "$out/high_speed_jerk.csv" \
    2>&1 | tee -a "$out/eval.log"
}

echo "[eval] timestamp=${TIMESTAMP}"
echo "[eval] v4_600=${V4_600}"
echo "[eval] v5b2_625=${V5B2_625}"
echo "[eval] v5b2_final=${V5B2_FINAL}"

run_fixed_eval "v4_model600" "$V4_600"
run_fixed_eval "v5b2_model625" "$V5B2_625"
run_fixed_eval "v5b2_final" "$V5B2_FINAL"

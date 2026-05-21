#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/booster_amp_lab}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/workspace/Isaac_uv_template/.venv/bin/activate}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-128}"
SEED="${SEED:-42}"
TASK="${TASK:-Booster-Run-AMP-TrackAdapter-v0}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
V5B2_DIR="${V5B2_DIR:-${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter/2026-05-11_11-21-18_track_adapter_pushdiag_v5b2_resume600_20260511_112114_ppo}"
CHECKPOINTS="${CHECKPOINTS:-700 750 800 825}"

cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

export PYTHONPATH="${REPO_ROOT}/rsl_rl:${REPO_ROOT}/source/booster_rl_tasks:${REPO_ROOT}/booster_assets/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export BOOSTER_TRACK_ADAPTER_FULL_TRAINING="${BOOSTER_TRACK_ADAPTER_FULL_TRAINING:-1}"
export BOOSTER_TRACK_ADAPTER_SMOKE_PASSED="${BOOSTER_TRACK_ADAPTER_SMOKE_PASSED:-1}"
export BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING="${BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING:-1}"
export BOOSTER_TRACK_ADAPTER_SMOKE_MODE="${BOOSTER_TRACK_ADAPTER_SMOKE_MODE:-0}"

echo "[jerk_sweep] timestamp=${TIMESTAMP}"
echo "[jerk_sweep] checkpoint_dir=${V5B2_DIR}"
echo "[jerk_sweep] checkpoints=${CHECKPOINTS}"

for iter in $CHECKPOINTS; do
  checkpoint="${V5B2_DIR}/model_${iter}.pt"
  out="${REPO_ROOT}/logs/fixed_eval_v5b2_model${iter}_jerk_${TIMESTAMP}"
  mkdir -p "$out"
  echo "[jerk_sweep] checkpoint=${checkpoint}"
  echo "[jerk_sweep] output=${out}"

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
    2>&1 | tee "$out/eval.log"
done

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/booster_amp_lab}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/workspace/Isaac_uv_template/.venv/bin/activate}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/logs/recovery_controller_gates_${TIMESTAMP}}"

TASK="${TASK:-Booster-Run-AMP-TrackAdapter-v0}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-32}"
SEED="${SEED:-321}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FAIL_ON_GATE="${FAIL_ON_GATE:-0}"

V14_CHECKPOINT="${V14_CHECKPOINT:-${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter/2026-05-15_10-17-58_track_adapter_81999_stage3_recovery_v14_capture_step_20260515_081803_ppo_capture_step/model_10000.pt}"
V15B_CHECKPOINT="${V15B_CHECKPOINT:-${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter/2026-05-16_11-44-40_track_adapter_81999_stage3_recovery_v15b_force_ladder_20260516_114435_ppo_force_ladder/model_16499.pt}"

find_latest_v44_checkpoint() {
  find "${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter" -type f -path "*v44*model_*.pt" 2>/dev/null | sort -V | tail -n 1
}

V44_CHECKPOINT="${V44_CHECKPOINT:-$(find_latest_v44_checkpoint || true)}"

if [[ -z "${CHECKPOINT_SPECS:-}" ]]; then
  if [[ -z "$V44_CHECKPOINT" ]]; then
    echo "[recovery_controller_gates] no v44 checkpoint found; set V44_CHECKPOINT or CHECKPOINT_SPECS" >&2
    echo "[recovery_controller_gates] CHECKPOINT_SPECS format: name=/path/model.pt=residual_scale[=load_mode][=hidden_dims][=adapter_mode]" >&2
    exit 1
  fi
  CHECKPOINT_SPECS="v14_10000=${V14_CHECKPOINT}=0.20=strict=128=layerwise v15b_16499=${V15B_CHECKPOINT}=0.23=strict=128=layerwise v44=${V44_CHECKPOINT}=0.24=strict=256,128=layerwise"
fi

NO_PUSH_COMMANDS="${NO_PUSH_COMMANDS:-0.0,0.0,0.0;0.2,0.0,0.0;0.5,0.0,0.0;1.0,0.0,0.0;1.5,0.0,0.0;2.0,0.0,0.0;0.0,0.4,0.0;0.0,1.0,0.0;0.0,0.0,0.8;0.0,0.0,1.5}"
NO_PUSH_STEPS="${NO_PUSH_STEPS:-360}"
NO_PUSH_SETTLE_STEPS="${NO_PUSH_SETTLE_STEPS:-120}"

SLOW_PUSH_FORCE_N="${SLOW_PUSH_FORCE_N:-40}"
SLOW_PUSH_START_STEP="${SLOW_PUSH_START_STEP:-80}"
SLOW_PUSH_DURATION_STEPS="${SLOW_PUSH_DURATION_STEPS:-150}"
SLOW_PUSH_POST_STEPS="${SLOW_PUSH_POST_STEPS:-120}"
SLOW_PUSH_RECOVERY_WINDOW_S="${SLOW_PUSH_RECOVERY_WINDOW_S:-4.0}"
SLOW_PUSH_METRIC_MODE="${SLOW_PUSH_METRIC_MODE:-none}"
SLOW_PUSH_FORCE_BODY_X="${SLOW_PUSH_FORCE_BODY_X:-1.0}"
SLOW_PUSH_FORCE_BODY_Y="${SLOW_PUSH_FORCE_BODY_Y:-0.0}"
SLOW_PUSH_FORCE_BODY_Z="${SLOW_PUSH_FORCE_BODY_Z:-0.0}"

WALK_PUSH_CMD="${WALK_PUSH_CMD:-0.5,0.0,0.0}"
WALK_PUSH_FORCE_N="${WALK_PUSH_FORCE_N:-80}"
WALK_PUSH_START_STEP="${WALK_PUSH_START_STEP:-100}"
WALK_PUSH_DURATION_STEPS="${WALK_PUSH_DURATION_STEPS:-75}"
WALK_PUSH_POST_STEPS="${WALK_PUSH_POST_STEPS:-150}"
WALK_PUSH_RECOVERY_WINDOW_S="${WALK_PUSH_RECOVERY_WINDOW_S:-3.0}"
WALK_PUSH_FORCE_BODY_X="${WALK_PUSH_FORCE_BODY_X:-1.0}"
WALK_PUSH_FORCE_BODY_Y="${WALK_PUSH_FORCE_BODY_Y:-0.0}"
WALK_PUSH_FORCE_BODY_Z="${WALK_PUSH_FORCE_BODY_Z:-0.0}"

SHORT_IMPACT_WARMUP_STEPS="${SHORT_IMPACT_WARMUP_STEPS:-100}"
SHORT_IMPACT_MOVE_STEPS="${SHORT_IMPACT_MOVE_STEPS:-150}"
SHORT_IMPACT_STOP_STEPS="${SHORT_IMPACT_STOP_STEPS:-150}"
SHORT_IMPACT_PUSH_SPEED="${SHORT_IMPACT_PUSH_SPEED:-0.60}"
SHORT_IMPACT_PUSH_YAW="${SHORT_IMPACT_PUSH_YAW:-0.10}"
SHORT_IMPACT_PUSH_STEP="${SHORT_IMPACT_PUSH_STEP:-0}"

MAX_NO_PUSH_FALL="${MAX_NO_PUSH_FALL:-0.02}"
MAX_NO_PUSH_CONTACT="${MAX_NO_PUSH_CONTACT:-0.02}"
MAX_WALK_PUSH_FALL="${MAX_WALK_PUSH_FALL:-0.05}"
MAX_WALK_PUSH_CONTACT="${MAX_WALK_PUSH_CONTACT:-0.05}"
MAX_SHORT_IMPACT_FALL="${MAX_SHORT_IMPACT_FALL:-0.05}"
MAX_SHORT_IMPACT_CONTACT="${MAX_SHORT_IMPACT_CONTACT:-0.05}"
MIN_SHORT_IMPACT_SETTLE="${MIN_SHORT_IMPACT_SETTLE:-0.85}"
MAX_SLOW_PUSH_FALL="${MAX_SLOW_PUSH_FALL:-0.05}"
MAX_SLOW_PUSH_CONTACT="${MAX_SLOW_PUSH_CONTACT:-0.05}"
MIN_SLOW_PUSH_CLEAN="${MIN_SLOW_PUSH_CLEAN:-0.90}"

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT"
source "$VENV_ACTIVATE"

export PYTHONPATH="${REPO_ROOT}/rsl_rl:${REPO_ROOT}/source/booster_rl_tasks:${REPO_ROOT}/booster_assets/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export BOOSTER_TRACK_ADAPTER_FULL_TRAINING="${BOOSTER_TRACK_ADAPTER_FULL_TRAINING:-1}"
export BOOSTER_TRACK_ADAPTER_SMOKE_PASSED="${BOOSTER_TRACK_ADAPTER_SMOKE_PASSED:-1}"
export BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING="${BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING:-1}"
export BOOSTER_TRACK_ADAPTER_SMOKE_MODE="${BOOSTER_TRACK_ADAPTER_SMOKE_MODE:-0}"
export BOOSTER_TRACK_ADAPTER_BASE_CHECKPOINT="${BOOSTER_TRACK_ADAPTER_BASE_CHECKPOINT:-${REPO_ROOT}/logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt}"
export BOOSTER_TRACK_ADAPTER_MODE="${BOOSTER_TRACK_ADAPTER_MODE:-layerwise}"
export BOOSTER_TRACK_ADAPTER_LAYERWISE_OUTPUT_INIT_SCALE="${BOOSTER_TRACK_ADAPTER_LAYERWISE_OUTPUT_INIT_SCALE:-0.0}"
export BOOSTER_TRACK_ADAPTER_BASE_CONTACT_DURATION_S="${BOOSTER_TRACK_ADAPTER_BASE_CONTACT_DURATION_S:-0.0}"
export BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_STABLE_FLOOR="${BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_STABLE_FLOOR:-0.0}"
export BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_STABLE_MAX="${BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_STABLE_MAX:-0.006}"
export BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PUSH_MIN="${BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PUSH_MIN:-0.94}"
export BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_SUDDEN_STOP_MIN="${BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_SUDDEN_STOP_MIN:-0.70}"
export BOOSTER_TRACK_ADAPTER_PUSH_RECOVERY_GATE_MIN="${BOOSTER_TRACK_ADAPTER_PUSH_RECOVERY_GATE_MIN:-0.94}"
export BOOSTER_TRACK_ADAPTER_PUSH_STYLE_GATE_MAX="${BOOSTER_TRACK_ADAPTER_PUSH_STYLE_GATE_MAX:-0.32}"
export BOOSTER_TRACK_ADAPTER_PUSH_COMMAND_GATE_MIN="${BOOSTER_TRACK_ADAPTER_PUSH_COMMAND_GATE_MIN:-0.48}"

log() {
  echo "[recovery_controller_gates] $*" | tee -a "$OUT_ROOT/eval_all.log"
}

require_json() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    log "missing expected JSON: $path"
    exit 1
  fi
}

run_no_push_gate() {
  local name="$1"
  local checkpoint="$2"
  local load_mode="$3"
  local out_dir="$4/no_push"
  local load_args=()
  if [[ "$load_mode" == "non_strict" ]]; then
    load_args+=(--non_strict_checkpoint_load)
  fi
  mkdir -p "$out_dir"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/no_push.json" ]]; then
    log "skip existing ${name} no_push"
    return
  fi
  log "start ${name} gate=no_push"
  BOOSTER_TRACK_ADAPTER_PUSH=0 \
  BOOSTER_TRACK_ADAPTER_FORCE_PUSH=0 \
  python scripts/rsl_rl/eval_velocity_command_grid.py \
    --task "$TASK" \
    --headless \
    --device "$DEVICE" \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --checkpoint "$checkpoint" \
    --disable_push \
    --steps "$NO_PUSH_STEPS" \
    --settle_steps "$NO_PUSH_SETTLE_STEPS" \
    --commands "$NO_PUSH_COMMANDS" \
    "${load_args[@]}" \
    --output_json "$out_dir/no_push.json" \
    --output_csv "$out_dir/no_push.csv" \
    2>&1 | tee "$out_dir/eval.log"
}

run_slow_push_gate() {
  local name="$1"
  local checkpoint="$2"
  local load_mode="$3"
  local out_dir="$4/zero_command_slow_sustained_push"
  local load_args=()
  if [[ "$load_mode" == "non_strict" ]]; then
    load_args+=(--non_strict_checkpoint_load)
  fi
  mkdir -p "$out_dir"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/zero_command_slow_sustained_push.json" ]]; then
    log "skip existing ${name} zero_command_slow_sustained_push"
    return
  fi
  log "start ${name} gate=zero_command_slow_sustained_push"
  BOOSTER_TRACK_ADAPTER_PUSH=0 \
  BOOSTER_TRACK_ADAPTER_FORCE_PUSH=0 \
  python scripts/rsl_rl/eval_track_adapter_static_push_step.py \
    --task "$TASK" \
    --headless \
    --device "$DEVICE" \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --checkpoint "$checkpoint" \
    --force_n "$SLOW_PUSH_FORCE_N" \
    --force_body_x "$SLOW_PUSH_FORCE_BODY_X" \
    --force_body_y "$SLOW_PUSH_FORCE_BODY_Y" \
    --force_body_z "$SLOW_PUSH_FORCE_BODY_Z" \
    --push_start_step "$SLOW_PUSH_START_STEP" \
    --push_duration_steps "$SLOW_PUSH_DURATION_STEPS" \
    --post_push_steps "$SLOW_PUSH_POST_STEPS" \
    --recovery_window_s "$SLOW_PUSH_RECOVERY_WINDOW_S" \
    --push_metric_mode "$SLOW_PUSH_METRIC_MODE" \
    "${load_args[@]}" \
    --output_json "$out_dir/zero_command_slow_sustained_push.json" \
    --output_csv "$out_dir/zero_command_slow_sustained_push.csv" \
    2>&1 | tee "$out_dir/eval.log"
}

run_walking_push_gate() {
  local name="$1"
  local checkpoint="$2"
  local load_mode="$3"
  local out_dir="$4/walking_push"
  local load_args=()
  if [[ "$load_mode" == "non_strict" ]]; then
    load_args+=(--non_strict_checkpoint_load)
  fi
  IFS="," read -r cmd_x cmd_y cmd_yaw <<< "$WALK_PUSH_CMD"
  mkdir -p "$out_dir"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/walking_push.json" ]]; then
    log "skip existing ${name} walking_push"
    return
  fi
  log "start ${name} gate=walking_push"
  BOOSTER_TRACK_ADAPTER_PUSH=0 \
  BOOSTER_TRACK_ADAPTER_FORCE_PUSH=0 \
  python scripts/rsl_rl/eval_track_adapter_static_push_step.py \
    --task "$TASK" \
    --headless \
    --device "$DEVICE" \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --checkpoint "$checkpoint" \
    --cmd_x "$cmd_x" \
    --cmd_y "$cmd_y" \
    --cmd_yaw "$cmd_yaw" \
    --force_n "$WALK_PUSH_FORCE_N" \
    --force_body_x "$WALK_PUSH_FORCE_BODY_X" \
    --force_body_y "$WALK_PUSH_FORCE_BODY_Y" \
    --force_body_z "$WALK_PUSH_FORCE_BODY_Z" \
    --push_start_step "$WALK_PUSH_START_STEP" \
    --push_duration_steps "$WALK_PUSH_DURATION_STEPS" \
    --post_push_steps "$WALK_PUSH_POST_STEPS" \
    --recovery_window_s "$WALK_PUSH_RECOVERY_WINDOW_S" \
    --push_metric_mode none \
    "${load_args[@]}" \
    --output_json "$out_dir/walking_push.json" \
    --output_csv "$out_dir/walking_push.csv" \
    2>&1 | tee "$out_dir/eval.log"
}

run_short_impact_gate() {
  local name="$1"
  local checkpoint="$2"
  local load_mode="$3"
  local out_dir="$4/short_impact"
  local load_args=()
  if [[ "$load_mode" == "non_strict" ]]; then
    load_args+=(--non_strict_checkpoint_load)
  fi
  mkdir -p "$out_dir"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/short_impact.json" ]]; then
    log "skip existing ${name} short_impact"
    return
  fi
  log "start ${name} gate=short_impact"
  BOOSTER_TRACK_ADAPTER_PUSH=0 \
  BOOSTER_TRACK_ADAPTER_FORCE_PUSH=0 \
  python scripts/rsl_rl/eval_track_adapter_recovery.py \
    --task "$TASK" \
    --headless \
    --device "$DEVICE" \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --checkpoint "$checkpoint" \
    --disable_push \
    --scenario_set high_speed_jerk \
    --push_modes push \
    --warmup_steps "$SHORT_IMPACT_WARMUP_STEPS" \
    --move_steps "$SHORT_IMPACT_MOVE_STEPS" \
    --stop_steps "$SHORT_IMPACT_STOP_STEPS" \
    --eval_push_speed "$SHORT_IMPACT_PUSH_SPEED" \
    --eval_push_yaw "$SHORT_IMPACT_PUSH_YAW" \
    --eval_push_step "$SHORT_IMPACT_PUSH_STEP" \
    "${load_args[@]}" \
    --output_json "$out_dir/short_impact.json" \
    --output_csv "$out_dir/short_impact.csv" \
    2>&1 | tee "$out_dir/eval.log"
}

log "timestamp=${TIMESTAMP}"
log "out_root=${OUT_ROOT}"
log "checkpoint_specs=${CHECKPOINT_SPECS}"
log "num_envs=${NUM_ENVS} seed=${SEED} fail_on_gate=${FAIL_ON_GATE}"

read -r -a specs <<< "$CHECKPOINT_SPECS"
for spec in "${specs[@]}"; do
  IFS="=" read -r name checkpoint residual_scale load_mode hidden_dims adapter_mode extra <<< "$spec"
  load_mode="${load_mode:-strict}"
  hidden_dims="${hidden_dims:-${BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS:-128}}"
  adapter_mode="${adapter_mode:-${BOOSTER_TRACK_ADAPTER_MODE:-layerwise}}"
  if [[ -z "${name:-}" || -z "${checkpoint:-}" || -z "${residual_scale:-}" || -n "${extra:-}" ]]; then
    log "invalid checkpoint spec: $spec"
    exit 1
  fi
  if [[ ! -f "$checkpoint" ]]; then
    log "missing checkpoint ${name}: ${checkpoint}"
    exit 1
  fi
  candidate_out="${OUT_ROOT}/${name}"
  mkdir -p "$candidate_out"
  log "candidate=${name} checkpoint=${checkpoint} residual_scale=${residual_scale} load_mode=${load_mode} hidden_dims=${hidden_dims} adapter_mode=${adapter_mode}"
  BOOSTER_TRACK_ADAPTER_MODE="$adapter_mode" BOOSTER_TRACK_ADAPTER_RESIDUAL_SCALE="$residual_scale" BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS="$hidden_dims" run_no_push_gate "$name" "$checkpoint" "$load_mode" "$candidate_out"
  BOOSTER_TRACK_ADAPTER_MODE="$adapter_mode" BOOSTER_TRACK_ADAPTER_RESIDUAL_SCALE="$residual_scale" BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS="$hidden_dims" run_slow_push_gate "$name" "$checkpoint" "$load_mode" "$candidate_out"
  BOOSTER_TRACK_ADAPTER_MODE="$adapter_mode" BOOSTER_TRACK_ADAPTER_RESIDUAL_SCALE="$residual_scale" BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS="$hidden_dims" run_walking_push_gate "$name" "$checkpoint" "$load_mode" "$candidate_out"
  BOOSTER_TRACK_ADAPTER_MODE="$adapter_mode" BOOSTER_TRACK_ADAPTER_RESIDUAL_SCALE="$residual_scale" BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS="$hidden_dims" run_short_impact_gate "$name" "$checkpoint" "$load_mode" "$candidate_out"

  require_json "$candidate_out/no_push/no_push.json"
  require_json "$candidate_out/zero_command_slow_sustained_push/zero_command_slow_sustained_push.json"
  require_json "$candidate_out/walking_push/walking_push.json"
  require_json "$candidate_out/short_impact/short_impact.json"
done

OUT_ROOT="$OUT_ROOT" \
FAIL_ON_GATE="$FAIL_ON_GATE" \
MAX_NO_PUSH_FALL="$MAX_NO_PUSH_FALL" \
MAX_NO_PUSH_CONTACT="$MAX_NO_PUSH_CONTACT" \
MAX_WALK_PUSH_FALL="$MAX_WALK_PUSH_FALL" \
MAX_WALK_PUSH_CONTACT="$MAX_WALK_PUSH_CONTACT" \
MAX_SHORT_IMPACT_FALL="$MAX_SHORT_IMPACT_FALL" \
MAX_SHORT_IMPACT_CONTACT="$MAX_SHORT_IMPACT_CONTACT" \
MIN_SHORT_IMPACT_SETTLE="$MIN_SHORT_IMPACT_SETTLE" \
MAX_SLOW_PUSH_FALL="$MAX_SLOW_PUSH_FALL" \
MAX_SLOW_PUSH_CONTACT="$MAX_SLOW_PUSH_CONTACT" \
MIN_SLOW_PUSH_CLEAN="$MIN_SLOW_PUSH_CLEAN" \
NO_PUSH_COMMANDS="$NO_PUSH_COMMANDS" \
SLOW_PUSH_FORCE_N="$SLOW_PUSH_FORCE_N" \
SLOW_PUSH_START_STEP="$SLOW_PUSH_START_STEP" \
SLOW_PUSH_DURATION_STEPS="$SLOW_PUSH_DURATION_STEPS" \
SLOW_PUSH_FORCE_BODY_X="$SLOW_PUSH_FORCE_BODY_X" \
SLOW_PUSH_FORCE_BODY_Y="$SLOW_PUSH_FORCE_BODY_Y" \
SLOW_PUSH_FORCE_BODY_Z="$SLOW_PUSH_FORCE_BODY_Z" \
WALK_PUSH_CMD="$WALK_PUSH_CMD" \
WALK_PUSH_FORCE_N="$WALK_PUSH_FORCE_N" \
WALK_PUSH_START_STEP="$WALK_PUSH_START_STEP" \
WALK_PUSH_DURATION_STEPS="$WALK_PUSH_DURATION_STEPS" \
WALK_PUSH_FORCE_BODY_X="$WALK_PUSH_FORCE_BODY_X" \
WALK_PUSH_FORCE_BODY_Y="$WALK_PUSH_FORCE_BODY_Y" \
WALK_PUSH_FORCE_BODY_Z="$WALK_PUSH_FORCE_BODY_Z" \
SHORT_IMPACT_PUSH_SPEED="$SHORT_IMPACT_PUSH_SPEED" \
SHORT_IMPACT_PUSH_YAW="$SHORT_IMPACT_PUSH_YAW" \
SHORT_IMPACT_PUSH_STEP="$SHORT_IMPACT_PUSH_STEP" \
python - <<'PY'
import json
import os
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def max_key(rows, key):
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return max(values) if values else None


def min_key(rows, key):
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return min(values) if values else None


def mean_key(rows, key):
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def no_push_summary(path: Path, max_fall: float, max_contact: float):
    payload = load(path)
    summary = payload["summary"]
    rows = summary.get("summaries", [])
    worst_fall = max_key(rows, "ever_fall_rate_mean")
    worst_contact = max_key(rows, "ever_base_contact_rate_mean")
    worst_vx_error = max_key(rows, "abs_vx_error_mean")
    worst_vy_error = max_key(rows, "abs_vy_error_mean")
    worst_wz_error = max_key(rows, "abs_wz_error_mean")
    residual = mean_key(rows, "residual_norm_mean")
    passed = (worst_fall or 0.0) <= max_fall and (worst_contact or 0.0) <= max_contact
    return {
        "gate": "no_push",
        "passed": passed,
        "worst_fall_rate": worst_fall,
        "worst_base_contact_rate": worst_contact,
        "worst_abs_vx_error": worst_vx_error,
        "worst_abs_vy_error": worst_vy_error,
        "worst_abs_wz_error": worst_wz_error,
        "mean_residual_norm": residual,
        "thresholds": {"max_fall": max_fall, "max_base_contact": max_contact},
        "json": str(path),
    }


def slow_push_summary(path: Path, max_fall: float, max_contact: float, min_clean: float):
    payload = load(path)
    summary = payload["summary"]
    fall = summary.get("ever_fall_rate")
    contact = summary.get("ever_base_contact_rate")
    clean = summary.get("clean_rate")
    passed = (fall or 0.0) <= max_fall and (contact or 0.0) <= max_contact and (clean or 0.0) >= min_clean
    return {
        "gate": "zero_command_slow_sustained_push",
        "passed": passed,
        "fall_rate": fall,
        "base_contact_rate": contact,
        "clean_rate": clean,
        "support_step_success_rate": summary.get("support_step_success_rate"),
        "clean_support_step_success_rate": summary.get("clean_support_step_success_rate"),
        "max_contacted_step_projection_mean": summary.get("max_contacted_step_projection_mean"),
        "max_speed_xy_mean": summary.get("max_speed_xy_mean"),
        "final_speed_xy_mean": summary.get("final_speed_xy_mean"),
        "force_n": summary.get("force_n"),
        "push_duration_s": summary.get("push_duration_s"),
        "push_metric_mode": summary.get("push_metric_mode"),
        "thresholds": {"max_fall": max_fall, "max_base_contact": max_contact, "min_clean": min_clean},
        "json": str(path),
    }


def walking_push_summary(path: Path, max_fall: float, max_contact: float):
    payload = load(path)
    summary = payload["summary"]
    rows = summary.get("summaries", [])
    if rows:
        worst_fall = max_key(rows, "ever_fall_rate_mean")
        worst_contact = max_key(rows, "ever_base_contact_rate_mean")
        worst_vx_error = max_key(rows, "abs_vx_error_mean")
        worst_vy_error = max_key(rows, "abs_vy_error_mean")
        worst_wz_error = max_key(rows, "abs_wz_error_mean")
        residual = mean_key(rows, "residual_norm_mean")
        support_step = None
        clean_support_step = None
    else:
        sample_rows = payload.get("rows", [])
        worst_fall = summary.get("ever_fall_rate")
        worst_contact = summary.get("ever_base_contact_rate")
        worst_vx_error = summary.get("final_abs_vx_error_mean")
        worst_vy_error = summary.get("final_abs_vy_error_mean")
        worst_wz_error = summary.get("final_abs_wz_error_mean")
        residual = mean_key(sample_rows, "residual_norm_mean")
        support_step = summary.get("support_step_success_rate")
        clean_support_step = summary.get("clean_support_step_success_rate")
    passed = (worst_fall or 0.0) <= max_fall and (worst_contact or 0.0) <= max_contact
    return {
        "gate": "walking_push",
        "passed": passed,
        "worst_fall_rate": worst_fall,
        "worst_base_contact_rate": worst_contact,
        "worst_abs_vx_error": worst_vx_error,
        "worst_abs_vy_error": worst_vy_error,
        "worst_abs_wz_error": worst_wz_error,
        "mean_residual_norm": residual,
        "support_step_success_rate": support_step,
        "clean_support_step_success_rate": clean_support_step,
        "force_push_enabled": summary.get("force_push_enabled"),
        "thresholds": {"max_fall": max_fall, "max_base_contact": max_contact},
        "json": str(path),
    }


def short_impact_summary(path: Path, max_fall: float, max_contact: float, min_settle: float):
    payload = load(path)
    summaries = as_list(payload.get("summary")) or payload.get("summaries", [])
    worst_fall = max_key(summaries, "fall_rate")
    worst_contact = max_key(summaries, "base_contact_rate")
    min_settle_valid = min_key(summaries, "settle_success_rate_valid_envs")
    worst_pre_stop = max_key(summaries, "pre_stop_failure_rate")
    passed = (
        (worst_fall or 0.0) <= max_fall
        and (worst_contact or 0.0) <= max_contact
        and (min_settle_valid if min_settle_valid is not None else 0.0) >= min_settle
    )
    return {
        "gate": "short_impact",
        "passed": passed,
        "worst_fall_rate": worst_fall,
        "worst_base_contact_rate": worst_contact,
        "worst_pre_stop_failure_rate": worst_pre_stop,
        "min_settle_success_rate_valid_envs": min_settle_valid,
        "thresholds": {
            "max_fall": max_fall,
            "max_base_contact": max_contact,
            "min_settle_success_valid_envs": min_settle,
        },
        "json": str(path),
    }


root = Path(os.environ["OUT_ROOT"])
thresholds = {
    "max_no_push_fall": float(os.environ["MAX_NO_PUSH_FALL"]),
    "max_no_push_contact": float(os.environ["MAX_NO_PUSH_CONTACT"]),
    "max_walking_push_fall": float(os.environ["MAX_WALK_PUSH_FALL"]),
    "max_walking_push_contact": float(os.environ["MAX_WALK_PUSH_CONTACT"]),
    "max_short_impact_fall": float(os.environ["MAX_SHORT_IMPACT_FALL"]),
    "max_short_impact_contact": float(os.environ["MAX_SHORT_IMPACT_CONTACT"]),
    "min_short_impact_settle": float(os.environ["MIN_SHORT_IMPACT_SETTLE"]),
    "max_slow_push_fall": float(os.environ["MAX_SLOW_PUSH_FALL"]),
    "max_slow_push_contact": float(os.environ["MAX_SLOW_PUSH_CONTACT"]),
    "min_slow_push_clean": float(os.environ["MIN_SLOW_PUSH_CLEAN"]),
}
candidate_results = []
flat_rows = []

for candidate_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    gates = [
        no_push_summary(
            candidate_dir / "no_push" / "no_push.json",
            thresholds["max_no_push_fall"],
            thresholds["max_no_push_contact"],
        ),
        slow_push_summary(
            candidate_dir / "zero_command_slow_sustained_push" / "zero_command_slow_sustained_push.json",
            thresholds["max_slow_push_fall"],
            thresholds["max_slow_push_contact"],
            thresholds["min_slow_push_clean"],
        ),
        walking_push_summary(
            candidate_dir / "walking_push" / "walking_push.json",
            thresholds["max_walking_push_fall"],
            thresholds["max_walking_push_contact"],
        ),
        short_impact_summary(
            candidate_dir / "short_impact" / "short_impact.json",
            thresholds["max_short_impact_fall"],
            thresholds["max_short_impact_contact"],
            thresholds["min_short_impact_settle"],
        ),
    ]
    for gate in gates:
        gate_path = candidate_dir / gate["gate"] / "gate_result.json"
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        flat_rows.append({"candidate": candidate_dir.name, **gate})
    candidate_results.append(
        {
            "candidate": candidate_dir.name,
            "passed": all(gate["passed"] for gate in gates),
            "gates": gates,
        }
    )

summary = {
    "output_root": str(root),
    "thresholds": thresholds,
    "passed": all(candidate["passed"] for candidate in candidate_results),
    "candidates": candidate_results,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

lines = [
    "# Recovery-controller fixed gate comparison",
    "",
    f"- output: `{root}`",
    f"- overall pass: `{summary['passed']}`",
    "",
    "## Push/eval strength",
    "",
    f"- no_push: no external push; commands `{os.environ.get('NO_PUSH_COMMANDS', '')}`",
    f"- zero_command_slow_sustained_push: `{os.environ.get('SLOW_PUSH_FORCE_N', '')}N` body force, duration `{os.environ.get('SLOW_PUSH_DURATION_STEPS', '')}` steps, start step `{os.environ.get('SLOW_PUSH_START_STEP', '')}`, direction body `({os.environ.get('SLOW_PUSH_FORCE_BODY_X', '')}, {os.environ.get('SLOW_PUSH_FORCE_BODY_Y', '')}, {os.environ.get('SLOW_PUSH_FORCE_BODY_Z', '')})`",
    f"- walking_push: command `{os.environ.get('WALK_PUSH_CMD', '')}`, `{os.environ.get('WALK_PUSH_FORCE_N', '')}N` body force, duration `{os.environ.get('WALK_PUSH_DURATION_STEPS', '')}` steps, start step `{os.environ.get('WALK_PUSH_START_STEP', '')}`, direction body `({os.environ.get('WALK_PUSH_FORCE_BODY_X', '')}, {os.environ.get('WALK_PUSH_FORCE_BODY_Y', '')}, {os.environ.get('WALK_PUSH_FORCE_BODY_Z', '')})`",
    f"- short_impact: root-velocity impulse speed `{os.environ.get('SHORT_IMPACT_PUSH_SPEED', '')}m/s`, yaw `{os.environ.get('SHORT_IMPACT_PUSH_YAW', '')}rad/s`, at stop step `{os.environ.get('SHORT_IMPACT_PUSH_STEP', '')}`",
    "",
    "| candidate | gate | pass | fall/contact | recovery | extra |",
    "|---|---|---:|---|---|---|",
]
for row in flat_rows:
    gate = row["gate"]
    if gate == "zero_command_slow_sustained_push":
        fall_contact = f"{row['fall_rate']:.4f}/{row['base_contact_rate']:.4f}"
        recovery = f"clean={row['clean_rate']:.4f}, support={row['support_step_success_rate']:.4f}"
        extra = f"final_speed={row['final_speed_xy_mean']:.4f}"
    elif gate == "short_impact":
        fall_contact = f"{row['worst_fall_rate']:.4f}/{row['worst_base_contact_rate']:.4f}"
        recovery_value = row["min_settle_success_rate_valid_envs"]
        recovery = "settle_valid=" + ("" if recovery_value is None else f"{recovery_value:.4f}")
        extra = f"pre_stop={row['worst_pre_stop_failure_rate']:.4f}"
    elif gate == "walking_push" and row.get("support_step_success_rate") is not None:
        fall_contact = f"{row['worst_fall_rate']:.4f}/{row['worst_base_contact_rate']:.4f}"
        recovery = f"support={row['support_step_success_rate']:.4f}, clean_support={row['clean_support_step_success_rate']:.4f}"
        extra = f"vx_err={row['worst_abs_vx_error']:.4f}, vy_err={row['worst_abs_vy_error']:.4f}"
    else:
        fall_contact = f"{row['worst_fall_rate']:.4f}/{row['worst_base_contact_rate']:.4f}"
        recovery = f"vx_err={row['worst_abs_vx_error']:.4f}, vy_err={row['worst_abs_vy_error']:.4f}"
        extra = f"wz_err={row['worst_abs_wz_error']:.4f}"
    lines.append(f"| {row['candidate']} | {gate} | {row['passed']} | {fall_contact} | {recovery} | {extra} |")

(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

if os.environ["FAIL_ON_GATE"] == "1" and not summary["passed"]:
    raise SystemExit("one or more recovery-controller gates failed")
PY

log "complete summary=${OUT_ROOT}/summary.json"

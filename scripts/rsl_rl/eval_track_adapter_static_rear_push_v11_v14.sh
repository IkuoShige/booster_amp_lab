#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/booster_amp_lab}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/workspace/Isaac_uv_template/.venv/bin/activate}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/logs/static_rear_push_v11_v14_${TIMESTAMP}}"

V11_CHECKPOINT="${V11_CHECKPOINT:-${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter/2026-05-14_15-16-50_track_adapter_81999_stage3_recovery_v11_adaptive_force_kick_20260514_151645_ppo_adaptive_force_kick/model_6500.pt}"
V14_CHECKPOINT="${V14_CHECKPOINT:-${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter/2026-05-15_10-17-58_track_adapter_81999_stage3_recovery_v14_capture_step_20260515_081803_ppo_capture_step/model_10000.pt}"

FORCES="${FORCES:-100 150}"
NUM_ENVS="${NUM_ENVS:-64}"
PUSH_START_STEP="${PUSH_START_STEP:-80}"
PUSH_DURATION_STEPS="${PUSH_DURATION_STEPS:-5}"
POST_PUSH_STEPS="${POST_PUSH_STEPS:-170}"
FORCE_BODY_X="${FORCE_BODY_X:-1.0}"
FORCE_BODY_Y="${FORCE_BODY_Y:-0.0}"
FORCE_BODY_Z="${FORCE_BODY_Z:-0.0}"
SEED="${SEED:-123}"

cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

export PYTHONPATH="${REPO_ROOT}/rsl_rl:${REPO_ROOT}/source/booster_rl_tasks:${REPO_ROOT}/booster_assets/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export BOOSTER_TRACK_ADAPTER_FULL_TRAINING=1
export BOOSTER_TRACK_ADAPTER_SMOKE_PASSED=1
export BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING=1
export BOOSTER_TRACK_ADAPTER_SMOKE_MODE=0
export BOOSTER_TRACK_ADAPTER_BASE_CHECKPOINT="${BOOSTER_TRACK_ADAPTER_BASE_CHECKPOINT:-${REPO_ROOT}/logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt}"
export BOOSTER_TRACK_ADAPTER_MODE="${BOOSTER_TRACK_ADAPTER_MODE:-layerwise}"
export BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS="${BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS:-128}"
export BOOSTER_TRACK_ADAPTER_LAYERWISE_OUTPUT_INIT_SCALE="${BOOSTER_TRACK_ADAPTER_LAYERWISE_OUTPUT_INIT_SCALE:-0.0}"
export BOOSTER_TRACK_ADAPTER_PUSH=0
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH=0
export BOOSTER_TRACK_ADAPTER_BASE_CONTACT_DURATION_S="${BOOSTER_TRACK_ADAPTER_BASE_CONTACT_DURATION_S:-0.0}"

mkdir -p "$OUT_ROOT"

echo "[static_rear_push] out=${OUT_ROOT}"
echo "[static_rear_push] forces=${FORCES}"
echo "[static_rear_push] direction_body=(${FORCE_BODY_X},${FORCE_BODY_Y},${FORCE_BODY_Z})"

run_one() {
  local name="$1"
  local checkpoint="$2"
  local residual_scale="$3"
  local force_n="$4"
  local out_dir="${OUT_ROOT}/${name}/F${force_n}"
  mkdir -p "$out_dir"
  echo "[static_rear_push] start ${name} force=${force_n} checkpoint=${checkpoint} residual_scale=${residual_scale}"
  BOOSTER_TRACK_ADAPTER_RESIDUAL_SCALE="$residual_scale" \
  python scripts/rsl_rl/eval_track_adapter_static_push_step.py \
    --task Booster-Run-AMP-TrackAdapter-v0 \
    --headless \
    --device cuda:0 \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --checkpoint "$checkpoint" \
    --force_n "$force_n" \
    --force_body_x "$FORCE_BODY_X" \
    --force_body_y "$FORCE_BODY_Y" \
    --force_body_z "$FORCE_BODY_Z" \
    --push_start_step "$PUSH_START_STEP" \
    --push_duration_steps "$PUSH_DURATION_STEPS" \
    --post_push_steps "$POST_PUSH_STEPS" \
    --output_json "${out_dir}/static_push_step.json" \
    --output_csv "${out_dir}/static_push_step.csv" \
    2>&1 | tee "${out_dir}/eval.log"
  echo "[static_rear_push] done ${name} force=${force_n}"
}

for force_n in $FORCES; do
  run_one "v11_model_6500" "$V11_CHECKPOINT" "0.16" "$force_n"
  run_one "v14_model_10000" "$V14_CHECKPOINT" "0.20" "$force_n"
done

OUT_ROOT="$OUT_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
rows = []
for path in sorted(root.glob("*/F*/static_push_step.json")):
    payload = json.loads(path.read_text())
    summary = payload["summary"]
    rows.append(
        {
            "candidate": path.parts[-3],
            "force_n": summary["force_n"],
            "fall": summary["ever_fall_rate"],
            "base_contact": summary["ever_base_contact_rate"],
            "clean": summary["clean_rate"],
            "support_step": summary["support_step_success_rate"],
            "clean_support_step": summary["clean_support_step_success_rate"],
            "max_contacted_step": summary["max_contacted_step_projection_mean"],
            "max_step": summary["max_step_projection_mean"],
            "final_speed": summary["final_speed_xy_mean"],
        }
    )

lines = [
    "# Static rear-push step evaluation",
    "",
    f"- output: `{root}`",
    "- command: zero velocity",
    "- force direction: body-frame +x by default. If the robot convention is reversed, rerun with `FORCE_BODY_X=-1.0`.",
    "",
    "| candidate | force N | fall | base contact | clean | support step | clean support step | contacted step m | max step m | final speed m/s |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {candidate} | {force_n:.0f} | {fall:.4f} | {base_contact:.4f} | {clean:.4f} | "
        "{support_step:.4f} | {clean_support_step:.4f} | {max_contacted_step:.4f} | "
        "{max_step:.4f} | {final_speed:.4f} |".format(**row)
    )

(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo "[static_rear_push] complete summary=${OUT_ROOT}/summary.md"

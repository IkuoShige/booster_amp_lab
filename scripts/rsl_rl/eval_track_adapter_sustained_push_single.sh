#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/booster_amp_lab}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/workspace/Isaac_uv_template/.venv/bin/activate}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
NAME="${NAME:-track_adapter}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/logs/sustained_rear_push_${NAME}_${TIMESTAMP}}"

: "${CHECKPOINT:?Set CHECKPOINT to the Track Adapter checkpoint to evaluate.}"

FORCES="${FORCES:-10 20 30 50}"
MODES="${MODES:-none event}"
NUM_ENVS="${NUM_ENVS:-32}"
SEED="${SEED:-321}"
PUSH_START_STEP="${PUSH_START_STEP:-80}"
PUSH_DURATION_STEPS="${PUSH_DURATION_STEPS:-125}"
POST_PUSH_STEPS="${POST_PUSH_STEPS:-100}"
RECOVERY_WINDOW_S="${RECOVERY_WINDOW_S:-4.0}"
FORCE_BODY_X="${FORCE_BODY_X:-1.0}"
FORCE_BODY_Y="${FORCE_BODY_Y:-0.0}"
FORCE_BODY_Z="${FORCE_BODY_Z:-0.0}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.165}"
ADAPTER_MODE="${ADAPTER_MODE:-layerwise}"

cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

export PYTHONPATH="${REPO_ROOT}/rsl_rl:${REPO_ROOT}/source/booster_rl_tasks:${REPO_ROOT}/booster_assets/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export BOOSTER_TRACK_ADAPTER_FULL_TRAINING=1
export BOOSTER_TRACK_ADAPTER_SMOKE_PASSED=1
export BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING=1
export BOOSTER_TRACK_ADAPTER_SMOKE_MODE=0
export BOOSTER_TRACK_ADAPTER_BASE_CHECKPOINT="${BOOSTER_TRACK_ADAPTER_BASE_CHECKPOINT:-${REPO_ROOT}/logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt}"
export BOOSTER_TRACK_ADAPTER_PUSH=0
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH=0
export BOOSTER_TRACK_ADAPTER_BASE_CONTACT_DURATION_S="${BOOSTER_TRACK_ADAPTER_BASE_CONTACT_DURATION_S:-0.0}"
export BOOSTER_TRACK_ADAPTER_MODE="$ADAPTER_MODE"
export BOOSTER_TRACK_ADAPTER_RESIDUAL_SCALE="$RESIDUAL_SCALE"
export BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS="${BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS:-128}"
export BOOSTER_TRACK_ADAPTER_LAYERWISE_OUTPUT_INIT_SCALE="${BOOSTER_TRACK_ADAPTER_LAYERWISE_OUTPUT_INIT_SCALE:-0.0}"

mkdir -p "$OUT_ROOT"

echo "[sustained_push_single] checkpoint=${CHECKPOINT}" | tee "$OUT_ROOT/eval_all.log"
echo "[sustained_push_single] out=${OUT_ROOT}" | tee -a "$OUT_ROOT/eval_all.log"
echo "[sustained_push_single] forces=${FORCES}" | tee -a "$OUT_ROOT/eval_all.log"
echo "[sustained_push_single] modes=${MODES}" | tee -a "$OUT_ROOT/eval_all.log"

for mode in $MODES; do
  for force_n in $FORCES; do
    out_dir="${OUT_ROOT}/${mode}/${NAME}/F${force_n}"
    mkdir -p "$out_dir"
    echo "[sustained_push_single] start mode=${mode} force=${force_n}" | tee -a "$OUT_ROOT/eval_all.log"
    python scripts/rsl_rl/eval_track_adapter_static_push_step.py \
      --task Booster-Run-AMP-TrackAdapter-v0 \
      --headless \
      --device cuda:0 \
      --num_envs "$NUM_ENVS" \
      --seed "$SEED" \
      --checkpoint "$CHECKPOINT" \
      --force_n "$force_n" \
      --force_body_x "$FORCE_BODY_X" \
      --force_body_y "$FORCE_BODY_Y" \
      --force_body_z "$FORCE_BODY_Z" \
      --push_start_step "$PUSH_START_STEP" \
      --push_duration_steps "$PUSH_DURATION_STEPS" \
      --post_push_steps "$POST_PUSH_STEPS" \
      --recovery_window_s "$RECOVERY_WINDOW_S" \
      --push_metric_mode "$mode" \
      --output_json "${out_dir}/sustained_push_step.json" \
      --output_csv "${out_dir}/sustained_push_step.csv" \
      2>&1 | tee "${out_dir}/eval.log"
    echo "[sustained_push_single] done mode=${mode} force=${force_n}" | tee -a "$OUT_ROOT/eval_all.log"
  done
done

OUT_ROOT="$OUT_ROOT" NAME="$NAME" CHECKPOINT="$CHECKPOINT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
name = os.environ["NAME"]
checkpoint = os.environ["CHECKPOINT"]
rows = []
for path in sorted(root.glob("*/*/F*/sustained_push_step.json")):
    payload = json.loads(path.read_text())
    summary = payload["summary"]
    rows.append(
        {
            "mode": path.parts[-4],
            "force_n": float(summary["force_n"]),
            "fall": float(summary["ever_fall_rate"]),
            "base_contact": float(summary["ever_base_contact_rate"]),
            "clean": float(summary["clean_rate"]),
            "support_step": float(summary["support_step_success_rate"]),
            "clean_support_step": float(summary["clean_support_step_success_rate"]),
            "contacted_step": float(summary["max_contacted_step_projection_mean"]),
            "max_speed": float(summary["max_speed_xy_mean"]),
            "final_speed": float(summary["final_speed_xy_mean"]),
            "residual_mean": summary.get("residual_norm_mean"),
        }
    )

lines = [
    "# Sustained rear-push evaluation",
    "",
    f"- output: `{root}`",
    f"- candidate: `{name}`",
    f"- checkpoint: `{checkpoint}`",
    "- command: zero velocity",
    "- force direction: body-frame +x",
    "- force duration: 2.5s by default",
    "",
    "| mode | force N | fall | base contact | clean | support step | clean support step | contacted step m | max speed m/s | final speed m/s | residual mean |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    residual = row["residual_mean"]
    residual_text = "" if residual is None else f"{float(residual):.4f}"
    lines.append(
        "| {mode} | {force_n:.0f} | {fall:.4f} | {base_contact:.4f} | {clean:.4f} | "
        "{support_step:.4f} | {clean_support_step:.4f} | {contacted_step:.4f} | "
        "{max_speed:.4f} | {final_speed:.4f} | {residual} |".format(
            residual=residual_text,
            **row,
        )
    )

(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo "[sustained_push_single] complete summary=${OUT_ROOT}/summary.md"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/booster_amp_lab}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/workspace/Isaac_uv_template/.venv/bin/activate}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
NAME="${NAME:-ta_v14_single_kick_gate}"
FORCES="${FORCES:-150 200 250 300 400 500}"
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/logs/single_kick_gate_${NAME}_${TIMESTAMP}}"
: "${CHECKPOINT:?Set CHECKPOINT to the Track Adapter checkpoint to evaluate.}"

cd "$REPO_ROOT"
source "$VENV_ACTIVATE"

export BOOSTER_TRACK_ADAPTER_BASE_CHECKPOINT="${BOOSTER_TRACK_ADAPTER_BASE_CHECKPOINT:-${REPO_ROOT}/logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt}"
export BOOSTER_TRACK_ADAPTER_MODE="${BOOSTER_TRACK_ADAPTER_MODE:-layerwise}"
export BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS="${BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS:-128}"
export BOOSTER_TRACK_ADAPTER_LAYERWISE_OUTPUT_INIT_SCALE="${BOOSTER_TRACK_ADAPTER_LAYERWISE_OUTPUT_INIT_SCALE:-0.0}"

export BOOSTER_TRACK_ADAPTER_PUSH=0
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH=1
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_MAX_PULSES_PER_EPISODE=1
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_GLOBAL_FRAME="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_GLOBAL_FRAME:-0}"
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_TORQUE_Z_ABS="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_TORQUE_Z_ABS:-0}"
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_TORQUE_Z_ABS="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_TORQUE_Z_ABS:-0}"
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_POS_X_ABS="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_POS_X_ABS:-0}"
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_POS_Y_ABS="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_POS_Y_ABS:-0}"
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FORCE_Z_ABS="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FORCE_Z_ABS:-0}"
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_EQUIV_YAW_ABS="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_EQUIV_YAW_ABS:-0}"
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_EQUIV_YAW_ABS="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_EQUIV_YAW_ABS:-0}"
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_INTERVAL_MIN_S="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_INTERVAL_MIN_S:-1.0}"
export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_INTERVAL_MAX_S="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_INTERVAL_MAX_S:-1.0}"
export BOOSTER_TRACK_ADAPTER_BASE_CONTACT_DURATION_S="${BOOSTER_TRACK_ADAPTER_BASE_CONTACT_DURATION_S:-0.0}"

mkdir -p "$OUT_BASE"

echo "[single_kick_gate] checkpoint=${CHECKPOINT}"
echo "[single_kick_gate] out_base=${OUT_BASE}"
echo "[single_kick_gate] forces=${FORCES}"

for force_n in $FORCES; do
  export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_MIN_N="$force_n"
  export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_MAX_N="$force_n"
  export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_MIN_N="$force_n"
  export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_MAX_N="$force_n"
  export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_DURATION_MIN_S="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_DURATION_MIN_S:-0.10}"
  export BOOSTER_TRACK_ADAPTER_FORCE_PUSH_DURATION_MAX_S="${BOOSTER_TRACK_ADAPTER_FORCE_PUSH_DURATION_MAX_S:-0.10}"

  force_out="${OUT_BASE}/F${force_n}"
  mkdir -p "$force_out"
  echo "[single_kick_gate] force=${force_n}N out=${force_out}"
  NUM_ENVS="${NUM_ENVS:-64}" \
  NAME="${NAME}_F${force_n}" \
  OUT="$force_out" \
  COMMANDS="${COMMANDS:-0.0,0.0,0.0;0.5,0.0,0.0;1.0,0.0,0.0;2.0,0.0,0.0;0.0,1.0,0.0;0.0,0.0,1.5}" \
  STEPS="${STEPS:-360}" \
  SETTLE_STEPS="${SETTLE_STEPS:-120}" \
  CHECKPOINT="$CHECKPOINT" \
  scripts/rsl_rl/eval_track_adapter_v10_force_kick_grid.sh
done

python - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["OUT_BASE"])
required_force = float(os.getenv("REQUIRED_FORCE_N", "0"))
max_worst_fall = float(os.getenv("MAX_WORST_FALL", "0.02"))
max_worst_contact = float(os.getenv("MAX_WORST_CONTACT", "0.02"))
rows = []
for path in sorted(base.glob("F*/velocity_force_kick.json"), key=lambda p: int(p.parent.name[1:])):
    force = int(path.parent.name[1:])
    data = json.loads(path.read_text())
    summaries = data["summary"]["summaries"]
    worst_fall = max(row["ever_fall_rate_mean"] for row in summaries)
    worst_contact = max(row["ever_base_contact_rate_mean"] for row in summaries)
    rows.append((force, worst_fall, worst_contact))
print("[single_kick_gate] summary force,worst_fall,worst_contact")
for row in rows:
    print(f"[single_kick_gate] {row[0]},{row[1]:.4f},{row[2]:.4f}")
if required_force > 0:
    passed = [
        force >= required_force and worst_fall <= max_worst_fall and worst_contact <= max_worst_contact
        for force, worst_fall, worst_contact in rows
    ]
    if not any(passed):
        raise SystemExit(
            f"single-kick gate failed: no force >= {required_force:g}N met "
            f"fall<={max_worst_fall:g}, contact<={max_worst_contact:g}"
        )
PY

echo "[single_kick_gate] done out_base=${OUT_BASE}"

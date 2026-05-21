#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/booster_amp_lab}"
VENV_ACTIVATE="${VENV_ACTIVATE:-/workspace/Isaac_uv_template/.venv/bin/activate}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="${REPO_ROOT}/logs/eval_track_adapter_v38_candidates_${TIMESTAMP}"
V38_DIR="${REPO_ROOT}/logs/rsl_rl/run_amp_track_adapter/2026-05-20_06-28-02_track_adapter_81999_stage3_recovery_v38_strict_survival_consolidate_20260520_062758_ppo_consolidate"

FORCES="${FORCES:-50 100 120}"
MODES="${MODES:-none event}"
NUM_ENVS="${NUM_ENVS:-16}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.32}"
ADAPTER_MODE="${ADAPTER_MODE:-layerwise}"

mkdir -p "$RUN_DIR"

evaluate_candidate() {
  local name="$1"
  local checkpoint="$2"
  local out_root="${RUN_DIR}/${name}"

  CHECKPOINT="$checkpoint" \
  NAME="$name" \
  OUT_ROOT="$out_root" \
  FORCES="$FORCES" \
  MODES="$MODES" \
  NUM_ENVS="$NUM_ENVS" \
  RESIDUAL_SCALE="$RESIDUAL_SCALE" \
  ADAPTER_MODE="$ADAPTER_MODE" \
  REPO_ROOT="$REPO_ROOT" \
  VENV_ACTIVATE="$VENV_ACTIVATE" \
  bash "${REPO_ROOT}/scripts/rsl_rl/eval_track_adapter_sustained_push_single.sh"
}

evaluate_candidate "v38_model_850" "${V38_DIR}/model_850.pt"
evaluate_candidate "v38_model_1100" "${V38_DIR}/model_1100.pt"

{
  echo "# v38 candidate sustained-push comparison"
  echo
  echo "- timestamp: \`${TIMESTAMP}\`"
  echo "- forces: \`${FORCES}\`"
  echo "- modes: \`${MODES}\`"
  echo "- num_envs: \`${NUM_ENVS}\`"
  echo "- residual_scale: \`${RESIDUAL_SCALE}\`"
  echo
  for summary in "${RUN_DIR}"/*/summary.md; do
    echo "## $(basename "$(dirname "$summary")")"
    echo
    tail -n +9 "$summary"
    echo
  done
} > "${RUN_DIR}/summary.md"

echo "[eval_v38_candidates] summary=${RUN_DIR}/summary.md"

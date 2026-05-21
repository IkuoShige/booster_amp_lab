# v44 Recovery Controller Reproduction Notes

Date: 2026-05-21

This note pins the current v44 recovery-controller chain and answers the
important baseline question: the frozen AMP base checkpoint `model_81999.pt`
already had AMP-side push/disturbance/recovery training primitives enabled, but
it did not include the later Track Adapter recovery-controller residual.

The many `scripts/rsl_rl/run_*.sh` and `scripts/rsl_rl/watch_*.sh` files are
local convenience wrappers and are intentionally not git-managed.  For a clean
reproduction, treat the saved run `params/*.yaml`, saved run `git/*.diff`, and
this document as the tracked source of truth.

## Checkpoint Chain

### AMP base fine-tune

Frozen base used by v44:

```text
logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt
```

It resumed from:

```text
logs/rsl_rl/run_amp_y/2026-05-04_19-10-18_resume56_yawlow_zero_stand/model_70000.pt
```

Evidence that `model_81999.pt` was a plain AMP fine-tune:

- `params/agent.yaml` uses `runner_class_name: AmpOnPolicyRunner`;
- policy is `ActorCritic`;
- algorithm is `AMPPPO`;
- optimizer was not restored from the source checkpoint.

Evidence that AMP-side disturbance/recovery primitives were already active:

- `params/env.yaml` includes `events.push_robot` as an interval root-velocity push;
- `push_robot.params.failure_mining_command_name` is `base_velocity`;
- push range is `x/y = +/-0.4`, `yaw = +/-0.25`, interval `5.0-8.0s`;
- `params/env.yaml` includes `rewards.push_recovery_velocity_track_exp` with weight `1.1`;
- zero-command velocity/joint-velocity penalties use `skip_push_active: true`;
- training logs recorded non-zero `failure_push_active` and non-zero
  `push_recovery_velocity_track_exp`.

So the answer is:

```text
AMP base model_81999.pt: yes, trained with AMP-side push/disturbance/recovery primitives.
AMP base model_81999.pt: no, not trained with v44 Track Adapter recovery-controller residual.
```

The AMP base improved low-speed/single-axis tracking, but was not promoted as a
standalone robust base because high-speed/no-push and jerk-smoke robustness
regressed.  The later adapter branch intentionally used `model_81999.pt` as the
best tracking base and assigned push/contact correction to Track Adapter.

### v44 Stage3 world model

Run:

```text
logs/rsl_rl/run_amp_track_adapter/2026-05-20_16-20-56_track_adapter_81999_stage3_recovery_v44_recovery_controller_20260520_162052_wm_pretrain_recovery_controller
```

Checkpoint kept:

```text
model_699.pt
```

Key settings:

- frozen base checkpoint: AMP `model_81999.pt`;
- history length: `79`;
- world-model horizon: `20`;
- pretrain iterations: `700`;
- pretrain envs: `1024`;
- pretrain steps per env: `128`;
- world-model replay buffer: `262144`.

### v44 PPO adapter

Run:

```text
logs/rsl_rl/run_amp_track_adapter/2026-05-20_18-04-18_track_adapter_81999_stage3_recovery_v44_recovery_controller_20260520_162052_ppo_recovery_controller
```

Current deploy-positive checkpoint:

```text
model_2475.pt
```

Key settings from `params/agent.yaml`:

- policy: `TrackAdapterActorCritic`;
- algorithm: `TrackAdapterAMPPPO`;
- `freeze_base: true`;
- `base_checkpoint_path`: AMP `model_81999.pt`;
- `history_length: 79`;
- `adapter_mode: layerwise`;
- `residual_scale: 0.22`;
- PPO steps per env: `48`;
- mini-batches: `48`;
- learning epochs: `2`;
- max iterations: `5200`;
- save interval: `25`.

The run stopped at iteration `2475/5200` by the adaptive safety gate because
scaled residual action norm exceeded the configured threshold.  This checkpoint
is not a fixed-eval promotion checkpoint, but it is the current real-robot
positive v44 artifact.

## Reproduction Commands

Always run training in tmux and use the Isaac uv environment:

```bash
cd /workspace/booster_amp_lab
source /workspace/Isaac_uv_template/.venv/bin/activate
export PYTHONPATH=/workspace/booster_amp_lab/rsl_rl:/workspace/booster_amp_lab/source/booster_rl_tasks:/workspace/booster_amp_lab/booster_assets/src:${PYTHONPATH:-}
```

### Rebuild AMP base `model_81999.pt`

The local wrapper used for the original run was:

```bash
tmux new -s amp_axis_ft_v2_repro \
  'cd /workspace/booster_amp_lab && source /workspace/Isaac_uv_template/.venv/bin/activate && TIMESTAMP=REPRO_AMP81999 RUN_NAME=amp_axis_ft_v2_REPRO MAX_ITERATIONS=12000 BASE_LOAD_RUN=2026-05-04_19-10-18_resume56_yawlow_zero_stand BASE_CHECKPOINT=model_70000.pt scripts/rsl_rl/run_amp_axis_ft.sh'
```

Equivalent critical settings:

```bash
BOOSTER_AMP_FT_ENABLE=1
BOOSTER_AMP_LOAD_OPTIMIZER=0
BOOSTER_AMP_FT_PUSH=1
BOOSTER_AMP_FT_PUSH_XY=0.40
BOOSTER_AMP_FT_PUSH_YAW=0.25
BOOSTER_AMP_FT_PUSH_INTERVAL_MIN=5.0
BOOSTER_AMP_FT_PUSH_INTERVAL_MAX=8.0
BOOSTER_AMP_FT_PUSH_ACTIVE_WINDOW_S=2.5
BOOSTER_AMP_FT_FAILURE_PUSH_SAMPLE_PROB=0.20
BOOSTER_AMP_FT_HARD_ENVS=0.55
BOOSTER_AMP_FT_STANDING_ENVS=0.12
BOOSTER_AMP_FT_LOW_SPEED_ENVS=0.16
BOOSTER_AMP_FT_SUDDEN_STOP_ENVS=0.0
BOOSTER_AMP_FT_LR=2.0e-4
BOOSTER_AMP_FT_ENTROPY=0.003
```

Training command shape:

```bash
python scripts/rsl_rl/train.py \
  --task Booster-Run-AMP-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --seed 42 \
  --resume \
  --load_run 2026-05-04_19-10-18_resume56_yawlow_zero_stand \
  --checkpoint model_70000.pt \
  --run_name amp_axis_ft_v2_REPRO \
  --max_iterations 12000 \
  --logger tensorboard
```

### Rebuild v44 adapter

Original local wrapper shape:

```bash
tmux new -s track_adapter_v44_repro \
  'cd /workspace/booster_amp_lab && source /workspace/Isaac_uv_template/.venv/bin/activate && scripts/rsl_rl/run_track_adapter_stage3_recovery_v44_recovery_controller.sh track_adapter_81999_stage3_recovery_v44_recovery_controller_REPRO'
```

Equivalent critical settings:

```bash
BOOSTER_TRACK_ADAPTER_BASE_CHECKPOINT=/workspace/booster_amp_lab/logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt
BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER=1
BOOSTER_TRACK_ADAPTER_RECOVERY_STATE_MACHINE=1
BOOSTER_TRACK_ADAPTER_RECOVERY_PHASES=nominal,recovery,return
BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE=1
BOOSTER_TRACK_ADAPTER_RESIDUAL_ACTION_GATE=1
BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_STABLE_MAX=0.0
BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PUSH_MIN=0.96
BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PHASE_RECOVERY_MIN=0.90
BOOSTER_TRACK_ADAPTER_FORCE_PUSH=1
BOOSTER_TRACK_ADAPTER_FORCE_DURATION_MIXTURE=10:80:1.00:4.00:0.15:0.80:0.44,80:160:0.50:2.00:0.08:0.35:0.36,150:250:0.05:0.15:0.00:0.03:0.20
BOOSTER_TRACK_ADAPTER_PUSH=1
BOOSTER_TRACK_ADAPTER_PUSH_XY=0.08
BOOSTER_TRACK_ADAPTER_PUSH_XY_FINAL=0.22
BOOSTER_TRACK_ADAPTER_RECOVERY_NO_FALL_WEIGHT=13.0
BOOSTER_TRACK_ADAPTER_RECOVERY_NO_BASE_CONTACT_WEIGHT=10.0
BOOSTER_TRACK_ADAPTER_RECOVERY_BASE_CONTACT_PENALTY_WEIGHT=-900.0
BOOSTER_TRACK_ADAPTER_RECOVERY_WEIGHT=14.0
BOOSTER_TRACK_ADAPTER_TASK_WEIGHT=0.10
BOOSTER_TRACK_ADAPTER_STYLE_WEIGHT=0.02
BOOSTER_TRACK_ADAPTER_REG_WEIGHT=0.025
```

The wrapper runs Stage3 first, then PPO.  The direct command shape is:

```bash
python scripts/rsl_rl/pretrain_track_adapter_world_model.py \
  --task Booster-Run-AMP-TrackAdapter-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 1024 \
  --seed 42 \
  --run_name track_adapter_81999_stage3_recovery_v44_recovery_controller_REPRO_wm_pretrain_recovery_controller \
  --max_iterations 700

python scripts/rsl_rl/train.py \
  --task Booster-Run-AMP-TrackAdapter-v0 \
  --headless \
  --device cuda:0 \
  --num_envs 4096 \
  --seed 42 \
  --run_name track_adapter_81999_stage3_recovery_v44_recovery_controller_REPRO_ppo_recovery_controller \
  --max_iterations 5200 \
  --logger tensorboard
```

## Evaluation Gate

Use the fixed recovery-controller gates:

```bash
V44_CHECKPOINT=/workspace/booster_amp_lab/logs/rsl_rl/run_amp_track_adapter/2026-05-20_18-04-18_track_adapter_81999_stage3_recovery_v44_recovery_controller_20260520_162052_ppo_recovery_controller/model_2475.pt \
scripts/rsl_rl/eval_track_adapter_recovery_controller_gates.sh
```

The gate separates:

- no-push command tracking;
- zero-command slow sustained push;
- walking push;
- short impact.

Current v44 status:

- passed no-push;
- passed short-impact;
- failed zero-command slow sustained push;
- failed walking push;
- real robot standing-push behavior was qualitatively better than deployed v11/v14;
- known deployment issue: mild transient actuator oscillation at policy start and
  when a leg/foot shifts inward.

## Export / Deploy Artifact

Exported deploy artifact in the runtime repo:

```text
/workspace/booster_k1_locomotion/assets/track_adapter_v44_model_2475.onnx
```

ONNX ABI:

```text
inputs:
  obs[1,75]
  history[1,79,97]
  residual_gate[1,1]
  reset[1,1]

outputs:
  action[1,22]
  next_history[1,79,97]
```

`booster_k1_locomotion` should run this with:

```text
track_adapter_gate_mode:=recovery_controller
```

## Reproducibility Standard

Do not expect bit-identical checkpoints from RL reruns.  The practical
reproducibility target is:

- same frozen AMP chain and v44 adapter chain;
- same saved configuration surface;
- same fixed-eval pass/fail pattern within noise;
- no regression on the observed real-robot improvement over v11/v14;
- no worsening of the known startup/cross-leg oscillation issue.

Before declaring a scratch rerun equivalent to current v44, compare it against
the retained checkpoints:

- v7 `model_1100.pt`;
- v11 `model_6500.pt`;
- v14 `model_10000.pt`;
- v15b `model_16499.pt`;
- v44 `model_2475.pt`.

## Log Retention

On 2026-05-21, checkpoint cleanup reduced `logs/` from about `42G` to `15G`.
The policy is now one checkpoint per run.  The cleanup explicitly preserved:

- AMP base `model_81999.pt`;
- AMP source `model_70000.pt`;
- v7 `model_1100.pt`;
- v11 `model_6500.pt`;
- v14 `model_10000.pt`;
- v15b `model_16499.pt`;
- v44 Stage3 `model_699.pt`;
- v44 PPO `model_2475.pt`.

Non-checkpoint artifacts such as `params/`, saved `git/` diffs, TensorBoard
events, and fixed-eval summaries were left in place because they are needed for
reproduction and audit.

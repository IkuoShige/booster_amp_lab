# Track Adapter Worklog

## 2026-05-12

### Stage3/PPO Branch: AMP Axis v2 `model_81999.pt`

- Changed the next Track Adapter training branch to use the AMP axis fine-tune final checkpoint as frozen base:

```text
logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt
```

- Rationale: compared with `model_70100.pt`, `model_81999.pt` is better on the highest-priority no-push single-axis tracking metrics, especially `vx-only` and `vy-only`. Its higher high-speed jerk/contact risk is now assigned to the adapter recovery objective.
- Updated Track Adapter environment sampling for this branch:
  - hard benchmark command points enabled for Stage3/PPO,
  - yaw command coverage widened to `|vyaw| <= 1.5`,
  - primary hard points include zero, `vx={0.2,0.5,1.0,1.5,2.0}`, `vy={+-0.2,+-0.4,+-1.0}`, and yaw-only points,
  - secondary hard points include `vx+vyaw` and `vy+vyaw`.
- Added Track Adapter reward terms for this branch:
  - `vx_only_track_vx_exp`,
  - `vx_only_crosstalk_l2`,
  - `vy_only_track_vy_exp`,
  - `vy_only_crosstalk_l2`,
  - `yaw_only_track_ang_vel_z_rel_exp`,
  - `yaw_only_translation_l2`,
  - `zero_command_velocity_l2`,
  - `zero_command_joint_vel_l2`,
  - `axis_yaw_x_track_exp`,
  - `axis_yaw_x_crosstalk_l2`,
  - `axis_yaw_y_track_exp`,
  - `axis_yaw_y_crosstalk_l2`,
  - `recovery_push_velocity_track_exp`.
- Added `scripts/rsl_rl/run_track_adapter_81999_stage3_to_ppo.sh` as the reproducible tmux entrypoint. It runs Stage3 world-model pretraining first, validates the resulting pretrain checkpoint, then launches full PPO with persistent world-model replay and alternating WM/PPO updates.
- Resource plan for the 5090 run:
  - first attempt with Stage3 `1024` envs and `4` pretrain mini-batches OOMed on the first backward pass,
  - relaunched with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  - Stage3 pretrain: `768` envs, `500` iterations, `96` steps/env, `32` mini-batches,
  - PPO: `3072` envs, `12000` iterations,
  - WM replay: `524288` samples, `4096` mini-batch, `4` WM updates per PPO iteration.
- Active tmux session:

```text
track_adapter_81999_stage3_to_ppo_20260512_102031
```

- Active command log:

```text
logs/track_adapter_81999_stage3_to_ppo_20260512_102031.log
```

- Startup check:
  - environment has `75` policy obs, `78` critic obs, `56` AMP obs,
  - `43` reward terms are active, including the new single-axis, zero-command, axis+yaw, and push-recovery terms,
  - Stage3 reached at least iteration `10/500`,
  - initial Stage3 metrics are finite: `wm` around `0.66-0.72`, `wm_valid_fraction` mostly about `0.90`,
  - GPU memory after successful relaunch was about `15.5 GB` used.
- PPO status check around 2026-05-12 15:55 UTC:
  - Stage3 completed and PPO is running in `logs/rsl_rl/run_amp_track_adapter/2026-05-12_11-03-59_track_adapter_81999_stage3_to_ppo_20260512_102031_ppo_81999`.
  - Latest observed PPO iteration: `2084/12000`.
  - Recent 50-iteration averages: reward `277.43`, `wm=0.3101`, `wm_valid_fraction=0.7690`, `base_contact=0.0113`, `error_vel_xy=0.2901`, `error_vel_yaw=0.3712`, `adapter_scaled_residual_action_l2=0.0372`, residual gate `0.3605`.
  - Latest checkpoint observed: `model_2075.pt`.
  - Status: healthy enough to continue; no safety-gate stop, no WM divergence, and residual is not exploding. Fixed command-grid evaluation is still required before judging real tracking/push recovery quality.
- Started a concurrent fixed evaluation for `model_2000.pt`:

```text
fixed_eval_ta81999_model2000_20260512_155924
```

- Output directory:

```text
logs/fixed_eval_ta81999_model2000_20260512_155924
```

- Evaluation settings:
  - `64` envs to avoid OOM while PPO keeps running,
  - no-push, interval-push, and high-speed-jerk suite via `eval_track_adapter_v6_grid.sh`,
  - command list widened from the default `vx/vy` grid to include yaw-only `|vyaw| <= 1.5` and secondary axis+yaw commands,
  - checkpoint base override set to AMP axis fine-tune `model_81999.pt`.
- Early eval status: process is running, not OOMed, but no-push JSON was still pending after startup/preload; PPO iteration time increased while eval shares the GPU.

## 2026-05-09

### Initial Findings

- Read `idea-adapter.md`.
- Confirmed the minimal requested architecture is action residual:

```text
a_t = pi_base(o_t, c_t) + xi(o_t, c_t, e_t)
```

- Confirmed current repo has no Track Adapter implementation.
- Confirmed current AMP stack uses one trainable `ActorCritic` and `AMPPPO` optimizes all policy params plus discriminator params.
- Confirmed frozen checkpoint target resolves to:

```text
logs/rsl_rl/run_amp_y/2026-05-04_19-10-18_resume56_yawlow_zero_stand/model_70000.pt
```

### Fork Review

- Reviewed `/workspace/forked_booster_amp_lab` Recovery-Gated AMP attempt.
- Useful pieces:
  - disturbance score and gate math,
  - recovery reward terms,
  - task config shape for reward groups.
- Not adopted:
  - trained artifacts,
  - broad multi-critic runner/storage changes,
  - RWM recovery-gated imagination path.
- Reason: fork execution log reports MuJoCo/SDK zero-command stability failure after the Recovery-Gated AMP pipeline.

### Implementation Plan

- Use new narrow Track Adapter path:
  - `TrackAdapterActorCritic`,
  - `TrackAdapterRolloutStorage`,
  - `TrackAdapterAMPPPO`,
  - `TrackAdapterRunner`,
  - dedicated K1 task/config.

### Parallel Work

- Worker A assigned policy module/export implementation.
- Worker B assigned storage/algorithm implementation.
- Worker C assigned recovery utilities/rewards/task config.
- Worker D assigned lightweight tests.

### Implementation Added

- Added `TrackAdapterActorCritic` with frozen base `ActorCritic`, trainable history encoder, zero-initialized residual actor, trainable critic, and trainable action noise.
- Added `TrackAdapterRolloutStorage` to store policy observation history with each PPO transition.
- Added `TrackAdapterAMPPPO` to pass history through action/value/log-prob recomputation and to support frozen discriminator training.
- Added `TrackAdapterRunner` and `scripts/rsl_rl/train.py` dispatch.
- Added `scripts/rsl_rl/play.py` dispatch so Track Adapter checkpoints can be loaded without forcing `AmpOnPolicyRunner`.
- Added `mdp.robustness` disturbance/gate helpers and exported them through `mdp/__init__.py`.
- Added `recovery_*` reward terms.
- Added `TrackAdapterRoughWoStateEstimationEnvCfg`, `TrackAdapterPPORunnerCfg`, and registered `Booster-Run-AMP-TrackAdapter-v0`.
- Added ignored local tests under `tests/` for policy, optimizer, and storage behavior.

### Integration Fixes

- Aligned runner history shape `[num_envs, history_length, num_obs]` with policy encoding by flattening higher-rank history tensors in `TrackAdapterActorCritic`.
- Added policy config aliases used by `TrackAdapterPPORunnerCfg`:
  - `history_encoder_hidden_dims`
  - `adapter_hidden_dims`
  - `history_embedding_dim`
  - `residual_output_init_scale`
  - `freeze_base`
- Made `TrackAdapterAMPPPO` accept `residual_penalty_cfg` from the config object.
- Made runner residual penalty accept both `stable_coef/recovery_coef` and `stable_weight/recovery_weight`.
- Fixed runner residual penalty config lookup to read `residual_penalty_cfg`.
- Added observation normalizer shape guard when loading the frozen base checkpoint.
- Made `TrackAdapterRunner.get_inference_policy()` maintain policy-observation history and accept `dones` for reset-aware inference.

### Sudden Stop Robustness

- Added sudden zero-command sampling to `OmniVelocityCommand`.
  - A configurable fraction of resampled non-zero commands is abruptly replaced with `[0, 0, 0]`.
  - `sudden_stop_timer` marks the post-stop recovery window.
- Enabled sudden-stop sampling for `TrackAdapterRoughWoStateEstimationEnvCfg`.
- Added stop-specific recovery rewards:
  - `recovery_sudden_stop_upright_exp`
  - `recovery_sudden_stop_velocity_damp_exp`
  - `recovery_sudden_stop_default_pose_exp`
- Updated `scripts/rsl_rl/play.py` to pass `dones` into Track Adapter inference so history resets correctly during play/eval.

### Verification

- `python -m py_compile` passed for Track Adapter modules, runner, train/play scripts, and touched MDP/config files.
- `git diff --check` passed.
- In `/workspace/Isaac_uv_template/.venv`, focused pytest passed:

```text
4 passed in 1.03s
```

- Import smoke passed for `TrackAdapterActorCritic`, `TrackAdapterAMPPPO`, `TrackAdapterRolloutStorage`, and `TrackAdapterRunner`.
- Frozen checkpoint smoke loaded `logs/rsl_rl/run_amp_y/2026-05-04_19-10-18_resume56_yawlow_zero_stand/model_70000.pt` and verified 75-dim actor input, 78-dim critic input, 22-dim action output, and frozen base params.

### Training Launch

- Started tmux session:

```text
track_adapter_20260509_234630
```

- Command log:

```text
logs/track_adapter_20260509_234630.log
```

- Training task:

```text
Booster-Run-AMP-TrackAdapter-v0
```

- Confirmed runtime setup:
  - 4096 envs
  - policy obs shape `(75,)`
  - critic obs shape `(78,)`
  - AMP obs shape `(56,)`
  - 29 active reward terms
  - sudden-stop rewards registered and became non-zero by iteration 13
- At iteration 13, training was running with finite losses and no startup crash.

### Handoff Check

- Confirmed tmux session `track_adapter_20260509_234630` is still alive.
- Latest observed log reached iteration 53/12000 with finite losses and reward.
- Sudden-stop rewards were active in the latest observed log:
  - `recovery_sudden_stop_upright_exp`
  - `recovery_sudden_stop_velocity_damp_exp`
  - `recovery_sudden_stop_default_pose_exp`

## 2026-05-10

### Correction: Premature Training Launch

- User pointed out that the run reached roughly iteration 480 while `base_contact` was around `0.8` or worse.
- Rechecked the log and observed `base_contact` around `0.93` at iterations 492-493.
- Stopped tmux session `track_adapter_20260509_234630`.
- The previous launch was premature because it treated the action-residual subset as sufficient for a full run before the complete `idea-adapter.md` plan was implemented and before stability gates were enforced.
- Full training is now blocked until the missing pieces below are implemented and a bounded smoke gate passes:
  - world-model auxiliary loss / pretraining path,
  - observation plus action history for the adapter embedding,
  - push and sudden-stop curriculum instead of immediate hard distribution,
  - failure/contact-aware acceptance criteria,
  - explicit evaluation metrics for robustness, AMP style, command recovery, safety, and adapter residual behavior.

### Follow-up Implementation

- Added code-side full-training protection:
  - Track Adapter defaults to bounded smoke mode.
  - Long runs are refused unless the full-training gate is explicitly unlocked.
  - Runner safety gate aborts when `base_contact` exceeds the configured threshold for repeated checks.
- Changed adapter history from observation-only to observation plus executed-action history in `TrackAdapterRunner`.
- Added GRU history encoder support in `TrackAdapterActorCritic`.
- Added a lightweight world-model auxiliary path:
  - rollout storage keeps explicit normalized `next_observations`,
  - storage produces short-horizon future targets plus validity masks,
  - PPO update logs `wm`, `wm_abs_error`, and `wm_valid_fraction`,
  - world-model gradients update the adapter/history side and not the frozen base.
- Added contact-aware disturbance gating for Track Adapter via `contact_weight` and contact sensor configuration.
- Added `scripts/rsl_rl/eval_track_adapter_recovery.py` for abrupt all-zero command evaluation:
  - fall rate,
  - base-contact rate,
  - settle success/recovery steps,
  - final speed,
  - residual norm,
  - AMP score.
- Tightened base checkpoint invariants:
  - missing base checkpoint is fatal,
  - missing/mismatched discriminator, AMP normalizer, or observation normalizer is fatal under strict mode.
- Made smoke defaults less aggressive:
  - pushes disabled in smoke mode by default,
  - sudden-stop probability and window reduced,
  - command/randomization stress reduced,
  - full disturbance run requires explicit gate variables.

### Verification After Correction

- `python -m py_compile` passed for Track Adapter RSL-RL modules, runner, train script, and touched run_amp/MDP files.
- `python -m py_compile` passed for `scripts/rsl_rl/eval_track_adapter_recovery.py`.
- `git diff --check` passed.
- Focused pytest in `/workspace/Isaac_uv_template/.venv` passed:

```text
13 passed in 1.25s
```

- Synthetic PPO update smoke passed and produced positive world-model loss:

```text
wm: 0.7550
wm_valid_fraction: 1.0
```

### Bounded Smoke Run

- Launched a bounded tmux smoke run only, with the requested Python environment:

```text
track_adapter_smoke_20260510_045025
```

- Command log:

```text
logs/track_adapter_smoke_20260510_045025.log
```

- Output run directory:

```text
logs/rsl_rl/run_amp_track_adapter/2026-05-10_04-50-29_track_adapter_smoke16
```

- Result:
  - completed iterations `0` through `15`,
  - `Episode_Termination/base_contact: 0.0000` throughout the smoke,
  - `wm_valid_fraction: 1.0000`,
  - world-model loss stayed finite around `0.79 -> 0.77`,
  - residual action norm stayed near zero because the residual head is zero-initialized.
- This is a startup/stability smoke, not a full training result. Full training remains locked.

### Abrupt Zero-Command Stop Evaluation

- Fixed evaluator dtype handling for Isaac returned `dones`.
- Ran abrupt all-zero command evaluation from the bounded smoke checkpoint:

```text
logs/rsl_rl/run_amp_track_adapter/2026-05-10_04-50-29_track_adapter_smoke16/model_15.pt
```

- Session and artifacts:

```text
track_adapter_stop_eval_20260510_045214
logs/track_adapter_stop_eval_20260510_045214.log
logs/track_adapter_stop_eval_20260510_045214.json
logs/track_adapter_stop_eval_20260510_045214.csv
```

- Summary:

```text
fall_rate: 0.0
base_contact_rate: 0.0
settle_success_rate: 1.0
mean_recovery_steps: 40.8359
final_stop_speed_mean: 0.0392
final_stop_speed_p95: 0.0792
final_residual_norm_mean: 0.0008
final_amp_score_mean: 0.1450
```

- Interpretation: the explicit abrupt-stop harness is functional and the frozen-base smoke checkpoint can settle from a sudden `0.8 -> 0.0` x-velocity command without falls or base contact in this short evaluation. This does not unlock full training by itself.

### Cleanup

- Made policy normalization config flags explicit in `TrackAdapterActorCritic` so Track Adapter does not emit an unknown-argument warning when launched from the runner config.
- Tightened `scripts/rsl_rl/train.py` so a long iteration request from `BOOSTER_TRACK_ADAPTER_MAX_ITERATIONS` is also rejected before full-training unlock, instead of being silently rounded down to smoke length.
- Addressed follow-up review blockers:
  - world-model predictor now conditions on current normalized actor observation as well as history latent and action,
  - enabled symmetry augmentation is rejected for Track Adapter until observation/action history mirroring is implemented,
  - optimizer construction uses explicit adapter/critic/noise/world-model parameter groups and structurally excludes the frozen base,
  - full-training unlock now uses one shared three-flag predicate across env config, runner config, train entrypoint, and runner,
  - Track Adapter checkpoint loading validates base checkpoint metadata, history length, policy keys, discriminator, AMP normalizer, and empirical normalizers under strict mode,
  - contact lookup for disturbance gating and failure mining now fails fast instead of reporting clean contact-free state on lookup errors,
  - abrupt-stop evaluator refreshes observations after forced command changes, tracks pre-stop failures, masks final metrics to valid environments, and resolves contact sensor body IDs via `SceneEntityCfg`.
- Re-ran lightweight verification after cleanup:

```text
py_compile: passed
focused pytest: 17 passed in 1.28s
git diff --check: passed
normalization kwargs smoke: passed
```

### Strict Bounded Smoke Run

- Re-ran bounded smoke after strict fixes and world-model input-shape change:

```text
track_adapter_smoke_20260510_050642
logs/track_adapter_smoke_20260510_050642.log
logs/rsl_rl/run_amp_track_adapter/2026-05-10_05-06-46_track_adapter_smoke16_strict
```

- Result:
  - completed iterations `0` through `15`,
  - `Episode_Termination/base_contact: 0.0000` throughout,
  - `wm_valid_fraction: 1.0000`,
  - world-model loss stayed finite,
  - no unknown Track Adapter config warning,
  - no safety-gate strike.

### Strict Abrupt Zero-Command Stop Evaluation

- Re-ran abrupt all-zero command evaluation after evaluator fixes:

```text
track_adapter_stop_eval_20260510_050725
logs/track_adapter_stop_eval_20260510_050725.log
logs/track_adapter_stop_eval_20260510_050725.json
logs/track_adapter_stop_eval_20260510_050725.csv
```

- Summary:

```text
fall_rate: 0.0
base_contact_rate: 0.0
pre_stop_failure_rate: 0.0
valid_stop_env_rate: 1.0
settle_success_rate: 1.0
settle_success_rate_valid_envs: 1.0
mean_recovery_steps: 42.5547
final_stop_speed_mean: 0.0388
final_stop_speed_p95: 0.0680
final_residual_norm_mean: 0.0007
final_amp_score_mean: 0.1452
```

- Interpretation: the stricter evaluation still passes for a sudden `0.8 -> 0.0` x-velocity command with no pre-stop reset/contact, no stop-phase reset/contact, and masked final metrics over valid environments. Full training remains locked.

### Re-scope To Full `idea-adapter.md`

- User clarified that the desired implementation is not only PPO-time auxiliary world-model loss.
- Re-read `idea-adapter.md` Stage 3/5/6 and confirmed the missing pieces:
  - explicit frozen-base world-model pretraining before adapter PPO,
  - PPO launch path that can load the pretrain checkpoint,
  - richer `psi_s(s)` target support beyond plain next actor observation,
  - failure-aware push mining rather than command-only failure metrics.
- Updated the roadmap so full training stays locked until pretrain smoke, PPO smoke, and evaluation gates all pass.
- Launched parallel implementation/review agents:
  - Stage 3 pretrain path,
  - richer world-model targets,
  - failure-aware push curriculum,
  - read-only compliance review,
  - read-only integration hazard review.

### Stage 3 World-Model Pretraining Path

- Added `TrackAdapterWorldModelPretrainRunner`.
  - Builds the current `TrackAdapterActorCritic` from the Track Adapter config.
  - Loads the frozen AMP base actor plus strict base auxiliary checkpoint state.
  - Collects rollouts with deterministic base-only actions (`base_residual_zero`) so the residual actor cannot perturb the base controller during pretraining.
  - Maintains observation plus executed-action history in the same layout as `TrackAdapterRunner`.
  - Optimizes only `history_encoder` and `world_model_predictor`.
  - Uses the existing policy L1 future target loss and validity masking.
- Added `scripts/rsl_rl/pretrain_track_adapter_world_model.py`.
  - Launches Isaac headless through the normal Hydra task config.
  - Requires a `TrackAdapterRunner` config.
  - Supports CLI overrides for pretrain iterations, rollout length, epochs, mini-batches, LR, grad clip, target source, target dim, horizon, and save interval.
  - Can initialize from an existing Track Adapter checkpoint via `--checkpoint`.
- Pretrain checkpoints intentionally save `pretrain_optimizer_state_dict` instead of `optimizer_state_dict` so `TrackAdapterRunner.load()` can consume the checkpoint without trying to load a mismatched PPO optimizer.
- Saved checkpoint metadata includes:
  - `model_state_dict`
  - `discriminator_state_dict`
  - `amp_normalizer`
  - `obs_norm_state_dict`
  - `privileged_obs_norm_state_dict`
  - `base_checkpoint_path`
  - `history_length`
  - `track_adapter_stage`
  - `world_model_pretrain`
  - `world_model_target`

### Stage 3 Verification

- `python -m py_compile` passed for:
  - `rsl_rl/rsl_rl/runners/track_adapter_world_model_pretrain_runner.py`
  - `scripts/rsl_rl/pretrain_track_adapter_world_model.py`
- Synthetic runner smoke in `/workspace/Isaac_uv_template/.venv` passed:
  - collected base-only rollouts,
  - produced finite L1 world-model loss,
  - updated the world-model predictor,
  - preserved frozen base actor weights,
  - saved TrackAdapterRunner-compatible metadata,
  - omitted PPO `optimizer_state_dict`.
- Initial focused pytest found an integration bug in the PPO storage call site after `world_model_state_shape` was added. This was fixed by passing `world_model_state_shape`, `rnd_state_shape`, and `device` by keyword from `TrackAdapterAMPPPO.init_storage()`.

### Full Stage 3/PPO Integration Hardening

- Switched the default Track Adapter world-model target from plain `next_observations` to structured `psi_s`:
  - normalized actor observation: 75 dims,
  - base linear velocity in body frame: 3 dims,
  - base height: 1 dim,
  - left/right foot contact bits: 2 dims,
  - total target: 81 dims.
- Added structured target metadata to both pretrain and PPO checkpoints.
- Tightened pretrain loading in `TrackAdapterRunner`:
  - requires `track_adapter_stage == "world_model_pretrain"`,
  - rejects PPO checkpoints as `BOOSTER_TRACK_ADAPTER_PRETRAINED_WM`,
  - requires `collection_policy == "base_residual_zero"`,
  - requires finite `wm` and positive `wm_valid_fraction`,
  - checks base checkpoint path, history length, structured target metadata, and tensor shapes.
- Added PPO checkpoint metadata:
  - `track_adapter_stage = "ppo"`,
  - `pretrained_world_model_path`,
  - `pretrained_world_model_loaded`,
  - `world_model_metadata`.
- Added a pretrain disturbance manifest. Under full-training flags, the runner now requires material, base-mass, base-CoM, sudden-stop, and push coverage in the pretrain checkpoint.
- Exported `TrackAdapterWorldModelPretrainRunner` through `rsl_rl.runners`.
- Updated the pretrain CLI so `--pretrain_target_source` can request `psi_s`, `world_model_state`, `wm_state`, or `structured`, and added `--pretrain_target_groups`.

### Verification After Hardening

- In `/workspace/Isaac_uv_template/.venv`, verification passed:

```text
py_compile: passed
focused pytest: 23 passed in 1.28s
git diff --check: passed
```

### Strict Stage 3 Pretrain Smoke

- Launched tmux session:

```text
track_adapter_wm_pretrain_strict_20260510_061347
logs/track_adapter_wm_pretrain_strict_20260510_061347.log
logs/rsl_rl/run_amp_track_adapter/2026-05-10_06-13-51_track_adapter_wm_pretrain_strict_smoke
```

- Result:

```text
iteration 0/2: wm=0.6974, wm_valid_fraction=1.0000
iteration 1/2: wm=0.7318, wm_valid_fraction=1.0000
```

- Saved checkpoint:

```text
logs/rsl_rl/run_amp_track_adapter/2026-05-10_06-13-51_track_adapter_wm_pretrain_strict_smoke/model_1.pt
```

- Checkpoint metadata:

```text
track_adapter_stage: world_model_pretrain
collection_policy: base_residual_zero
source: psi_s
target_dim: 81
state_schema: normalized_actor_obs[75] + base_lin_vel_b[3] + base_height[1] + foot_contacts[2]
last_loss_dict: finite, wm_valid_fraction=1.0
disturbance_manifest: material/mass/CoM/sudden-stop present, push disabled in this smoke
```

### Pretrain-Loaded PPO Smoke

- Launched tmux session:

```text
track_adapter_pretrain_ppo_strict_20260510_061431
logs/track_adapter_pretrain_ppo_strict_20260510_061431.log
logs/rsl_rl/run_amp_track_adapter/2026-05-10_06-14-35_track_adapter_pretrain_strict_ppo_smoke
```

- Confirmed PPO runner loaded:

```text
[TrackAdapterRunner] Loaded world-model pretrain: .../track_adapter_wm_pretrain_strict_smoke/model_1.pt
```

- Result:
  - completed iterations `0` through `3`,
  - `Episode_Termination/base_contact: 0.0000`,
  - `wm_valid_fraction: 1.0000`,
  - `wm` stayed finite around `0.75 -> 0.71`,
  - residual norm remained near zero.
- PPO checkpoint metadata confirms:

```text
track_adapter_stage: ppo
pretrained_world_model_loaded: True
world_model_metadata.source: psi_s
world_model_metadata.target_dim: 81
```

### Strict Abrupt Stop Evaluation After Stage 3 Integration

- Evaluated:

```text
logs/rsl_rl/run_amp_track_adapter/2026-05-10_06-14-35_track_adapter_pretrain_strict_ppo_smoke/model_3.pt
```

- Session and artifacts:

```text
track_adapter_stop_eval_strict_20260510_061503
logs/track_adapter_stop_eval_strict_20260510_061503.log
logs/track_adapter_stop_eval_strict_20260510_061503.json
logs/track_adapter_stop_eval_strict_20260510_061503.csv
```

- Summary:

```text
fall_rate: 0.0
base_contact_rate: 0.0
pre_stop_failure_rate: 0.0
valid_stop_env_rate: 1.0
settle_success_rate: 1.0
settle_success_rate_valid_envs: 1.0
mean_recovery_steps: 43.46875
final_stop_speed_mean: 0.0375
final_stop_speed_p95: 0.0602
final_residual_norm_mean: 0.000777
final_amp_score_mean: 0.1452
```

- Interpretation: the fully wired path now runs as intended in bounded smoke form:
  frozen AMP base -> Stage 3 structured world-model pretrain -> pretrain-loaded Track Adapter PPO -> abrupt all-zero command evaluation. Full-length training remains locked until a disturbance pretrain with push coverage is produced and the manual full-training flags are set.

### Stage 3 Push-Coverage Pretrain And PPO Launch

- User requested running Stage 3 "Pretrain history encoder + world model" and then training.
- Launch policy:
  - run a non-smoke Stage 3 pretrain with push coverage and full disturbance flags,
  - validate the resulting checkpoint metadata before PPO starts,
  - start PPO only if the checkpoint is a `world_model_pretrain` artifact with finite world-model loss, positive valid fraction, base-only collection, and push-enabled disturbance manifest.
- PPO is launched with the same full-training unlock flags and the validated pretrain checkpoint.

### Stage 3 Push-Coverage Pretrain Completed

- Launched chained tmux session:

```text
track_adapter_stage3_to_ppo_20260510_063457
logs/track_adapter_stage3_to_ppo_20260510_063457.log
```

- Stage 3 pretrain run:

```text
logs/rsl_rl/run_amp_track_adapter/2026-05-10_06-35-01_track_adapter_stage3_to_ppo_20260510_063457_wm_pretrain_push/model_299.pt
```

- Checkpoint metadata confirms:

```text
track_adapter_stage: world_model_pretrain
world_model_metadata.source: psi_s
world_model_metadata.target_dim: 81
world_model_metadata.state_schema: normalized_actor_obs[75] + base_lin_vel_b[3] + base_height[1] + foot_contacts[2]
world_model_pretrain.collection_policy: base_residual_zero
world_model_pretrain.last_loss_dict.wm: 0.527325302362442
world_model_pretrain.last_loss_dict.wm_valid_fraction: 1.0
disturbance_manifest.push_enabled: true
disturbance_manifest.material_randomization_enabled: true
disturbance_manifest.base_mass_randomization_enabled: true
disturbance_manifest.base_com_randomization_enabled: true
disturbance_manifest.sudden_stop_enabled: true
```

- PPO run started from the validated pretrain checkpoint:

```text
logs/rsl_rl/run_amp_track_adapter/2026-05-10_06-40-57_track_adapter_stage3_to_ppo_20260510_063457_ppo_full/model_0.pt
```

- PPO checkpoint metadata confirms:

```text
track_adapter_stage: ppo
pretrained_world_model_loaded: True
pretrained_world_model_path: logs/rsl_rl/run_amp_track_adapter/2026-05-10_06-35-01_track_adapter_stage3_to_ppo_20260510_063457_wm_pretrain_push/model_299.pt
world_model_metadata.source: psi_s
world_model_metadata.target_dim: 81
```

- Early PPO status at iteration `34/12000`:
  - `Episode_Termination/base_contact: 0.0010`;
  - `Mean wm loss: 0.5856`;
  - `Mean wm_valid_fraction loss: 1.0000`;
  - `Mean adapter_residual_action_l2 loss: 0.0021`;
  - `Metrics/base_velocity/failure_push_active: 1.0000`;
  - failure mining has started recording nonzero command-failure mass.

### First Full PPO Attempt Stopped By Safety Gate

- PPO attempt:

```text
track_adapter_stage3_to_ppo_20260510_063457
logs/rsl_rl/run_amp_track_adapter/2026-05-10_06-40-57_track_adapter_stage3_to_ppo_20260510_063457_ppo_full
```

- The run was intentionally stopped by the Track Adapter safety gate at iteration `128/12000`.
- Trigger:

```text
iteration 127: Episode_Termination/base_contact = 0.2029
iteration 128: Episode_Termination/base_contact = 0.2132
safety threshold: 0.2000
patience: 2
```

- Preceding failure mode:
  - world-model loss stayed finite and improving (`wm` around `0.50`);
  - `failure_base_contact` remained `0.0000`;
  - adapter residual grew too quickly (`Mean adapter_scaled_residual_action_l2 loss` reached `2.9891`);
  - mean reward and episode length collapsed after roughly iteration `100`.
- Saved checkpoints:

```text
model_0.pt
model_100.pt
model_128_aborted.pt
```

- Follow-up hardening:
  - capped `BOOSTER_TRACK_ADAPTER_SMOKE_ITERATIONS` so it cannot bypass the full-training unlock;
  - require complete `history_encoder.*` and `world_model_predictor.*` key coverage when loading Stage 3 pretrain weights;
  - copy loaded pretrain metadata/manifest into future PPO checkpoints;
  - lowered default adapter residual scale, PPO learning rate, entropy, and initial noise;
  - increased residual penalty and added a `scaled_residual_action_l2` safety gate.
- Verification after hardening:

```text
py_compile: passed
bash -n scripts/rsl_rl/run_track_adapter_stage3_to_ppo.sh: passed
pytest -q tests/test_track_adapter_module.py tests/test_track_adapter_storage.py tests/test_track_adapter_optimizer.py: 24 passed
```

### H/N Configuration Correction

- User pointed out that `idea-adapter.md` describes a dynamics embedding built from `H=79` history steps and an autoregressive world-model prediction window `N=20`.
- Confirmed issue: the first push-coverage Stage 3 pretrain and PPO attempts used `H=20,N=1`.
- Action taken:
  - stopped the conservative PPO rerun before it became a long invalid run;
  - changed Track Adapter defaults to `history_length=79` and `world_model_horizon=20`;
  - changed PPO rollout length default to `48` steps to keep a useful valid fraction for `N=20`;
  - changed the chained Stage3-to-PPO script defaults to `PRETRAIN_NUM_ENVS=512`, `PRETRAIN_STEPS_PER_ENV=96`, `PRETRAIN_ITERATIONS=500`, `PPO_NUM_ENVS=2048`, and `BOOSTER_TRACK_ADAPTER_PPO_STEPS_PER_ENV=48`;
  - added script-side checkpoint validation for `history_length=79` and `world_model_metadata.horizon=20`;
  - added a storage test covering `horizon=20` tail validity masks.
- Consequence: the previous `model_299.pt` Stage 3 checkpoint remains useful as evidence that the pipeline works, but it must not be used for corrected full PPO because its metadata is `H=20,N=1`.

### Corrected H79/N20 Stage 3 Launch

- Launched corrected chained tmux session:

```text
track_adapter_h79n20_stage3_to_ppo_20260510_065629
logs/track_adapter_h79n20_stage3_to_ppo_20260510_065629.log
```

- Corrected defaults used by the chained script:

```text
BOOSTER_TRACK_ADAPTER_HISTORY_LENGTH=79
BOOSTER_TRACK_ADAPTER_WM_HORIZON=20
BOOSTER_TRACK_ADAPTER_PPO_STEPS_PER_ENV=48
PRETRAIN_NUM_ENVS=512
PRETRAIN_STEPS_PER_ENV=96
PRETRAIN_ITERATIONS=500
PPO_NUM_ENVS=2048
```

- Initial corrected Stage 3 checkpoint:

```text
logs/rsl_rl/run_amp_track_adapter/2026-05-10_06-56-33_track_adapter_h79n20_stage3_to_ppo_20260510_065629_wm_pretrain_push/model_0.pt
```

- Metadata confirms:

```text
track_adapter_stage: world_model_pretrain
history_length: 79
world_model_metadata.source: psi_s
world_model_metadata.target_dim: 81
world_model_metadata.horizon: 20
world_model_pretrain.num_steps_per_env: 96
world_model_pretrain.collection_policy: base_residual_zero
world_model_pretrain.last_loss_dict.wm: 0.7662730887532234
world_model_pretrain.last_loss_dict.wm_valid_fraction: 0.901041716337204
```

- Progress checkpoint:

```text
model_100.pt
history_length: 79
world_model_metadata.horizon: 20
world_model_pretrain.last_loss_dict.wm: 0.7320364639163017
world_model_pretrain.last_loss_dict.wm_valid_fraction: 0.9009349048137665
```

- Progress at `model_150.pt`:

```text
history_length: 79
world_model_metadata.horizon: 20
world_model_pretrain.last_loss_dict.wm: 0.7335809245705605
world_model_pretrain.last_loss_dict.wm_valid_fraction: 0.900828093290329
```

- Progress at `model_250.pt`:

```text
history_length: 79
world_model_metadata.horizon: 20
world_model_pretrain.last_loss_dict.wm: 0.6611831486225128
world_model_pretrain.last_loss_dict.wm_valid_fraction: 0.9004008546471596
```

### Corrected H79/N20 Stage 3 Completion And PPO Restart

- Corrected Stage 3 completed 500 iterations and produced:

```text
logs/rsl_rl/run_amp_track_adapter/2026-05-10_06-56-33_track_adapter_h79n20_stage3_to_ppo_20260510_065629_wm_pretrain_push/model_499.pt
```

- Manual validation passed:

```text
track_adapter_stage: world_model_pretrain
history_length: 79
world_model_metadata.source: psi_s
world_model_metadata.target_dim: 81
world_model_metadata.horizon: 20
world_model_pretrain.collection_policy: base_residual_zero
world_model_pretrain.last_loss_dict.wm: 0.6431065872311592
world_model_pretrain.last_loss_dict.wm_valid_fraction: 0.8906260654330254
disturbance_manifest.push_enabled: true
disturbance_manifest.material_randomization_enabled: true
disturbance_manifest.base_mass_randomization_enabled: true
disturbance_manifest.base_com_randomization_enabled: true
disturbance_manifest.sudden_stop_enabled: true
```

- The chained script did not enter PPO after Stage 3 because the shell process hit a quoting parse error after the file was edited while the long-running script was still open. The current script passes `bash -n`, and the checkpoint itself is valid.
- Started PPO-only from the validated H79/N20 checkpoint:

```text
tmux session: track_adapter_h79n20_ppo_from499_20260510_073208
log: logs/track_adapter_h79n20_ppo_from499_20260510_073208.log
pretrain checkpoint: model_499.pt
BOOSTER_TRACK_ADAPTER_HISTORY_LENGTH=79
BOOSTER_TRACK_ADAPTER_WM_HORIZON=20
BOOSTER_TRACK_ADAPTER_PPO_STEPS_PER_ENV=48
BOOSTER_TRACK_ADAPTER_PRETRAINED_WM=model_499.pt
```

- That PPO run loaded the Stage 3 checkpoint correctly, then failed in the first PPO update with CUDA OOM while evaluating the GRU history encoder. The failing shape used `2048` envs, `48` rollout steps, and only `4` PPO mini-batches, which is too large for `H=79`.
- Mitigation applied:

```text
TrackAdapter PPO default num_learning_epochs: 2
TrackAdapter PPO default num_mini_batches: 32
New env overrides:
  BOOSTER_TRACK_ADAPTER_NUM_LEARNING_EPOCHS
  BOOSTER_TRACK_ADAPTER_NUM_MINI_BATCHES
Chained script PPO_NUM_ENVS default: 1024
Chained script BOOSTER_TRACK_ADAPTER_NUM_MINI_BATCHES default: 32
```

- Verification after mitigation:

```text
bash -n scripts/rsl_rl/run_track_adapter_stage3_to_ppo.sh: passed
py_compile ppo_cfg.py: passed
pytest tests/test_track_adapter_module.py tests/test_track_adapter_storage.py tests/test_track_adapter_optimizer.py: 25 passed
```

- Restarted PPO with the memory-safe H79/N20 shape:

```text
tmux session: track_adapter_h79n20_ppo_from499_mb32_20260510_073429
log: logs/track_adapter_h79n20_ppo_from499_mb32_20260510_073429.log
run dir: logs/rsl_rl/run_amp_track_adapter/2026-05-10_07-34-33_track_adapter_h79n20_ppo_from499_mb32_20260510_073429_ppo_full
num_envs: 1024
num_steps_per_env: 48
num_learning_epochs: 2
num_mini_batches: 32
PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
```

- Early PPO status:

```text
iteration 0: wm=0.6879, wm_valid_fraction=0.8021, base_contact=0.0000
iteration 1: wm=0.6532, wm_valid_fraction=0.8021, base_contact=0.0000
iteration 2: wm=0.6561, wm_valid_fraction=0.8021, base_contact=0.0000
iteration 17: wm=0.6465, wm_valid_fraction=0.8021, base_contact=0.0010
iteration 18: wm=0.6447, wm_valid_fraction=0.8021, base_contact=0.0010
iteration 19: wm=0.6374, wm_valid_fraction=0.8021, base_contact=0.0010
iteration 43: wm=0.6073, wm_valid_fraction=0.8021, base_contact=0.0020
iteration 45: wm=0.6102, wm_valid_fraction=0.8021, base_contact=0.0020
iteration 56: wm=0.6151, wm_valid_fraction=0.8021, base_contact=0.0020
```

- At iteration 56, OOM had not recurred, `failure_push_active` was logging, sudden-stop sample metrics were present, and no safety gate abort had fired.
- Stopped the PPO run at iteration 88 before it became a long training run because a stricter review of `idea-adapter.md` found a remaining implementation gap: the world model was H79/N20 but direct multi-step prediction, while the document writes the dynamics as an autoregressive rollout.

```text
Stopped tmux session: track_adapter_h79n20_ppo_from499_mb32_20260510_073429
Last observed iteration: 88
Last observed wm: 0.6054
Last observed base_contact: 0.0000
Last observed scaled residual L2: 0.0013
Saved PPO checkpoints: model_0.pt only
Reason: replace direct-horizon world model with autoregressive dynamics before continuing full PPO.
```

### Autoregressive World Model Implementation

- Replaced the direct H-step prediction head with an autoregressive transition model:

```text
predictor input: dynamics embedding e_t + current predicted psi_s + action a_{t+i}
predictor output: one-step delta psi_s
rollout: repeat N=20 times and stack predicted psi_s trajectory
metadata: world_model_metadata.prediction_mode = autoregressive
```

- Added current `psi_s(s_t)` reference buffers and future action sequence generation to Track Adapter rollout storage.
- Updated PPO and Stage 3 pretrain paths to pass:

```text
reference: current psi_s(s_t)
actions: future sequence [a_t ... a_{t+N-1}]
targets: future psi_s(s_{t+1} ... s_{t+N})
valid_mask: existing terminal/tail mask
```

- Added validation so old direct-horizon Stage 3 checkpoints, including the earlier H79/N20 `model_499.pt`, are rejected because they have no `prediction_mode=autoregressive`.
- Verification:

```text
py_compile Track Adapter module/storage/algorithm/runners/tests: passed
bash -n scripts/rsl_rl/run_track_adapter_stage3_to_ppo.sh: passed
pytest tests/test_track_adapter_module.py tests/test_track_adapter_storage.py tests/test_track_adapter_optimizer.py: 28 passed
```

- AR Stage 3 smoke:

```text
tmux session: track_adapter_ar_wm_smoke_20260510_074531
log: logs/track_adapter_ar_wm_smoke_20260510_074531.log
run dir: logs/rsl_rl/run_amp_track_adapter/2026-05-10_07-45-35_track_adapter_ar_wm_smoke_20260510_074531
num_envs: 64
steps_per_env: 48
iterations: 3
```

- Smoke completed and saved `model_2.pt` with:

```text
history_length: 79
world_model_metadata.source: psi_s
world_model_metadata.target_dim: 81
world_model_metadata.horizon: 20
world_model_metadata.prediction_mode: autoregressive
world_model_predictor.0.weight: (256, 167)
world_model_predictor.4.weight: (81, 128)
last wm: 0.5027257800102234
last wm_valid_fraction: 0.8020833730697632
```

### Autoregressive H79/N20 Full Stage 3 Launch

- Started corrected AR Stage3-to-PPO chain:

```text
tmux session: track_adapter_ar_stage3_to_ppo_20260510_074627
log: logs/track_adapter_ar_stage3_to_ppo_20260510_074627.log
script: scripts/rsl_rl/run_track_adapter_stage3_to_ppo.sh
```

- Expected default chain settings:

```text
BOOSTER_TRACK_ADAPTER_HISTORY_LENGTH=79
BOOSTER_TRACK_ADAPTER_WM_HORIZON=20
PRETRAIN_NUM_ENVS=512
PRETRAIN_STEPS_PER_ENV=96
PRETRAIN_ITERATIONS=500
PPO_NUM_ENVS=1024
BOOSTER_TRACK_ADAPTER_PPO_STEPS_PER_ENV=48
BOOSTER_TRACK_ADAPTER_NUM_LEARNING_EPOCHS=2
BOOSTER_TRACK_ADAPTER_NUM_MINI_BATCHES=32
```

- Initial full Stage 3 checkpoint:

```text
run dir: logs/rsl_rl/run_amp_track_adapter/2026-05-10_07-46-30_track_adapter_ar_stage3_to_ppo_20260510_074627_wm_pretrain_push
model_0.pt metadata.prediction_mode: autoregressive
model_0.pt metadata.horizon: 20
model_0.pt metadata.target_dim: 81
model_0.pt wm: 0.6806761398911476
model_0.pt wm_valid_fraction: 0.901041716337204
```

- Early progress:

```text
iteration 0: wm=0.6807, valid=0.9010
iteration 3: wm=0.6547, valid=0.9010
iteration 6: wm=0.6128, valid=0.9010
iteration 20: wm=0.6341, valid=0.7948
iteration 29: wm=0.5985, valid=0.9010
```

- At iteration 29 the full AR Stage 3 run was still active. GPU memory use was about `22164 MiB`; no traceback/OOM had appeared.
- `model_50.pt` validation:

```text
prediction_mode: autoregressive
horizon: 20
target_dim: 81
wm: 0.5972001552581787
wm_valid_fraction: 0.9010417088866234
disturbance manifest: push/material/mass/CoM/sudden-stop/failure-mining enabled
```

- Confirmed the old direct H79/N20 checkpoint is no longer load-compatible:

```text
legacy prediction_mode: None
legacy world_model_predictor.0.weight: (256, 161)
legacy world_model_predictor.4.weight: (1620, 128)
current AR expected predictor input/output: 167 -> 81
```

- `model_100.pt` validation:

```text
prediction_mode: autoregressive
horizon: 20
target_dim: 81
wm: 0.5579968020319939
wm_valid_fraction: 0.9009349048137665
```

- Stage 3 continued through `model_150.pt`:

```text
iteration 150: wm=0.5668, valid=0.9008
iteration 152: wm=0.5316, valid=0.9007
```

- `model_200.pt` validation:

```text
prediction_mode: autoregressive
horizon: 20
target_dim: 81
wm: 0.49980806559324265
wm_valid_fraction: 0.900828093290329
```

- `model_250.pt` validation:

```text
prediction_mode: autoregressive
horizon: 20
target_dim: 81
wm: 0.49980494752526283
wm_valid_fraction: 0.9004008546471596
post-checkpoint low points: iteration 251 wm=0.4555, iteration 257 wm=0.4507
```

- H/N sizing check:

```text
sim.dt: 0.005
decimation: 4
policy step: 0.020 s
H=79 history window: 1.58 s
N=20 autoregressive prediction window: 0.40 s
assessment: not too small for the current Track Adapter run; the earlier H=20,N=1 path was too small and is no longer compatible.
```

- `model_300.pt` validation:

```text
prediction_mode: autoregressive
horizon: 20
target_dim: 81
wm: 0.4603327102959156
wm_valid_fraction: 0.9009664356708527
nearby lows: iteration 298 wm=0.4425, iteration 299 wm=0.4467
```

- `model_350.pt` validation:

```text
prediction_mode: autoregressive
horizon: 20
target_dim: 81
wm: 0.4347095340490341
wm_valid_fraction: 0.9008952230215073
disturbance manifest: push/material/mass/CoM/sudden-stop/failure-mining enabled
```

- `model_400.pt` validation:

```text
prediction_mode: autoregressive
horizon: 20
target_dim: 81
wm: 0.4355729892849922
wm_valid_fraction: 0.9007212817668915
recent trend: wm improved from 0.6807 at iteration 0 to roughly 0.42-0.44 around iterations 371-400.
PPO status: not started yet; waiting for final Stage 3 checkpoint and script validation.
```

- `model_450.pt` validation:

```text
prediction_mode: autoregressive
horizon: 20
target_dim: 81
wm: 0.41068674996495247
wm_valid_fraction: 0.9006144553422928
recent progress: iterations 433-450 include several wm values around 0.406-0.421.
PPO status: not started yet; Stage 3 still active at iteration 466/500.
```

- Stage 3 finished:

```text
checkpoint: logs/rsl_rl/run_amp_track_adapter/2026-05-10_07-46-30_track_adapter_ar_stage3_to_ppo_20260510_074627_wm_pretrain_push/model_499.pt
validation: passed by scripts/rsl_rl/run_track_adapter_stage3_to_ppo.sh
prediction_mode: autoregressive
horizon: 20
target_dim: 81
wm: 0.4397998936474323
wm_valid_fraction: 0.8906260654330254
manifest: push/material/mass/CoM/sudden-stop/failure-mining enabled
```

- Stopped the automatically chained PPO after iteration 0 started, because a stricter review found that the PPO path was still the simplified `PPO loss + WM auxiliary loss` implementation, not the AnyAdapter-style replay-buffer plus alternating WM/PPO loop.
- Implemented the missing AnyAdapter-style PPO update path:

```text
new replay buffer: TrackAdapterWorldModelReplayBuffer
WM replay samples: history, psi_s reference, future action sequence, psi_s targets, valid mask
default PPO path: add rollout samples to persistent WM replay, update history_encoder + world_model_predictor from replay, then update adapter/critic with PPO
joint PPO+WM loss: disabled by default
PPO history encoder gradient: detached by default while WM replay owns dynamics embedding updates
checkpoint save: ppo optimizer alias plus separate world_model_optimizer_state_dict
```

- Verification after alternating-update implementation:

```text
py_compile: passed
bash -n scripts/rsl_rl/run_track_adapter_stage3_to_ppo.sh: passed
git diff --check: passed
focused pytest: 30 passed
```

- Alternating PPO smoke from `model_499.pt`:

```text
tmux session: track_adapter_ar_alt_ppo_smoke_20260510_083001
run dir: logs/rsl_rl/run_amp_track_adapter/2026-05-10_08-30-05_track_adapter_ar_alt_ppo_smoke_20260510_083001
iterations: 24
checkpoint: model_23.pt
wm_replay_inserted: 24576 per iteration
wm_replay_size: 65536 after fill
wm_updates: 2 per iteration
base_contact: 0.0000 through iteration 16, 0.0020 by iteration 23
final wm: 0.4943
checkpoint optimizer states: optimizer_state_dict, ppo_optimizer_state_dict, world_model_optimizer_state_dict
```

- Initial full PPO attempt with conservative memory use:

```text
tmux session: track_adapter_ar_alt_ppo_full_20260510_083242
num_envs: 1024
wm_replay_buffer_size: 65536
status: intentionally stopped at iteration 17 because GPU memory was only about 11.9 GB and the machine can tolerate roughly 26 GB.
base_contact: about 0.0010 at stop
reason for restart: scale to 2048 envs and a larger persistent WM replay buffer for the actual full run.
```

- Restarted full PPO with larger GPU budget:

```text
tmux session: track_adapter_ar_alt_ppo_full2048_20260510_083437
log: logs/track_adapter_ar_alt_ppo_full2048_20260510_083437.log
pretrain checkpoint: model_499.pt from the AR Stage 3 run
num_envs: 2048
history/horizon: H=79,N=20
wm_replay_buffer_size: 262144
wm_updates_per_iteration: 4
wm_mini_batch_size: 2048
ppo_mini_batches: 32
observed GPU memory at iteration 4: about 17.3 GB
base_contact through iteration 4: 0.0000
```

- 2048-env full PPO early monitoring:

```text
iteration: 29/12000
wm_replay_size: 262144
wm_updates: 4 per iteration
wm range after replay fill: roughly 0.4879-0.5204
base_contact: 0.0024
adapter_scaled_residual_action_l2: 0.0000 during the first 30 iterations
sudden_stop metrics: sampled events appeared around iterations 20-22 without failure_during_sudden_stop in the monitored tail
GPU memory: about 17.3 GB
status: continuing
```

- 2048-env full PPO passed the first 50-iteration gate:

```text
iteration: 54/12000
wm_replay_size: 262144
wm_updates: 4 per iteration
wm: roughly 0.48-0.52 after replay fill
base_contact: about 0.0020-0.0024
adapter_scaled_residual_action_l2: about 0.0001 after iteration 44
failure_during_sudden_stop: 0.0000 in the monitored tail
GPU memory: about 17.3 GB
status: continuing toward the first nonzero PPO checkpoint
```

- `model_100.pt` validation:

```text
stage: ppo
iter: 100
pretrained_world_model_loaded: True
pretrained_world_model_iter: 499
optimizer states: optimizer_state_dict, ppo_optimizer_state_dict, world_model_optimizer_state_dict
world_model_metadata.prediction_mode: autoregressive
world_model_metadata.horizon: 20
```

- 2048-env full PPO monitoring after checkpoint:

```text
iteration: 131/12000
wm_replay_size: 262144
wm_updates: 4 per iteration
adapter_scaled_residual_action_l2: 0.3379
base_contact: 0.0047
failure_during_sudden_stop: 0.0000
status: continuing; stop condition for manual intervention is base_contact > 0.02 early or scaled residual approaching 1.0.
```

- Stopped the 2048-env full PPO run for early residual/contact intervention:

```text
stopped run: track_adapter_ar_alt_ppo_full2048_20260510_083437
last observed iteration: 168/12000
base_contact: grew past the manual early-intervention threshold, reaching about 0.05
adapter_scaled_residual_action_l2: grew to about 0.84
saved safe branch point: model_100.pt
decision: restart from model_100.pt with lower LR/residual scale and without loading old optimizer momentum.
```

- Added resume control:

```text
script: scripts/rsl_rl/train.py
env: BOOSTER_TRACK_ADAPTER_LOAD_OPTIMIZER=0
behavior: TrackAdapterRunner loads PPO checkpoint weights/normalizers but skips optimizer state so lower LR settings take effect cleanly.
```

- Restarted from the safe PPO branch point:

```text
tmux session: track_adapter_ar_alt_ppo_full2048_soft_20260510_085257
resume source: track_adapter_ar_alt_ppo_full2048_20260510_083437/model_100.pt
load optimizer: false
num_envs: 2048
policy LR: 5.0e-7
WM LR: 1.0e-6
residual_scale: 0.05
init_noise_std: 0.20
residual penalty stable/recovery/max: 0.30 / 0.15 / 2.0
early check: iteration 106, base_contact=0.0000, adapter_scaled_residual_action_l2 about 0.0107
status: continuing as the current full PPO run
```

- Low-LR resumed full PPO monitoring:

```text
iteration: 139/12100
run dir: logs/rsl_rl/run_amp_track_adapter/2026-05-10_08-53-01_track_adapter_ar_alt_ppo_full2048_soft_20260510_085257
checkpoint present: model_100.pt
wm_replay_size: 262144
wm_updates: 4 per iteration
adapter_scaled_residual_action_l2: 0.0230
base_contact: 0.0014
failure_during_sudden_stop: 0.0000
GPU memory: about 17.3 GB
status: continuing; this is the active full PPO run.
```

- Verified the AnyAdapter-style per-iteration update order in code:

```text
TrackAdapterAMPPPO.update order:
1. use the current rollout storage to insert WM samples into TrackAdapterWorldModelReplayBuffer
2. run K replay-sampled WM optimizer steps
3. build PPO minibatches from the current rollout storage
4. run PPO optimizer steps on the current rollout

current full run:
wm_replay_inserted: about 98288-98304 samples per iteration
wm_replay_size: 262144
wm_updates: 4 per iteration
world_model_joint_loss: false
```

- Added explicit numeric full-training gates to the roadmap and checked the active low-LR run against them:

```text
tmux session: track_adapter_ar_alt_ppo_full2048_soft_20260510_085257
iteration: 473/12100
base_contact: 0.0127, below the early <0.02 pass gate
adapter_scaled_residual_action_l2: 0.3975, below the early manual-stop >=0.80 gate
wm_valid_fraction: 0.7783, above the >=0.70 replay gate
wm loss: 0.4715, finite and not exploding in the monitored tail
failure_during_sudden_stop: 0.0000
decision: keep the current full PPO run alive and continue monitoring; it is not in the same failure regime as the earlier stopped run.
```

- Active full PPO status check:

```text
tmux session: track_adapter_ar_alt_ppo_full2048_soft_20260510_085257
iteration: 701/12100
checkpoint present: model_700.pt
GPU memory: 17.3 GB / 32.6 GB
base_contact: 0.0184, still below the <0.02 early pass gate but close enough to monitor
adapter_scaled_residual_action_l2: 0.5343, below the >=0.80 intervention gate but trending upward
wm_valid_fraction: 0.7782, above the >=0.70 replay gate
wm loss: 0.4529, finite and stable in the monitored tail
sudden_stop_sampled: 0.0625
failure_during_sudden_stop: 0.0000
decision: keep training running; treat contact/residual as a yellow-zone trend and recheck before/around the next few checkpoints.
```

- Rechecked residual/contact trend after the user noted the residual climb:

```text
iteration window: 660-739
adapter_scaled_residual_action_l2: roughly 0.51 -> 0.56, with a short-term plateau around 0.54-0.56
base_contact: repeatedly near or above 0.02, latest observed 0.0210 at iteration 739
wm_valid_fraction: still around 0.77-0.78
failure_during_sudden_stop: still 0.0000
assessment: not a clean green run; do not rely on this run reaching final training successfully without intervention.
decision: keep the current checkpoint history, but treat model_700+ as a diagnostic branch unless contact drops back below 0.02 and residual stops climbing.
```

- Stopped the active low-LR full PPO run after the trend worsened:

```text
stopped tmux session: track_adapter_ar_alt_ppo_full2048_soft_20260510_085257
last logged iteration: 748/12100
saved checkpoints: model_100.pt through model_700.pt
latest base_contact: 0.0240, above the 0.02 caution gate for multiple recent iterations
latest adapter_scaled_residual_action_l2: 0.5657, below the hard intervention gate but no longer a clean/stable residual regime
latest reward: 61.72, degraded from the earlier 110+ range
wm_valid_fraction: 0.7787
failure_during_sudden_stop: 0.0000
decision: do not let this branch continue toward a likely high-residual solution; use the saved checkpoints for diagnosis and restart from an earlier safer point with stronger residual control.
```

- Added env-configurable Track Adapter safety gates so the next branch can stop earlier than the permissive default:

```text
file: source/booster_rl_tasks/booster_rl_tasks/tasks/manager_based/beyond_mimic/robots/k1/run_amp/ppo_cfg.py
envs:
  BOOSTER_TRACK_ADAPTER_SAFETY_MIN_ITERATION
  BOOSTER_TRACK_ADAPTER_SAFETY_MAX_BASE_CONTACT
  BOOSTER_TRACK_ADAPTER_SAFETY_MAX_SCALED_RESIDUAL_ACTION_L2
  BOOSTER_TRACK_ADAPTER_SAFETY_PATIENCE
reason: the default base_contact=0.20 and scaled_residual=1.25 gates are too loose for early branch triage.
```

- Started a guarded restart from the safer `model_300.pt` branch, then stopped it immediately after finding that resume was still using the checkpointed action noise:

```text
tmux session: track_adapter_ar_alt_ppo_guarded_from300_20260510_100049
resume source: model_300.pt from the low-LR branch
requested action std: 0.12
observed action std after resume: 0.25
reason: checkpoint loading restores the trainable std parameter after config construction, so init_noise_std does not affect resumed runs.
decision: stop the guarded restart before it becomes a real branch; add an explicit resume-time action-std override.
```

- Added resume-time action noise override:

```text
file: rsl_rl/rsl_rl/runners/track_adapter_runner.py
env: BOOSTER_TRACK_ADAPTER_OVERRIDE_ACTION_STD
behavior: after loading a Track Adapter PPO checkpoint, overwrite the scalar/log std parameter before training continues.
```

- Restarted guarded branch with resume-time action std override confirmed:

```text
tmux session: track_adapter_ar_alt_ppo_guarded_from300_std012_20260510_100253
resume source: low-LR branch model_300.pt
load optimizer: false
action std override: BOOSTER_TRACK_ADAPTER_OVERRIDE_ACTION_STD=0.12
observed action std: 0.12 at iterations 301-302
residual_scale: 0.03
policy LR: 2.0e-7
PPO epochs: 1
residual penalty stable/recovery/max: 0.80 / 0.25 / 6.0
safety gate: starts at iter 330, base_contact > 0.025 or scaled_residual > 0.65 for 3 checks aborts
early metrics: scaled_residual_action_l2 about 0.067, wm_valid_fraction 0.78-0.80, wm loss finite
status: active; first episode-level contact/reward metrics are still warming up.
```

- Guarded branch status check:

```text
tmux session: track_adapter_ar_alt_ppo_guarded_from300_std012_20260510_100253
latest observed iteration: 540/12100
checkpoints present: model_300.pt, model_400.pt, model_500.pt
action std: 0.12, override still active
adapter_scaled_residual_action_l2 trend: 0.0741 at iter 300 -> 0.0994 at iter 540
base_contact trend: 0.0000-0.0020 through iter 540
reward trend: 185.52 at iter 350, 177.33 at iter 400, 169.04 at iter 500, 164.58 at iter 540
wm loss: 0.4178 at iter 540, finite
wm_valid_fraction: 0.7742 at iter 540
failure_during_sudden_stop: 0.0000
assessment: much healthier than the stopped branch at the same phase; continue training, while watching the slow reward decline and residual creep toward 0.10.
```

- Interpreted the slow reward decline in the guarded branch:

```text
latest observed iteration: 567/12100
reward: about 185 at iter 350 -> about 162 at iter 567
base_contact: still low, about 0.001-0.003
scaled_residual_action_l2: about 0.074 -> 0.101
wm_valid_fraction: still about 0.77-0.78
failure_during_sudden_stop: 0.0000
likely causes:
  - the restart intentionally uses lower action std, lower LR, lower residual scale, stronger residual penalty, and PPO epochs=1;
  - failure/push mining is active, so recent episode-return samples include harder perturbation windows;
  - RSL-RL Mean reward is the mean completed episode return buffer, not a fixed-distribution evaluation score;
  - some recent windows show tracking/recovery dips, but not contact collapse.
decision: continue; monitor whether reward stabilizes around 160-170 or keeps falling together with residual/contact.
```

- Guarded branch status check:

```text
tmux session: track_adapter_ar_alt_ppo_guarded_from300_std012_20260510_100253
latest observed iteration: 1119/12100
checkpoints present: model_300.pt through model_1100.pt
GPU memory: about 17.3 GB / 32.6 GB
action std: 0.12
adapter_scaled_residual_action_l2 trend: 0.0741 at iter 300 -> 0.1528 at iter 1119
adapter_scaled_residual_action_l2 max so far: 0.1662
base_contact latest: 0.0010, still far below the 0.02 caution gate
reward trend: 177.33 at iter 400, 153.50 at iter 700, 139.88 at iter 1000, 128.50 at iter 1119
wm loss latest: 0.3781, finite
wm_valid_fraction latest: 0.7775; minimum seen in this branch: 0.6962
failure_during_sudden_stop: 0.0000
assessment: locomotion safety/contact remains healthy, but reward is still drifting downward and WM valid briefly touched the 0.70 gate; continue for now, but this is not a clean performance-green run.
```

- Diagnosed the apparent reward decline:

```text
source: TensorBoard scalars in the guarded branch run directory
key point: console Episode_Reward/* values are raw env episode metrics; Train/mean_reward is the final runner reward after recovery gates, AMP style reward, reward-group weights, and residual penalty.

iter 400:
  RewardGroup/task_mean: 0.1393
  RewardGroup/style_mean: 0.1070
  RewardGroup/recovery_mean: 0.0291
  RewardGroup/reg_mean: -0.0047
  Adapter/residual_penalty_mean: 0.0563
  RewardGroup/total_mean: 0.1775
  Train/mean_reward: 177.33

iter 1119:
  RewardGroup/task_mean: 0.1404
  RewardGroup/style_mean: 0.1072
  RewardGroup/recovery_mean: 0.0286
  RewardGroup/reg_mean: -0.0054
  Adapter/residual_penalty_mean: 0.1036
  RewardGroup/total_mean: 0.1306
  Train/mean_reward: 128.50

interpretation: task/style/recovery rewards are mostly stable; the drop in final return is almost entirely explained by residual_penalty_mean increasing by about 0.047 per step, which is about 47 episode-return points over a 1000-step episode.
decision: velocity tracking is not the main issue; monitor residual usage and evaluate fixed checkpoints instead of reading Train/mean_reward alone as tracking performance.
```

- Ran fixed abrupt-stop recovery evaluation for guarded branch checkpoints:

```text
eval dir: logs/track_adapter_fixed_eval_20260510_115151
script: scripts/rsl_rl/eval_track_adapter_recovery.py
num_envs: 256
seed: 123
command: move at x=0.8, then abrupt all-zero command
timing: warmup 100, move 150, stop 150
background disturbance: push enabled, failure mining disabled
action std override: 0.12

checkpoint | fall_rate | base_contact_rate | settle_success_rate | valid_stop_env_rate | mean_recovery_steps | final_stop_speed_mean | final_stop_speed_p95 | final_residual_norm_mean
model_700  | 0.0000    | 0.0000            | 1.0000              | 1.0000              | 27.49               | 0.0985                | 0.2370               | 0.9419
model_1000 | 0.0078    | 0.0078            | 0.9922              | 0.9922              | 27.96               | 0.1140                | 0.2559               | 1.1122
model_1200 | 0.0508    | 0.0312            | 0.9492              | 0.9492              | 27.83               | 0.1170                | 0.2676               | 1.1777

interpretation: model_700 is the best fixed-eval checkpoint among the three. Later checkpoints use more residual, stop slightly worse, and begin to show fall/base-contact failures.
decision: stopped the active guarded training branch after it reached model_1400 because fixed evaluation showed degradation after model_700/model_1000 rather than recovery improvement.
```

- Evaluated fixed low-speed command tracking and push response for the selected guarded checkpoint:

```text
eval dir: logs/velocity_grid_eval_20260510_121031
script: scripts/rsl_rl/eval_velocity_command_grid.py
adapter checkpoint: guarded branch model_700.pt
base checkpoint: run_amp_y model_70000.pt
commands: vx = 0.0, 0.2, 0.4, 0.8 with vy=0.0, yaw=0.0
num_envs: 128
steps / settled samples: 350 / last 200
push eval: interval push enabled every 1.0-2.0 s, x/y +/-0.30, yaw +/-0.15

adapter no-push:
  vx=0.0 -> actual -0.011, abs_vx_error 0.029, no falls/contact, residual 0.886
  vx=0.2 -> actual  0.087, abs_vx_error 0.122, no falls/contact, residual 0.960
  vx=0.4 -> actual  0.337, abs_vx_error 0.096, no falls/contact, residual 1.009
  vx=0.8 -> actual  0.705, abs_vx_error 0.121, no falls/contact, residual 0.944

adapter push:
  vx=0.0 -> actual -0.026, abs_vx_error 0.055, no falls/contact, residual 0.919
  vx=0.2 -> actual  0.105, abs_vx_error 0.113, no falls/contact, residual 0.994
  vx=0.4 -> actual  0.322, abs_vx_error 0.115, no falls/contact, residual 1.007
  vx=0.8 -> actual  0.662, abs_vx_error 0.173, ever_fall_rate 0.008, base_contact 0.000, residual 0.949

base no-push:
  vx=0.0 -> actual -0.001, abs_vx_error 0.018, no falls/contact
  vx=0.2 -> actual  0.065, abs_vx_error 0.161, no falls/contact
  vx=0.4 -> actual  0.422, abs_vx_error 0.067, no falls/contact
  vx=0.8 -> actual  0.752, abs_vx_error 0.087, no falls/contact

base push:
  vx=0.0 -> actual  0.002, abs_vx_error 0.045, no falls/contact
  vx=0.2 -> actual  0.098, abs_vx_error 0.140, no falls/contact
  vx=0.4 -> actual  0.419, abs_vx_error 0.078, no falls/contact
  vx=0.8 -> actual  0.756, abs_vx_error 0.101, no falls/contact

interpretation:
  - zero command does not produce sustained walking in mean velocity; adapter still uses too much residual at zero/low speed.
  - low-speed vx=0.2 tracking is poor in the frozen base itself, so this is not purely adapter-induced.
  - adapter model_700 preserves safety in this grid, but slightly worsens vx=0.4/0.8 tracking and shows one push fall at vx=0.8.
  - current command sampling is continuous over the full range but does not sufficiently stratify low-speed straight walking; the narrow vx=0.15-0.45, low-y, low-yaw region is underrepresented.
```

- Implemented low-speed and zero-command follow-up changes:

```text
command sampler:
  added OmniVelocityCommand low-speed straight-walk stratum
  env vars:
    BOOSTER_TRACK_ADAPTER_LOW_SPEED_ENVS default 0.22 full / 0.10 smoke
    BOOSTER_TRACK_ADAPTER_LOW_SPEED_X_MIN default 0.15
    BOOSTER_TRACK_ADAPTER_LOW_SPEED_X_MAX default 0.45
    BOOSTER_TRACK_ADAPTER_LOW_SPEED_Y_ABS_MAX default 0.08
    BOOSTER_TRACK_ADAPTER_LOW_SPEED_YAW_ABS_MAX default 0.06
    BOOSTER_TRACK_ADAPTER_LOW_SPEED_FORWARD_PROB default 0.90
  metric: Metrics/base_velocity/low_speed_sampled

zero-command residual discipline:
  added stable-only zero-command residual penalty boost
  env vars:
    BOOSTER_TRACK_ADAPTER_RESIDUAL_ZERO_COMMAND_WEIGHT default 0.20
    BOOSTER_TRACK_ADAPTER_RESIDUAL_ZERO_COMMAND_THRESHOLD default 0.08
  implementation: extra residual coefficient is multiplied by style_gate, so push/recovery states are not aggressively suppressed.

play/eval ergonomics:
  scripts/rsl_rl/play.py now supports --fixed_command VX VY WZ and --disable_push
  scripts/rsl_rl/eval_velocity_command_grid.py now logs push_enabled separately from Track Adapter's failure_push_active metric.

verification:
  py_compile passed for changed Python files
  git diff --check passed
  pytest passed: tests/test_track_adapter_module.py tests/test_track_adapter_storage.py tests/test_track_adapter_optimizer.py -> 31 passed
```

- Reframed the practical objective and strengthened external-force robustness:

```text
objective:
  AMP is treated as a useful locomotion prior, not a hard behavioral constraint.
  Primary priorities are push recovery, velocity tracking, and survival under jerk commands.
  Zero command should stay quiet only when stable and unpushed; during external disturbance it may step or move to recover.

runner gate changes:
  added push-active reward gating from Metrics/base_velocity/failure_push_active
  during the post-push window:
    style_gate is capped by BOOSTER_TRACK_ADAPTER_PUSH_STYLE_GATE_MAX, default 0.60
    recovery_gate is raised to at least BOOSTER_TRACK_ADAPTER_PUSH_RECOVERY_GATE_MIN, default 0.45
    command_gate is raised to at least BOOSTER_TRACK_ADAPTER_PUSH_COMMAND_GATE_MIN, default 0.85
  logged Gate/push_active_mean

curriculum/reward changes:
  default full-training interval pushes changed to every 4-7 s
  default full-training push delta increased to xy +/-0.40, yaw +/-0.22
  push failure-mining sample prob increased to 0.30
  push active/failure window increased to 2.5 s
  recovery group weight increased to 1.15 and task weight to 1.30
  upright/no-fall/support recovery term defaults increased

evaluation changes:
  scripts/rsl_rl/eval_track_adapter_recovery.py now supports:
    --scenario_set high_speed_jerk
    --commands "vx,vy,wz;..."
    --push_modes none|push|both
    deterministic eval push at stop time via --eval_push_speed/--eval_push_yaw/--eval_push_step
  preset high-speed jerk scenarios:
    vx=2.0 -> 0
    vx=1.0 -> 0
    vy=1.0 -> 0
    vx=1.0, vy=1.0 -> 0

training decision:
  previous run track_adapter_practical_lowspd_jerk_20260510_125047 was healthy at iteration 113
    base_contact around 0.002
    wm_valid_fraction around 0.776
    low-speed samples present
    residual still 0.0
  stopped it because the new push-active gate and stronger push curriculum require process restart.

verification:
  py_compile passed for changed Python files
  bash -n passed for run_track_adapter_stage3_to_ppo.sh
  git diff --check passed
  pytest passed: tests/test_track_adapter_module.py tests/test_track_adapter_storage.py tests/test_track_adapter_optimizer.py -> 32 passed
```

- Started the new push-robust PPO branch:

```text
tmux session: track_adapter_pushrobust_lowspd_jerk_20260510_130532
log file: logs/track_adapter_pushrobust_lowspd_jerk_20260510_130532.log
run dir: logs/rsl_rl/run_amp_track_adapter/2026-05-10_13-05-36_track_adapter_pushrobust_lowspd_jerk_20260510_130532_ppo
pretrain checkpoint: logs/rsl_rl/run_amp_track_adapter/2026-05-10_07-46-30_track_adapter_ar_stage3_to_ppo_20260510_074627_wm_pretrain_push/model_499.pt
note: tmux pipe-pane logging was attached after startup, so the log file starts from about iteration 10; earlier pane output was checked from tmux capture.

confirmed launch config:
  num_envs: 2048
  command range: vx -1.0..2.0, vy -1.0..1.0, yaw -0.45..0.45
  low-speed stratum: 0.25
  sudden-stop stratum: 0.10
  interval push: every 4.0-7.0 s
  push delta: x/y +/-0.40, yaw +/-0.22
  push failure-mining sample prob: 0.30
  push active/failure window: 2.5 s
  reward group weights: task 1.30, style 0.45, recovery 1.15, reg 0.35
  push gates: recovery >=0.45, style <=0.60, command >=0.85 while push-active
  residual scale/std/lr: 0.06 / 0.12 / 5e-7
  WM replay: 262144, 4 updates/iteration, batch 2048

initial health check:
  latest checked around iteration 12/12000
  wm loss: about 0.48-0.50
  wm_valid_fraction: about 0.775-0.782
  adapter_scaled_residual_action_l2: 0.0000
  base_contact: about 0.0034
  low_speed_sampled/command: active in the logged windows
  failure_push_active: active in early windows, confirming stronger push curriculum is present
  assessment: healthy enough to continue; this is still before residual learning meaningfully starts.
```

- Applied review fixes and restarted the push-robust branch again:

```text
fixes:
  push-active recovery gates now fail fast if Metrics/base_velocity/failure_push_active is missing or shape-mismatched
  zero-command residual penalty now ignores environments that are in sudden-stop recovery or push-active recovery
  eval_track_adapter_recovery disables interval env pushes by default for clean fixed-command tests; --keep_env_push opts back in

verification:
  py_compile passed for track_adapter_runner.py and eval_track_adapter_recovery.py
  git diff --check passed
  pytest passed: tests/test_track_adapter_module.py tests/test_track_adapter_storage.py tests/test_track_adapter_optimizer.py -> 34 passed

stopped stale run:
  tmux session: track_adapter_pushrobust_lowspd_jerk_20260510_130532
  reason: it was launched before the fail-fast/zero-penalty recovery fixes.

active run:
  tmux session: track_adapter_pushrobust_lowspd_jerk_v2_20260510_130938
  log file: logs/track_adapter_pushrobust_lowspd_jerk_v2_20260510_130938.log
  run dir: logs/rsl_rl/run_amp_track_adapter/2026-05-10_13-09-42_track_adapter_pushrobust_lowspd_jerk_v2_20260510_130938_ppo
  confirmed params:
    low_speed_envs: 0.25
    sudden_stop_envs: 0.10
    push failure-mining sample prob: 0.30
    push window: 2.5 s
    push gates: recovery >=0.45, style <=0.60, command >=0.85

initial v2 health:
  checked through about iteration 23/12000
  wm loss: about 0.47-0.51
  wm_valid_fraction: about 0.776-0.785
  adapter_scaled_residual_action_l2: 0.0000
  base_contact: about 0.0039
  low_speed_sampled/command: active in logged windows; latest checked low_speed_sampled 0.3015, low_speed_command 0.2466
  sudden-stop active in latest checked window; failure_during_sudden_stop 0.0000
  push_active: present by iteration 4 and in later windows; latest checked failure_push_active 0.2990
  TensorBoard scalar Gate/push_active_mean is present; latest checked value at step 21 was 0.00049
  zero-command residual penalty active excludes push/sudden-stop recovery and was logged at 0.116 at step 21
  assessment: continue; still too early to judge learned residual behavior.
```

- Safety gate stopped the v2 branch and a safer resume branch was started:

```text
stopped run:
  tmux session: track_adapter_pushrobust_lowspd_jerk_v2_20260510_130938
  stop reason: TrackAdapterRunner safety gate, not a simulator crash
  stopped at: iteration 1884
  gate: base_contact > 0.035 for 3 consecutive checks

stop-window metrics:
  iter 1882 base_contact 0.0353, scaled_residual_l2 0.7212
  iter 1883 base_contact 0.0363, scaled_residual_l2 0.7248
  iter 1884 base_contact 0.0377, scaled_residual_l2 0.7266
  wm_valid_fraction: about 0.778
  failure_during_sudden_stop: 0.0000
  failure_push_active: 0.24-0.35 in the stop window
  interpretation: residual finally learned, but residual magnitude and contact risk rose together; treat as over-intervention, not WM failure.

trend from TensorBoard:
  scaled_residual_l2:
    step 1000: 0.0166
    step 1500: 0.2874
    step 1800: 0.6461
    step 1884: 0.7266
  base_contact:
    step 1000: 0.0000
    step 1500: 0.0039
    step 1800: 0.0229
    step 1884: 0.0377

resume decision:
  checkpoint chosen: model_1700.pt
  reason: model_1800 already had base_contact 0.0229, while model_1700 is before the contact ramp.
  optimizer state: not loaded
  push curriculum: kept strong
  residual changes:
    residual_scale 0.06 -> 0.05
    init_noise_std 0.12 -> 0.10
    LR 5e-7 -> 3e-7
    entropy 0.00008 -> 0.00005
    stable residual penalty 0.08 -> 0.12
    recovery residual penalty 0.015 -> 0.03
    reg weight 0.35 -> 0.38
    reg_recovery_scale 0.65 -> 0.80
  safety:
    max_base_contact stays 0.035
    max_scaled_residual_l2 tightened to 1.2

active resumed run:
  tmux session: track_adapter_pushrobust_lowspd_jerk_v3_resume1700_20260510_161501
  log file: logs/track_adapter_pushrobust_lowspd_jerk_v3_resume1700_20260510_161501.log
  resume source: logs/rsl_rl/run_amp_track_adapter/2026-05-10_13-09-42_track_adapter_pushrobust_lowspd_jerk_v2_20260510_130938_ppo/model_1700.pt

initial resume health:
  checked through displayed iteration 1705/13700
  wm loss: about 0.36
  wm_valid_fraction: about 0.778-0.785
  scaled_residual_l2: about 0.34 after lowering residual_scale
  base_contact: about 0.0011
  push active: present by iteration 1705
  assessment: resumed branch is running and initially stable; monitor whether contact remains below 0.02 as residual adapts.
```

- Checked the resumed v3 branch on 2026-05-11:

```text
run:
  tmux session: track_adapter_pushrobust_lowspd_jerk_v3_resume1700_20260510_161501
  status: stopped by TrackAdapterRunner safety gate, not currently training
  stopped at: displayed iteration 6570/13700
  run dir: logs/rsl_rl/run_amp_track_adapter/2026-05-10_16-15-05_track_adapter_pushrobust_lowspd_jerk_v3_resume1700_20260510_161501_ppo
  checkpoints: regular checkpoints through model_6500.pt plus model_6570_aborted.pt

stop reason:
  adapter_scaled_residual_action_l2 exceeded the hard gate of 1.2 for 3 consecutive checks
  safety strikes ended at scaled_residual_action_l2 about 1.2071
  base_contact did not trip the configured 0.035 gate in the stop window

latest stop-window metrics:
  mean_reward: 130.6424
  wm loss: 0.2250
  wm_valid_fraction: 0.7737
  adapter_scaled_residual_action_l2: 1.2034
  base_contact: 0.0236
  error_vel_xy: 0.3760
  error_vel_yaw: 0.4220
  failure_push_active: 0.2648
  failure_during_sudden_stop: 0.0000
  sudden_stop_sampled: 0.0451
  low_speed_sampled: 0.2370
  low_speed_command: 0.1936

trend summary:
  mean_reward decreased from about 212.67 at step 2000 to 130.64 at step 6570
  scaled_residual_l2 rose from about 0.52 at step 2000 to 1.20 at step 6570
  base_contact rose late from about 0.0095 at step 6000 to about 0.0236 at step 6570
  wm loss improved from about 0.324 at step 2000 to about 0.225 at step 6570
  wm_valid_fraction stayed healthy around 0.77

current machine state:
  no train.py process is active
  no Track Adapter training tmux session is active
  a play.py process is not active at the latest check
  GPU memory usage is low, about 419 MiB

assessment:
  world-model replay/alternating training stayed numerically healthy
  sudden-stop failures stayed at zero in logged windows
  the branch learned substantial intervention, but residual usage continued to grow and hit the explicit residual gate
  model_6500.pt is the latest regular checkpoint for evaluation; do not blindly resume past it without fixed-grid push/jerk/low-speed evaluation
```

- Ran fixed evaluation for the v3 branch after qualitative degradation was suspected:

```text
eval dir: logs/fixed_eval_v3_20260511_070051
tmux session: fixed_eval_v3_20260511_070051
status: completed

instrumentation:
  scripts/rsl_rl/eval_velocity_command_grid.py now logs left/right hand relative position means and hand x-range summaries.

velocity grid protocol:
  num_envs: 192
  steps / settle_steps: 320 / 140
  seed: 421
  commands:
    (0.0, 0.0, 0.0)
    (0.2, 0.0, 0.0)
    (0.4, 0.0, 0.0)
    (0.8, 0.0, 0.0)
    (1.0, 0.0, 0.0)
    (2.0, 0.0, 0.0)
    (0.0, 1.0, 0.0)
    (1.0, 1.0, 0.0)
    (0.0, 0.0, 0.3)
  no-push checkpoints:
    frozen base model_70000
    v3 model_5000
    v3 model_6000
    v3 model_6500
  push checkpoints:
    frozen base model_70000
    v3 model_5000
    v3 model_6500
  push eval: interval push every 1.0-2.0 s, x/y +/-0.30, yaw +/-0.15

high-speed jerk protocol:
  num_envs: 192
  warmup/move/stop: 80 / 120 / 160
  seed: 422
  scenario_set: high_speed_jerk
  push_modes: none and deterministic push
  eval push: speed 0.60, yaw 0.10 at stop step 0
  checkpoints:
    frozen base model_70000
    v3 model_5000
    v3 model_6500

key velocity-grid results:
  frozen base no-push:
    zero command: actual vx/vy/wz = -0.001/-0.001/-0.001, fall/base_contact = 0/0
    vx=0.2: actual vx = 0.058, confirming base under-tracks very low speed
    vx=2.0: actual vx = 1.931, fall/base_contact = 0/0
    vx=1.0,vy=1.0: fall/base_contact = 0.010/0.010
  frozen base push:
    vx=2.0: actual vx = 1.950, fall/base_contact = 0/0
    vx=1.0,vy=1.0: fall/base_contact = 0.018/0.009

  v3 model_5000 no-push:
    zero command: actual vx/vy/wz = 0.040/0.132/0.068, residual = 1.519
    vx=0.8: actual vx = 0.612, abs_vx_error = 0.207, residual = 1.441
    vx=2.0: actual vx = 1.782, fall/base_contact = 0.003/0.003, residual = 1.518
    vx=1.0,vy=1.0: fall/base_contact = 0.005/0.005, residual = 1.447
  v3 model_6000 no-push:
    zero command: actual vx/vy/wz = 0.079/0.102/0.163, residual = 1.534
    vx=2.0: fall/base_contact = 0.045/0.045, residual = 1.798
    vx=1.0,vy=1.0: fall/base_contact = 0.049/0.038, residual = 1.603
  v3 model_6500 no-push:
    zero command: actual vx/vy/wz = 0.076/0.103/0.198, residual = 1.624
    vx=2.0: actual vx/vy/wz = 1.710/0.223/-0.123, fall/base_contact = 0.071/0.066, residual = 1.978
    vx=1.0,vy=1.0: fall/base_contact = 0.246/0.211, residual = 1.740
  v3 model_6500 push:
    zero command: actual vx/vy/wz = 0.075/0.097/0.200, residual = 1.646
    vx=2.0: fall/base_contact = 0.095/0.084, residual = 1.999
    vx=1.0,vy=1.0: fall/base_contact = 0.234/0.196, residual = 1.744

key high-speed jerk results:
  frozen base:
    all no-push scenarios: fall/base_contact = 0/0, final speed about 0.034-0.036
    push scenarios: only vx=2.0 and vy=1.0 cases showed about 0.005 fall; final speed about 0.033-0.036
  v3 model_5000:
    final stop speed about 0.178-0.193, residual about 1.51-1.54
    fall/base_contact mostly 0-0.021/0-0.016
  v3 model_6500:
    vx=2.0 -> 0 no-push: fall/base_contact = 0.062/0.052, final speed = 0.196, residual = 1.631
    vx=2.0 -> 0 push: fall/base_contact = 0.078/0.068, final speed = 0.192, residual = 1.611
    vx=1.0,vy=1.0 -> 0 no-push: fall/base_contact = 0.328/0.276, final speed = 0.196, residual = 1.625
    vx=1.0,vy=1.0 -> 0 push: fall/base_contact = 0.266/0.224, final speed = 0.185, residual = 1.602

assessment:
  The user's qualitative impression is supported by fixed evaluation.
  The v3 branch does not merely trade reward for robustness; it degrades no-push tracking, zero-command quietness, diagonal/high-speed safety, and stop settling.
  Residual intervention is high even when no push is present, so the adapter is overriding the base locomotion rather than staying as a recovery adapter.
  The frozen base is stronger than v3 model_6500 on the fixed high-speed jerk suite under these conditions.
  Do not promote model_6500 and do not resume this branch as-is.
  If any v3 checkpoint is kept for reference, model_5000 is less bad than model_6500 but still fails the residual/zero-command/tracking quality bar.
```

## 2026-05-11 07:45 UTC - Residual action gate for v4 recovery branch

User decision:
  Do not promote v3 and change the training setup so the adapter cannot freely override the frozen base during normal no-push locomotion.

Implemented:
  - Added an action-side residual gate to `TrackAdapterActorCritic`.
    - The residual head still predicts an ungated residual for diagnostics/regularization.
    - The action mean now uses `base_action + residual_gate * residual_scale * residual`.
    - Diagnostics now expose both gated `scaled_residual_action_l2` and `ungated_scaled_residual_action_l2`.
  - Threaded `residual_action_gate` through PPO rollout storage.
    - The rollout-time gate is stored in `TrackAdapterRolloutStorage`.
    - PPO log-prob recomputation reuses the same gate, so the distribution used for PPO matches the sampled action distribution.
  - Added runner-side residual gate computation.
    - Stable no-push states are capped to a small gate.
    - Stable zero-command states can close the residual gate fully.
    - Push-active and sudden-stop windows reopen the gate with configurable minimums.
    - Runner logs `Gate/residual_action_mean/min/max`, stable-limited fraction, and sudden-stop-active fraction.
  - Added an ungated stable residual penalty so the hidden residual head is still discouraged from growing in stable states even when the action gate masks it.
  - Added config/env knobs:
    - `BOOSTER_TRACK_ADAPTER_RESIDUAL_ACTION_GATE`
    - `BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_STABLE_FLOOR`
    - `BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_STABLE_MAX`
    - `BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_ZERO_COMMAND`
    - `BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PUSH_MIN`
    - `BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_SUDDEN_STOP_MIN`
    - `BOOSTER_TRACK_ADAPTER_RESIDUAL_UNGATED_STABLE_WEIGHT`
  - Updated tests for storage tuple shift, gate preservation/defaulting, actor gate behavior, and runner stable-vs-recovery gate behavior.

Verification:
  - `python -m py_compile ...` passed for modified Track Adapter modules/config/tests/eval script.
  - `pytest -q tests/test_track_adapter_module.py tests/test_track_adapter_storage.py tests/test_track_adapter_optimizer.py` passed: `37 passed`.
  - `git diff --check` passed.

Next run:
  Start a new v4 gated branch from the validated Stage 3 world-model pretrain, not from v3 PPO checkpoints.
  Monitor early `adapter_scaled_residual_action_l2`, `adapter_ungated_scaled_residual_action_l2`, `Gate/residual_action_mean`, zero-command drift, `base_contact`, and high-speed jerk safety before any promotion.

## 2026-05-11 07:57 UTC - v4 gate review fixes before restart

Review findings addressed:
  - Stopped the first `track_adapter_gated_recovery_v4_20260511_075052` run after startup because the safety gate was still reading the policy cache after PPO minibatch recomputation instead of rollout-time intervention.
    - That run reached early training and showed low `base_contact` (`~0.001`) before it was manually interrupted, but it is not a candidate branch because code changed immediately after.
  - Added rollout-time adapter diagnostics:
    - `AdapterRollout/scaled_residual_action_l2`
    - `AdapterRollout/ungated_scaled_residual_action_l2`
    - `AdapterRollout/residual_action_gate_mean/min/max`
    - safety now uses the rollout aggregate, not the last PPO minibatch cache.
  - Added a differentiable PPO auxiliary loss for low-gate samples:
    - `Loss/ungated_residual_policy_penalty`
    - controlled by `BOOSTER_TRACK_ADAPTER_RESIDUAL_UNGATED_POLICY_LOSS_WEIGHT`
    - gated by `BOOSTER_TRACK_ADAPTER_RESIDUAL_UNGATED_POLICY_GATE_THRESHOLD`
    - this keeps `zero_command_gate=0.0` from hiding residual-head growth from gradients.
  - Made residual gate width fail fast:
    - actor accepts only scalar per-env gate or per-action gate;
    - rollout storage intentionally stores scalar gates only.
  - Changed recovery-gated reward plumbing so residual penalty can use the pre-action gate/command/recovery info for the action that was just taken.
  - Added `scripts/rsl_rl/run_track_adapter_gated_ppo.sh` so the v4 PPO-only restart from a validated Stage 3 pretrain is reproducible.

Verification:
  - `python -m py_compile` passed for modified Python files.
  - `bash -n` passed for Track Adapter run scripts.
  - `pytest -q tests/test_track_adapter_module.py tests/test_track_adapter_storage.py tests/test_track_adapter_optimizer.py` passed: `40 passed`.
  - `git diff --check` passed.

## 2026-05-11 08:27 UTC - Interrupted v4 restart handling

Observation:
  - `track_adapter_gated_recovery_v4_20260511_075902` stopped around iteration `32/1200`.
  - No traceback, safety abort, or Python exception was present in the run log.
  - Last TensorBoard values were healthy for a start-from-base run:
    - `Episode_Termination/base_contact`: `0.0039`
    - `AdapterRollout/scaled_residual_action_l2`: `2.3e-10`
    - `AdapterRollout/ungated_scaled_residual_action_l2`: `1.0e-9`
    - `Loss/wm`: `0.454`
    - `Loss/wm_valid_fraction`: `0.773`
    - `Metrics/base_velocity/failure_during_sudden_stop`: `0.0`
  - Only `model_0.pt` was saved, so this branch should be treated as interrupted rather than resumable from useful progress.

Change:
  - Added `BOOSTER_TRACK_ADAPTER_SAVE_INTERVAL` support to Track Adapter PPO config.
  - Set the v4 and stage3-to-PPO scripts to default `BOOSTER_TRACK_ADAPTER_SAVE_INTERVAL=25` so future interruptions leave usable short-interval checkpoints.

Verification:
  - `py_compile` passed for `ppo_cfg.py`.
  - `bash -n` passed for both Track Adapter run scripts.
  - `git diff --check` passed.

## 2026-05-11 08:32 UTC - v4 gated PPO restarted

Run:
  - tmux session: `track_adapter_gated_recovery_v4_20260511_082805`
  - log: `logs/track_adapter_gated_recovery_v4_20260511_082805.log`
  - run dir: `logs/rsl_rl/run_amp_track_adapter/2026-05-11_08-28-29_track_adapter_gated_recovery_v4_20260511_082805_ppo`
  - start: Stage 3 pretrain `model_499.pt`, no v3 PPO checkpoint.
  - save interval: `25`

Status at TensorBoard step `39`:
  - session running
  - checkpoint exists: `model_25.pt`
  - `Episode_Termination/base_contact`: `0.0034`
  - `AdapterRollout/scaled_residual_action_l2`: `3.1e-10`
  - `AdapterRollout/ungated_scaled_residual_action_l2`: `1.3e-9`
  - `AdapterRollout/residual_action_gate_mean`: `0.386`
  - `AdapterRollout/residual_action_gate_max`: `0.700`
  - `Gate/push_active_mean`: `0.391`
  - `Loss/wm`: `0.449`
  - `Loss/wm_valid_fraction`: `0.780`
  - `Metrics/base_velocity/error_vel_xy`: `0.220`
  - `Metrics/base_velocity/error_vel_yaw`: `0.266`
  - `Metrics/base_velocity/failure_during_sudden_stop`: `0.0`

Assessment:
  Initial behavior is intentionally conservative: the action gate opens in push-active windows, but the residual actor is still effectively zero. Continue through the safety-gate window before making a promotion decision.

## 2026-05-11 08:41 UTC - v4 step 108 status

Status:
  - session still running
  - checkpoints saved: `model_0.pt`, `model_25.pt`, `model_50.pt`, `model_75.pt`, `model_100.pt`
  - `Episode_Termination/base_contact`: `0.0015`
  - `Safety/base_contact`: `0.0012`
  - `Safety/scaled_residual_action_l2`: `1.1e-10`
  - `AdapterRollout/scaled_residual_action_l2`: `2.3e-10`
  - `AdapterRollout/ungated_scaled_residual_action_l2`: `3.4e-9`
  - `AdapterRollout/residual_action_gate_mean`: `0.165`
  - `AdapterRollout/residual_action_gate_max`: `0.700`
  - `Loss/wm`: `0.437`
  - `Loss/wm_valid_fraction`: `0.781`
  - `Train/mean_reward`: `274.8`
  - `Metrics/base_velocity/error_vel_xy`: `0.230`
  - `Metrics/base_velocity/error_vel_yaw`: `0.274`
  - `Metrics/base_velocity/failure_push_active`: `0.125`
  - `Metrics/base_velocity/failure_during_sudden_stop`: `0.0`

Parameter movement from `model_0.pt` to `model_100.pt`:
  - `residual_actor` delta L2: `0.00163`
  - `critic` delta L2: `0.428`
  - `history_encoder` delta L2: `0.102`
  - `world_model_predictor` delta L2: `0.170`
  - `log_std` delta L2: `6.6e-5`

Assessment:
  Training is moving, but residual behavior is intentionally still near zero. This branch currently preserves the frozen base and has not yet demonstrated adapter recovery improvements.

## 2026-05-11 08:50 UTC - v4 step 188 status

Status:
  - session still running
  - checkpoints saved through `model_175.pt`
  - `Episode_Termination/base_contact`: `0.0029`
  - `Safety/base_contact`: `0.0024`
  - `Safety/scaled_residual_action_l2`: `8.5e-10`
  - `AdapterRollout/scaled_residual_action_l2`: `2.2e-10`
  - `AdapterRollout/ungated_scaled_residual_action_l2`: `6.1e-9`
  - `AdapterRollout/residual_action_gate_mean`: `0.133`
  - `AdapterRollout/residual_action_gate_max`: `0.716`
  - `Loss/wm`: `0.440`
  - `Loss/wm_valid_fraction`: `0.705`
  - `Train/mean_reward`: `273.5`
  - `Metrics/base_velocity/error_vel_xy`: `0.105`
  - `Metrics/base_velocity/error_vel_yaw`: `0.080`
  - `Metrics/base_velocity/failure_push_active`: `0.038`
  - `Metrics/base_velocity/failure_during_sudden_stop`: `0.0`

Assessment:
  This is much safer than the over-intervention v3 branch at the same early phase. Velocity tracking is currently good and base contact remains low. WM valid fraction is close to the `0.70` lower gate, so keep monitoring. Residual action remains near zero; the branch has not yet demonstrated a learned recovery override.

## 2026-05-11 09:07 UTC - v4 around iteration 388-398

Status:
  - session still running
  - checkpoints saved through `model_375.pt`
  - GPU memory: about `17.1 GB`
  - Latest TensorBoard sample checked: step `388`; tmux pane was at iteration `398/1200`.

Latest scalar snapshot:
  - `Episode_Termination/base_contact`: `0.0024`
  - `Safety/base_contact`: `0.0024`
  - `Safety/scaled_residual_action_l2`: `1.46e-8`
  - `AdapterRollout/scaled_residual_action_l2`: `1.51e-8`
  - `AdapterRollout/ungated_scaled_residual_action_l2`: `5.06e-8`
  - `AdapterRollout/residual_action_gate_mean`: `0.456`
  - `AdapterRollout/residual_action_gate_max`: `0.700`
  - `Loss/wm`: `0.394`
  - `Loss/wm_valid_fraction`: `0.785`
  - `Train/mean_reward`: `274.9`
  - `Metrics/base_velocity/error_vel_xy`: `0.239`
  - `Metrics/base_velocity/error_vel_yaw`: `0.300`
  - `Metrics/base_velocity/failure_push_active`: `0.313`
  - `Metrics/base_velocity/failure_during_sudden_stop`: `0.0`

Trend notes:
  - Last 50-step `base_contact` average: about `0.0020`, max about `0.0034`.
  - Last 50-step `wm_valid_fraction` average: about `0.773`, min about `0.725`.
  - Last 50-step `error_vel_xy` average: about `0.281`; `error_vel_yaw` average: about `0.289`.
  - Push-active samples are present; last 50-step average was about `0.204`.
  - `failure_during_sudden_stop` stayed `0.0`.

Parameter movement from `model_0.pt` to `model_375.pt`:
  - `residual_actor` delta L2: `0.0105`
  - `critic` delta L2: `1.55`
  - `history_encoder` delta L2: `0.366`
  - `world_model_predictor` delta L2: `0.613`

Assessment:
  This run is still safe and should continue. It is not showing v3-style residual over-intervention. The residual actor is moving, but actual residual action remains tiny, so the adapter has not yet shown meaningful recovery control. Continue to at least `model_500` or `model_600`, then run a fixed eval before deciding whether to loosen the action gate or residual scale.

## 2026-05-11 09:33 UTC - v4 reached model_600, continuing past 650

Status:
  - training session: `track_adapter_gated_recovery_v4_20260511_082805`
  - run dir: `logs/rsl_rl/run_amp_track_adapter/2026-05-11_08-28-29_track_adapter_gated_recovery_v4_20260511_082805_ppo`
  - session still running
  - checkpoints saved through at least `model_650.pt`
  - GPU memory while training: about `17.1 GB`

Latest TensorBoard snapshot checked at step `663`:
  - `Train/mean_reward`: `278.87`
  - `Episode_Termination/base_contact`: `0.00195`
  - `Safety/base_contact`: `0.00170`
  - `AdapterRollout/scaled_residual_action_l2`: `1.22e-7`
  - `AdapterRollout/ungated_scaled_residual_action_l2`: `5.22e-7`
  - `AdapterRollout/residual_action_gate_mean`: `0.381`
  - `Loss/wm`: `0.385`
  - `Loss/wm_valid_fraction`: `0.779`
  - `Metrics/base_velocity/error_vel_xy`: `0.251`
  - `Metrics/base_velocity/error_vel_yaw`: `0.297`
  - `Metrics/base_velocity/failure_during_sudden_stop`: `0.0`

Last 50-step trend:
  - `base_contact` average about `0.00159`, max about `0.00244`
  - `scaled_residual_action_l2` average about `8.18e-8`
  - `ungated_scaled_residual_action_l2` average about `4.37e-7`
  - `residual_action_gate_mean` average about `0.322`
  - `wm_valid_fraction` average about `0.774`
  - `error_vel_xy` average about `0.288`
  - `error_vel_yaw` average about `0.302`
  - `failure_during_sudden_stop` stayed `0.0`

Assessment:
  The run is still safe and healthier than the previous over-intervention branch. Base contact is comfortably below the manual gate, world-model training is valid, and velocity tracking remains usable. Residual intervention is still very small, so this checkpoint is not yet strong evidence of learned push-recovery override; fixed recovery eval is required before changing residual limits.

Fixed eval:
  - Initial eval session `fixed_eval_v4_model600_20260511_093040` failed immediately because full-training guard env vars were missing.
  - Relaunched as `fixed_eval_v4_model600_full_20260511_093307` with `BOOSTER_TRACK_ADAPTER_FULL_TRAINING=1`, `BOOSTER_TRACK_ADAPTER_SMOKE_PASSED=1`, and `BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING=1`.
  - Output dir: `logs/fixed_eval_v4_model600_full_20260511_093307`
  - Evaluating `model_600.pt` with no-push velocity grid and high-speed jerk/push recovery scenarios.

## 2026-05-11 09:41 UTC - fixed eval results for v4 model_600

No-push velocity grid:
  - `cmd=(0.0, 0.0, 0.0)`: actual `(-0.001, -0.000, -0.000)`, no fall/contact, residual `0.000000`
  - `cmd=(0.2, 0.0, 0.0)`: actual `(0.045, -0.002, -0.001)`, no fall/contact, residual `0.000061`
  - `cmd=(0.4, 0.0, 0.0)`: actual `(0.363, -0.020, -0.010)`, no fall/contact, residual `0.000090`
  - `cmd=(0.8, 0.0, 0.0)`: actual `(0.748, -0.024, 0.009)`, no fall/contact, residual `0.000123`
  - `cmd=(1.0, 0.0, 0.0)`: actual `(0.924, -0.015, 0.033)`, no fall/contact, residual `0.000128`
  - `cmd=(2.0, 0.0, 0.0)`: actual `(1.858, -0.034, -0.034)`, no fall/contact, residual `0.000246`
  - `cmd=(0.0, 1.0, 0.0)`: actual `(-0.004, 0.906, 0.032)`, no fall/contact, residual `0.000222`
  - `cmd=(1.0, 1.0, 0.0)`: actual `(0.886, 0.897, 0.102)`, instant fall/contact `0.0001`, ever-fall/contact `0.0158`, residual `0.000144`
  - `cmd=(0.0, 0.0, 0.3)`: actual `(0.034, -0.008, 0.309)`, no fall/contact, residual `0.000085`

High-speed jerk/recovery eval:
  - `vx2_vy0_yaw0_to_0_no_push`: fall/contact `0.0`, settle success `1.0`, final stop speed mean `0.0296`, p95 `0.0493`
  - `vx2_vy0_yaw0_to_0_push`: fall/contact `0.0`, settle success `1.0`, final stop speed mean `0.0314`, p95 `0.0530`
  - `vx1_vy0_yaw0_to_0_no_push`: fall/contact `0.0`, settle success `1.0`, final stop speed mean `0.0355`, p95 `0.0779`
  - `vx1_vy0_yaw0_to_0_push`: fall/contact `0.0`, settle success `1.0`, final stop speed mean `0.0318`, p95 `0.0623`
  - `vx0_vy1_yaw0_to_0_no_push`: fall/contact `0.0`, settle success `1.0`, final stop speed mean `0.0333`, p95 `0.0634`
  - `vx0_vy1_yaw0_to_0_push`: fall/contact `0.0`, settle success `1.0`, final stop speed mean `0.0332`, p95 `0.0621`
  - `vx1_vy1_yaw0_to_0_no_push`: fall/contact `0.0`, settle success `1.0`, final stop speed mean `0.0352`, p95 `0.0592`
  - `vx1_vy1_yaw0_to_0_push`: fall/contact `0.0`, settle success `1.0`, final stop speed mean `0.0367`, p95 `0.0616`

Assessment:
  Fixed eval confirms that `model_600.pt` stops cleanly from the high-speed jerk cases, including explicit push at the stop transition. The zero-command no-push case stays essentially stationary. Low-speed forward tracking remains weak at `vx=0.2`, and diagonal `vx=1.0, vy=1.0` has a small ever-fall/contact rate. Residual norms are effectively zero in fixed eval, so the current robustness appears to come mostly from the frozen base policy plus the command/recovery environment, not from a strong learned residual override yet.

Training status after eval:
  - training session still running
  - checkpoints saved through at least `model_725.pt`
  - latest checked step `731`: reward `277.19`, base contact `0.00098`, WM loss `0.378`, WM valid fraction `0.777`, xy/yaw errors `0.338/0.370`

## 2026-05-11 10:29 UTC - v4 full run completed at model_1199

Status:
  - training tmux session `track_adapter_gated_recovery_v4_20260511_082805` is no longer present
  - no training Python process is running
  - final checkpoint saved: `logs/rsl_rl/run_amp_track_adapter/2026-05-11_08-28-29_track_adapter_gated_recovery_v4_20260511_082805_ppo/model_1199.pt`
  - the log reached `Learning iteration 1199/1200`, so this is a normal max-iteration completion, not a safety-gate stop or crash

Final scalar snapshot at step `1199`:
  - `Train/mean_reward`: `274.02`
  - `Episode_Termination/base_contact`: `0.00293`
  - `Safety/base_contact`: `0.00283`
  - `AdapterRollout/scaled_residual_action_l2`: `7.50e-6`
  - `AdapterRollout/ungated_scaled_residual_action_l2`: `3.07e-5`
  - `AdapterRollout/residual_action_gate_mean`: `0.398`
  - `Loss/wm`: `0.351`
  - `Loss/wm_valid_fraction`: `0.779`
  - `Metrics/base_velocity/error_vel_xy`: `0.243`
  - `Metrics/base_velocity/error_vel_yaw`: `0.305`
  - `Metrics/base_velocity/failure_during_sudden_stop`: `0.0`
  - `Metrics/base_velocity/failure_push_active`: `0.417`

Last 50-step trend:
  - `base_contact` average about `0.00240`, max about `0.00391`
  - `scaled_residual_action_l2` average about `4.69e-6`
  - `ungated_scaled_residual_action_l2` average about `2.53e-5`
  - `wm_valid_fraction` average about `0.769`
  - `error_vel_xy` average about `0.279`
  - `error_vel_yaw` average about `0.297`
  - `failure_during_sudden_stop` stayed `0.0`

Assessment:
  The full v4 run completed safely. It did not reproduce the v3 residual blow-up; residual action is still tiny at the end. The final checkpoint now needs the same fixed velocity/recovery evaluation as `model_600.pt` to see whether later training improved low-speed tracking or push recovery.

Final eval:
  - started tmux session `fixed_eval_v4_model1199_20260511_103006`
  - output dir: `logs/fixed_eval_v4_model1199_20260511_103006`
  - checkpoint: `model_1199.pt`
  - running the same no-push velocity grid and high-speed jerk/push recovery scenarios used for `model_600.pt`

## 2026-05-11 10:34 UTC - fixed eval results for v4 model_1199

No-push velocity grid:
  - `cmd=(0.0, 0.0, 0.0)`: actual `(-0.000, -0.000, -0.001)`, no fall/contact, residual `0.000000`
  - `cmd=(0.2, 0.0, 0.0)`: actual `(0.069, -0.002, 0.014)`, no fall/contact, residual `0.000608`
  - `cmd=(0.4, 0.0, 0.0)`: actual `(0.417, -0.019, -0.015)`, no fall/contact, residual `0.000938`
  - `cmd=(0.8, 0.0, 0.0)`: actual `(0.728, -0.024, 0.019)`, no fall/contact, residual `0.001178`
  - `cmd=(1.0, 0.0, 0.0)`: actual `(0.910, -0.018, 0.026)`, no fall/contact, residual `0.001286`
  - `cmd=(2.0, 0.0, 0.0)`: actual `(1.898, -0.025, -0.019)`, no fall/contact, residual `0.002371`
  - `cmd=(0.0, 1.0, 0.0)`: actual `(-0.001, 0.917, 0.034)`, no fall/contact, residual `0.002313`
  - `cmd=(1.0, 1.0, 0.0)`: actual `(0.876, 0.864, 0.065)`, instant fall/contact `0.0001`, ever-fall/contact `0.0205`, residual `0.001494`
  - `cmd=(0.0, 0.0, 0.3)`: actual `(0.034, -0.006, 0.293)`, no fall/contact, residual `0.000803`

High-speed jerk/recovery eval:
  - `vx2_vy0_yaw0_to_0_no_push`: fall/contact `0.0`, settle success `1.0`, final stop speed mean `0.0340`, p95 `0.0618`
  - `vx2_vy0_yaw0_to_0_push`: fall/contact `0.0`, settle success `1.0`, final stop speed mean `0.0345`, p95 `0.0609`
  - `vx1_vy0_yaw0_to_0_no_push`: fall/contact `0.0`, settle success `1.0`, final stop speed mean `0.0340`, p95 `0.0596`
  - `vx1_vy0_yaw0_to_0_push`: fall/contact `0.0`, settle success `1.0`, final stop speed mean `0.0336`, p95 `0.0585`
  - `vx0_vy1_yaw0_to_0_no_push`: fall/contact `0.0`, settle success `1.0`, final stop speed mean `0.0327`, p95 `0.0610`
  - `vx0_vy1_yaw0_to_0_push`: fall/contact `0.0`, settle success `1.0`, final stop speed mean `0.0338`, p95 `0.0579`
  - `vx1_vy1_yaw0_to_0_no_push`: fall/contact `0.0078`, pre-stop failure `0.0078`, settle success `0.9922`, final stop speed mean `0.0335`, p95 `0.0617`
  - `vx1_vy1_yaw0_to_0_push`: fall/contact `0.0078`, pre-stop failure `0.0078`, settle success `0.9922`, final stop speed mean `0.0350`, p95 `0.0612`

Assessment:
  The final checkpoint remains safe for zero-command, straight high-speed stop, lateral stop, and yaw tracking. Compared with `model_600.pt`, low-speed forward tracking improved slightly (`vx=0.2` actual from about `0.045` to `0.069`) and `vx=2.0` tracking improved slightly, but diagonal `vx=1.0, vy=1.0` is worse: grid ever-fall/contact rose from about `0.0158` to `0.0205`, and jerk eval now has `1/128` pre-stop/base-contact failures. Residual is still tiny, so the final checkpoint still does not demonstrate a strong learned residual recovery policy.

## 2026-05-11 10:55 UTC - v5 push/diagonal branch implementation

Implemented v5 changes:
  - Added a diagonal command stratum in `OmniVelocityCommand` with metrics `diagonal_sampled` and `diagonal_command`.
  - Added optional horizontal command norm cap `lin_vel_xy_norm_max` to avoid over-hard rectangular corner samples while preserving `vx=2.0,vy=0` and `vx=1.0,vy=1.0`.
  - Added `failure_mining_min_command_norm` so zero/near-zero command failures do not seed command-failure mining bins.
  - Added Track Adapter env knobs for diagonal sampling, XY norm cap, failure-mining min command norm, and low-speed tracking reward.
  - Added `low_speed_track_lin_vel_xy_exp`, a low-speed-only XY tracking reward that excludes true zero command.
  - Added manual stop recovery detection: zero command plus non-trivial actual speed now opens stop/recovery activity even when the command was set manually instead of sampled by the sudden-stop timer.
  - Added rollout diagnostics split by stable, push-active, sudden-stop, and active-recovery masks.
  - Added an optional recovery-window residual activation loss, controlled by env knobs, to break the all-zero residual solution only when the residual action gate is high.
  - Added v5 runner script `scripts/rsl_rl/run_track_adapter_pushdiag_v5.sh`, resuming from v4 `model_600.pt` without optimizer state.

Verification:
  - `py_compile` passed for changed Python files.
  - `bash -n scripts/rsl_rl/run_track_adapter_pushdiag_v5.sh` passed.
  - `pytest -q tests/test_track_adapter_module.py tests/test_track_adapter_storage.py tests/test_track_adapter_optimizer.py` passed: `42 passed`.

v5 intent:
  Resume from the safer v4 `model_600.pt`, increase diagonal and low-speed coverage, open residual authority primarily during push/stop recovery, and keep stable/zero-command residual suppression tighter than v4.

v5 smoke started:
  - tmux session: `track_adapter_pushdiag_v5_resume600_20260511_105549`
  - command: `PPO_ITERATIONS=250 bash scripts/rsl_rl/run_track_adapter_pushdiag_v5.sh track_adapter_pushdiag_v5_resume600_20260511_105549`
  - resume checkpoint: v4 `model_600.pt`
  - optimizer state intentionally not loaded

Initial v5 status:
  - Isaac startup succeeded.
  - v4 `model_600.pt` loaded successfully.
  - optimizer state was intentionally skipped.
  - resumed action std was overridden to `0.14`.
  - Reward manager now has 30 terms including `low_speed_track_lin_vel_xy_exp`.
  - First observed iterations around `610-612/850`: `base_contact` about `0.0029`, `wm_valid_fraction` about `0.77-0.79`, `error_vel_xy` about `0.234`, `error_vel_yaw` about `0.368`.
  - Low-speed/diagonal sampled metrics are still `0.0` in the first logged window, likely because command resampling has not yet occurred after resume; monitor later windows before judging the new command strata.

v5 status around step `649`:
  - session still running
  - checkpoints saved through `model_625.pt`
  - `Train/mean_reward`: `313.09`
  - `Episode_Termination/base_contact`: `0.00195`; last 20 average about `0.00289`
  - `AdapterRollout/scaled_residual_action_l2`: `4.87e-6`
  - `AdapterRollout/ungated_scaled_residual_action_l2`: `9.80e-6`
  - `AdapterRollout/scaled_residual_action_l2_push_active`: `8.79e-6`
  - `AdapterRollout/residual_action_gate_mean`: `0.568`
  - `AdapterRollout/residual_action_gate_push_active`: `0.95`
  - `Loss/recovery_residual_activation`: `5.7e-7`
  - `Loss/wm`: `0.414`
  - `Loss/wm_valid_fraction`: `0.779`
  - `Metrics/base_velocity/error_vel_xy`: `0.294`
  - `Metrics/base_velocity/error_vel_yaw`: `0.292`
  - `Metrics/base_velocity/low_speed_sampled`: `1.0`
  - `Metrics/base_velocity/diagonal_sampled`: `1.0`
  - `Metrics/base_velocity/diagonal_command`: `0.0` in the last row but last 20 average about `0.565`
  - `Metrics/base_velocity/failure_push_active`: `1.0`
  - `Metrics/base_velocity/failure_during_sudden_stop`: `0.0`

Assessment:
  v5 command strata and push-active windows are now present. Safety remains good, WM replay is valid, and the new recovery gate opens to the intended `0.95` during push-active windows. Residual is still tiny but larger than v4 at the same checkpoint scale; continue the bounded 250-iteration v5 run and fixed-evaluate after completion.

## 2026-05-11 11:13 UTC - v5 stopped for targeted v5b fixes

Stopped tmux session `track_adapter_pushdiag_v5_resume600_20260511_105549` intentionally after it had saved through `model_725.pt`.

Last captured v5 status around iteration `733/850`:
  - `Train/mean_reward`: about `309-311`
  - `Episode_Termination/base_contact`: about `0.0029-0.0031`
  - `Mean wm loss`: about `0.418-0.425`
  - `Mean wm_valid_fraction`: about `0.776-0.778`
  - `Mean adapter_residual_action_l2`: about `0.083-0.093`
  - `Mean adapter_scaled_residual_action_l2`: about `0.0003`
  - `Metrics/base_velocity/failure_push_active`: non-zero and often high
  - `Metrics/base_velocity/failure_during_sudden_stop`: `0.0`

Reason for stopping:
  v5 was safe, but review found implementation details that can bias the training signal before it is worth extending:
  - push-active failure windows were being cleared on ordinary command resampling, not only on true env reset;
  - push failure-mining could record near-zero-command failures, which is not aligned with the push-recovery objective;
  - low-speed, diagonal, and yaw-only sampled flags were not strictly exclusive after command overrides;
  - the low-speed reward used total XY command norm, so it could miss intended low-speed forward samples near `vx=0.45` with small lateral noise;
  - resume-time `residual_output_init_scale` does not affect an already-loaded v4 checkpoint, so a tiny explicit output-layer perturbation is needed if we want to break the exact zero-residual solution without loosening stable-state gates.

v5b code changes implemented:
  - preserve push-active state across ordinary command resamples and clear it only for reset/terminated/episode-length-zero envs;
  - add `failure_mining_push_min_command_norm` and `BOOSTER_TRACK_ADAPTER_FAILURE_PUSH_MIN_COMMAND_NORM`;
  - store the command bin and scaled command norm at push time, so push failures are attributed to the command active when the disturbance was applied instead of a later resampled command;
  - compute failure-mining command norm as `sqrt(vx^2 + vy^2 + (yaw_scale*wz)^2)` through `BOOSTER_TRACK_ADAPTER_FAILURE_COMMAND_YAW_SCALE`;
  - make low-speed/diagonal/yaw-only sample metrics mutually exclusive when later command strata override earlier samples;
  - add `yaw_only_sampled` command metric for command-stratum observability;
  - change `low_speed_track_lin_vel_xy_exp` to target low-speed straight commands by `abs(vx)` plus small `abs(vy)`/yaw thresholds;
  - add `BOOSTER_TRACK_ADAPTER_LOW_SPEED_TRACK_MAX_LATERAL`;
  - add opt-in resume perturbation for the residual actor output layer through `BOOSTER_TRACK_ADAPTER_RESIDUAL_RESUME_OUTPUT_NOISE_STD` and `BOOSTER_TRACK_ADAPTER_RESIDUAL_RESUME_BIAS_NOISE_STD`;
  - skip optimizer-state loading when resume perturbation is active, avoiding stale optimizer moments on the perturbed output layer;
  - set the v5 script defaults to small resume perturbation (`0.001`) and a slightly stronger recovery activation target (`weight=0.005`, `target_norm=0.03`).

Verification:
  - `py_compile` passed for updated Track Adapter command/reward/env/runner files.
  - `bash -n scripts/rsl_rl/run_track_adapter_pushdiag_v5.sh` passed.
  - `PYTHONPATH=/workspace/booster_amp_lab/rsl_rl pytest -q tests/test_track_adapter_module.py tests/test_track_adapter_storage.py tests/test_track_adapter_optimizer.py` passed: `45 passed`.

Next step:
  Start a fresh bounded v5b resume from v4 `model_600.pt` rather than continuing the flawed v5 branch.

## 2026-05-11 11:23 UTC - v5b first launch stopped for metric cleanup

Started tmux session `track_adapter_pushdiag_v5b_resume600_20260511_111741` from v4 `model_600.pt`.

Confirmed startup:
  - Stage 3 world-model pretrain loaded.
  - v4 `model_600.pt` loaded.
  - action std overridden to `0.14`.
  - residual output perturb applied with `weight_std=0.001`, `bias_std=0.001`.

Initial metrics around iterations `604-617/850`:
  - `Episode_Termination/base_contact`: `0.0005` to `0.0054`
  - `Mean wm loss`: about `0.417-0.435`
  - `Mean wm_valid_fraction`: about `0.776-0.779`
  - `failure_push_active`: present
  - `failure_during_sudden_stop`: `0.0`
  - `low_speed_sampled` became present after command resampling

Stopped this first v5b session intentionally because sudden-stop command override could still leave earlier `failure_mining_sampled`/low/diagonal/yaw flags set in the same resample. This was primarily a logging/diagnostic consistency problem, but it can mislead gate decisions.

Fix added:
  - when sudden-stop overrides the sampled command to zero, clear low-speed, diagonal, yaw-only, and failure-mining sampled flags for that env.

Verification after the fix:
  - `py_compile` passed for `commands.py`.
  - focused pytest passed: `46 passed`.

## 2026-05-11 11:28 UTC - v5b2 clean run started

Started tmux session `track_adapter_pushdiag_v5b2_resume600_20260511_112114`.

Run details:
  - run dir: `logs/rsl_rl/run_amp_track_adapter/2026-05-11_11-21-18_track_adapter_pushdiag_v5b2_resume600_20260511_112114_ppo`
  - resume checkpoint: v4 `model_600.pt`
  - optimizer state: not loaded
  - action std override: `0.14`
  - residual output perturb: `weight_std=0.001`, `bias_std=0.001`
  - checkpoint saved: `model_625.pt`

Early status around iterations `625-627/850`:
  - `Train/mean_reward`: about `317-320`
  - `Episode_Termination/base_contact`: `0.0041-0.0046`
  - `Mean wm loss`: about `0.405-0.413`
  - `Mean wm_valid_fraction`: `0.774-0.783`
  - `Mean adapter_scaled_residual_action_l2`: still effectively `0.0`
  - `failure_push_active`: present
  - `failure_during_sudden_stop`: `0.0`
  - `diagonal_sampled` and `diagonal_command`: present, about `0.50-0.54` in observed windows
  - `sudden_stop_sampled`: present, including a full sudden-stop window with low/diagonal/yaw/failure-mining sampled flags cleared as intended

Assessment:
  v5b2 startup is clean. The run is safe so far, the replay world model remains healthy, push-active windows are present, and command-stratum metrics now match the final command after overrides. Residual action remains extremely small at this early point, so the main remaining thing to watch is whether the perturb/recovery activation lets residual usage emerge specifically in push/stop windows without growing in stable walking.

v5b2 status around iteration `653/850`:
  - session still running
  - latest saved checkpoint: `model_625.pt`
  - `Train/mean_reward`: about `306-309`
  - `Episode_Termination/base_contact`: about `0.0031-0.0038`
  - `Mean wm loss`: about `0.426-0.433`
  - `Mean wm_valid_fraction`: about `0.778-0.780`
  - `Mean adapter_residual_action_l2`: about `0.0015-0.0016`
  - `Mean adapter_scaled_residual_action_l2`: still effectively `0.0`
  - `low_speed_sampled`, `diagonal_sampled`, and push-active windows are present
  - `failure_during_sudden_stop`: `0.0`

Assessment:
  The run remains safe and diagnostically clean. Residual usage is increasing only from near-zero to tiny values so far; do not claim improved push recovery until fixed evaluation or a later checkpoint shows behavior changes under push.

v5b2 status around iteration `679/850`:
  - session still running
  - GPU memory: about `17.1GB / 32.6GB`
  - `Train/mean_reward`: about `307.7-308.5`
  - `Episode_Termination/base_contact`: about `0.0024`
  - `Mean wm loss`: about `0.404-0.412`
  - `Mean wm_valid_fraction`: about `0.775-0.783`
  - `Mean adapter_residual_action_l2`: about `0.0036-0.0039`
  - `Mean adapter_scaled_residual_action_l2`: still printed as `0.0000`
  - diagonal/yaw-only/push-active windows are present
  - `failure_during_sudden_stop`: `0.0`

Assessment:
  Continue. Contact and WM gates remain healthy. Velocity error is higher in diagonal-heavy windows, so fixed diagonal evaluation will be decisive. Residual action is growing slowly from the tiny perturbation but has not yet become a strong intervention.

v5b2 status around iteration `714/850`:
  - session still running
  - checkpoints saved through `model_700.pt`
  - `Train/mean_reward`: about `305.5-308.6`
  - `Episode_Termination/base_contact`: about `0.0049-0.0053`
  - `Mean wm loss`: about `0.425-0.438`
  - `Mean wm_valid_fraction`: about `0.774-0.778`
  - `Mean adapter_residual_action_l2`: about `0.0147-0.0164`
  - `Mean adapter_scaled_residual_action_l2`: about `0.0000-0.0001`
  - diagonal/yaw-only/push-active windows are present
  - `failure_during_sudden_stop`: `0.0`

Assessment:
  Continue. Residual is finally moving off the exact-zero solution, but it is still behaviorally tiny. Contact is still far below the safety stop threshold, and no sudden-stop failure is showing.

v5b2 completed:
  - tmux session `track_adapter_pushdiag_v5b2_resume600_20260511_112114` exited normally.
  - final checkpoint: `logs/rsl_rl/run_amp_track_adapter/2026-05-11_11-21-18_track_adapter_pushdiag_v5b2_resume600_20260511_112114_ppo/model_849.pt`
  - no safety-gate failure, traceback, or aborted checkpoint was found in the log.
  - final iteration `849/850`:
    - `Train/mean_reward`: `319.91`
    - `Episode_Termination/base_contact`: `0.0046`
    - `Mean wm loss`: `0.4165`
    - `Mean wm_valid_fraction`: `0.7761`
    - `Mean adapter_residual_action_l2`: `2.5161`
    - `Mean adapter_scaled_residual_action_l2`: `0.0108`
    - `Mean adapter_ungated_scaled_residual_action_l2`: `0.0161`
    - `Metrics/base_velocity/error_vel_xy`: `0.4281`
    - `Metrics/base_velocity/error_vel_yaw`: `0.3971`
    - `failure_during_sudden_stop`: `0.0`

Assessment:
  Training completed safely and residual usage emerged without tripping the safety gates. This is still not proof of better push recovery; proceed to fixed evaluation against v4 `model_600.pt`, v5b2 `model_625.pt`, and v5b2 final `model_849.pt`.

## 2026-05-11 11:49 UTC - fixed evaluation started

Added `scripts/rsl_rl/eval_track_adapter_v5b2_fixed.sh` to run the existing fixed evaluators consistently across:
  - v4 `model_600.pt`
  - v5b2 `model_625.pt`
  - v5b2 final `model_849.pt`

Evaluation session:
  - tmux: `track_adapter_v5b2_fixed_eval_20260511_114934`
  - outputs use timestamp `20260511_114934`
  - first output dir: `logs/fixed_eval_v4_model600_20260511_114934`

Evaluation contents:
  - no-push velocity grid for zero, low-speed, straight, lateral, diagonal, and yaw commands;
  - high-speed jerk suite for `vx=2`, `vx=1`, `vy=1`, and `vx=1,vy=1`, both no-push and explicit stop-time push.

Fixed evaluation results:
  - v4 `model_600.pt`
    - velocity grid max ever fall/base-contact: `0.0055`, only diagonal `vx=1,vy=1`;
    - high-speed jerk max failure/contact/pre-stop: `0.0078`;
    - zero command actual velocity: about `(-0.001, -0.000, 0.002)`;
    - zero command residual: effectively `0.0`.
  - v5b2 `model_625.pt`
    - velocity grid max ever fall/base-contact: `0.0060`, diagonal `vx=1,vy=1`;
    - high-speed jerk max failure/contact/pre-stop: `0.0234`, diagonal no-push;
    - zero command actual velocity: about `(-0.001, -0.000, 0.003)`;
    - zero command residual: effectively `0.0`.
  - v5b2 final `model_849.pt`
    - velocity grid max ever fall/base-contact: `0.0066`, diagonal `vx=1,vy=1`;
    - high-speed jerk max failure/contact/pre-stop: `0.0156`, diagonal push;
    - zero command actual velocity: about `(-0.000, -0.000, 0.003)`;
    - zero command residual: about `0.0005`;
    - residual is active in normal no-push grid, e.g. about `0.033` at `vx=2` and `0.0325` at `vy=1`.

Assessment:
  v5b2 proved that residual usage can emerge without training-time safety failure, but neither `model_625.pt` nor `model_849.pt` beats v4 `model_600.pt` on the promotion gates. The final checkpoint is still quiet at zero command, but it introduces more high-speed jerk failures than the v4 baseline. Do not promote v5b2 final yet.

Next step:
  Run high-speed jerk-only sweep on intermediate v5b2 checkpoints (`model_700`, `750`, `800`, `825`) to see whether there is a better residual sweet spot before final over-adaptation.

## 2026-05-11 12:14 UTC - v5b2 intermediate jerk sweep completed

Added and ran `scripts/rsl_rl/eval_track_adapter_v5b2_jerk_sweep.sh`.

Evaluation session:
  - tmux: `track_adapter_v5b2_jerk_sweep_20260511_120336`
  - outputs:
    - `logs/fixed_eval_v5b2_model700_jerk_20260511_120336`
    - `logs/fixed_eval_v5b2_model750_jerk_20260511_120336`
    - `logs/fixed_eval_v5b2_model800_jerk_20260511_120336`
    - `logs/fixed_eval_v5b2_model825_jerk_20260511_120336`

High-speed jerk max failure summary:
  - v4 `model_600`: max `0.0078`; failures in `vx1 push` and `vx1_vy1 no_push`.
  - v5b2 `model_625`: max `0.0234`; worsened diagonal no-push.
  - v5b2 `model_700`: max `0.0078`; failures in `vx2 push`, `vx1_vy1 no_push`, `vx1_vy1 push`.
  - v5b2 `model_750`: max `0.0234`; worsened `vx1_vy1 no_push`.
  - v5b2 `model_800`: max `0.0234`; worsened `vx1_vy1 no_push` and `vx2 no_push`.
  - v5b2 `model_825`: max `0.0078`; failures in `vx2 push`, `vy1 no_push`, `vx1_vy1 no_push`.
  - v5b2 `model_849`: max `0.0156`; worsened `vx1_vy1 push`.

Assessment:
  No v5b2 checkpoint beats v4 `model_600` on the jerk promotion gate. `model700` and `model825` match the v4 maximum failure rate, but they spread failures across more scenarios. Do not promote v5b2. Keep v4 `model_600.pt` as the current safest deployment candidate.

Technical interpretation:
  v5b2 successfully moved residuals off the exact-zero solution during PPO, but the residual did not translate into better fixed push/jerk recovery. The final checkpoint also carries measurable residual in stable no-push velocity grid while not improving the failure gates, so additional residual freedom alone is not the right next step.

Next branch direction:
  Shift from broad residual activation to a more targeted recovery objective: emphasize explicit post-push survival/return-to-command samples, reduce diagonal over-sampling pressure during early residual emergence, and gate residual learning by push/stop windows rather than letting stable no-push residual grow.

## 2026-05-11 - v6 objective reset

User objective was narrowed to two primary goals:
  - external-force push recovery without falling/base contact;
  - better command tracking at `vx = 0.2, 0.5, 1.0, 1.5, 2.0` and `vy = 0.2, 0.4, 1.0`, with special attention to `vx`-only low-speed straight walking.

Current evidence:
  - v4 `model_600.pt` remains the safest adapter checkpoint.
  - v5b2 learned non-zero residual but did not improve fixed push/jerk gates.
  - fixed velocity grid already shows severe `vx=0.2` under-tracking: v4 mean `actual_vx` about `0.060` for command `0.2`; v5b2 final about `0.057`.
  - existing grid did not yet include all user-critical points (`vx=0.5/1.5`, `vy=0.2/0.4`), and it needs explicit lateral displacement/yaw-drift metrics to judge straightness.

Decision:
  Do not continue broad residual-growth training. First add/perform frozen-base-only and adapter fixed evaluations on the user-critical command grid. If the frozen AMP base is already biased or asymmetric at low `vx`, velocity tracking should be improved in the AMP base/fine-tune stage or the adapter must be deliberately allowed to become a stronger residual locomotion controller.

Planned v6 strategy:
  - anchor on v4 `model_600.pt`;
  - evaluate frozen base, v4, and any v6 candidate on the same no-push and push grid;
  - oversample exact straight-command bins during training;
  - keep zero-command no-push quietness;
  - open residual authority mainly during external-push, command-jerk, or high velocity-error recovery windows;
  - promote only if push/jerk failures decrease and straight low-speed tracking improves without stable no-push residual leakage.

## 2026-05-11 16:14 UTC - frozen AMP base fixed evaluation started

Added straightness diagnostics to `scripts/rsl_rl/eval_velocity_command_grid.py`:
  - local horizontal displacement from the post-settle reference pose;
  - absolute lateral displacement;
  - yaw drift and absolute yaw drift;
  - final/max summary fields for drift metrics.

Added `scripts/rsl_rl/eval_amp_base_v6_grid.sh` for the frozen AMP base checkpoint:
  - task: `Booster-Run-AMP-v0`;
  - checkpoint: last model in `logs/rsl_rl/run_amp_y/2026-05-04_19-10-18_resume56_yawlow_zero_stand`;
  - commands: `vx = 0.0, 0.2, 0.5, 1.0, 1.5, 2.0` with `vy=0`, plus `vy = 0.2, 0.4, 1.0` with `vx=0`;
  - evaluations: no-push velocity grid, interval-push velocity grid, and high-speed jerk stop/push suite.

Verification before launch:
  - `python -m py_compile scripts/rsl_rl/eval_velocity_command_grid.py scripts/rsl_rl/eval_track_adapter_recovery.py`;
  - `bash -n scripts/rsl_rl/eval_amp_base_v6_grid.sh`.

Completed output:
  - `logs/fixed_eval_amp_base_v6_grid_20260511_161417/velocity_no_push.json`
  - `logs/fixed_eval_amp_base_v6_grid_20260511_161417/velocity_interval_push.json`
  - `logs/fixed_eval_amp_base_v6_grid_20260511_161417/high_speed_jerk.json`

No-push base tracking summary:
  - `vx=0.2`: mean `actual_vx=0.060`, `abs_vx_error=0.163`, max mean absolute lateral displacement `0.085 m`, max mean absolute yaw drift `0.204 rad`.
  - `vx=0.5`: mean `actual_vx=0.502`, `abs_vx_error=0.068`, max mean absolute lateral displacement `0.643 m`, max mean absolute yaw drift `0.464 rad`.
  - `vx=1.0`: mean `actual_vx=0.910`, `abs_vx_error=0.118`, max mean absolute lateral displacement `1.113 m`, max mean absolute yaw drift `0.430 rad`.
  - `vx=1.5`: mean `actual_vx=1.355`, `abs_vx_error=0.155`, max mean absolute lateral displacement `1.429 m`, ever fall/base contact about `0.0066`.
  - `vx=2.0`: mean `actual_vx=1.975`, `abs_vx_error=0.093`, max mean absolute lateral displacement `2.549 m`, ever fall/base contact about `0.0026`.
  - `vy=0.2`: mean `actual_vy=0.161`, `abs_vy_error=0.112`, max mean absolute lateral displacement `0.874 m`, max mean absolute yaw drift `0.472 rad`.
  - `vy=0.4`: mean `actual_vy=0.315`, `abs_vy_error=0.140`, max mean absolute lateral displacement `1.636 m`, max mean absolute yaw drift `0.536 rad`.
  - `vy=1.0`: mean `actual_vy=0.940`, `abs_vy_error=0.210`, max mean absolute lateral displacement `4.807 m`, max mean absolute yaw drift `0.631 rad`.

Interval-push base tracking summary:
  - The interval-push grid did not create failures for most commands, but `vx=2.0` degraded to ever fall about `0.052` and ever base contact about `0.040`.
  - Tracking errors increased broadly under interval pushes, e.g. `vx=2.0` had `abs_vx_error=0.210`, `abs_vy_error=0.184`, `abs_wz_error=0.377`.

High-speed jerk stop/push summary:
  - Fixed `vx=2 -> 0`, `vx=1 -> 0`, `vy=1 -> 0`, and `vx=1,vy=1 -> 0` all passed with fall/base/pre-stop failure `0.0` in both no-push and stop-time-push modes.
  - Stop settling stayed valid for all envs; final stop speed p95 was about `0.064-0.084`.

Interpretation:
  The frozen AMP base is not failing the simple abrupt-stop gate under this fixed condition, but it has clear tracking/straightness limitations. `vx=0.2` is effectively under-commanded, and `vx=0.5+` can show substantial lateral/yaw drift even when mean forward velocity looks acceptable. This supports treating low-speed straight walking as a base-policy coverage issue or a deliberate stronger-adapter objective, not just a generic push-recovery residual issue.

## 2026-05-11 16:29 UTC - v4 anchor fixed evaluation started

Added `scripts/rsl_rl/eval_track_adapter_v6_grid.sh` to run the same v6 command grid and drift diagnostics on a Track Adapter checkpoint.

Launch target:
  - checkpoint: `logs/rsl_rl/run_amp_track_adapter/2026-05-11_08-28-29_track_adapter_gated_recovery_v4_20260511_082805_ppo/model_600.pt`;
  - task: `Booster-Run-AMP-TrackAdapter-v0`;
  - commands: same v6 grid as frozen-base evaluation;
  - outputs: no-push grid, interval-push grid, high-speed jerk stop/push suite.

Verification before launch:
  - `bash -n scripts/rsl_rl/eval_track_adapter_v6_grid.sh`.

Completed output:
  - `logs/fixed_eval_v4_model600_v6_grid_20260511_162918/velocity_no_push.json`
  - `logs/fixed_eval_v4_model600_v6_grid_20260511_162918/velocity_interval_push.json`
  - `logs/fixed_eval_v4_model600_v6_grid_20260511_162918/high_speed_jerk.json`

No-push comparison against frozen AMP base:
  - `vx=0.2`: base `actual_vx=0.060`, v4 `actual_vx=0.053`; v4 does not fix the low-speed under-tracking.
  - `vx=0.5`: base `actual_vx=0.502`, v4 `actual_vx=0.486`; both have substantial lateral/yaw drift, though v4 slightly reduces the drift metrics.
  - `vx=1.0`: base `actual_vx=0.910`, v4 `actual_vx=0.910`; v4 is essentially unchanged.
  - `vx=1.5`: base `actual_vx=1.355`, v4 `actual_vx=1.302`; v4 removes the small no-push ever fall/base contact seen in base, but slows the command more.
  - `vx=2.0`: base `actual_vx=1.975`, v4 `actual_vx=1.856`; v4 removes the small no-push ever fall/base contact seen in base, but loses forward tracking.
  - `vy=0.2/0.4/1.0`: v4 is very close to base and does not materially improve lateral command tracking.
  - v4 residual norms in no-push grid are about `0.0000-0.0003`; this checkpoint is effectively base-only in stable commands.

Interval-push comparison:
  - `vx=2.0`: base max ever fall/base-contact about `0.052`, v4 about `0.045`; v4 is slightly safer but not a clear recovery improvement.
  - `vx=1.0`: base `0.0`, v4 about `0.0073`; v4 regresses this interval-push command.
  - velocity errors under interval push remain almost identical to base; v4 residual norms are about `0.0005-0.0007`.

High-speed jerk stop/push comparison:
  - frozen base passed all fixed jerk stop/push cases with max failure `0.0`.
  - v4 max failure was `0.0078`, with failures in `vx=1.0` push and `vx=1.0,vy=1.0` no-push.
  - v4 final stop speed p95 is slightly lower than base, but that does not compensate for the new rare failures.

Interpretation:
  v4 `model_600.pt` is still the safest Track Adapter anchor among trained adapter checkpoints because it avoids broad residual leakage, but it is not solving the user's two core goals. It mostly preserves the frozen AMP behavior, slightly damps some high-speed no-push failures, and leaves low-speed straight tracking as a base-policy problem. The next branch should either fine-tune the AMP base for exact low-speed straight commands or intentionally train a stronger adapter with residual authority opened for command-tracking error, not only push/stop recovery.

## 2026-05-11 - optimization priority reset

User priority update:
  - High-speed jerk is not a primary benchmark. It may be kept only as an OOD/smoke check for not falling under bad command transitions.
  - Primary benchmark is single-axis velocity tracking, ordered by importance: `vx-only`, then `vy-only`, then `vyaw-only`.
  - `vx-only` must not drift in `y` or yaw.
  - `vy-only` must not drift in `x` or yaw.
  - Secondary benchmark is `vx + vyaw` and `vy + vyaw`; these should be made reliable after the single-axis commands.
  - Other random mixed command combinations are lower priority and should not dominate training or checkpoint promotion.

Optimization implication:
  Train/fine-tune the AMP base first with exact command bins and crosstalk penalties. Avoid making high-speed jerk robustness a reward objective if it trades off against tracking. Keep OOD external pushes in training/eval, but measure success as "does not fall and returns to the commanded single-axis motion", not as abrupt-stop perfection.

## 2026-05-11 17:38 UTC - AMP axis fine-tune implementation

Implemented the base AMP fine-tune plan:
  - `OmniVelocityCommand` now supports `rel_hard_envs` and `hard_command_points`, with `hard_command_sampled` metrics.
  - `RoughWoStateEstimationEnvCfg` has an opt-in `BOOSTER_AMP_FT_ENABLE=1` mode.
  - Fine-tune command distribution emphasizes exact zero command, `vx-only`, `vy-only`, and `vyaw-only` points, with `|vyaw|` up to `1.5 rad/s`; secondary `vx+vyaw` and `vy+vyaw` points are present at lower effective weight.
  - Random mixed commands remain in the background, but diagonal/random mixtures are not emphasized.
  - High-speed jerk/sudden-stop sampling is disabled by default for this branch.
  - External push randomization remains enabled; push success is trained as return-to-command, not abrupt-stop perfection.
  - Added reward terms for:
    - `vx-only` and `vy-only` axis tracking;
    - crosstalk suppression for unused linear/yaw axes;
    - relative yaw-only tracking and translation suppression;
    - zero-command no-push base/joint velocity damping;
    - push-active return-to-command.
  - Added `BOOSTER_AMP_LOAD_OPTIMIZER=0` support so AMP fine-tuning can resume weights with a fresh optimizer.
  - Added `scripts/rsl_rl/run_amp_axis_ft.sh` to launch the full fine-tune through tmux.

Verification before launch:
  - `python -m py_compile` on changed train/config/command/reward/eval files passed.
  - `bash -n` on the new/changed shell scripts passed.
  - `source /workspace/Isaac_uv_template/.venv/bin/activate && PYTHONPATH=/workspace/booster_amp_lab/rsl_rl pytest -q tests/test_track_adapter_module.py tests/test_track_adapter_storage.py tests/test_track_adapter_optimizer.py`
  - result: `47 passed`.

## 2026-05-11 17:45 UTC - AMP axis fine-tune full run launched

Started the first full AMP base-policy fine-tune for the revised objective:
  - tmux session: `amp_axis_ft_v1_20260511_173907`;
  - run directory: `logs/rsl_rl/run_amp_y/2026-05-11_17-39-11_amp_axis_ft_v1_20260511_173907`;
  - base checkpoint: `logs/rsl_rl/run_amp_y/2026-05-04_19-10-18_resume56_yawlow_zero_stand/model_70000.pt`;
  - optimizer: not restored (`BOOSTER_AMP_LOAD_OPTIMIZER=0`), so this is a fresh optimizer fine-tune from the AMP weights;
  - target iterations: `12000` more PPO iterations, i.e. `70000 -> 82000`;
  - launch wrapper: `scripts/rsl_rl/run_amp_axis_ft.sh`.

Early status:
  - `model_70100.pt` has been saved.
  - Around iteration `70117-70119`, `Episode_Termination/base_contact` is about `0.0076-0.0081`, below the early caution threshold.
  - `Metrics/base_velocity/hard_command_sampled` is non-zero, so exact hard command bins are active.
  - Axis-specific reward terms are logging non-zero values for `vx-only`, `vy-only`, and `vyaw-only` samples.
  - `Metrics/base_velocity/sudden_stop_sampled` is `0.0`, as intended for this branch.
  - Push failure metrics are still `0.0` in the first few minutes. Continue monitoring; this may be episode-window timing, but if push-active metrics stay absent after a longer window, inspect the push event configuration before trusting robustness conclusions.

Promotion criterion for this branch:
  Do not promote based on training reward alone. Promote only after fixed evaluation on no-push and interval-push command grids covering `vx={0.0,0.2,0.5,1.0,1.5,2.0}`, `vy={0.0,0.2,0.4,1.0}`, and `|vyaw|<=1.5`, with explicit straightness/crosstalk metrics and zero-command no-push stillness.

## 2026-05-11 17:54 UTC - AMP axis fine-tune v2 restart with push-active tracking

Stopped the v1 fine-tune early after detecting a configuration gap:
  - physical interval pushes were enabled;
  - however, `failure_mining_command_name` was not passed to `push_by_setting_velocity`;
  - therefore `failure_push_active` stayed at `0.0`, `push_recovery_velocity_track_exp` stayed at `0.0`, and zero-command no-push penalties could not skip push-active recovery windows.

Fix applied:
  - AMP fine-tune mode now sets `self.events.push_robot.params["failure_mining_command_name"] = "base_velocity"`;
  - AMP fine-tune mode now exposes and sets `BOOSTER_AMP_FT_PUSH_ACTIVE_WINDOW_S` default `2.5`;
  - AMP fine-tune mode now exposes `BOOSTER_AMP_FT_FAILURE_PUSH_SAMPLE_PROB` default `0.20`, so failed push bins can bias later push samples after failures are observed;
  - launch wrapper default run name moved from `amp_axis_ft_v1_*` to `amp_axis_ft_v2_*`.

Verification after the fix:
  - `python -m py_compile source/.../run_amp/env_cfg.py scripts/rsl_rl/train.py` passed;
  - `bash -n scripts/rsl_rl/run_amp_axis_ft.sh` passed;
  - `source /workspace/Isaac_uv_template/.venv/bin/activate && PYTHONPATH=/workspace/booster_amp_lab/rsl_rl pytest -q tests/test_track_adapter_module.py tests/test_track_adapter_storage.py tests/test_track_adapter_optimizer.py` passed with `47 passed`.

Restarted full run:
  - tmux session: `amp_axis_ft_v2_20260511_174702`;
  - run directory: `logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702`;
  - base checkpoint: `logs/rsl_rl/run_amp_y/2026-05-04_19-10-18_resume56_yawlow_zero_stand/model_70000.pt`;
  - optimizer: not restored (`BOOSTER_AMP_LOAD_OPTIMIZER=0`);
  - target iterations: `70000 -> 82000`.

Early v2 status:
  - `push_robot` interval event is active with `(5.0, 8.0)` seconds.
  - Around iteration `70008`, `Metrics/base_velocity/failure_push_active` became non-zero (`0.1562`) and `Episode_Reward/push_recovery_velocity_track_exp` became non-zero.
  - Around iteration `70008`, `Episode_Termination/base_contact` was about `0.0025`, below the early caution threshold.
  - `hard_command_sampled` is non-zero, so exact command bins are active.
  - Around iteration `70021`, `failure_push_active` was still appearing (`0.2500` in the sampled logging window), `push_recovery_velocity_track_exp` was about `0.1221`, and `base_contact` was about `0.0052`.

## 2026-05-12 07:59 UTC - AMP axis fine-tune v2 completed and final eval started

Training completion:
  - tmux session `amp_axis_ft_v2_20260511_174702` has exited.
  - Final checkpoint: `logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt`.
  - The run completed the planned `70000 -> 82000` range.

Training-log health:
  - final `Episode_Termination/base_contact`: `0.01094`;
  - max `Episode_Termination/base_contact` over the run: about `0.01707`;
  - final `Metrics/base_velocity/failure_base_contact`: `0.0`;
  - final `failure_push_active`: about `0.468`, with last-100 average about `0.517`;
  - final `hard_command_sampled`: about `0.601`, with last-100 average about `0.552`;
  - final `push_recovery_velocity_track_exp`: about `0.277`, with last-100 average about `0.282`.

Final checkpoint fixed evaluation:
  - output directory: `logs/fixed_eval_amp_axis_ft_v2_final_20260512_073810`;
  - checkpoint: `model_81999.pt`;
  - no-push, interval-push, and high-speed jerk smoke all completed.

Key no-push results versus the frozen base eval:
  - `vx=0.2` improved strongly: actual `vx` from about `0.060` to `0.192`, absolute `vx` error from about `0.163` to `0.033`.
  - `vx=0.5/1.0/1.5/2.0` forward tracking improved or stayed accurate.
  - `vy=0.2/0.4/1.0` lateral tracking improved in mean command error.
  - zero command remained near-stationary: actual `vx/vy/wz` about `0.0067/0.0009/-0.0046`, fall/contact `0.0`.
  - Straight-line drift/yaw did not improve; in several `vx-only` rows, integrated lateral/yaw drift is larger than the frozen base.
  - No-push safety regressed at the high end: `vx=2.0` ever fall/base-contact about `0.0234`, and `vx=1.5` ever fall about `0.0078`.

Key interval-push results:
  - Overlapping command tracking generally improved in linear command error, especially low-speed `vx` and `vy`.
  - Safety is mixed: `vx=2.0` interval-push fall/contact improved versus base, but new small failures appeared at zero, `vx=1.0`, `vx=1.5`, `vy=0.2`, `vy=1.0`, and some yaw/axis-yaw rows.

Key jerk-smoke results:
  - Frozen base had zero fall/contact in the same jerk scenarios.
  - `model_81999.pt` regressed jerk robustness:
    - `vx=2.0 -> 0` no-push fall/contact `0.0469`;
    - `vx=2.0 -> 0` push fall `0.0234`, contact `0.0156`;
    - `vx=1.0 -> 0` no-push/push fall/contact `0.0078`;
    - `vx=1.0,vy=1.0 -> 0` push fall/contact `0.0156`.

Interpretation:
  `model_81999.pt` solves the most visible low-speed tracking problem, but it is not safe enough to promote as the new base because high-speed and jerk robustness regressed. A checkpoint sweep is required to find an earlier model with most of the tracking gain and less safety loss.

Started lightweight checkpoint sweep:
  - tmux session: `eval_amp_axis_ft_v2_sweep_20260512_075951`;
  - output root: `logs/fixed_eval_amp_axis_ft_v2_sweep_20260512_075951`;
  - checkpoints: `model_74000.pt`, `model_76000.pt`, `model_78000.pt`, `model_80000.pt`, `model_81000.pt`;
  - lightweight settings: `NUM_ENVS=64`, `STEPS=300`, `SETTLE_STEPS=80`, no interval-push grid, high-speed jerk smoke included.

Additional lightweight early sweep:
  - output root: `logs/fixed_eval_amp_axis_ft_v2_early_sweep_20260512_081754`;
  - checkpoints: `model_70000.pt`, `model_70100.pt`, `model_71000.pt`, `model_72000.pt`, `model_73000.pt`;
  - same lightweight settings as the later sweep.

Sweep interpretation:
  - `model_70000.pt` remains the only swept checkpoint with zero jerk-smoke failures, but it retains the original low-speed `vx=0.2` under-tracking.
  - `model_70100.pt` already fixes most of `vx=0.2` tracking in the lightweight grid, but introduces small jerk-smoke failures.
  - `model_73000.pt` looks like the best coarse tradeoff among the swept fine-tuned checkpoints: low-speed tracking remains improved, no-push base-contact in the lightweight grid is better than many later checkpoints, and jerk-smoke max failure is lower than most later checkpoints.
  - Later checkpoints keep tracking gains but do not recover jerk robustness; `model_81999.pt` is not the promotion target.

Started full candidate evaluation:
  - tmux session: `eval_amp_axis_ft_v2_candidates_20260512_083603`;
  - output root: `logs/fixed_eval_amp_axis_ft_v2_candidates_20260512_083603`;
  - candidates: `model_70100.pt` and `model_73000.pt`;
  - settings: full 24-command grid, `NUM_ENVS=128`, `STEPS=400`, `SETTLE_STEPS=120`, interval-push grid enabled, high-speed jerk smoke included.

Candidate evaluation result:
  - `model_70100.pt` is the best current promotion candidate from this v2 branch.
  - `model_73000.pt` improves tracking further, but no-push `vx=2.0` safety is worse and jerk-smoke failures are broader.
  - `model_81999.pt` has the best broad tracking and better interval-push `vx=2.0` than the base, but jerk-smoke robustness regresses too much for promotion.

Full-eval comparison:
  - frozen base:
    - no-push `vx=0.2`: actual `0.060`, abs `vx` error `0.163`;
    - no-push max fall/contact in the v6 grid: about `0.0066/0.0066`;
    - interval-push max fall/contact: about `0.0517/0.0399`;
    - high-speed jerk smoke: fall/contact `0.0`.
  - `model_70100.pt`:
    - no-push `vx=0.2`: actual `0.191`, abs `vx` error `0.047`;
    - no-push `vx` error sum over `0.2/0.5/1.0/1.5/2.0`: `0.398` versus base `0.597`;
    - no-push `vy` error sum over `0.2/0.4/1.0`: `0.402` versus base `0.462`;
    - zero-command actual `vx/vy/wz`: about `-0.003/-0.001/-0.002`;
    - no-push max fall/contact: `0.0156/0.0156`, mainly `vx=2.0`;
    - interval-push max fall/contact: `0.0423/0.0423`, mainly `vx=2.0`, slightly better than base fall but still not clean;
    - high-speed jerk smoke max fall/contact/pre-stop failure: `0.0078/0.0078/0.0078`, only in `vx=1.0,vy=1.0 -> 0` no-push.
  - `model_73000.pt`:
    - no-push `vx=0.2`: actual `0.203`, abs `vx` error `0.035`;
    - no-push `vx` error sum: `0.324`, `vy` error sum: `0.311`;
    - no-push max fall/contact: `0.0444/0.0444`, mainly `vx=2.0`;
    - interval-push max fall/contact: `0.0359/0.0359`;
    - jerk-smoke max fall/contact/pre-stop failure: `0.0234/0.0156/0.0234`.
  - `model_81999.pt`:
    - no-push `vx=0.2`: actual `0.192`, abs `vx` error `0.033`;
    - no-push `vx` error sum: `0.307`, `vy` error sum: `0.282`;
    - no-push max fall/contact: `0.0234/0.0234`;
    - interval-push max fall/contact: `0.0218/0.0206`;
    - jerk-smoke max fall/contact/pre-stop failure: `0.0469/0.0469/0.0156`.

Current recommendation:
  Use `model_70100.pt` as the conservative AMP axis fine-tune candidate if a better base is needed immediately. It fixes the `vx=0.2` under-tracking without the severe final-checkpoint jerk regression. Do not promote `model_73000.pt` or `model_81999.pt` without another stability repair stage.

Next branch recommendation:
  Resume from `model_70100.pt`, not from `model_81999.pt`, for a short stability/push-recovery repair run. Keep the exact low-speed and lateral bins, reduce the learning rate, reduce high-speed `vx=2.0` over-optimization pressure, and add explicit high-speed no-push safety/jerk-smoke validation before allowing further tracking gains to accumulate.

## 2026-05-12 09:04 UTC - AMP axis repair run from model_70100 started

Started a short repair run from the current conservative candidate:
  - tmux session: `amp_axis_ft_repair70100_20260512_090043`;
  - run directory: `logs/rsl_rl/run_amp_y/2026-05-12_09-00-46_amp_axis_ft_repair70100_20260512_090043`;
  - base checkpoint: `logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_70100.pt`;
  - optimizer: not restored (`BOOSTER_AMP_LOAD_OPTIMIZER=0`);
  - target iterations: `70100 -> 74100`.

Repair settings:
  - lower LR: `BOOSTER_AMP_FT_LR=5.0e-5`;
  - lower entropy: `BOOSTER_AMP_FT_ENTROPY=0.0015`;
  - fewer PPO epochs: `BOOSTER_AMP_FT_NUM_LEARNING_EPOCHS=3`;
  - reduced hard command pressure: `BOOSTER_AMP_FT_HARD_ENVS=0.45`;
  - low-speed and standing retained: `LOW_SPEED_ENVS=0.20`, `STANDING_ENVS=0.14`;
  - small sudden-stop exposure: `BOOSTER_AMP_FT_SUDDEN_STOP_ENVS=0.06`;
  - stronger push exposure: push interval `(4.0, 7.0)`, `PUSH_XY=0.45`, push return-command weight `1.3`.

Early status:
  - startup and checkpoint load succeeded;
  - `push_robot` interval event is active;
  - around iteration `70109`, `base_contact` was about `0.0013`;
  - `failure_push_active` and `push_recovery_velocity_track_exp` are non-zero;
  - `hard_command_sampled` is non-zero.

Monitoring:
  Stop or replan if early `Episode_Termination/base_contact` rises toward `0.02`, if sudden-stop failures appear, or if fixed evaluation after the first repair checkpoint shows worse `vx=2.0` no-push/jerk safety than `model_70100.pt`.

## 2026-05-12 09:38 UTC - Base repair run stopped; move jerk recovery to Track Adapter

Stopped the base repair run intentionally after `model_70900.pt` was saved.

Reasoning:
  - The AMP axis fine-tune already established the useful base-policy correction at `model_70100.pt`: low-speed `vx=0.2` tracking improves substantially while zero-command remains quiet.
  - Further base-policy optimization tends to trade nominal tracking against high-speed/jerk safety.
  - The user correctly pointed out that residual correction for jerk/fall recovery is a better fit for the Track Adapter than continuing to reshape the AMP base.

Stopped run:
  - tmux session: `amp_axis_ft_repair70100_20260512_090043`;
  - saved checkpoints: `model_70100.pt` through `model_70900.pt`;
  - last observed early training state before stop: `base_contact` around `0.006`, `failure_push_active` non-zero, `failure_during_sudden_stop` `0.0`.

Decision:
  - Treat `logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_70100.pt` as the current frozen base candidate for Track Adapter work.
  - Do not promote `model_81999.pt` or continue base repair as the main path.
  - Next Track Adapter branch should freeze `model_70100.pt`, rebuild/revalidate Stage 3 world-model pretraining against that base, and train residual recovery for push/jerk/high-velocity-error windows while keeping stable/no-push residual authority tightly gated.

## 2026-05-13 10:08 UTC - v7 adapter run checked around iteration 2700

Current run:
  - tmux session: `track_adapter_81999_stage3_recovery_v7_20260513_0453`;
  - PPO directory: `logs/rsl_rl/run_amp_track_adapter/2026-05-13_05-24-25_track_adapter_81999_stage3_recovery_v7_20260513_0453_ppo_full`;
  - latest saved checkpoint at check time: `model_2700.pt`.

Rolling TensorBoard status at iteration `2718`:
  - last-200 mean reward: `307.98`;
  - `Episode_Termination/base_contact`: `0.0085`;
  - `Metrics/base_velocity/error_vel_xy`: `0.2915`;
  - `Metrics/base_velocity/error_vel_yaw`: `0.3643`;
  - `failure_base_contact`, `failure_terminated`, and `failure_during_sudden_stop`: all `0.0`;
  - `wm_valid_fraction`: `0.7706`, `wm loss`: `0.3637`;
  - gated scaled residual: `0.0019`;
  - ungated scaled residual: `0.0107`;
  - push-window scaled residual: `0.0054`, stable scaled residual: `0.0003`.

Interpretation:
  - The v7 restart is much safer than the previous `model_7000` branch: contact is low, sudden-stop failures are absent, and stable residual leakage is suppressed.
  - Residual is now emerging slowly and mostly through push/recovery windows, but it is still small. This is acceptable at this stage, but a fixed evaluation around `model_3000` should confirm whether push recovery improves without reintroducing the earlier `model_7000` over-intervention pattern.

## 2026-05-13 11:55 UTC - v7 model_3400 fixed evaluation

Evaluated:
  - checkpoint: `logs/rsl_rl/run_amp_track_adapter/2026-05-13_05-24-25_track_adapter_81999_stage3_recovery_v7_20260513_0453_ppo_full/model_3400.pt`;
  - output directory: `logs/fixed_eval_ta81999_v7_model3400_20260513_114216`;
  - eval used the correct frozen base override: `logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt`;
  - first eval attempt failed as intended because the base checkpoint override was missing and checkpoint metadata rejected the mismatch.

No-push fixed grid, common nine benchmark commands:
  - avg abs error `vx/vy/wz`: `0.0757 / 0.0829 / 0.1668`;
  - avg residual norm: `0.0400`;
  - max fall/contact: `0.0001 / 0.0000`;
  - zero command actual `vx/vy/wz`: about `0.001 / -0.001 / 0.002`, residual `0.002`;
  - compared with previous `model_7000`, residual is much lower (`0.0400` vs `0.0910`) with similar tracking.

Interval-push fixed grid:
  - avg abs error `vx/vy/wz`: `0.1460 / 0.1458 / 0.2226`;
  - avg residual norm: `0.2461`, much lower than previous `model_7000` (`0.5469`);
  - max instantaneous fall/contact: `0.0002 / 0.0001`;
  - max ever fall/contact remains weak at `0.0521 / 0.0384`, mainly `vx=2.0` under interval pushes.

High-speed jerk suite:
  - max fall/contact/pre-stop failure: `0.0156 / 0.0156 / 0.0156`;
  - this is better than previous `model_7000` on aggregate jerk max fall/contact (`0.0312 / 0.0156`) and with far lower final residual norm (`0.0042` vs `0.0216`);
  - failures remain in explicit stop-time push cases, especially `vy=1.0 -> 0` push and `vx=2.0 -> 0` push.

Current training status near iteration `3580`:
  - last-200 reward: `306.35`;
  - `error_vel_xy/yaw`: `0.2885 / 0.3612`;
  - `base_contact`: `0.0078`;
  - gated/ungated scaled residual: `0.0082 / 0.0469`;
  - `failure_during_sudden_stop`: `0.0`;
  - `wm_valid_fraction`: `0.7708`.

Decision:
  - Continue the v7 run. It has avoided the previous `model_7000` over-intervention pattern while beginning to use residual in recovery windows.
  - Do not promote `model_3400` as final yet: interval-push `vx=2.0` ever-fall/contact is still too high.
  - Re-evaluate around `model_4500` or earlier if gated residual approaches `0.03`, ungated residual approaches `0.10`, or base contact trends toward `0.02`.

## 2026-05-13 13:04 UTC - v7 model_4000 fixed evaluation and stop

Triggered early evaluation because the residual growth condition was reached before `model_4500`:
  - at iteration `4059`, last-200 gated scaled residual was about `0.0180`;
  - last-200 ungated scaled residual was about `0.1018`;
  - base contact stayed low at about `0.0102`;
  - sudden-stop failure stayed `0.0`.

Evaluated:
  - checkpoint: `logs/rsl_rl/run_amp_track_adapter/2026-05-13_05-24-25_track_adapter_81999_stage3_recovery_v7_20260513_0453_ppo_full/model_4000.pt`;
  - output directory: `logs/fixed_eval_ta81999_v7_model4000_20260513_124811`;
  - frozen base override: `logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt`.

No-push fixed grid, common nine benchmark commands:
  - `model_3400` avg abs error `vx/vy/wz`: `0.0757 / 0.0829 / 0.1668`, avg residual `0.0400`, max ever fall/contact `0.0129 / 0.0000`;
  - `model_4000` avg abs error `vx/vy/wz`: `0.0755 / 0.0811 / 0.1651`, avg residual `0.0646`, max ever fall/contact `0.0000 / 0.0000`;
  - no-push behavior is slightly better at `model_4000`, but it uses more residual.

Interval-push fixed grid:
  - `model_3400` avg abs error `vx/vy/wz`: `0.1460 / 0.1458 / 0.2226`, avg residual `0.2461`, max ever fall/contact `0.0521 / 0.0384`;
  - `model_4000` avg abs error `vx/vy/wz`: `0.1518 / 0.1500 / 0.2256`, avg residual `0.4150`, max ever fall/contact `0.0467 / 0.0278`;
  - `model_4000` slightly improves the worst `vx=2.0` interval-push ever-fall/contact, but residual use rises sharply and failures spread to `vx=1.5`, `vy=0.4`, and `vy=1.0` cases.

High-speed jerk suite:
  - `model_3400` max fall/contact/pre-stop: `0.0156 / 0.0156 / 0.0156`, avg stop p95 `0.0707`, avg final residual `0.0042`;
  - `model_4000` max fall/contact/pre-stop: `0.0156 / 0.0156 / 0.0156`, avg stop p95 `0.0670`, avg final residual `0.0032`;
  - jerk behavior is modestly better than `model_3400` and clearly better than previous `model_7000` on aggregate, but failures still occur in stop-time push cases.

Stopped:
  - tmux session `track_adapter_81999_stage3_recovery_v7_20260513_0453` was stopped after the `model_4000` evaluation completed;
  - saved checkpoints reached `model_4150.pt`;
  - the evaluated candidate remains `model_4000.pt`, not the later unevaluated checkpoints.

Decision:
  - Do not continue this exact run toward `model_7000`: residual growth is trending toward the previous over-intervention branch.
  - Keep `model_4000.pt` as a tentative candidate only if jerk/no-push priority dominates.
  - Keep `model_3400.pt` as the safer candidate if interval-push average tracking and lower residual are preferred.
  - Next branch should keep the v7 safety balance but add stronger push-specific residual shaping so residual increases only in true recovery windows and does not spread into broad interval-push locomotion.

## 2026-05-13 13:14 UTC - v8 push-selective residual branch prepared

Clarification:
  - residual use during continuous external push is allowed and expected;
  - the failure mode to avoid is not "any push-time intervention";
  - the target is avoiding high residual in already-stable push windows when it does not reduce fall/contact or improve command tracking.

Implementation change:
  - added push-stable residual shaping in `TrackAdapterRunner._compute_residual_penalty`;
  - new residual penalty fields:
    - `push_stable_weight`;
    - `push_stable_ungated_weight`;
    - `push_stable_score_threshold`;
  - new environment variables:
    - `BOOSTER_TRACK_ADAPTER_RESIDUAL_PUSH_STABLE_WEIGHT`;
    - `BOOSTER_TRACK_ADAPTER_RESIDUAL_PUSH_STABLE_UNGATED_WEIGHT`;
    - `BOOSTER_TRACK_ADAPTER_RESIDUAL_PUSH_STABLE_SCORE_THRESHOLD`;
  - the penalty only applies when `failure_push_active` is true and the disturbance score is below the configured threshold, so high-disturbance push recovery can still use residual authority.

New run script:
  - `scripts/rsl_rl/run_track_adapter_stage3_recovery_v8_resume3400.sh`;
  - resumes from v7 `model_3400.pt`;
  - reuses the validated v7 H79/N20 Stage 3 world-model pretrain;
  - loads the PPO checkpoint without optimizer state;
  - lowers LR to `3.0e-7`;
  - keeps residual scale at `0.045`;
  - keeps push residual gate open but lower than v7: `push_min_gate=0.50`;
  - adds push-stable residual shaping: `push_stable_weight=0.32`, `push_stable_ungated_weight=0.16`, score threshold `0.85`;
  - slightly strengthens push command/recovery tracking while keeping no-push/zero-command constraints.

Verification:
  - `bash -n scripts/rsl_rl/run_track_adapter_stage3_recovery_v8_resume3400.sh` passed;
  - `py_compile` passed for `track_adapter_runner.py` and Track Adapter PPO config.

Planned gate:
  - monitor `Adapter/push_stable_residual_penalty_active`, gated/ungated residual, and `base_contact`;
  - fixed-evaluate around resumed iteration `4000`-`4500`, or earlier if gated residual exceeds `0.03`, ungated residual exceeds `0.12`, or base contact trends toward `0.02`.

Started:
  - tmux session: `track_adapter_81999_stage3_recovery_v8_resume3400_20260513_131531`;
  - run directory: `logs/rsl_rl/run_amp_track_adapter/2026-05-13_13-15-35_track_adapter_81999_stage3_recovery_v8_resume3400_20260513_131531_ppo`;
  - source checkpoint: `logs/rsl_rl/run_amp_track_adapter/2026-05-13_05-24-25_track_adapter_81999_stage3_recovery_v7_20260513_0453_ppo_full/model_3400.pt`;
  - configured extra PPO iterations: `3200`, so the run logs as `3400/6600`;
  - optimizer state is not loaded.

Early status around iteration `3426`:
  - reward recovered to `323.86` after the short resume-start episode statistics;
  - `base_contact`: `0.0065`;
  - `error_vel_xy/yaw`: `0.2852 / 0.4366`;
  - gated/ungated scaled residual: `0.0123 / 0.0430`;
  - `Adapter/push_stable_residual_penalty_active`: `0.2787`;
  - `failure_during_sudden_stop`: `0.0`;
  - `wm_valid_fraction`: `0.7815`.

## 2026-05-13 16:12 UTC - v8 model_5000 fixed evaluation requested

Training status at request:
  - tmux session `track_adapter_81999_stage3_recovery_v8_resume3400_20260513_131531` was around iteration `5018/6600`;
  - checkpoint `model_5000.pt` was available;
  - fixed evaluation was requested because residual usage crossed the earlier warning range while failure metrics remained zero.

Evaluation plan:
  - keep training running;
  - run a separate tmux evaluation for `model_5000.pt`;
  - use the frozen AMP base checkpoint `run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt`;
  - include the priority command grid: zero command, vx-only, vy-only, and vyaw-only commands up to `1.5 rad/s`;
  - run no-push, interval-push, and high-speed jerk smoke outputs.

Result:
  - evaluation session: `eval_ta81999_v8_model5000_20260513_161223`;
  - output directory: `logs/fixed_eval_ta81999_v8_model5000_20260513_161223`;
  - no-push 12-command grid:
    - average absolute errors vx/vy/wz: `0.0701 / 0.0846 / 0.1747`;
    - average residual norm: `0.0780`;
    - max ever fall/base-contact: `0.0156 / 0.0156`, observed at high forward speeds.
  - interval-push 12-command grid:
    - average absolute errors vx/vy/wz: `0.1530 / 0.1504 / 0.2507`;
    - average residual norm: `0.5417`;
    - max ever fall/base-contact: `0.2073 / 0.1124`, worst at `vx=2.0`;
    - this is worse than both v7 `model_3400.pt` and v7 `model_4000.pt` on the common 9-command grid.
  - high-speed jerk smoke:
    - max fall/base-contact/pre-stop failure: `0.0156 / 0.0156 / 0.0156`;
    - average final stop p95 speed: `0.0794`;
    - average final residual norm: `0.0085`.

Common 9-command comparison:
  - v7 `model_3400.pt` no-push avg errors vx/vy/wz: `0.0757 / 0.0829 / 0.1668`, interval-push max ever fall/base-contact: `0.0521 / 0.0384`;
  - v7 `model_4000.pt` no-push avg errors vx/vy/wz: `0.0755 / 0.0811 / 0.1651`, interval-push max ever fall/base-contact: `0.0467 / 0.0278`;
  - v8 `model_5000.pt` no-push avg errors vx/vy/wz: `0.0754 / 0.0843 / 0.1666`, interval-push max ever fall/base-contact: `0.2073 / 0.1124`.

Decision:
  - v8 `model_5000.pt` is not an upgrade candidate;
  - the push-stable penalty did not prevent residual escalation by iteration 5000;
  - the active v8 training run was manually stopped after `model_5100.pt` had been saved, because continuing toward `6600` was likely to amplify the same failure mode.

## 2026-05-13 16:37 UTC - v9 layer-wise Track Adapter implemented

Reason:
  - the Any2Track paper uses layer-wise zero-initialized adapters rather than a single action residual head;
  - v8 showed that action residuals can still escalate under interval pushes even when stable-push penalties are active.

Implementation:
  - added `adapter_mode` to `TrackAdapterActorCritic`;
  - default remains `action_residual` for compatibility;
  - new `adapter_mode=layerwise` creates zero-initialized adapter MLPs after every frozen base actor `Linear` layer;
  - existing residual gate is applied to layer-wise adapter outputs, so stable/no-push and zero-command gating still constrain behavior;
  - diagnostics still measure final action deviation from the frozen base action;
  - PPO optimizer uses `layer_adapters` instead of the unused action residual head in layer-wise mode;
  - runner residual-output perturbation now supports both action-residual and layer-wise adapters;
  - config/env additions:
    - `BOOSTER_TRACK_ADAPTER_MODE`;
    - `BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS`;
    - `BOOSTER_TRACK_ADAPTER_LAYERWISE_OUTPUT_INIT_SCALE`.

Verification:
  - `py_compile` passed for the changed Track Adapter module, runner, and PPO config;
  - `pytest -q tests/test_track_adapter_module.py tests/test_track_adapter_optimizer.py` passed with `39 passed`;
  - layer-wise smoke tmux session `track_adapter_v9_layerwise_smoke_20260513_163702` completed 3 iterations;
  - smoke loaded the frozen `model_81999.pt` base and the H79/N20 Stage 3 world-model pretrain `model_499.pt`;
  - smoke output directory: `logs/rsl_rl/run_amp_track_adapter/2026-05-13_16-37-06_track_adapter_81999_stage3_recovery_v9_layerwise_smoke_20260513_163702_ppo`.

Training script:
  - added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v9_layerwise.sh`;
  - starts from the frozen `model_81999.pt` base plus the existing Stage 3 world-model pretrain;
  - does not resume an action-residual PPO checkpoint because the adapter architecture changed;
  - uses layer-wise hidden dim `128`, residual scale `0.08`, LR `7.5e-7`, and safety gate `scaled_residual_action_l2 <= 0.12`.

Started:
  - tmux session: `track_adapter_v9_layerwise_20260513_163804`;
  - run directory: `logs/rsl_rl/run_amp_track_adapter/2026-05-13_16-38-08_track_adapter_81999_stage3_recovery_v9_layerwise_20260513_163804_ppo`;
  - configured iterations: `5000`;
  - environment count: `2048`;
  - initial iteration `0` loaded the pretrain and inserted about `98304` world-model replay samples;
  - iteration `1` remained safe with `base_contact` about `0.0008`, `failure_base_contact=0`, `wm_valid_fraction` about `0.802`, and residual effectively zero from zero initialization.

Early status around iteration `12`:
  - mean reward last-10: about `43.6` while early episodes are still short;
  - `Episode_Termination/base_contact` last-10: about `0.0034`;
  - `failure_base_contact`, `failure_terminated`, and `failure_during_sudden_stop`: `0.0`;
  - push-active fraction last-10: about `0.443`;
  - gated/ungated scaled residual last-10: about `0.000001 / 0.000002`;
  - `wm_valid_fraction` last-10: about `0.778`;
  - no early collapse from layer-wise adapter insertion was observed.

Status around iteration `919`:
  - tmux session was still running and checkpoint `model_900.pt` was saved;
  - last-200 mean reward: about `328.0`;
  - last-200 velocity errors vx/vy aggregate and yaw: `0.3045 / 0.3946`;
  - last-200 `Episode_Termination/base_contact`: about `0.0078`;
  - `failure_base_contact`, `failure_terminated`, and `failure_during_sudden_stop`: `0.0`;
  - last-200 push-active fraction: about `0.368`;
  - last-200 gated/ungated scaled residual: about `0.0008 / 0.0046`;
  - last-200 push-window scaled residual: about `0.0020`;
  - last-200 stable scaled residual: about `0.00012`;
  - last-200 `wm_valid_fraction`: about `0.771`;
  - interpretation: layer-wise adapter remains much more conservative than the action-residual v7/v8 branches at the same stage. This is safe, but fixed evaluation should wait until residual recovery has emerged more clearly or until about `model_1500`-`model_2000`.

## 2026-05-14 04:10 UTC - v9 layer-wise run completed and fixed-evaluated

Training completion:
  - tmux training session `track_adapter_v9_layerwise_20260513_163804` completed the configured `5000` iterations;
  - final checkpoint: `logs/rsl_rl/run_amp_track_adapter/2026-05-13_16-38-08_track_adapter_81999_stage3_recovery_v9_layerwise_20260513_163804_ppo/model_4999.pt`;
  - final iteration `4999`:
    - `Train/mean_reward`: `310.80`;
    - `Train/mean_episode_length`: `1000.00`;
    - `Episode_Termination/base_contact`: `0.0138`;
    - `Metrics/base_velocity/failure_base_contact`: `0.0`;
    - `Metrics/base_velocity/failure_terminated`: `0.0`;
    - `Metrics/base_velocity/failure_during_sudden_stop`: `0.0`;
    - `Metrics/base_velocity/error_vel_xy`: `0.2939`;
    - `Metrics/base_velocity/error_vel_yaw`: `0.3695`;
    - `Loss/adapter_scaled_residual_action_l2`: `0.0298`;
    - `Loss/adapter_ungated_scaled_residual_action_l2`: `0.1599`;
    - `Loss/wm`: `0.2899`;
    - `Loss/wm_valid_fraction`: `0.7784`.

Fixed evaluation:
  - the first eval attempt at `logs/fixed_eval_ta81999_v9_layerwise_model4999_20260514_040820` failed because the eval process did not pass `BOOSTER_TRACK_ADAPTER_MODE=layerwise` and `BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS=128`;
  - rerun output: `logs/fixed_eval_ta81999_v9_layerwise_model4999_20260514_041014`;
  - corrected eval used `BOOSTER_TRACK_ADAPTER_MODE=layerwise`, `BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS=128`, and `BOOSTER_TRACK_ADAPTER_RESIDUAL_SCALE=0.08`.

No-push fixed 12-command grid:
  - average absolute errors vx/vy/wz: `0.0700 / 0.0815 / 0.1747`;
  - max ever fall/base-contact: `0.0000 / 0.0000`;
  - zero command actual vx/vy/wz: about `0.002 / -0.000 / 0.001`;
  - `vx=0.2` actual vx/vy/wz: about `0.183 / -0.005 / 0.018`;
  - `vx=0.5` actual vx/vy/wz: about `0.469 / 0.001 / 0.024`;
  - `vx=2.0` actual vx/vy/wz: about `1.929 / -0.023 / 0.010`;
  - average residual norm: `0.0420`.

Interval-push fixed 12-command grid:
  - average absolute errors vx/vy/wz: `0.1280 / 0.1459 / 0.2273`;
  - average ever fall/base-contact: `0.0031 / 0.0027`;
  - max ever fall/base-contact: `0.0326 / 0.0326`, worst at `vx=2.0`;
  - zero command under interval push remained upright with max fall/base-contact `0.0000 / 0.0000`;
  - average residual norm: `0.2839`.

High-speed jerk fixed smoke:
  - no-push jerk from `vx=2.0`, `vx=1.0`, `vy=1.0`, and `vx=1.0,vy=1.0` all had fall/base-contact `0.0000 / 0.0000`;
  - push-at-stop jerk average fall/base-contact: `0.0078 / 0.0078`;
  - push-at-stop jerk max fall/base-contact: `0.0156 / 0.0156`, seen at `vx=2.0 -> 0` and `vx=1.0,vy=1.0 -> 0`;
  - all-scenario average final stop speed: `0.0365`;
  - no-push average final stop speed: `0.0346`.

Decision:
  - v9 `model_4999.pt` is the best Track Adapter promotion candidate so far for the user's priority set;
  - it preserves no-push command tracking and zero-command quietness while materially improving interval-push robustness over the v8 final branch;
  - remaining risk is concentrated at high forward speed under repeated interval pushes (`vx=2.0`) and explicit push-at-stop jerk, where failures are rare but non-zero.

## 2026-05-14 05:06 UTC - v9 external-push strength sweep

Purpose:
  - quantify how much external push the promoted v9 `model_4999.pt` can tolerate;
  - the push implementation is `root velocity += push_delta`, not a Newton-force impulse;
  - interval-push sweep used 12 fixed commands, `64` envs, `400` steps with `120` settle steps, and a very aggressive `1.0 s` push interval.

Reference result from the previous fixed eval:
  - `push_xy=0.60`, `push_yaw=0.10`, interval `1.0 s`;
  - average fall/base-contact: `0.0031 / 0.0027`;
  - max fall/base-contact: `0.0326 / 0.0326`, worst at `vx=2.0`.

Stronger sweep with larger yaw disturbance:
  - output root: `logs/push_sweep_ta81999_v9_layerwise_model4999_20260514_050658`;
  - `push_xy=0.80`, `push_yaw=0.22`:
    - average fall/base-contact: `0.0384 / 0.0324`;
    - max fall/base-contact: `0.1834 / 0.1326`, worst at `vx=2.0`;
    - average abs errors vx/vy/wz: `0.1922 / 0.1799 / 0.2829`;
  - `push_xy=1.00`, `push_yaw=0.22`:
    - average fall/base-contact: `0.0822 / 0.0621`;
    - max fall/base-contact: `0.2838 / 0.2204`, worst at `vx=2.0`;
    - average abs errors vx/vy/wz: `0.2424 / 0.2330 / 0.3291`;
  - `push_xy=1.20`, `push_yaw=0.22`:
    - average fall/base-contact: `0.2172 / 0.1790`;
    - max fall/base-contact: `0.4621 / 0.3664`, worst at `vx=2.0`;
    - average abs errors vx/vy/wz: `0.3102 / 0.2957 / 0.4022`.

Interpretation:
  - current v9 adapter is robust to repeated `0.6 m/s` horizontal velocity pushes with small yaw disturbance;
  - under the harsher yaw disturbance `0.22 rad/s`, `0.8 m/s` horizontal pushes are already beyond the clean-promotion range, especially while commanded at `vx=2.0`;
  - `1.0 m/s` and `1.2 m/s` repeated pushes are outside the reliable operating envelope for this checkpoint.

## 2026-05-14 05:55 UTC - v10 recovery-first force-kick branch prepared

Purpose:
  - start a new recovery-first adapter branch rather than extending v9;
  - target push recovery under both velocity-delta disturbances and finite-duration physical trunk kicks;
  - keep no-push command tracking and zero-command stillness, but allow much larger adapter authority only during recovery.

Implementation:
  - added `apply_external_wrench_pulse` as a per-env finite-duration external force/torque event on the trunk;
  - force pulses use internal timers, clear on reset, support world-frame forces, randomized contact point, yaw torque, vertical force, curriculum, and failure-mining attribution through `OmniVelocityCommand`;
  - added Track Adapter env wiring for the force event, disabled by default unless `BOOSTER_TRACK_ADAPTER_FORCE_PUSH=1`;
  - added real Track Adapter root-velocity push curriculum wiring, so `BOOSTER_TRACK_ADAPTER_PUSH_XY_FINAL` and `BOOSTER_TRACK_ADAPTER_PUSH_YAW_FINAL` now affect the Track Adapter event;
  - added safe Stage3 disturbance-manifest serialization for force-push params;
  - added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v10_force_kick.sh` as the reproducible v10 entrypoint;
  - added `scripts/rsl_rl/eval_track_adapter_v10_force_kick_grid.sh` to evaluate physical kicks separately from root-velocity jump pushes.

v10 defaults:
  - frozen AMP base: `logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt`;
  - Track Adapter mode: `layerwise`, hidden dim `128`;
  - fresh Stage3 H79/N20 pretrain: `768` envs, `700` iterations, `128` steps/env;
  - PPO: `2048` envs, `12000` iterations, `48` steps/env;
  - WM replay: `786432` samples, `6` WM updates per PPO iteration;
  - velocity push curriculum: `push_xy 0.40 -> 1.00`, `push_yaw 0.10 -> 0.30`;
  - physical kick curriculum: `160-420 N -> 320-900 N`, duration `0.06-0.16 s`, yaw torque `24 -> 70 Nm`, world-frame force direction;
  - recovery authority: residual scale `0.16`, push gate min `0.92`, stable gate max `0.025`, recovery residual penalty weight `0.005`.

Verification before full run:
  - `python -m py_compile` passed for modified env/event/command/pretrain runner files;
  - `bash -n` passed for v10 train/eval scripts and chained Stage3-to-PPO script;
  - `pytest -q tests/test_track_adapter_module.py tests/test_track_adapter_optimizer.py` passed: `39 passed`;
  - bounded v10 smoke completed Stage3 and PPO startup after manifest serialization fix.

## 2026-05-14 05:50 UTC - v10 full Stage3 run started

First full-run attempt:
  - session `track_adapter_v10_force_kick_20260514_054819`;
  - stopped during early Stage3 because IsaacLab emitted repeated external-wrench frame toggle warnings when calling `set_external_force_and_torque(..., is_global=True)`;
  - no checkpoint from this attempt should be used.

Fix:
  - kept the v10 semantic intent of world-directional kicks, but changed `apply_external_wrench_pulse` so `is_global=True` means "sample force/torque in world frame, then convert to link frame before passing to IsaacLab";
  - the IsaacLab API call now uses `is_global=False`, avoiding per-step wrench-frame toggles and log spam;
  - reran `py_compile`, `bash -n`, and `pytest -q tests/test_track_adapter_module.py tests/test_track_adapter_optimizer.py`; all passed.

Active full run:
  - tmux session: `track_adapter_v10_force_kick_20260514_055015`;
  - run id: `track_adapter_81999_stage3_recovery_v10_force_kick_20260514_055015`;
  - log: `logs/track_adapter_81999_stage3_recovery_v10_force_kick_20260514_055015.log`;
  - Stage3 is running with `external_wrench_push` active at interval `(0.02, 0.02)`;
  - early Stage3 metrics through iteration `8/700` are finite: `wm` roughly `0.70-0.90`, `wm_valid_fraction` roughly `0.88-0.93`;
  - observed GPU memory during Stage3: about `15.6 / 32.6 GB`.

## 2026-05-14 05:54 UTC - v10 final full run restarted after review fixes

Review fixes:
  - added a small recovery residual activation loss for PPO (`weight=0.01`, gate threshold `0.85`, target norm `0.025`) so recovery windows do not merely permit adapter action but also mildly encourage using it when the gate is open;
  - changed `eval_track_adapter_v10_force_kick_grid.sh` to require an explicit `CHECKPOINT`, avoiding accidental evaluation of the old v9 checkpoint;
  - added `force_push_enabled` telemetry to `eval_velocity_command_grid.py` so physical kick eval is not confused with root-velocity push eval.

Final active full run:
  - stopped the intermediate run `track_adapter_v10_force_kick_20260514_055015` before PPO;
  - tmux session: `track_adapter_v10_force_kick_20260514_055403`;
  - run id: `track_adapter_81999_stage3_recovery_v10_force_kick_20260514_055403`;
  - log: `logs/track_adapter_81999_stage3_recovery_v10_force_kick_20260514_055403.log`;
  - Stage3 entered cleanly; early metrics through iteration `2/700`: `wm=0.7059 -> 0.8817`, `wm_valid_fraction=0.9258 -> 0.8833`;
  - no external-wrench frame spam, no traceback, and GPU memory remains about `15.6 / 32.6 GB` during Stage3.

## 2026-05-14 11:50 UTC - v10 failed early; v10b warm-start PPO launched

v10 outcome:
  - Stage3 completed normally and saved `model_699.pt`;
  - Stage3 final metrics: `wm=0.3915`, `wm_valid_fraction=0.8948`;
  - PPO run directory: `logs/rsl_rl/run_amp_track_adapter/2026-05-14_07-33-51_track_adapter_81999_stage3_recovery_v10_force_kick_20260514_055403_ppo_force_kick`;
  - PPO did not complete; it stopped at iteration `123/12000`;
  - checkpoint saved on abort: `model_123_aborted.pt`;
  - stop reason: safety gate, `Episode_Termination/base_contact` was about `0.997`, far above the configured `0.18` threshold.

Interpretation:
  - this was not a successful full training;
  - the physical force kick started too hard and too frequently for a zero-initialized layerwise adapter;
  - adapter residual stayed essentially zero in v10 PPO, so the policy was not actually intervening before repeated trunk-contact terminations dominated.

Fixes:
  - added `BOOSTER_TRACK_ADAPTER_SKIP_STAGE3=1` support so the verified v10 Stage3 checkpoint can be reused without spending another long pretrain pass;
  - added `BOOSTER_TRACK_ADAPTER_ADAPTER_WARM_START`, loading compatible `layer_adapters.*` and `residual_actor.*` keys from a PPO Track Adapter checkpoint after the Stage3 world model is loaded;
  - added force-kick curriculum fields for duration, interval, and activation probability;
  - updated v10 defaults to warm-start from v9 `model_4999.pt`, start physical force kick at `50-160 N`, ramp to `320-900 N`, start with probability `0.20`, and ramp to `0.85`;
  - updated safety gate for the stronger curriculum: min iteration `300`, base-contact max `0.35`, patience `8`.

v10b active run:
  - tmux session: `track_adapter_v10b_force_kick_warm_20260514_115031`;
  - run id: `track_adapter_81999_stage3_recovery_v10b_force_kick_warm_20260514_115031`;
  - log: `logs/track_adapter_81999_stage3_recovery_v10b_force_kick_warm_20260514_115031.log`;
  - reused Stage3 checkpoint: `logs/rsl_rl/run_amp_track_adapter/2026-05-14_05-54-07_track_adapter_81999_stage3_recovery_v10_force_kick_20260514_055403_wm_pretrain_force_kick/model_699.pt`;
  - warm-start loaded `22` adapter keys from v9 `model_4999.pt`;
  - iteration `40/12000` snapshot: `Mean reward=407.54`, `Mean episode length` near timeout, `Episode_Termination/base_contact=0.1051`, `error_vel_xy=0.2435`, `error_vel_yaw=0.3015`, `adapter_scaled_residual_action_l2=0.2081`;
  - iteration `50/12000` snapshot: `Mean reward=355.97`, `Mean episode length=826.65`, `Episode_Termination/base_contact=0.0998`, `error_vel_xy=0.2398`, `error_vel_yaw=0.3056`, `adapter_scaled_residual_action_l2=0.1492`;
  - latest monitored values after iteration `50` show `base_contact` drifting down toward about `0.078-0.083` while force-push active windows remain present;
  - compared with the failed v10 run, base-contact at the same early phase is much lower and adapter intervention is active.

## 2026-05-14 12:27 UTC - v10b status at iteration 326

Current run:
  - tmux session `track_adapter_v10b_force_kick_warm_20260514_115031` is still running;
  - latest observed iteration: `326/12000`;
  - no safety-gate strike, traceback, or runtime error observed.

Latest snapshot:
  - `Mean reward`: about `446`;
  - `Mean episode length`: about `959`;
  - `Episode_Termination/base_contact`: about `0.071`;
  - `Metrics/base_velocity/error_vel_xy`: about `0.26`;
  - `Metrics/base_velocity/error_vel_yaw`: about `0.32`;
  - `failure_push_active`: ranged roughly `0.36-0.74` in the latest windows;
  - `adapter_scaled_residual_action_l2`: about `0.06-0.08`;
  - `wm`: about `0.275`, `wm_valid_fraction`: about `0.79`;
  - GPU memory: about `16.5 / 32.6 GB`.

Interpretation:
  - v10b is much healthier than failed v10: it has not hit the safety gate and base-contact is around `7%` instead of nearly `100%`;
  - adapter intervention is active but no longer explosively large;
  - push recovery is improving enough to keep most episodes near timeout, but base contact is not yet clean enough for promotion;
  - fixed evaluation should wait until at least several hundred more iterations unless the trend reverses.

## 2026-05-14 14:52 UTC - v10b stopped by safety gate at iteration 1573

Outcome:
  - tmux session `track_adapter_v10b_force_kick_warm_20260514_115031` ended;
  - PPO stopped at iteration `1573/12000`;
  - stop reason: safety gate failure, `Episode_Termination/base_contact` exceeded the configured `0.35` threshold for `8` strikes;
  - final abort checkpoint: `logs/rsl_rl/run_amp_track_adapter/2026-05-14_11-50-36_track_adapter_81999_stage3_recovery_v10b_force_kick_warm_20260514_115031_ppo_force_kick/model_1573_aborted.pt`;
  - latest regular checkpoint before abort: `model_1550.pt`.

Trend:
  - iteration `500`: reward `471.66`, episode length `987.69`, base_contact `0.0762`, xy/yaw error `0.2833/0.3495`, scaled residual `0.0451`;
  - iteration `750`: reward `452.55`, episode length `966.64`, base_contact `0.1101`, xy/yaw error `0.2661/0.3062`, scaled residual `0.0813`;
  - iteration `1000`: reward `425.81`, episode length `939.98`, base_contact `0.1949`, xy/yaw error `0.2671/0.3400`, scaled residual `0.1360`;
  - iteration `1250`: reward `385.05`, episode length `872.95`, base_contact `0.2844`, xy/yaw error `0.2700/0.3375`, scaled residual `0.1610`;
  - iteration `1500`: reward `379.48`, episode length `873.20`, base_contact `0.3289`, xy/yaw error `0.2770/0.3493`, scaled residual `0.1773`;
  - iteration `1573`: reward `364.62`, episode length `847.04`, base_contact `0.3590`, xy/yaw error `0.2635/0.3426`, scaled residual `0.1789`.

Interpretation:
  - v10b was a meaningful improvement over failed v10, but it still did not survive the stronger late physical-kick curriculum;
  - as force-kick curriculum strengthened, base-contact steadily rose while speed tracking stayed roughly flat;
  - the branch should not be promoted as-is;
  - candidate checkpoints for fixed evaluation are likely pre-failure midpoints such as `model_500.pt`, `model_750.pt`, and possibly `model_1000.pt`; later checkpoints carry too much base-contact risk.

## 2026-05-14 15:20 UTC - v11 adaptive force-kick branch prepared

Implemented before the next full run:
  - termination-time failure recording: trunk/base contact now calls the command term immediately, so command and push failure-mining bins can update even when the episode terminates before the next command update;
  - training-only contact grace: Track Adapter training can allow a short trunk-contact recovery window before reset while still recording contact immediately for mining;
  - adaptive physical-kick curriculum: force magnitude, duration, interval, and activation probability can be promoted/demoted from recent training feedback instead of only wall-clock iteration;
  - resume-relative safety gate: v11 can wait `N` iterations after a resumed checkpoint before enforcing the safety gate;
  - push-state cleanup: expired push windows now clear their sampled flag, and reset-only push state clears direction/magnitude/command bins;
  - eval telemetry cleanup: force-kick eval requires an explicit checkpoint, reports `force_push_enabled`, uses strict contact termination by default, and defaults to local-frame wrench application;
  - v11 launch defaults now use local-frame physical kicks, `model_81999.pt` frozen base, v10 Stage3 `model_699.pt`, and v10b `model_750.pt` as the PPO resume point.

Verification:
  - `py_compile` passed for command, robustness, events, env cfg, PPO cfg, runner, and velocity-grid eval files;
  - `bash -n` passed for v11/v10/stage3/eval shell entrypoints;
  - focused tests passed: `39 passed in 1.35s`;
  - tmux smoke completed with `PPO_NUM_ENVS=256`, `PPO_ITERATIONS=6`, Stage3 skipped/reused, and PPO resumed from `model_750.pt`;
  - smoke run directory: `logs/rsl_rl/run_amp_track_adapter/2026-05-14_15-15-12_track_adapter_81999_stage3_recovery_v11_adaptive_force_kick_smoke_20260514_151507_ppo_adaptive_force_kick`;
  - smoke checkpoints saved: `model_750.pt`, `model_753.pt`, `model_755.pt`;
  - smoke latest metrics at iteration `755/756`: `wm=0.2490`, `wm_valid_fraction=0.8006`, `adapter_scaled_residual_action_l2=0.1435`, `base_contact=0.0000` in the short smoke window.

Full-run intent:
  - run v11 as a recovery-first adapter branch, not a v10 continuation;
  - prioritize survival under physical kicks and root-velocity push recovery while keeping no-push residual authority tightly capped;
  - promote/demote kick difficulty from recent `base_contact` and episode-length feedback;
  - fixed evaluation, not training reward alone, decides promotion.

## 2026-05-14 15:18 UTC - v11 full training launched

Active tmux:
  - session: `track_adapter_v11_adaptive_force_kick_20260514_151645`;
  - run id: `track_adapter_81999_stage3_recovery_v11_adaptive_force_kick_20260514_151645`;
  - log: `logs/track_adapter_81999_stage3_recovery_v11_adaptive_force_kick_20260514_151645.log`;
  - PPO run directory: `logs/rsl_rl/run_amp_track_adapter/2026-05-14_15-16-50_track_adapter_81999_stage3_recovery_v11_adaptive_force_kick_20260514_151645_ppo_adaptive_force_kick`;
  - frozen base: `logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt`;
  - Stage3 pretrain: `logs/rsl_rl/run_amp_track_adapter/2026-05-14_05-54-07_track_adapter_81999_stage3_recovery_v10_force_kick_20260514_055403_wm_pretrain_force_kick/model_699.pt`;
  - PPO resume: `logs/rsl_rl/run_amp_track_adapter/2026-05-14_11-50-36_track_adapter_81999_stage3_recovery_v10b_force_kick_warm_20260514_115031_ppo_force_kick/model_750.pt`;
  - optimizer resume disabled.

Initial monitored status:
  - iteration `757/12750`: `wm=0.2498`, `wm_valid_fraction=0.8017`, replay at `786430`, scaled residual `0.0882`, `base_contact=0.0010`;
  - iteration `765/12750`: reward `298.20`, episode length `642.94`, `error_vel_xy=0.2978`, `error_vel_yaw=0.4678`, scaled residual `0.1122`, `base_contact=0.1040`;
  - failure mining is now active: `failure_mining_total=162.1615`, `failure_push_mining_total=96.0246`, `failure_push_mining_recorded=0.9375`;
  - GPU memory observed at about `16.5 / 32.6 GB`.

Watch criteria:
  - expected early range: `base_contact` may be nonzero while the adaptive force curriculum is settling, but it should not sustain above the v11 safety threshold `0.22` after the relative warmup;
  - the branch is healthy only if episode length climbs while `base_contact` and residual growth stay bounded;
  - fixed evaluation is required before any promotion, especially force-kick recovery, no-push zero command, and the user-priority single-axis velocity grid.

## 2026-05-14 19:12 UTC - v11 training status at iteration 2787

Active run is still running:
  - tmux session: `track_adapter_v11_adaptive_force_kick_20260514_151645`;
  - latest checkpoint observed: `model_2775.pt`;
  - GPU memory: about `16.5 / 32.6 GB`.

Latest snapshot:
  - iteration: `2787/12750`;
  - mean reward: `436.30`;
  - mean episode length: `961.60`;
  - `Episode_Termination/base_contact`: `0.0884`;
  - `Metrics/base_velocity/error_vel_xy`: `0.3084`;
  - `Metrics/base_velocity/error_vel_yaw`: `0.3520`;
  - `adapter_scaled_residual_action_l2`: `0.3129`;
  - `wm`: `0.2028`;
  - `wm_valid_fraction`: `0.7958`;
  - `failure_mining_total`: `114.6307`;
  - `failure_push_mining_total`: `88.7499`.

Interpretation:
  - the run is much healthier than v10/v10b late failure: episode length is near timeout and base contact is below the v11 safety threshold;
  - failure mining is active and no longer stuck at zero;
  - residual usage is rising, so this should continue to be watched. The branch is not yet promotable without fixed evaluation.

## 2026-05-15 00:34 UTC - v11 training status at iteration 5556

Active run is still running:
  - tmux session: `track_adapter_v11_adaptive_force_kick_20260514_151645`;
  - latest checkpoint observed: `model_5550.pt`;
  - GPU memory: about `16.5 / 32.6 GB`;
  - ETA in log: about `13:54` remaining.

Latest snapshot:
  - iteration: `5556/12750`;
  - mean reward: `438.23`;
  - mean episode length: `972.07`;
  - `Episode_Termination/base_contact`: `0.0896`;
  - `Metrics/base_velocity/error_vel_xy`: `0.3205`;
  - `Metrics/base_velocity/error_vel_yaw`: `0.3705`;
  - `adapter_scaled_residual_action_l2`: `0.2868`;
  - `adapter_ungated_scaled_residual_action_l2`: `0.6330`;
  - `wm`: `0.1720`;
  - `wm_valid_fraction`: `0.7932`;
  - `failure_mining_total`: `63.5543`;
  - `failure_push_mining_total`: `41.9577`.

Interpretation:
  - still healthy compared with v10b: base contact remains around `0.09` and episode length stays near timeout;
  - world-model loss improved from about `0.20` at iteration `2787` to about `0.17`;
  - residual behavior is not exploding, but ungated residual remains high, so fixed no-push/zero-command evaluation is still required before promotion.

## 2026-05-15 02:23 UTC - v11 training status at iteration 6493

Active run is still running:
  - tmux session: `track_adapter_v11_adaptive_force_kick_20260514_151645`;
  - latest checkpoint observed: `model_6475.pt`;
  - GPU memory: about `16.5 / 32.6 GB`;
  - ETA in log: about `12:06` remaining.

Latest snapshot:
  - iteration: `6493/12750`;
  - mean reward: `422.65`;
  - mean episode length: `954.42`;
  - `Episode_Termination/base_contact`: `0.0939`;
  - `Metrics/base_velocity/error_vel_xy`: `0.3096`;
  - `Metrics/base_velocity/error_vel_yaw`: `0.3648`;
  - `adapter_scaled_residual_action_l2`: `0.3110`;
  - `adapter_ungated_scaled_residual_action_l2`: `0.6564`;
  - `wm`: `0.1631`;
  - `wm_valid_fraction`: `0.7941`;
  - `failure_mining_total`: `63.7673`;
  - `failure_push_mining_total`: `37.2855`.

Interpretation:
  - still running safely below the configured `base_contact` safety threshold;
  - world-model loss continues to improve slowly;
  - reward/episode length fluctuate but do not show the v10b-style collapse;
  - residual use remains material, so promotion still requires fixed no-push, force-kick, and single-axis tracking evaluation.

## 2026-05-15 03:00 UTC - v11 model_6500 fixed evaluation and stop decision

Evaluated checkpoint:
  - `logs/rsl_rl/run_amp_track_adapter/2026-05-14_15-16-50_track_adapter_81999_stage3_recovery_v11_adaptive_force_kick_20260514_151645_ppo_adaptive_force_kick/model_6500.pt`.

No-push priority grid:
  - output: `logs/fixed_eval_ta_v11_model6500_priority_20260515_023615`;
  - zero command was quiet: actual `vx/vy/wz = 0.003/0.001/-0.003`, fall/contact `0.0`, residual `0.009`;
  - single-axis tracking was much better than older adapter branches:
    - `vx=0.2`: actual `0.183`;
    - `vx=0.5`: actual `0.477`;
    - `vx=1.0`: actual `0.947`;
    - `vx=2.0`: actual `1.932`;
    - `vy=1.0`: actual `0.913`;
    - `yaw=1.5`: actual `1.341`;
  - issue: strict no-push `vx=1.5` produced one fall/contact out of 64 envs, `ever_fall/contact=0.015625`.

Force-kick grid:
  - output: `logs/fixed_eval_ta_v11_model6500_forcekick_20260515_024913`;
  - configuration: 500N, 0.10s, 1.0s interval, strict contact, 64 envs;
  - result was not acceptable for the target "recover walking after a human kick":
    - every command had `ever_fall_rate` around `0.99-1.00`;
    - every command had `ever_base_contact_rate` around `0.90-0.97`;
    - residual opened to about `0.46-0.70`, but the policy did not recover.

Decision:
  - current v11 training was stopped by user-specified criterion because a real correction target was found;
  - latest regular checkpoint at stop time: `model_6725.pt`;
  - v11 is not promoted, despite good no-push tracking, because dense physical kick recovery fails.

v12 correction:
  - added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v12_dense_force_kick.sh`;
  - resumes from v11 `model_6500.pt` to preserve the good no-push tracking behavior;
  - increases recovery-only adapter authority with `residual_scale=0.22` while keeping stable gate tight;
  - shifts training to dense finite-duration physical kicks: `220-420N -> 500-760N`, interval `2.6-4.2s -> 1.0-1.8s`, probability `0.45 -> 0.95`;
  - raises survival/upright/base-height/angvel/foot-placement recovery weights;
  - keeps root-velocity pushes secondary and milder;
  - allows a slightly longer training contact grace window, `0.18s`;
  - syntax checks passed before launch.

## 2026-05-15 03:08 UTC - v12 failed immediately; v12b launched

v12 dense-force attempt:
  - tmux session: `track_adapter_v12_dense_force_kick_20260515_030328`;
  - result: stopped manually after the first monitored iterations;
  - reason: initial curriculum was too hard. It produced `Episode_Termination/base_contact=0.9288-0.9421`, episode length around `267`, and `adapter_scaled_residual_action_l2=1.3782`;
  - interpretation: this is not a useful learning signal; it is immediate collapse.

v12b correction:
  - added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v12b_ramped_force_kick.sh`;
  - still resumes from v11 `model_6500.pt`;
  - reduces initial physical kick to `60-180N`, long interval `6-9s`, probability `0.18`;
  - ramps toward dense kick target `500-680N`, interval `1.2-2.2s`, probability `0.86`;
  - lowers initial adaptive alpha to `0.08`, with min `0.04`;
  - keeps residual scale moderate at `0.18` and enables early safety gates after `50` relative iterations.

v12b active run:
  - tmux session: `track_adapter_v12b_ramped_force_kick_20260515_030711`;
  - run id: `track_adapter_81999_stage3_recovery_v12b_ramped_force_kick_20260515_030711`;
  - log: `logs/track_adapter_81999_stage3_recovery_v12b_ramped_force_kick_20260515_030711.log`;
  - iteration `6524/12500` snapshot: reward `480.06`, episode length `970.13`, `Episode_Termination/base_contact=0.1272`, `error_vel_xy=0.1538`, `error_vel_yaw=0.3727`, `adapter_scaled_residual_action_l2=0.1064`, `wm=0.1562`, `wm_valid_fraction=0.7822`;
  - interpretation: v12b did not suffer the v12 immediate-collapse failure. Continue monitoring; fixed force-kick eval should be repeated after several hundred to a thousand v12b iterations, not immediately.

## 2026-05-15 05:42 UTC - v12b status at iteration 7825

Active run:
  - tmux session: `track_adapter_v12b_ramped_force_kick_20260515_030711`;
  - latest checkpoint observed: `model_7825.pt`;
  - GPU memory: about `16.5 / 32.6 GB`.

Latest snapshot:
  - iteration: `7825/12500`;
  - mean reward: `472.94`;
  - mean episode length: `923.61`;
  - `Episode_Termination/base_contact`: `0.1904`;
  - `Metrics/base_velocity/error_vel_xy`: `0.2904`;
  - `Metrics/base_velocity/error_vel_yaw`: `0.3742`;
  - `adapter_scaled_residual_action_l2`: `0.2967`;
  - `adapter_ungated_scaled_residual_action_l2`: `0.6009`;
  - `wm`: `0.1577`;
  - `wm_valid_fraction`: `0.7969`.

Interpretation:
  - v12b remains running and did not collapse like v12;
  - base contact is higher than v11, as expected under denser force-kick exposure, but still below the v12b safety threshold `0.30`;
  - episode length remains above `900`, and reward is around the v11/v12b healthy range;
  - this branch now needs fixed force-kick evaluation after more adaptation, likely around `model_8000` or `model_8500`, to see whether dense-kick survival actually improves.

## 2026-05-15 06:10 UTC - v12b reached iteration 8000; fixed kick eval started

Active run:
  - tmux session: `track_adapter_v12b_ramped_force_kick_20260515_030711`;
  - checkpoint: `logs/rsl_rl/run_amp_track_adapter/2026-05-15_03-07-15_track_adapter_81999_stage3_recovery_v12b_ramped_force_kick_20260515_030711_ppo_ramped_force_kick/model_8000.pt`;
  - latest observed training snapshot around `8018-8020` remains alive, with GPU memory around `22.6 / 32.6 GB` while a parallel eval is running.

Training interpretation:
  - v12b is not in the immediate-collapse regime seen in v12;
  - current train-time `Episode_Termination/base_contact` is around `0.20-0.22`, which is still below the relative safety stop threshold but too high to promote without fixed eval;
  - `error_vel_xy` is around `0.28-0.29`, `error_vel_yaw` around `0.37-0.41`;
  - `adapter_scaled_residual_action_l2` is around `0.29-0.32`, meaning the recovery gate is actively using the adapter but not fully saturated;
  - `wm` remains finite around `0.156-0.159`, and `wm_valid_fraction` stays around `0.79`.

Fixed eval:
  - tmux session: `eval_v12b_8000_fk`;
  - output directory: `logs/fixed_eval_ta_v12b_model8000_forcekick_20260515_060610`;
  - configuration: 32 envs, 500N physical force-kick, 0.10s pulse, 1.0s interval, commands `[zero, vx=0.2/0.5/1.0/1.5/2.0, vy=1.0, yaw=1.5]`;
  - decision criterion: continue v12b if fixed kick `ever_fall/base_contact` clearly improves over v11 while no-push tracking remains recoverable; stop and redesign if 500N still produces broad near-certain failure.

Result:
  - evaluation completed and showed broad failure under the strict repeated-kick test;
  - `ever_fall_rate_mean` was `0.9905-1.0000` across all tested commands;
  - `ever_base_contact_rate_mean` was `0.9060-0.9653`;
  - stepwise `fall_rate_mean` was about `0.0122-0.0132`, enough to make almost every env fail over the rollout;
  - residual opened substantially, `residual_norm_mean=0.510-0.769`, so the issue is not that the adapter gate stayed closed;
  - tracking during repeated 500N kicks was poor, e.g. `vx=2.0` produced actual `vx=0.964` with `abs_vx_error=1.180`.

Decision:
  - stopped tmux session `track_adapter_v12b_ramped_force_kick_20260515_030711`;
  - latest regular checkpoint before stop: `model_8050.pt`;
  - v12b is not promoted. It is stable enough to train, but it has not learned the intended physical kick recovery behavior;
  - next branch should avoid simply increasing force density. The failure suggests the task is becoming "continuous repeated hits" rather than "single strong disturbance followed by recovery". v13 should emphasize recoverable strong single/low-frequency kicks, delayed return-to-command, stricter anti-fall/contact reward, and higher recovery-only adapter authority.

## 2026-05-15 06:32 UTC - v13 single-kick recovery branch prepared

Code changes:
  - added `mdp.push_recovery_delayed_velocity_track_exp`, which rewards velocity tracking only after a configurable post-kick delay;
  - added `mdp.recovery_no_base_contact`, a recovery-group reward for avoiding trunk/body contact during recovery windows;
  - wired both rewards into the Track Adapter env config behind env vars:
    - `BOOSTER_TRACK_ADAPTER_PUSH_RECOVERY_DELAYED_TRACK_WEIGHT`;
    - `BOOSTER_TRACK_ADAPTER_PUSH_RECOVERY_DELAYED_MIN_S`;
    - `BOOSTER_TRACK_ADAPTER_PUSH_RECOVERY_DELAYED_MAX_S`;
    - `BOOSTER_TRACK_ADAPTER_RECOVERY_NO_BASE_CONTACT_WEIGHT`.

v13 training script:
  - added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v13_single_kick_recovery.sh`;
  - resumes from the good no-push v11 checkpoint `model_6500.pt`, not v12b, to avoid inheriting dense-contact habits;
  - reuses the validated Stage3 world-model checkpoint;
  - changes force curriculum to recoverable strong single/low-frequency kicks:
    - `90-220N -> 480-640N`;
    - `0.05-0.09s -> 0.08-0.12s`;
    - interval `8-12s -> 3.2-5.0s`;
    - probability `0.22 -> 0.72`;
  - increases recovery-only adapter authority with `residual_scale=0.26`, push residual gate `1.0`, and tight stable gate `0.006`;
  - keeps zero-command/stable intervention strongly suppressed;
  - sets immediate push tracking weight to `0.0` and uses delayed tracking weight `1.60` after `0.42s`;
  - strengthens no-fall/no-contact/upright/height/angvel/foot-placement recovery rewards.

Verification:
  - `bash -n scripts/rsl_rl/run_track_adapter_stage3_recovery_v13_single_kick_recovery.sh` passed;
  - `python -m py_compile` passed for `mdp/rewards.py` and `run_amp/env_cfg.py`.

Initial v13 launch:
  - tmux session: `track_adapter_v13_single_kick_recovery_20260515_062004`;
  - stopped manually after early monitoring;
  - reason: v13 opened the layer-wise adapter too aggressively immediately after resume. At iteration `6510`, `adapter_scaled_residual_action_l2=1.6201`, `adapter_residual_action_gate=0.7455`, and `Episode_Termination/base_contact=0.0486`;
  - interpretation: the new delayed-recovery rewards are active, but `residual_scale=0.26` plus full push gate is too much for a v11 resume checkpoint. Continuing would risk learning another over-intervention branch.

v13b correction:
  - added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v13b_single_kick_recovery.sh`;
  - keeps v13 delayed-return/no-contact reward structure;
  - reduces recovery authority to `residual_scale=0.17`, push residual gate `0.94`, push recovery gate `0.94`, and recovery activation target `0.035`;
  - softens initial force exposure to `70-170N`, final `460-600N`, and final interval `4-6s`;
  - keeps stable gate tight at `0.006`;
  - syntax and `py_compile` checks passed.

v13b launch:
  - tmux session: `track_adapter_v13b_single_kick_recovery_20260515_062314`;
  - run id: `track_adapter_81999_stage3_recovery_v13b_single_kick_recovery_20260515_062314`;
  - log: `logs/track_adapter_81999_stage3_recovery_v13b_single_kick_recovery_20260515_062314.log`;
  - initial checkpoint resume: v11 `model_6500.pt`, optimizer not loaded.

Early v13b status:
  - iteration `6500`: `adapter_scaled_residual_action_l2=0.1742`, `Episode_Termination/base_contact=0.0000`;
  - iteration `6511`: `adapter_scaled_residual_action_l2=0.5114`, `Episode_Termination/base_contact=0.0011`, `error_vel_xy=0.2021`, `error_vel_yaw=0.3470`, `wm=0.1524`, `wm_valid_fraction=0.8048`;
  - interpretation: v13b removed v13's immediate residual explosion while still opening the adapter during push-active windows. Continue monitoring before fixed evaluation.

## 2026-05-15 06:50 UTC - v13b running, iteration 6742

Active run:
  - tmux session: `track_adapter_v13b_single_kick_recovery_20260515_062314`;
  - latest checkpoint observed: `model_6725.pt`;
  - GPU memory: about `16.5 / 32.6 GB`.

Latest snapshot:
  - iteration: `6742/13000`;
  - mean reward: `544.40`;
  - mean episode length: `979.09`;
  - `Episode_Termination/base_contact`: `0.0515`;
  - `Metrics/base_velocity/error_vel_xy`: `0.2628`;
  - `Metrics/base_velocity/error_vel_yaw`: `0.3327`;
  - `adapter_residual_action_gate`: `0.4556`;
  - `adapter_scaled_residual_action_l2`: `0.2770`;
  - `adapter_ungated_scaled_residual_action_l2`: `0.7567`;
  - `wm`: `0.1533`;
  - `wm_valid_fraction`: `0.7970`;
  - `failure_push_active`: `0.4351`;
  - `failure_push_mining_recorded`: `0.0852`.

Interpretation:
  - v13b is alive and substantially healthier than the first v13 launch;
  - residual usage is active but not exploding;
  - world-model replay remains stable;
  - train-time base contact is not clean yet (`~0.05`), so do not promote or run a strict 500N repeated-kick judgment yet;
  - continue training unless base contact trends upward or residual L2 approaches the configured stop gate.

## 2026-05-15 07:13 UTC - v13b base-contact trend check

Active run:
  - tmux session: `track_adapter_v13b_single_kick_recovery_20260515_062314`;
  - latest checkpoint observed: `model_6925.pt`.

Trend:
  - `Episode_Termination/base_contact` rose from about `0.0515` at iteration `6742` to the `0.07-0.08` band by iterations `6850-6930`;
  - recent samples are not monotonically increasing: around `6887` it reached `0.0795`, then stayed mostly around `0.0676-0.0771`;
  - latest parsed sample near iteration `6930`: `base_contact=0.0727`, `adapter_scaled_residual_action_l2=0.2950`;
  - near iterations `6926-6927`, residual temporarily rose to `0.4360-0.4654`, then dropped again.

Interpretation:
  - user concern is valid: v13b has more base contact than the early `6500-6742` window;
  - it is not yet an obvious runaway, but it is no longer clean enough to promote without fixed eval;
  - continue only while it stays below the configured safety region and plan a fixed eval around `model_7000`.

## 2026-05-15 07:34 UTC - pure horizontal force sweep started

Motivation:
  - the previous physical kick included yaw disturbance through explicit yaw torque and off-center force application;
  - K1 mass is present in the asset files, with total mass about `19.666 kg` and Trunk mass `6.5 kg`;
  - a `500N x 0.10s` kick is therefore roughly `50 Ns`, or about `2.54 m/s` whole-body impulse equivalent before contacts, so it is a severe disturbance for this robot.

Pure horizontal sweep:
  - added `scripts/rsl_rl/eval_track_adapter_pure_horizontal_force_sweep.sh`;
  - tmux session: `pure_horizontal_force_sweep_v13b_20260515_073350`;
  - checkpoint: v13b `model_7050.pt`;
  - output: `logs/pure_horizontal_force_sweep_v13b_model7050_20260515_073350`;
  - force levels: `150, 250, 350, 450, 550N`;
  - pulse duration: `0.10s`;
  - root-velocity push disabled;
  - yaw torque disabled;
  - force application offset disabled;
  - vertical force disabled;
  - yaw-equivalent mining disabled.

## 2026-05-15 07:49 UTC - pure horizontal force sweep result

Evaluation condition:
  - checkpoint: v13b `model_7050.pt`;
  - force direction: horizontal XY random, not strict body-y-only;
  - no yaw torque, no off-center application offset, no vertical force, no root-velocity push;
  - force duration: `0.10s`;
  - force interval: `3.0s`, so the `320`-step eval includes repeated pulses rather than a single isolated kick;
  - commands: zero, `vx=0.5`, `vx=1.0`, `vx=2.0`, `vy=1.0`, `yaw=1.5`.

Summary:

| force | worst ever-fall | worst ever-base-contact | zero fall/contact | vx0.5 fall | vx1.0 fall | vx2.0 fall | vy1.0 fall | yaw1.5 fall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150N | 0.372 | 0.339 | 0.000 / 0.000 | 0.051 | 0.138 | 0.372 | 0.107 | 0.093 |
| 250N | 0.573 | 0.467 | 0.435 / 0.322 | 0.573 | 0.558 | 0.546 | 0.511 | 0.372 |
| 350N | 0.676 | 0.611 | 0.642 / 0.518 | 0.642 | 0.676 | 0.624 | 0.608 | 0.647 |
| 450N | 0.702 | 0.554 | 0.676 / 0.504 | 0.661 | 0.651 | 0.702 | 0.671 | 0.648 |
| 550N | 0.682 | 0.615 | 0.664 / 0.478 | 0.640 | 0.622 | 0.669 | 0.682 | 0.635 |

Interpretation:
  - current v13b `model_7050.pt` is only clean at `150N x 0.10s` for zero command;
  - at `150N`, moving commands already show non-trivial failures, especially `vx=2.0` with `ever_fall=0.372` and `ever_base_contact=0.339`;
  - at `250N+`, even zero command has large fall/contact, so this checkpoint is not yet at the desired human-kick recovery level;
  - because this eval uses repeated pulses every `3.0s`, a single-isolated-kick envelope may be higher, but repeated-kick robustness is currently far below the target.

Active v13b training status at the same check:
  - latest checkpoint observed: `model_7150.pt`;
  - recent train-time `Episode_Termination/base_contact=0.0890`;
  - recent `adapter_scaled_residual_action_l2=0.4405`;
  - recent `error_vel_xy=0.2595`, `error_vel_yaw=0.3470`;
  - interpretation: the branch is still learning/intervening, but train-time contact remains too high for promotion.

## 2026-05-15 08:00 UTC - recovery failure diagnosis

Latest v13b status:
  - active tmux session: `track_adapter_v13b_single_kick_recovery_20260515_062314`;
  - latest checkpoint observed: `model_7275.pt`;
  - latest parsed iteration: `7281/13000`;
  - recent `Episode_Termination/base_contact`: about `0.08`;
  - recent `failure_push_active`: about `0.40-0.52`;
  - recent `adapter_residual_action_gate`: about `0.46-0.51`;
  - recent `adapter_scaled_residual_action_l2`: about `0.36-0.42`;
  - recent `adapter_ungated_scaled_residual_action_l2`: about `1.00`;
  - `wm_valid_fraction`: about `0.794-0.797`, so the world-model replay update is numerically stable.

Diagnosis:
  - the adapter is not failing because the gate is fully closed; residual intervention is active during push windows;
  - it is failing because the learned intervention is not yet a coordinated capture-step/recovery behavior;
  - the current reward mix strongly rewards being upright/alive/near nominal height, but base contact is still tolerated enough that PPO can keep a policy with about `8%` train-time base-contact termination;
  - v13b's physical-force curriculum still exposes forces that are beyond the current learned envelope, so the replay/rollout stream contains many failed recovery trajectories instead of mostly successful recoveries to reinforce;
  - the reused Stage3 world-model/history encoder remains stable, but it was not rebuilt around strong physical force transitions, so the dynamics latent may not encode the fast kick impulse and capture-step requirement well enough;
  - the fixed pure-horizontal sweep shows the practical envelope today: clean only for `150N x 0.10s` at zero command, not for moving commands or `250N+`.

Missing pieces for a real recovery branch:
  - competence-gated force curriculum that only increases force after fixed single-kick pass rates are clean;
  - explicit single-kick eval gates before repeated-kick eval gates;
  - stronger hard failure treatment for base contact after a short grace period, not only a positive no-contact reward;
  - more direct capture-step rewards: capture-point foot placement, directional step into the impulse, stance width in body frame, swing-foot clearance, and angular-momentum damping;
  - push-heavy Stage3 replay/pretrain with physical force transitions and mass/friction/CoM variation;
  - recovery-only adapter authority can probably be higher, but only after the disturbance detector opens it, with stable/no-push residual remaining tightly suppressed.

## 2026-05-15 08:25 UTC - v14 capture-step implementation

Implemented v14 recovery changes:
  - `apply_external_wrench_pulse` now supports `max_pulses_per_episode`, enabling true single-kick training/evaluation instead of merely low-frequency repeated kicks;
  - adaptive force curriculum now also accepts push-window failure stats (`adaptive_promote_push_failure`, `adaptive_demote_push_failure`, `adaptive_min_push_active`), not only aggregate base contact and episode length;
  - `TrackAdapterRunner` publishes `push_failure` and `push_active` into `_booster_track_adapter_training_stats` for competence-gated force progression;
  - `OmniVelocityCommand.record_external_push` now stores continuous push delta XY and exposes `failure_push_delta_x/y/norm` metrics for recovery rewards and eval telemetry;
  - added recovery rewards:
    - `recovery_base_contact_penalty`;
    - `recovery_push_capture_point_exp`;
    - `recovery_directional_step_exp`;
    - `recovery_body_frame_stance_width`;
  - wired the new rewards and force curriculum parameters in Track Adapter env config;
  - added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v14_capture_step.sh`;
  - added `scripts/rsl_rl/eval_track_adapter_v14_single_kick_gate.sh`.

v14 training design:
  - frozen base: AMP axis-v2 `model_81999.pt`;
  - PPO resume anchor: v11 `model_6500.pt`, not v13b;
  - Stage3 is rebuilt (`BOOSTER_TRACK_ADAPTER_SKIP_STAGE3=0`) under push-heavy/single-kick collection;
  - physical force is horizontal-only by default: no yaw torque, no off-center application, no vertical force;
  - physical force is limited to one pulse per episode;
  - initial force range: `60-150N`, final `420-650N`, with adaptive alpha initially `0.03`;
  - PPO env count: `3072`, intended to use the RTX 5090 headroom.

Verification before launch:
  - `bash -n` passed for v14 train/eval scripts;
  - `python -m py_compile` passed for modified events, commands, rewards, env config, runner, and eval script;
  - direct import without Isaac `SimulationApp` still fails as expected in this environment, so runtime validation will be the tmux Stage3/PPO launch.

Launch:
  - stopped v13b tmux session `track_adapter_v13b_single_kick_recovery_20260515_062314`;
  - started tmux session `track_adapter_v14_capture_step_20260515_081803`;
  - run id: `track_adapter_81999_stage3_recovery_v14_capture_step_20260515_081803`;
  - log: `logs/track_adapter_81999_stage3_recovery_v14_capture_step_20260515_081803.log`;
  - Stage3 pretrain is running at `800` iterations;
  - initial Stage3 readings:
    - iteration `0`: `wm=0.6984`, `wm_valid_fraction=0.9258`;
    - iteration `1`: `wm=0.6762`, `wm_valid_fraction=0.9254`;
    - iteration `12`: `wm=0.6658`, `wm_valid_fraction=0.9254`;
  - GPU memory observed after launch: about `18.7 / 32.6 GB`.

## 2026-05-15 09:52 UTC - current best policy status

Current v14 status:
  - v14 is still in Stage3 world-model pretrain, not PPO policy training yet;
  - latest observed Stage3 iteration: around `659/800`;
  - `wm` has improved to about `0.31`, with `wm_valid_fraction` mostly around `0.90-0.92`;
  - therefore there is no v14 policy checkpoint to promote yet.

Best policy by use case:
  - best current Track Adapter promotion/deployment candidate: v9 layer-wise `model_4999.pt`;
  - path: `logs/rsl_rl/run_amp_track_adapter/2026-05-13_16-38-08_track_adapter_81999_stage3_recovery_v9_layerwise_20260513_163804_ppo/model_4999.pt`;
  - reason: fixed eval has no-push max fall/base-contact `0.0/0.0`, good single-axis tracking, near-static zero command, and interval-push average fall/base-contact `0.0031/0.0027`;
  - best PPO resume anchor for new recovery branches: v11 `model_6500.pt`;
  - path: `logs/rsl_rl/run_amp_track_adapter/2026-05-14_15-16-50_track_adapter_81999_stage3_recovery_v11_adaptive_force_kick_20260514_151645_ppo_adaptive_force_kick/model_6500.pt`;
  - reason: no-push tracking is very strong and it is the chosen anchor for v14, but its 500N repeated physical-force eval failed broadly, so it is not a deployment/promotion checkpoint.

## 2026-05-15 10:03 UTC - robust-priority policy ranking

Current v14 status:
  - v14 is still Stage3-only at the latest check, around iteration `703/800`;
  - latest observed Stage3 values are `wm` about `0.296`, `wm_valid_fraction` about `0.907`;
  - no v14 PPO checkpoint exists yet, so v14 cannot be ranked as a policy candidate yet.

Robustness-priority checkpoint decision:
  - best current robust/deployable Track Adapter checkpoint remains v9 layer-wise `model_4999.pt`;
  - path: `logs/rsl_rl/run_amp_track_adapter/2026-05-13_16-38-08_track_adapter_81999_stage3_recovery_v9_layerwise_20260513_163804_ppo/model_4999.pt`;
  - reason: among evaluated checkpoints, it has the best balance of no-push cleanliness and push recovery: no-push max fall/base-contact `0.0/0.0`, interval root-velocity push average fall/base-contact `0.0031/0.0027`, and worst interval-push fall/base-contact `0.0326/0.0326` at `vx=2.0`;
  - v11 `model_6500.pt` remains a training/resume anchor, not the robust deployment choice, because its repeated 500N physical-force evaluation failed broadly despite strong no-push tracking;
  - v12b and v13b are not promoted because their physical-force evaluations still show high fall/base-contact rates;
  - if "robust" means human-kick-like physical force recovery, no current checkpoint has reached the target yet. v14 is the active attempt to close that gap.

## 2026-05-15 10:20 UTC - comparable 100N physical-force evaluation started

Reason:
  - the previous robust-priority ranking mixed v9 root-velocity-push evidence with later physical-force-kick evidence;
  - to decide whether the stronger-push branches are actually better around the current survivable envelope, all candidate policies need the same physical-force test.

Evaluation protocol:
  - tmux session: `force100_compare_20260515_102011`;
  - log: `logs/force100_compare_20260515_102011.log`;
  - output root: `logs/force100_compare_20260515_102011`;
  - pure horizontal physical force only: no root-velocity push, no yaw torque, no off-center force offset, no vertical force;
  - force levels: `100N` and `150N`;
  - duration: `0.10s`;
  - interval: `3.0s`;
  - envs: `32`;
  - steps/settle: `320/100`;
  - commands: zero, `vx=0.2/0.5/1.0/1.5/2.0`, `vy=0.4/1.0`, `yaw=1.5`.

Checkpoints:
  - v9 layer-wise `model_4999.pt`;
  - v10b `model_750.pt`;
  - v11 `model_6500.pt`;
  - v12b `model_8050.pt`;
  - v13b `model_7400.pt`.

## 2026-05-15 10:38 UTC - comparable 100N physical-force evaluation result

Execution:
  - first launch attempted `100N` and `150N`, but was narrowed to `100N` only after v9 `F100` completed because concurrent v14 training made the full sweep too slow;
  - completed output root: `logs/force100_compare_20260515_102112`;
  - resume log: `logs/force100_compare_f100_resume_20260515_102610.log`;
  - protocol: pure horizontal physical force, `100N x 0.10s`, `3.0s` interval, no yaw torque, no force offset, no vertical force, no root-velocity push, `32` envs, `320/100` steps/settle, nine command grid.

Summary table:

| checkpoint | avg ever fall/contact | worst ever fall/contact | worst command | avg abs err vx/vy/wz | avg residual |
|---|---:|---:|---|---:|---:|
| v9 `model_4999.pt` | `0.3419 / 0.2773` | `0.9189 / 0.8439` | `vy=0.4` | `0.1289 / 0.1276 / 0.2120` | `0.4658` |
| v10b `model_750.pt` | `0.0072 / 0.0000` | `0.0330 / 0.0000` | `vx=1.0` | `0.1182 / 0.1390 / 0.1974` | `0.3335` |
| v11 `model_6500.pt` | `0.0039 / 0.0039` | `0.0176 / 0.0176` | `yaw=1.5` | `0.1261 / 0.1357 / 0.2110` | `0.6255` |
| v12b `model_8050.pt` | `0.0058 / 0.0040` | `0.0185 / 0.0185` | `vy=0.4` | `0.1476 / 0.1347 / 0.2085` | `0.5628` |
| v13b `model_7400.pt` | `0.0232 / 0.0161` | `0.0814 / 0.0791` | `vx=2.0` | `0.1247 / 0.1416 / 0.2204` | `0.7280` |

Interpretation:
  - the user's objection was correct: the earlier robust-priority statement over-weighted v9's root-velocity-push evidence and under-tested v9 on the physical-force regime;
  - under comparable `100N` physical force, v9 is not the best robust policy and is clearly worse than the later force-kick branches;
  - v11 `model_6500.pt` is the best balanced candidate on this `100N` physical-force test if fall/contact are primary and no-push tracking remains important;
  - v10b `model_750.pt` is also strong and has zero measured base-contact in this test, but its worst fall rate is slightly higher than v11;
  - v12b is close to v11 on fall/contact but has worse average `vx` tracking;
  - v13b `model_7400.pt` regresses relative to v10b/v11/v12b at `100N`, especially at `vx=2.0` and `yaw=1.5`;
  - v9 remains useful as a root-velocity-push comparison point, but should not be called the best robust policy for physical-force push recovery.

## 2026-05-15 11:11 UTC - v11 Track Adapter C++ ONNX deployment support

User request:
  - deploy the current robust-priority v11 Track Adapter checkpoint `model_6500.pt` through C++ ONNX inference in `booster_k1_locomotion` / `nomadz_deploy`;
  - checkpoint path:
    `logs/rsl_rl/run_amp_track_adapter/2026-05-14_15-16-50_track_adapter_81999_stage3_recovery_v11_adaptive_force_kick_20260514_151645_ppo_adaptive_force_kick/model_6500.pt`.

Implemented:
  - added layer-wise Track Adapter support to the deploy runtime used by ONNX export;
  - exported v11 as:
    `/workspace/booster_k1_locomotion/assets/track_adapter_v11_model_6500.onnx`;
  - ONNX ABI:
    `obs[1,75]`, `history[1,79,97]`, `residual_gate[1,1]`, `reset[1,1]`
    -> `action[1,22]`, `next_history[1,79,97]`;
  - baked checkpoint observation normalization into the ONNX graph;
  - added reset-aware history handling and explicit residual gate input;
  - updated `amp_running_policy_node_cpp` to auto-detect legacy 2-input TrackAdapter ONNX and new 4-input gate/reset ONNX;
  - added C++ launch parameters:
    `track_adapter_gate_mode`, `track_adapter_residual_gate`, `track_adapter_stable_gate`;
  - changed C++ launch defaults to v11 ONNX with `residual_scale=0.16` and `residual_gate=1.0`;
  - updated C++ yaw command limit to `|vyaw| <= 1.5`;
  - updated the Python raw `.pt` TrackAdapter runtime in `nomadz_deploy` to load v11 layer-wise checkpoints with `residual_scale=0.16`.

Verification:
  - ONNX export numerical check passed:
    `action max_abs_err=3.8147e-06`, `history max_abs_err=3.8147e-06`;
  - ONNX metadata checked:
    IR `8`, opset `17`, inputs `obs/history/residual_gate/reset`, outputs `action/next_history`;
  - Python syntax checks passed for the modified deploy/export/runtime files;
  - raw v11 runtime smoke checks passed in both `booster_k1_locomotion` and `nomadz_deploy`;
  - C++ dependency setup completed for MuJoCo, ONNX Runtime, Booster SDK, and GLFW under `/workspace/booster_k1_locomotion/third_party`;
  - full C++ `colcon build` could not be run in this container because ROS2/colcon are not installed here.

Operational note:
  - `nomadz_deploy` still has no native C++ inference node. The C++ ONNX path is now `booster_k1_locomotion`;
  - `nomadz_deploy` is kept compatible for Python/raw-checkpoint inference of the new layer-wise adapter.

## 2026-05-15 12:24 UTC - nomadz_deploy Track Adapter v11 ONNX runtime support

User request:
  - apply the v11 Track Adapter ONNX/gate support to `nomadz_deploy` as well.

Implemented:
  - added `tasks/beyond_mimic/track_adapter_onnx_runtime.py`, an ONNXRuntime-backed Track Adapter runtime with external history state;
  - supported the same deploy ABI as the C++ runner:
    `obs[1,75]`, `history[1,79,97]`, `residual_gate[1,1]`, `reset[1,1]`
    -> `action[1,22]`, `next_history[1,79,97]`;
  - updated `TrackAdapterVelocityPolicy` so `checkpoint_path` ending in `.onnx` uses ONNXRuntime, while raw `.pt` checkpoints still use the existing raw checkpoint runtime;
  - added `track_adapter_backend`, `track_adapter_gate_mode`, `track_adapter_residual_gate`, and `track_adapter_stable_gate` to the nomadz Track Adapter policy config;
  - changed `k1_run_amp_track_adapter_latest` to point to `/workspace/booster_k1_locomotion/assets/track_adapter_v11_model_6500.onnx`;
  - added explicit task aliases:
    `k1_run_amp_track_adapter_v11_onnx` and `k1_run_amp_track_adapter_model1100`;
  - added deploy/sim script CLI overrides for Track Adapter gate mode and gate values;
  - added `onnxruntime` to `nomadz_deploy/requirements.txt`.

Verification:
  - Python syntax checks passed for modified nomadz files;
  - direct ONNX runtime smoke produced `action (1,22)` and kept `history (1,79,97)`;
  - raw `.pt` v11 runtime with explicit `residual_gate` still works;
  - a lightweight fake-controller policy smoke using `k1_run_amp_track_adapter_latest` produced a finite `(22,)` target vector;
  - `scripts/deploy.py --list` shows the new Track Adapter tasks.

Operational note:
  - `nomadz_deploy` still runs as a Python controller process, but v11 inference now executes through ONNXRuntime's backend instead of PyTorch when using the `.onnx` task/checkpoint.

## 2026-05-15 15:47 UTC - nomadz sim2sim_booster_sdk_mjviser v11 ONNX support

User request:
  - make the v11 Track Adapter ONNX runnable through `nomadz_deploy/scripts/sim2sim_booster_sdk_mjviser.py`.

Implemented:
  - added `--track-adapter-backend {auto,onnx,raw}` to `sim2sim_booster_sdk_mjviser.py`;
  - kept `--track-adapter-gate-mode`, `--track-adapter-residual-gate`, and `--track-adapter-stable-gate` available in the SDK split-loop runner;
  - added startup logging for Track Adapter backend and gate settings so the chosen ONNX/gate path is visible at launch;
  - added the same backend override to `deploy.py`, `sim2sim_mj_trace.py`, and `sim2sim_mjviser.py` for consistency;
  - documented the `sim2sim_booster_sdk_mjviser.py` command in `nomadz_deploy/README.md`.

Verification:
  - `sim2sim_booster_sdk_mjviser.py --help` shows the Track Adapter backend/gate arguments;
  - short SDK split-loop smoke passed with:
    `--task k1_run_amp_track_adapter_latest --duration 0.2 --port 8100 --vx 0.5 --track-adapter-backend onnx --track-adapter-gate-mode constant --track-adapter-residual-gate 1.0`;
  - smoke reached Viser startup and stopped cleanly:
    `sim_time=0.180s`, `root_z=0.528`, finite final roll/pitch;
  - Python syntax checks passed for modified nomadz deploy scripts and Track Adapter runtime files.

## 2026-05-16 05:05 UTC - v14 fixed evaluation started

User request:
  - v14 training appears complete; run fixed evaluations and summarize the result.

Evaluation plan:
  - compare the previous physical-force robustness anchor v11 `model_6500.pt` against v14 checkpoints
    `model_10000.pt`, `model_11000.pt`, and final `model_13499.pt`;
  - use identical pure-horizontal physical-force settings at `100N` and `150N`, `0.10s` pulse duration,
    `3.0s` interval, no yaw torque, no off-center force, no vertical force;
  - use benchmark commands focused on user priorities:
    `0,0,0`, `vx in {0.2,0.5,1.0,1.5,2.0}`, `vy in {0.4,1.0}`, and `wz=1.5`;
  - also run a no-push fixed velocity grid for the same checkpoints to separate robustness from normal tracking quality.

Runtime:
  - tmux session will be created as `v14_fixed_eval_<timestamp>`;
  - outputs will be written under `logs/v14_fixed_eval_<timestamp>/`.

Results:
  - tmux session: `v14_fixed_eval_20260516_055722`;
  - output root: `logs/v14_fixed_eval_20260516_055722/`;
  - all force and no-push evaluations completed.

Pure-horizontal physical force, `100N/150N x 0.10s`, `3.0s` interval:

| checkpoint | force | avg fall | max fall | avg contact | max contact | worst command |
|---|---:|---:|---:|---:|---:|---|
| v11 `model_6500.pt` | 100N | 0.0039 | 0.0176 | 0.0039 | 0.0176 | `0,0,1.5` |
| v11 `model_6500.pt` | 150N | 0.0875 | 0.2984 | 0.0643 | 0.2214 | `2.0,0,0` |
| v14 `model_10000.pt` | 100N | 0.0092 | 0.0318 | 0.0039 | 0.0185 | `0.5,0,0` |
| v14 `model_10000.pt` | 150N | 0.0727 | 0.2028 | 0.0497 | 0.1274 | `2.0,0,0` |
| v14 `model_11000.pt` | 100N | 0.0032 | 0.0161 | 0.0018 | 0.0161 | `1.0,0,0` |
| v14 `model_11000.pt` | 150N | 0.0613 | 0.1857 | 0.0513 | 0.1670 | `2.0,0,0` |
| v14 `model_13499.pt` | 100N | 0.0059 | 0.0357 | 0.0039 | 0.0180 | `2.0,0,0` |
| v14 `model_13499.pt` | 150N | 0.0798 | 0.2094 | 0.0638 | 0.1562 | `0,0,1.5` |

No-push fixed velocity grid:

| checkpoint | avg fall | avg contact | vx-only `|vx err|` | vx cross `|vy|` | vx cross `|wz|` | vy-only `|vy err|` | zero speed | avg residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v11 `model_6500.pt` | 0.0000 | 0.0000 | 0.095 | 0.076 | 0.155 | 0.118 | 0.002 | 0.031 |
| v14 `model_10000.pt` | 0.0000 | 0.0000 | 0.094 | 0.076 | 0.154 | 0.118 | 0.001 | 0.015 |
| v14 `model_11000.pt` | 0.0000 | 0.0000 | 0.094 | 0.077 | 0.154 | 0.117 | 0.001 | 0.014 |
| v14 `model_13499.pt` | 0.0000 | 0.0000 | 0.095 | 0.076 | 0.154 | 0.118 | 0.001 | 0.013 |

Low-speed straight `vx` details:

| checkpoint | `vx=0.2` actual / `|vy|` / `|wz|` | `vx=0.5` actual / `|vy|` / `|wz|` |
|---|---|---|
| v11 `model_6500.pt` | `0.174 / 0.062 / 0.107` | `0.461 / 0.070 / 0.125` |
| v14 `model_10000.pt` | `0.175 / 0.063 / 0.106` | `0.461 / 0.070 / 0.125` |
| v14 `model_11000.pt` | `0.175 / 0.063 / 0.105` | `0.461 / 0.071 / 0.125` |
| v14 `model_13499.pt` | `0.174 / 0.063 / 0.106` | `0.461 / 0.071 / 0.126` |

Conclusion:
  - v14 improves the `150N` pure-horizontal force envelope relative to v11, but the final checkpoint is not the best;
  - `model_10000.pt` is the most conservative balanced candidate because it has the lowest `150N` average contact and lowest `150N` worst contact;
  - `model_11000.pt` is a close robustness candidate because it has the lowest `150N` average and worst fall, but its worst contact is higher than `model_10000.pt`;
  - normal no-push tracking is essentially unchanged from v11 and remains clean for fall/contact;
  - the low-speed `vx` straightness issue is not materially solved by v14. The adapter residual is very low in no-push, so this looks like a base-policy/command-tracking limitation rather than a v14 recovery branch improvement;
  - no evaluated checkpoint yet satisfies a human-kick-level target. The current evidence supports promoting v14 `model_10000.pt` or `model_11000.pt` only for a moderate `150N x 0.10s` pure-horizontal repeated-push envelope.

## 2026-05-16 11:45 UTC - v15 stronger push-recovery branch prepared

User request:
  - make the Track Adapter stronger against physical push recovery than the v14 `150N` envelope.

Design decision:
  - do not extend the v14 final checkpoint; resume from v14 `model_10000.pt`, which had the best `150N` contact profile;
  - reuse the validated v14 Stage3 world-model pretrain `model_799.pt` and skip Stage3 for this branch;
  - train a competence ladder over pure-horizontal physical trunk force, starting from `110-180N x 0.07-0.10s` and adaptively promoting toward `210-340N x 0.08-0.12s`;
  - keep no yaw torque, no off-center offset, no vertical force, and no root-velocity push in v15, so the policy learns the force-push regime directly;
  - keep velocity tracking as a delayed post-recovery objective, while the first recovery window prioritizes fall/base-contact avoidance, upright posture, height, angular damping, capture/step placement, and push-direction velocity cancellation.

Implemented:
  - added force-push telemetry to `OmniVelocityCommand` metrics:
    `force_push_active`, `force_push_started`, `force_push_curriculum_alpha`,
    active force/duration ranges, and activation probability;
  - wired `apply_external_wrench_pulse` to publish those metrics every interval step;
  - added `recovery_push_velocity_cancel_exp`, which rewards cancelling velocity that continues in the physical push direction during the early recovery window;
  - added the reward term to the Track Adapter K1 env config behind
    `BOOSTER_TRACK_ADAPTER_RECOVERY_PUSH_VELOCITY_CANCEL_*`;
  - added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v15_force_ladder.sh`;
  - added phase-aware residual action gating:
    push recovery reward gates can stay strong for the full push window, but the actual residual action gate now uses raw disturbance recovery plus a push phase floor;
    the floor is near `push_min_gate` for the first `0.85s`, tapers toward `0.35`, and reopens toward `0.93` when the disturbance score remains high.

Verification:
  - `py_compile` passed for the modified runner, config, command, event, reward, and env files;
  - `bash -n` and `git diff --check` passed for the v15 script and touched Python files;
  - checkpoint existence checks passed for base `model_81999.pt`, v14 pretrain `model_799.pt`, and v14 resume `model_10000.pt`;
  - normal 2-iteration PPO smoke passed from the v14 checkpoint with finite WM loss and no startup/runtime errors;
  - force-active smoke passed with artificially short force intervals, confirming `failure_push_active` and recovery rewards become non-zero. That smoke intentionally used unrealistic near-continuous pushes and reached high base-contact rates, so it is a runtime validation, not a policy-quality result.

Next action:
  - launch full v15 PPO in tmux with the standard force ladder intervals and monitor early base-contact, residual usage, WM validity, and force-push curriculum alpha before evaluating fixed checkpoints.

## 2026-05-16 11:53 UTC - v15 first launch stopped and curriculum softened

Observation:
  - the first v15 full PPO launch reached iteration `10007`;
  - force pushes became active and `recovery_*` rewards were non-zero, but `Episode_Termination/base_contact` quickly rose to about `0.14-0.16`;
  - adaptive force alpha had already demoted to its minimum, so the initial ladder was still too hard for useful learning.

Action:
  - stopped the first v15 tmux run before spending a long training window on excessive contact failures;
  - changed the v15 ladder to start inside the current survival envelope:
    initial physical force `85-145N`, duration `0.06-0.10s`, probability `0.46`, equivalent push velocity `0.45`;
  - kept the final target stronger than v14:
    final physical force `220-360N`, duration `0.08-0.12s`, probability `0.86`, equivalent push velocity `1.20`;
  - changed adaptive alpha to start at `0.05`, allow demotion to `0.0`, and promote more gradually.

Rationale:
  - the branch still targets stronger-than-v14 recovery, but the first curriculum rung must produce enough survivable rollouts for the adapter to learn coordinated capture/recovery instead of only collecting contact failures.

## 2026-05-16 11:58 UTC - v15b full PPO relaunched

Runtime:
  - tmux session: `track_adapter_v15b_force_ladder_20260516_114435`;
  - run id: `track_adapter_81999_stage3_recovery_v15b_force_ladder_20260516_114435`;
  - log: `logs/track_adapter_81999_stage3_recovery_v15b_force_ladder_20260516_114435.log`;
  - PPO run directory:
    `logs/rsl_rl/run_amp_track_adapter/2026-05-16_11-44-40_track_adapter_81999_stage3_recovery_v15b_force_ladder_20260516_114435_ppo_force_ladder`.

Early status:
  - startup, Stage3 checkpoint validation, and v14 `model_10000.pt` resume all passed;
  - first push-active iterations show force curriculum alpha demoted to `0.0`, active physical force `85-145N`, and activation probability `0.46`;
  - early `Episode_Termination/base_contact` after push activation is about `0.013-0.021`, much healthier than the stopped v15 launch's `0.14-0.16`;
  - `wm_valid_fraction` is about `0.79-0.80`, `wm loss` remains finite, and scaled residual is increasing under push without immediate explosion.

Decision:
  - keep v15b running and monitor whether the adaptive force ladder can raise force strength without pushing base-contact above the safety threshold.

## 2026-05-17 04:55 UTC - v15b training completed

Runtime result:
  - tmux session exited after completing the requested PPO window;
  - latest checkpoint:
    `logs/rsl_rl/run_amp_track_adapter/2026-05-16_11-44-40_track_adapter_81999_stage3_recovery_v15b_force_ladder_20260516_114435_ppo_force_ladder/model_16499.pt`;
  - final log timestamp: `2026-05-17 04:53 UTC`;
  - final logged iteration: `16499/16500`.

Final observed metrics:
  - `Mean reward`: about `903`;
  - `wm loss`: about `0.114`;
  - `wm_valid_fraction`: about `0.785`;
  - `adapter_residual_action_gate`: about `0.165`;
  - `adapter_scaled_residual_action_l2`: about `0.35`;
  - `error_vel_xy`: about `0.258`;
  - `error_vel_yaw`: about `0.307`;
  - `Episode_Termination/base_contact`: about `0.054`.

Important outcome:
  - adaptive force curriculum stayed at `alpha=0.0` at the end;
  - active force range stayed at the softened initial rung `85-145N`;
  - the branch did not promote toward the intended stronger `220-360N` force range;
  - this means v15b completed stably enough to evaluate, but it should not yet be considered a successful stronger-than-v14 push-recovery solution.

Next evaluation needed:
  - compare v15b checkpoints against v14 `model_10000.pt` and `model_11000.pt` on fixed pure-horizontal physical force sweeps at `100N`, `150N`, `175N`, `200N`, and `250N`;
  - include no-push velocity tracking to check whether the higher residual usage degraded normal tracking.

## 2026-05-17 07:20 UTC - v15b fixed evaluation and diagnosis

Evaluation:
  - output root: `logs/v15b_fixed_eval_20260517_063306`;
  - compared v14 `model_10000.pt`, v14 `model_11000.pt`, and v15b final `model_16499.pt`;
  - pure-horizontal physical force sweep: `100, 150, 175, 200, 250N`, `0.10s`, `3.0s` interval, no yaw torque, no off-center offset, no vertical force;
  - commands: zero, `vx={0.2,0.5,1.0,1.5,2.0}`, `vy={0.4,1.0}`, and `wz=1.5`;
  - no-push command-grid evaluation also completed for all three checkpoints;
  - the main comparison uses the existing fixed-eval convention `residual_scale=0.17`. A v15b-only native-scale check with `residual_scale=0.23` was also run.

Pure-horizontal force summary:

| policy | F | avg fall | max fall | avg contact | max contact | worst command |
|---|---:|---:|---:|---:|---:|---|
| v14 `model_10000` | 100 | 0.0085 | 0.0205 | 0.0085 | 0.0205 | `vx=2.0` |
| v14 `model_10000` | 150 | 0.0929 | 0.2749 | 0.0790 | 0.2749 | `vx=2.0` |
| v14 `model_10000` | 175 | 0.1621 | 0.3335 | 0.1327 | 0.2874 | `vx=1.0` |
| v14 `model_10000` | 200 | 0.2700 | 0.3947 | 0.2208 | 0.3214 | `vx=2.0` |
| v14 `model_10000` | 250 | 0.4609 | 0.5719 | 0.3830 | 0.5149 | `vx=2.0` |
| v14 `model_11000` | 100 | 0.0058 | 0.0384 | 0.0035 | 0.0178 | `vx=2.0` |
| v14 `model_11000` | 150 | 0.0813 | 0.2759 | 0.0620 | 0.2210 | `vx=2.0` |
| v14 `model_11000` | 175 | 0.1555 | 0.4241 | 0.1219 | 0.4028 | `vx=2.0` |
| v14 `model_11000` | 200 | 0.2759 | 0.4541 | 0.2145 | 0.3744 | `vx=2.0` |
| v14 `model_11000` | 250 | 0.4575 | 0.6061 | 0.3559 | 0.5469 | `vx=2.0` |
| v15b `model_16499` | 100 | 0.0016 | 0.0148 | 0.0016 | 0.0148 | `vx=2.0` |
| v15b `model_16499` | 150 | 0.0712 | 0.1946 | 0.0484 | 0.1344 | `vx=2.0` |
| v15b `model_16499` | 175 | 0.1794 | 0.4386 | 0.1575 | 0.4004 | `vx=2.0` |
| v15b `model_16499` | 200 | 0.2708 | 0.4741 | 0.2233 | 0.3756 | `vx=2.0` |
| v15b `model_16499` | 250 | 0.4726 | 0.5959 | 0.3699 | 0.4339 | `vx=2.0` |

No-push summary:

| policy | avg fall/contact | vx-only error | vx crosstalk `|vy|/|wz|` | vy-only error | zero speed | avg residual |
|---|---:|---:|---:|---:|---:|---:|
| v14 `model_10000` | 0.0000 / 0.0000 | 0.0569 | 0.0061 / 0.0216 | 0.0904 | 0.0019 | 0.0251 |
| v14 `model_11000` | 0.0000 / 0.0000 | 0.0563 | 0.0061 / 0.0216 | 0.0855 | 0.0034 | 0.0245 |
| v15b `model_16499` | 0.0000 / 0.0010 | 0.0522 | 0.0059 / 0.0229 | 0.0939 | 0.0026 | 0.0207 |

Native-scale v15b check:
  - with `residual_scale=0.23`, v15b became worse, not better;
  - `150N` avg/max fall rose to `0.1155/0.4320`, avg/max contact to `0.0946/0.3899`;
  - `175N`, `200N`, and `250N` were also worse than the fixed-eval `0.17` scale.

Conclusion:
  - v15b did improve the mild envelope: it is better than v14 at `100N` and `150N`, especially contact at `150N`;
  - v15b did not improve the stronger envelope: at `175N+` it is similar to or worse than v14, and `vx=2.0` remains the dominant worst case;
  - no-push tracking stayed mostly intact, so the failure is not broad nominal-locomotion collapse;
  - simply increasing adapter authority is not the answer, because the native `0.23` scale over-intervenes and worsens recovery.

Diagnosis:
  - the training curriculum never promoted: final training logs stayed at adaptive alpha `0.0`, force `85-145N`;
  - therefore the policy was not trained on the `175N+` distribution being evaluated;
  - the residual learned useful mild-force corrections but not a coordinated high-force capture-step controller;
  - the reused v14 world-model pretrain was also mild relative to the target and was not rebuilt for `175-250N+` transitions;
  - the capture/directional stepping rewards remained weak in practice, and the worst cases are high-speed `vx=2.0`, where the base gait is already near its dynamic limit.

Next branch implication:
  - do not extend v15b as-is for human-kick-level recovery;
  - use v15b only as a mild `100-150N` anchor, then run a staged curriculum with explicit fixed-eval gates:
    train `100-150N` until fixed eval passes, resume at `150-175N`, then `175-200N`, instead of relying on a single noisy adaptive alpha;
  - rebuild Stage3/world-model pretrain with `150-250N` physical-force rollouts and measured post-push root velocity/capture features;
  - strengthen high-speed forward-running recovery specifically, because `vx=2.0` is the limiting command across policies.

## 2026-05-17 07:55 UTC - v16 force-mixture implementation

Implemented fixes for the v15b diagnosis:

- added `BOOSTER_TRACK_ADAPTER_FORCE_PUSH_MIXTURE`, parsed as `min:max:weight` components, for physical-force pulse sampling;
- `apply_external_wrench_pulse` now samples force magnitudes from that mixture and reports the active min/max envelope for diagnostics;
- added `BOOSTER_TRACK_ADAPTER_KEEP_PRETRAINED_WM_ON_RESUME=1`, which reapplies the requested Stage 3 history encoder/world model after loading a PPO checkpoint and skips the stale resumed world-model optimizer state;
- added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v16_force_mixture.sh`.

v16 run intent:

- rebuild Stage 3 from scratch with `H=79,N=20`, frozen AMP axis-v2 `model_81999.pt`, and a push-heavy force mixture `60-150N:30%`, `150-220N:40%`, `220-300N:25%`, `300-380N:5%`;
- resume adapter PPO from v15b `model_16499.pt`, but keep the newly rebuilt Stage 3 world model on resume;
- disable the adaptive force curriculum for this branch, because v15b stayed at alpha `0.0`; instead use explicit mixture exposure during PPO;
- keep one physical force pulse per episode, pure horizontal/no yaw torque/no off-center offset, with failure mining enabled;
- monitor whether `force_push_force_max_n` reaches `380N`, `wm_valid_fraction >= 0.70`, and `Episode_Termination/base_contact` stays below the v16 safety gate before trusting later checkpoints.

Launch/restart:

- tmux session: `ta_v16_force_mixture`;
- first run id `track_adapter_81999_stage3_recovery_v16_force_mixture_20260517_081118` reached Stage 3 iteration `3/900` successfully, then was intentionally stopped and restarted after adding the `force_push_applied_force_n` metric so actual applied force can be monitored instead of only the configured force envelope;
- second run id `track_adapter_81999_stage3_recovery_v16_force_mixture_20260517_081352` reached Stage 3 iteration `8/900`, then was intentionally stopped because review caught that `BOOSTER_TRACK_ADAPTER_PUSH=0` would make Stage 3 manifest validation fail after pretrain; v16 now keeps a weak root-velocity proxy push enabled (`push_xy 0.03 -> 0.08`, yaw `0`) while physical force pulses remain dominant;
- v16 safety was tightened after review from `base_contact < 0.13` after relative iteration `450` to `base_contact < 0.11` after relative iteration `350`;
- active run id: `track_adapter_81999_stage3_recovery_v16_force_mixture_20260517_081701`;
- active log: `logs/track_adapter_81999_stage3_recovery_v16_force_mixture_20260517_081701.log`;
- active Stage 3 reached iteration `6/900`; `wm` moved from `0.6861` to about `0.6154`, and `wm_valid_fraction` stayed around `0.915-0.926`;
- log confirms both interval events are active: weak `push_robot` at `(12.0,18.0)s` for manifest coverage and `external_wrench_push` at `0.02s` for physical-force pulses.

## 2026-05-17 13:34 UTC - v16 PPO stopped after health check

Status:

- Stage 3 completed and saved the final pretrain checkpoint `model_899.pt`;
- PPO started in `logs/rsl_rl/run_amp_track_adapter/2026-05-17_10-40-24_track_adapter_81999_stage3_recovery_v16_force_mixture_20260517_081701_ppo_force_mixture`;
- latest saved PPO checkpoint before stop: `model_16675.pt`;
- the tmux session `ta_v16_force_mixture` was stopped manually after the health check.

Reason for stop:

- PPO entered a bad regime immediately after resume: recent `Episode_Termination/base_contact` stayed around `0.46-0.47`;
- `wm_valid_fraction` was healthy around `0.78`, and force/replay metrics were finite, so this was not a WM numerical divergence;
- likely cause is the v16 resume design: it loaded v15b adapter weights but then reapplied a newly trained Stage 3 history encoder/world model. For a layer-wise adapter, changing the history latent distribution under an already-trained adapter can invalidate the adapter input distribution and produce unsafe interventions;
- therefore do not promote or continue the v16 PPO checkpoints from this run.

Next correction:

- keep the v16 force-mixture Stage 3 checkpoint as a candidate pretrain artifact;
- do not reapply a new history encoder under an old layer-wise adapter unless the adapter is also reset/warm-started compatibly;
- the next recovery run should either:
  - resume the full v15b/v14 policy state and only continue WM replay online, or
  - start a fresh/near-zero adapter with the new Stage 3 encoder and use a much milder first PPO force rung.

## 2026-05-17 13:50 UTC - v17 stable force-mixture branch prepared

Implemented the safer continuation path after the v16 stop:

- added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v17_stable_force_mixture.sh`;
- resumes the full v15b PPO checkpoint `model_16499.pt`;
- explicitly sets `BOOSTER_TRACK_ADAPTER_KEEP_PRETRAINED_WM_ON_RESUME=0`, so the old adapter keeps the history encoder/world model distribution it was trained with;
- skips Stage 3 and reuses the validated v14/v15 world-model pretrain path only for the common Stage3 validation path;
- replaces v15b's stuck adaptive curriculum with an explicit moderate force mixture:
  - `80-150N`: 45%;
  - `150-190N`: 45%;
  - `190-230N`: 10%;
- keeps pure-horizontal, one-pulse physical force training and no root-velocity push during PPO;
- reduces residual scale from v15/v16's high-authority values to `0.18`, lowers LR to `2.5e-7`, and tightens early safety:
  - `base_contact < 0.12` after relative iteration `40`;
  - `scaled_residual_action_l2 < 0.75`;
  - patience `3`.

Goal:

- train past the current `150N` envelope without immediately flooding PPO with `250N+` failures;
- first target is improved fixed eval at `175N`, while preserving v15b's no-push tracking.

Launch:

- tmux session: `ta_v17_stable_force`;
- run id: `track_adapter_81999_stage3_recovery_v17_stable_force_mixture_20260517_112808`;
- log: `logs/track_adapter_81999_stage3_recovery_v17_stable_force_mixture_20260517_112808.log`;
- PPO run directory: `logs/rsl_rl/run_amp_track_adapter/2026-05-17_11-28-13_track_adapter_81999_stage3_recovery_v17_stable_force_mixture_20260517_112808_ppo_stable_force_mixture`;
- startup verified: v14 pretrain checkpoint validated, v15b `model_16499.pt` loaded without optimizer, action std overridden to `0.08`, and `external_wrench_push` is active.

## 2026-05-17 14:05 UTC - v17 stopped by safety gate; v18 mild-force continuation

v17 outcome:

- tmux process exited; no training process remains;
- latest saved checkpoint before abort:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-17_11-28-13_track_adapter_81999_stage3_recovery_v17_stable_force_mixture_20260517_112808_ppo_stable_force_mixture/model_16541_aborted.pt`;
- stop reason: safety gate, `Episode_Termination/base_contact` reached `0.1455` after relative iteration `40`, above the configured `0.12` threshold for `3/3` checks;
- `wm_valid_fraction` remained healthy and the residual norm was low, so this was not a numerical/world-model divergence;
- compared with v16's immediate `base_contact ~= 0.46-0.47`, keeping the resumed encoder/world-model distribution fixed was the right correction, but the v17 force mixture (`80-230N`, probability `0.42`, interval `3.2-5.0s`) was still too contact-heavy for a stable continuation run.

Next correction:

- add a v18 mild-force wrapper that resumes the same v15b policy state without changing the encoder/world-model;
- reduce early physical-force exposure to `60-205N`, with most samples below `175N`;
- lower pulse probability to `0.30` and lengthen the interval to `4.2-6.2s`;
- keep a small high-force tail so the replay does not forget `175-200N`, but avoid flooding PPO with failures before gait recovery improves.

Launch:

- tmux session: `ta_v18_mild_force`;
- run id: `track_adapter_81999_stage3_recovery_v18_mild_force_mixture_20260517_113854`;
- log: `logs/track_adapter_81999_stage3_recovery_v18_mild_force_mixture_20260517_113854.log`;
- script: `scripts/rsl_rl/run_track_adapter_stage3_recovery_v18_mild_force_mixture.sh`;
- PPO run directory:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-17_11-38-59_track_adapter_81999_stage3_recovery_v18_mild_force_mixture_20260517_113854_ppo_mild_force_mixture`.

Early health check:

- startup verified: v15b `model_16499.pt` loaded without optimizer, `BOOSTER_TRACK_ADAPTER_KEEP_PRETRAINED_WM_ON_RESUME=0`, action std overridden to `0.07`;
- safety gate passed the first active window: at iteration `16543`, `Episode_Termination/base_contact=0.0387`, below the `0.12` threshold;
- by iteration `16558`, `Episode_Termination/base_contact=0.0409`, `wm_valid_fraction=0.7903`, `wm loss=0.1077`, `adapter_scaled_residual_action_l2=0.0938`, and episode length was about `888`;
- first checkpoint saved after launch: `model_16550.pt`;
- decision: keep v18 running. Do not raise force yet; wait for a longer window and then run fixed `150N/175N` recovery evaluation.

Metric follow-up:

- reviewed the `force_push_active` / `force_push_started` / `force_push_applied_force_n` diagnostics;
- force pulses are active in v18, but the live-pulse metrics often print as `0.0000` because each pulse is only `0.055-0.09s` and is averaged over many envs/timesteps;
- added future-run diagnostics `force_push_active_count`, `force_push_started_count`, and `force_push_applied_force_max_n` so short pulses can be seen more reliably in logs;
- syntax check passed with `python -m py_compile` for `events.py` and `commands.py`;
- this code change does not affect the already-running v18 process until a later restart.

## 2026-05-17 16:00 UTC - v18 status check

Status:

- tmux session `ta_v18_mild_force` is still running;
- latest observed iteration: `17385/19499`;
- latest saved checkpoint: `model_17375.pt`;
- ETA from the trainer: about `5h30m` remaining.

Recent health:

- `Episode_Termination/base_contact` is around `0.053-0.055`, below the safety stop threshold `0.12`;
- `Mean wm loss` is about `0.106-0.108`, and `wm_valid_fraction` is about `0.781-0.787`;
- `adapter_scaled_residual_action_l2` is mostly `0.07-0.10`, with one recent sample around `0.125`;
- velocity errors remain in the rough band `error_vel_xy ~= 0.23-0.26`, `error_vel_yaw ~= 0.27-0.31`;
- force pulse metrics show non-zero applied-force samples, e.g. recent `force_push_applied_force_n=0.7318` and `1.2951` in the averaged log display.

Decision:

- continue v18. It is no longer in the v17-style early failure regime;
- keep watching `base_contact` trend. If it climbs toward `0.08-0.10`, run a fixed evaluation before allowing it to continue much longer; if it remains near `0.05`, evaluate around `model_17500` or `model_17600`.

## 2026-05-17 18:58 UTC - v18 late-run status check

Status:

- tmux session `ta_v18_mild_force` is still running;
- latest observed iteration: `18498/19499`;
- latest saved checkpoint: `model_18475.pt`;
- ETA from the trainer: about `2h38m` remaining.

Recent health:

- `Episode_Termination/base_contact` is back around `0.049-0.050`; the earlier rise into the `0.053-0.055` band did not continue upward;
- safety threshold remains `0.12`, so the run is still below the configured stop condition;
- `Mean wm loss` is about `0.105-0.106`, and `wm_valid_fraction` is about `0.784`;
- `adapter_scaled_residual_action_l2` is around `0.126-0.137` in the latest samples, higher than the early run but still far below the residual safety threshold;
- velocity errors remain stable around `error_vel_xy ~= 0.236-0.247`, `error_vel_yaw ~= 0.291-0.296`;
- force pulse logging shows actual force exposure, including a recent non-zero `force_push_active=0.0042`, `force_push_started=0.0042`, and `force_push_applied_force_n=0.6347`.

Decision:

- continue v18 to the planned end. The late-run signal is stable enough to finish;
- after completion, fixed evaluation should compare `model_18475.pt`, final checkpoint, and the older robust anchors on `100/150/175/200N` pure-horizontal force plus no-push command tracking.

## 2026-05-17 19:33 UTC - v18 force metric and contact caution

Status:

- latest observed iteration: `18718/19499`;
- latest saved checkpoint: `model_18700.pt`;
- ETA from the trainer: about `2h03m` remaining.

Interpretation:

- `force_push_active=0.0000`, `force_push_started=0.0000`, and `force_push_applied_force_n=0.0000` in a single displayed block do not mean the physical force event is disabled;
- nearby iterations show non-zero force metrics, including:
  - iteration `18699`: `force_push_active=0.0098`, `force_push_started=0.0046`, `force_push_applied_force_n=0.8681`;
  - iteration `18701`: `force_push_active=0.0174`, `force_push_applied_force_n=1.8784`;
  - iteration `18707`: `force_push_active=0.0042`, `force_push_started=0.0042`, `force_push_applied_force_n=0.6341`;
  - iteration `18710`: `force_push_active=0.0104`, `force_push_applied_force_n=1.5007`;
- because each force pulse is only `0.055-0.09s`, the printed averaged metrics frequently round to zero.

Concern:

- `Episode_Termination/base_contact` has risen to `0.060-0.062`, up from the earlier `0.049-0.055` band;
- this is still below the safety threshold `0.12`, and `time_out` remains high around `0.938-0.940`, but it is no longer a clean improvement trend;
- decision: keep training for now, but treat `0.07-0.08` sustained base contact as a fixed-evaluation trigger and do not promote the final checkpoint without comparing it against `model_18475.pt` and `model_18700.pt`.

## 2026-05-17 20:31 UTC - v18 status check

Status:

- tmux session `ta_v18_mild_force` is still running;
- latest observed iteration: `18866/19499`;
- latest saved checkpoint: `model_18850.pt`;
- ETA from the trainer: about `1h40m` remaining.

Recent health:

- `Episode_Termination/base_contact` moved back down to about `0.0558-0.0571` after the prior `0.060-0.062` caution window;
- `time_out` remains high around `0.943-0.944`;
- `wm loss` is about `0.1061-0.1062`, and `wm_valid_fraction` is about `0.783-0.784`;
- `adapter_scaled_residual_action_l2` is about `0.115-0.129`;
- velocity errors remain stable around `error_vel_xy ~= 0.231-0.242`, `error_vel_yaw ~= 0.282-0.290`.

Decision:

- continue v18. The contact rise has not continued into the `0.07-0.08` trigger band;
- keep `model_18475.pt`, `model_18700.pt`, `model_18850.pt`, and final checkpoint as fixed-evaluation candidates.

## 2026-05-17 21:29 UTC - v18 auto-eval watcher installed

Implemented automatic post-training evaluation:

- added `scripts/rsl_rl/watch_track_adapter_v18_auto_eval.sh`;
- launched tmux session `ta_v18_auto_eval`;
- output root:
  `logs/track_adapter_81999_stage3_recovery_v18_mild_force_mixture_20260517_113854_auto_eval_20260517_182927`;
- watcher log:
  `logs/track_adapter_81999_stage3_recovery_v18_mild_force_mixture_20260517_113854_auto_eval_20260517_182927/watch.log`.

Watcher behavior:

- waits for the current v18 training PID to exit;
- after checkpoint flush, resolves the latest/final checkpoint automatically;
- evaluates v18 candidates `model_18475.pt`, `model_18700.pt`, `model_18850.pt`, and final;
- also evaluates comparison anchors:
  - v14 `model_10000.pt`;
  - v15b `model_16499.pt`;
- force sweep uses pure-horizontal physical force at `100/150/175/200N`;
- no-push grid covers zero command, vx-only, vy-only, and yaw-only commands;
- writes `summary.json` and `summary.md` after all evaluations finish.

Launch verification:

- tmux `ta_v18_auto_eval` started successfully;
- first watcher line confirmed training is still alive and latest checkpoint is `model_19075.pt`.

## 2026-05-18 00:40 UTC - v18 completed; auto-eval completed

Training outcome:

- v18 completed normally;
- final/latest checkpoint: `model_19498.pt`;
- final train snapshot at iteration `19498/19499`:
  - `Episode_Termination/base_contact`: `0.0487`;
  - `time_out`: `0.9513`;
  - `wm loss`: `0.1069`;
  - `wm_valid_fraction`: `0.7852`;
  - `adapter_scaled_residual_action_l2`: `0.1318`;
  - `error_vel_xy/error_vel_yaw`: `0.2368/0.2907`.

Auto-eval outcome:

- output root:
  `logs/track_adapter_81999_stage3_recovery_v18_mild_force_mixture_20260517_113854_auto_eval_20260517_182927`;
- summary:
  `logs/track_adapter_81999_stage3_recovery_v18_mild_force_mixture_20260517_113854_auto_eval_20260517_182927/summary.md`;
- evaluated candidates:
  - v18 `model_18475.pt`;
  - v18 `model_18700.pt`;
  - v18 `model_18850.pt`;
  - v18 final `model_19498.pt`;
  - v14 `model_10000.pt`;
  - v15b `model_16499.pt`.

Key fixed-eval result:

- v18 did not improve physical-force recovery over the older anchors;
- v14 `model_10000.pt` remains the best aggregate fixed-eval candidate across `100/150/200N`;
- v15b `model_16499.pt` remains slightly better at `175N` in this sweep;
- v18 final regressed relative to v18 mid-run checkpoints and to the anchors.

Selected numbers from the auto-eval summary:

```text
candidate        100N fall/base   150N fall/base   175N fall/base   200N fall/base   no-push fall/base
v14_10000        0.003/0.003      0.081/0.064      0.158/0.127      0.264/0.206      0.0008/0.0009
v15b_16499       0.009/0.007      0.098/0.073      0.148/0.107      0.270/0.222      0.0000/0.0000
v18_18475        0.036/0.026      0.106/0.076      0.182/0.141      0.271/0.216      0.0000/0.0000
v18_18700        0.036/0.024      0.089/0.076      0.166/0.133      0.292/0.229      0.0003/0.0005
v18_18850        0.059/0.040      0.109/0.080      0.177/0.140      0.299/0.228      0.0012/0.0012
v18_19498        0.060/0.054      0.144/0.122      0.187/0.140      0.313/0.249      0.0022/0.0022
```

Interpretation:

- the v18 mild-force run was stable enough to finish, but it did not learn a better recovery strategy;
- residual usage during force eval became much larger than v14/v15b, especially around `100-150N`, while fall/contact did not improve;
- this suggests v18 learned higher intervention, not better recovery;
- do not promote v18 final;
- if using an existing checkpoint now, keep v14 `model_10000.pt` as the aggregate robust baseline and v15b `model_16499.pt` as the `175N` comparison anchor.

## 2026-05-18 00:58 UTC - v19 survive-then-return branch prepared

Implemented the next recovery branch after the v18 fixed-eval regression:

- added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v19_survive_then_return.sh`;
- resumes from v14 `model_10000.pt`, not from v18;
- keeps the v14 Stage3/world-model distribution fixed on resume;
- changes reward priority during push-active recovery:
  - immediate `push_recovery_velocity_track_exp` is disabled with weight `0.0`;
  - delayed command tracking starts at `1.10s` after push;
  - `push_command_gate_min` is reduced to `0.035`, so task/velocity tracking is almost suppressed during early recovery;
  - `recovery_weight` is raised to `7.80`;
  - no-fall, no-base-contact, base-contact penalty, upright, base height, angular-velocity damping, capture point, directional step, support contact, and stance-width rewards are all raised;
  - residual recovery penalty is raised versus v18 so success must come from useful recovery behavior, not just larger residual intervention.
- force mixture for v19:
  - `80-140N`: 50%;
  - `140-180N`: 36%;
  - `180-220N`: 12%;
  - `220-260N`: 2%.

Gate:

- early safety checks start after `60` resume-relative PPO iterations;
- hard stop if `Episode_Termination/base_contact > 0.11` for `4` strikes or scaled residual exceeds `0.85`;
- caution if train-time `base_contact` sustains above `0.07`.

Launch:

- tmux session: `ta_v19_survive_return`;
- run id: `track_adapter_81999_stage3_recovery_v19_survive_then_return_20260518_060451`;
- log: `logs/track_adapter_81999_stage3_recovery_v19_survive_then_return_20260518_060451.log`.

## 2026-05-18 06:08 UTC - v19 early health check

Training is running in tmux:

- session: `ta_v19_survive_return`;
- run directory:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-18_06-04-55_track_adapter_81999_stage3_recovery_v19_survive_then_return_20260518_060451_ppo_survive_then_return`;
- process: `scripts/rsl_rl/train.py --task Booster-Run-AMP-TrackAdapter-v0 ... --resume --load_run ...v14... --checkpoint model_10000.pt`.

Early snapshot at iteration `10020/13000`:

- `Episode_Termination/base_contact`: `0.0799`;
- `time_out`: `0.1724`;
- `force_push_started_count`: `2.8958`;
- `force_push_active_count`: `10.6458`;
- `force_push_applied_force_max_n`: `201.9204`;
- `wm loss`: `0.1278`;
- `wm_valid_fraction`: `0.7092`;
- `adapter_scaled_residual_action_l2`: `0.0263`;
- `error_vel_xy/error_vel_yaw`: `0.3205/0.3527`.

Interpretation:

- force push is firing; the instantaneous `force_push_active` value can be near zero because it is averaged over all steps/envs, so the count/max-force metrics are the reliable startup check;
- the survive-then-return gate is active: immediate `recovery_push_velocity_track_exp` is `0.0000`, while delayed tracking is still small;
- recovery rewards now dominate early push handling: upright, base height, angular velocity damping, alive/no-base-contact, and support terms are large;
- base contact is higher than the desired early value and is now a caution metric, but still below the v19 automatic stop threshold of `0.11`.

Next watch point:

- safety checks start after relative iteration `60` (`10060` absolute);
- stop or revise if base contact sustains above `0.11`, scaled residual approaches `0.85`, or force-push fall/contact worsens in fixed eval.

## 2026-05-18 06:12 UTC - v19 auto-eval watcher attached

Added and launched:

- script: `scripts/rsl_rl/watch_track_adapter_v19_auto_eval.sh`;
- tmux session: `ta_v19_auto_eval`;
- output root:
  `logs/track_adapter_81999_stage3_recovery_v19_survive_then_return_20260518_060451_auto_eval_20260518_061144`.

The watcher waits for the v19 training process to finish, then evaluates:

- the last `4` v19 checkpoints found in the PPO run directory;
- anchor v14 `model_10000.pt`;
- anchor v15b `model_16499.pt`.

Evaluation uses the v19 runtime recovery gates, especially:

- `residual_scale=0.19`;
- `push_command_gate_min=0.035`;
- `push_style_gate_max=0.06`;
- `push_recovery_gate_min=1.00`;
- phase-aware push residual gate enabled.

Fixed eval set:

- pure-horizontal physical force at `100/150/175/200/225N`;
- no-push command grid including zero command, `vx` single-axis commands, `vy` single-axis commands, and yaw-only commands.

## 2026-05-18 06:14 UTC - v19 stopped and v20 launched

Issue found by code review:

- `BOOSTER_TRACK_ADAPTER_PUSH_COMMAND_GATE_MIN` is a floor, not a cap;
- because `command_min_weight` stayed at the default `0.55`, v19 still kept roughly half of normal task/velocity tracking during push-active recovery;
- this contradicted the intended survive-first reward priority.

Code fix:

- added `push_command_gate_max` support to `rsl_rl/rsl_rl/runners/track_adapter_runner.py`;
- added `BOOSTER_TRACK_ADAPTER_PUSH_COMMAND_GATE_MAX` config plumbing in
  `source/booster_rl_tasks/booster_rl_tasks/tasks/manager_based/beyond_mimic/robots/k1/run_amp/ppo_cfg.py`;
- py_compile passed for both modified Python files.

Training action:

- stopped flawed v19 at about `10047` absolute iterations;
- last observed v19 `base_contact` was `0.0988`, still below the stop threshold but already near the caution zone;
- launched v20 script:
  `scripts/rsl_rl/run_track_adapter_stage3_recovery_v20_survive_first_command_cap.sh`;
- tmux session: `ta_v20_survive_cap`;
- run id:
  `track_adapter_81999_stage3_recovery_v20_survive_first_command_cap_20260518_061447`;
- PPO run directory:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-18_06-14-52_track_adapter_81999_stage3_recovery_v20_survive_first_command_cap_20260518_061447_ppo_survive_first_command_cap`.

Verified v20 saved config:

- `command_min_weight`: `0.55`;
- `push_recovery_gate_min`: `1.0`;
- `push_style_gate_max`: `0.045`;
- `push_command_gate_min`: `0.025`;
- `push_command_gate_max`: `0.06`.

Early v20 snapshot at iteration `10004/13000`:

- `Episode_Termination/base_contact`: `0.0006`;
- `force_push_started_count`: `2.7083`;
- `force_push_active_count`: `10.0208`;
- `force_push_applied_force_max_n`: `43.2349` in the printed window;
- `adapter_scaled_residual_action_l2`: `0.0316`;
- `wm loss`: `0.1199`;
- `wm_valid_fraction`: `0.7780`.

Auto-eval:

- added `scripts/rsl_rl/watch_track_adapter_v20_auto_eval.sh`;
- tmux session: `ta_v20_auto_eval`;
- output root:
  `logs/track_adapter_81999_stage3_recovery_v20_survive_first_command_cap_20260518_061447_auto_eval_20260518_061543`;
- fixed eval will use the v20 runtime gate, including `push_command_gate_max=0.06`.

Verification:

- added `test_push_active_can_cap_command_gate_without_affecting_stable_tracking` to
  `tests/test_track_adapter_optimizer.py`;
- command gate behavior is now tested for both push-active cap and stable/no-push tracking preservation;
- `source /workspace/Isaac_uv_template/.venv/bin/activate && pytest -q tests/test_track_adapter_optimizer.py -q` passed: `15` tests.

Follow-up early v20 snapshot at iteration `10016/13000`:

- `Episode_Termination/base_contact`: `0.0747`;
- `force_push_started_count`: `3.1042`;
- `force_push_active_count`: `13.9167`;
- `force_push_applied_force_max_n`: `207.8472`;
- `adapter_scaled_residual_action_l2`: `0.0479`;
- `wm loss`: `0.1255`;
- `wm_valid_fraction`: `0.7792`.

Interpretation:

- v20 is now actually using the survive-first command-gate cap;
- force exposure is active and includes 200N-class samples;
- contact is already in the caution band, but safety gate has not started yet and the automatic stop threshold remains `0.11` after relative iteration `60`.

## 2026-05-18 06:19 UTC - v20 caution checkpoint

Iteration `10028/13000` snapshot:

- `model_10025.pt` was saved;
- `Episode_Termination/base_contact`: `0.1027`;
- `time_out`: `0.8973`;
- `force_push_started_count`: `0.3125`;
- `force_push_active_count`: `0.4167`;
- `force_push_applied_force_max_n`: `55.9911` in that printed window;
- `adapter_scaled_residual_action_l2`: `0.0353`;
- `wm loss`: `0.1228`;
- `wm_valid_fraction`: `0.7764`.

Interpretation:

- base contact is very close to the v20 automatic stop threshold (`0.11`);
- the run is allowed to continue until the relative safety gate starts at absolute iteration `10060`;
- if contact remains this high or crosses `0.11` after the safety gate, the run should stop and fixed-eval the saved early checkpoints instead of burning the full training budget.

## 2026-05-18 08:05 UTC - v20 safety stop and auto-eval in progress

Training status:

- v20 training stopped automatically at iteration `10503/13000`;
- stop reason: Track Adapter safety gate failed after `4/4` strikes;
- final safety-gate values:
  - iteration `10502`: `Episode_Termination/base_contact=0.1129`;
  - iteration `10503`: `Episode_Termination/base_contact=0.1126`;
  - threshold: `0.1100`;
- latest normal checkpoint: `model_10500.pt`;
- aborted checkpoint marker exists: `model_10503_aborted.pt`.

Auto-eval status:

- tmux session: `ta_v20_auto_eval`;
- output root:
  `logs/track_adapter_81999_stage3_recovery_v20_survive_first_command_cap_20260518_061447_auto_eval_20260518_061543`;
- note: candidate directory names are currently `v19_model_*` due to watcher naming inherited from the v19 script, but the checkpoint paths are v20 checkpoints.

Completed partial fixed-eval rows:

```text
candidate          eval      force  fall    base_contact  vxerr   vyerr   wzerr   residual
v20 model_10425    no_push          0.0013  0.0013        0.0600  0.0816  0.1776  0.0162
v20 model_10425    force     100    0.0057  0.0057        0.1164  0.1358  0.2005  0.2349
v20 model_10425    force     150    0.0767  0.0650        0.1932  0.1684  0.2677  0.2418
v20 model_10425    force     175    0.1838  0.1506        0.2524  0.2008  0.3189  0.2413
v20 model_10425    force     200    0.2703  0.2111        0.2593  0.2047  0.3544  0.2330
v20 model_10425    force     225    0.3697  0.2943        0.2712  0.2288  0.3693  0.2134
v20 model_10450    no_push          0.0013  0.0013        0.0609  0.0874  0.1775  0.0164
v20 model_10450    force     100    0.0063  0.0020        0.1320  0.1245  0.2039  0.2326
v20 model_10450    force     150    0.0906  0.0725        0.1814  0.1696  0.2723  0.2380
v20 model_10450    force     175    0.1697  0.1345        0.2255  0.1945  0.3227  0.2344
v20 model_10450    force     200    0.2845  0.2424        0.2519  0.2214  0.3749  0.2392
v20 model_10450    force     225    0.3639  0.2844        0.2797  0.2252  0.3726  0.2156
v20 model_10475    no_push          0.0038  0.0039        0.0694  0.0847  0.1715  0.0164
v20 model_10475    force     100    0.0085  0.0058        0.1377  0.1445  0.1939  0.2267
v20 model_10475    force     150    0.0700  0.0475        0.1832  0.1684  0.2758  0.2412
v20 model_10475    force     175    0.1836  0.1465        0.2085  0.1983  0.3245  0.2411
v20 model_10475    force     200    0.2789  0.2167        0.2554  0.2245  0.3641  0.2264
v20 model_10475    force     225    0.3551  0.2924        0.2848  0.2244  0.3697  0.2136
```

Immediate interpretation:

- v20 did learn/use intervention under force (`residual ~= 0.23-0.24` in force eval) while staying quiet in no-push (`residual ~= 0.016`);
- no-push degradation is small but not zero;
- completed v20 checkpoints so far do not clearly beat the prior v14 `model_10000.pt` aggregate benchmark at `150-200N`;
- wait for `model_10500`, v14, and v15b anchor evaluations before choosing the next branch.

## 2026-05-18 08:36 UTC - v20 auto-eval complete

Auto-eval completed:

- summary:
  `logs/track_adapter_81999_stage3_recovery_v20_survive_first_command_cap_20260518_061447_auto_eval_20260518_061543/summary.md`;
- note: summary title/candidate prefix says `v19`, but these `v19_model_*` rows are v20 checkpoints from the v20 PPO run.

Key rows:

```text
candidate       100N fall/base   150N fall/base   175N fall/base   200N fall/base   225N fall/base   no-push fall/base
v14_10000       0.0083/0.0083    0.0812/0.0598    0.1669/0.1330    0.2535/0.1958    0.3794/0.3070    0.0013/0.0013
v15b_16499      0.0036/0.0036    0.0630/0.0529    0.1640/0.1325    0.2917/0.2425    0.3674/0.2862    0.0013/0.0013
v20_10425       0.0057/0.0057    0.0767/0.0650    0.1838/0.1506    0.2703/0.2111    0.3697/0.2943    0.0013/0.0013
v20_10450       0.0063/0.0020    0.0906/0.0725    0.1697/0.1345    0.2845/0.2424    0.3639/0.2844    0.0013/0.0013
v20_10475       0.0085/0.0058    0.0700/0.0475    0.1836/0.1465    0.2789/0.2167    0.3551/0.2924    0.0038/0.0039
v20_10500       0.0047/0.0047    0.0786/0.0576    0.1718/0.1359    0.2800/0.2045    0.3874/0.3108    0.0013/0.0013
```

Residual usage:

```text
candidate       no-push residual   force residual range
v14_10000       0.0208             0.2583-0.3232
v15b_16499      0.0167             0.3737-0.4911
v20_10425       0.0162             0.2134-0.2418
v20_10450       0.0164             0.2156-0.2392
v20_10475       0.0164             0.2136-0.2412
v20_10500       0.0155             0.2131-0.2400
```

Interpretation:

- v20 achieved the intended gate behavior: no-push residual stayed low and force residual was lower than v14/v15b while still reacting to pushes;
- however, lower residual did not translate into clearly better fall/base-contact rates;
- v14 `model_10000.pt` remains the best aggregate benchmark over `100-225N` when fall+base-contact are averaged;
- v15b `model_16499.pt` remains strongest at `100-175N` fall rate and good at `225N` contact;
- v20 `model_10475.pt` is the best v20 candidate if prioritizing high-force `225N` fall rate, but it has worse no-push fall/base and weaker `175N`;
- do not promote v20 as the new best robust policy.

Conclusion:

- v20 fixed the reward-gating implementation bug, but the branch still did not produce human-kick-level recovery;
- the remaining problem is not just "adapter can intervene"; it needs better recovery behavior/capture-step learning rather than lower command gate alone.

## 2026-05-18 09:11 UTC - v21 directional-support branch implemented and launched

Implemented:

- added `recovery_directional_support_contact_exp` in
  `source/booster_rl_tasks/booster_rl_tasks/tasks/manager_based/beyond_mimic/mdp/rewards.py`;
- wired it into Track Adapter env config as `recovery_directional_support_contact_exp`;
- added phase-aware push command gate fields:
  - `push_command_gate_late_max`;
  - `push_command_gate_full_s`;
  - `push_command_gate_taper_s`;
- exposed corresponding env overrides in `ppo_cfg.py`;
- added tests:
  - push command gate cap still preserves no-push tracking;
  - push command gate cap can taper back toward tracking;
  - directional support reward requires a contacted support foot in the push direction.

Verification:

- bash syntax check passed for v21 and watcher scripts;
- py_compile passed for modified runner/reward/env/ppo config files;
- `source /workspace/Isaac_uv_template/.venv/bin/activate && pytest -q tests/test_track_adapter_optimizer.py tests/test_track_adapter_module.py -q` passed: `42` tests.

Training launch:

- tmux session: `ta_v21_directional_support`;
- auto-eval session: `ta_v21_auto_eval`;
- run id:
  `track_adapter_81999_stage3_recovery_v21_directional_support_20260518_091110`;
- PPO run directory:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-18_09-11-15_track_adapter_81999_stage3_recovery_v21_directional_support_20260518_091110_ppo_directional_support`;
- auto-eval output root:
  `logs/track_adapter_81999_stage3_recovery_v21_directional_support_20260518_091110_auto_eval_20260518_091137`.

Parent:

- resumes from v15b:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-16_11-44-40_track_adapter_81999_stage3_recovery_v15b_force_ladder_20260516_114435_ppo_force_ladder/model_16499.pt`;
- rationale: v20 lowered residual use but did not improve recovery; v15b is the stronger `100-175N` fall-rate anchor and may give the new contacted support reward a better intervention prior.

Verified saved config:

- `push_command_gate_min`: `0.03`;
- `push_command_gate_max`: `0.065`;
- `push_command_gate_late_max`: `0.55`;
- `push_command_gate_full_s`: `0.65`;
- `push_command_gate_taper_s`: `1.55`;
- `recovery_directional_support_contact_exp` is present in `env.yaml`;
- safety gate `max_base_contact`: `0.13`, patience `5`.

Early health check at iteration `16508/19499`:

- `Episode_Termination/base_contact`: `0.0345`;
- `force_push_started_count`: `1.9792`;
- `force_push_active_count`: `8.4375`;
- `force_push_force_min/max_n`: `70/250`;
- `force_push_applied_force_max_n`: `149.8965`;
- `recovery_directional_support_contact_exp`: `0.1376`;
- `recovery_directional_step_exp`: `0.0813`;
- `recovery_push_capture_point_exp`: `0.1793`;
- `adapter_scaled_residual_action_l2`: `0.0387`;
- `wm loss`: `0.1138`;
- `wm_valid_fraction`: `0.7763`.

Interpretation:

- v21 startup is healthier than v20 at the same early stage;
- the new directional support-contact reward is non-zero under push, so the signal is connected;
- force exposure is active, but the printed window has so far sampled mostly moderate force; continue watching after safety gate starts at absolute iteration `16559`.

## 2026-05-18 09:59 UTC - v21 mid-run status

Training is still running:

- tmux session: `ta_v21_directional_support`;
- current process elapsed: about `48 min`;
- latest saved checkpoint: `model_16800.pt`;
- no safety-gate strike or aborted checkpoint found.

Latest observed training window around iteration `16805/19499`:

- `Episode_Termination/base_contact`: `0.1143`;
- `time_out`: `0.8857`;
- `force_push_started_count`: `2.0000`;
- `force_push_active_count`: `8.9583`;
- `force_push_applied_force_max_n`: `204.8349`;
- `adapter_scaled_residual_action_l2`: `0.0834`;
- `adapter_ungated_scaled_residual_action_l2`: `0.1954`;
- `wm loss`: `0.1088`;
- `wm_valid_fraction`: `0.7697`;
- `error_vel_xy/error_vel_yaw`: `0.2539/0.2935`;
- `recovery_directional_support_contact_exp`: `0.0352`;
- `recovery_push_capture_point_exp`: `0.1240`;
- `recovery_return_to_command_exp`: `0.1582`.

Interpretation:

- v21 remains within the relaxed safety gate (`base_contact < 0.13`) and has not stopped;
- base contact is still high relative to the desired final policy, so this is not a clean success yet;
- residual usage is moderate and not exploding;
- world-model replay remains healthy;
- the directional support reward is active, but its magnitude has dropped from the initial `0.1376` to about `0.035`, so fixed eval is still required before claiming recovery improvement.

## 2026-05-18 10:28 UTC - sim2real static rear-push regression hypothesis

User sim2real observation:

- compared deployed ONNX files:
  - `/workspace/booster_k1_locomotion/assets/track_adapter_model_1100.onnx`;
  - `/workspace/booster_k1_locomotion/assets/track_adapter_v11_model_6500.onnx`;
  - `/workspace/booster_k1_locomotion/assets/track_adapter_v14_model_10000.onnx`;
- under zero-command/static rear push, the old `track_adapter_model_1100.onnx` appears better:
  it steps a foot, while v11/v14 tend to respond by toe/ankle bracing;
- this can be a genuine behavioral regression even if v11/v14 score better on averaged
  fall/base-contact fixed evals, because the current eval does not explicitly score static
  rear-push capture stepping.

Local inspection:

- `track_adapter_model_1100.onnx`:
  - timestamp `2026-05-13 08:58:01`;
  - size `1,264,816` bytes;
  - inputs `obs[1,75]`, `history[1,79,97]`;
  - outputs `action[1,22]`, `next_history[1,79,97]`;
  - initializer count `19`, parameter elements `314,626`;
  - architecture names show a plain `residual_actor`;
- v11/v14 ONNX:
  - size `2,064,020` bytes;
  - inputs `obs[1,75]`, `history[1,79,97]`, `residual_gate[1,1]`, `reset[1,1]`;
  - initializer count `29`, parameter elements `513,410`;
  - architecture names show `layer_adapters`;
- likely implication: `model_1100.onnx` is an older non-layerwise residual-adapter export, while
  v11/v14 are layerwise gated adapters.

Deployment-tuning caveat:

- v11/v14 behavior depends on the supplied `residual_gate`;
- C++ and nomadz default to `track_adapter_gate_mode=constant` and
  `track_adapter_residual_gate=1.0`, but if real testing used `heuristic` or a low stable gate,
  a static push may under-open the adapter;
- real PD/action gains can also mask a capture step, but the fact that `model_1100.onnx` steps
  under the same robot setup points first to adapter output/gating/training differences.

Decision:

- do not treat v14/v21 promotion as settled by aggregate force fall/contact metrics alone;
- add/perform a static zero-command rear-push stepping benchmark before promoting another ONNX:
  fall/contact, support-foot contact in push direction, foot step projection, base velocity arrest,
  residual norm, and gate value;
- compare `model_1100.onnx`, v11 `model_6500`, v14 `model_10000`, and the best v21 checkpoints;
- keep the current v21 training running for now because it specifically adds contacted
  directional support reward, but require the new static rear-push gate before promotion.

## 2026-05-18 10:38 UTC - v11 vs v14 static rear-push step eval

Implemented:

- added `scripts/rsl_rl/eval_track_adapter_static_push_step.py`;
- added `scripts/rsl_rl/eval_track_adapter_static_rear_push_v11_v14.sh`;
- evaluation condition:
  - zero velocity command;
  - random push/root-velocity events disabled;
  - body-frame `+x` trunk force, interpreted as rear push under the current convention;
  - force duration `5` policy steps = `0.10s`;
  - push starts at step `80` = `1.60s`;
  - post-push measurement window `170` steps;
  - `64` envs, seed `123`;
  - v11 evaluated with `residual_scale=0.16`;
  - v14 evaluated with `residual_scale=0.20`;
  - push-active metrics are injected so the Track Adapter recovery/residual gate opens during the recovery window.

Output:

- `logs/static_rear_push_v11_v14_20260518_103715/summary.md`.

Results:

| candidate | force N | fall | base contact | clean | support step | clean support step | contacted step m | max step m | final speed m/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v11 `model_6500` | 100 | 0.0156 | 0.0156 | 0.9844 | 0.1406 | 0.1250 | 0.0501 | 0.0522 | 0.0208 |
| v14 `model_10000` | 100 | 0.0000 | 0.0000 | 1.0000 | 0.2656 | 0.2656 | 0.0595 | 0.0618 | 0.0267 |
| v11 `model_6500` | 150 | 0.0312 | 0.0312 | 0.9688 | 0.5625 | 0.5312 | 0.0869 | 0.0939 | 0.0232 |
| v14 `model_10000` | 150 | 0.0312 | 0.0312 | 0.9688 | 0.5625 | 0.5312 | 0.0866 | 0.0924 | 0.0247 |

Interpretation:

- under this Isaac raw-checkpoint eval, v14 is not worse than v11 for static rear-push stepping;
- at `100N`, v14 is slightly better: no fall/base contact and a higher contacted-step success rate;
- at `150N`, v11 and v14 are essentially tied;
- both policies still look weak as explicit capture-step controllers: `100N` contacted-step success is only
  `0.14-0.27`, and even `150N` reaches only `0.56`;
- the gate is not closed during this eval: during the active recovery window residual action gate is about
  `0.70`, with residual norm about `0.31-0.33`;
- therefore, if sim2real `model_1100.onnx` clearly steps better than both v11/v14, the likely gap is
  `model_1100` versus the newer layerwise/gated adapters, not v11 versus v14 alone. The next comparison
  should include `track_adapter_model_1100.onnx` or its source checkpoint under the same static rear-push metric.

## 2026-05-18 11:00 UTC - deploy observation reframed as weak sustained rear push

User clarified the real-robot failure mode:

- `track_adapter_model_1100.onnx` is clearly better when the robot is standing still and the user pushes it;
- v11/v14 ONNX tend to lean and move toes/ankles rather than taking a recovery step;
- the manual push is not a sharp kick: it is a weak force applied over several seconds;
- rear-to-front push is the problematic direction, while front-to-back and lateral pushes look more acceptable.

Interpretation:

- the previous `100/150N x 0.10s` static eval was still an impulse/kick-style test and does not match the deploy
  failure mode;
- v11/v14 were trained and evaluated mainly on short physical kicks or explicit push-event recovery windows, so
  they may have converged to impact recovery and ankle/toe bracing instead of quasi-static capture stepping;
- deployment has no Isaac event telling the adapter "push is active". If the deployed gate is heuristic or if the
  learned layerwise adapter requires event-like recovery context, a slow push can stay in a near-stable/no-step
  regime for too long;
- even with constant ONNX gate, the policy itself may not have learned the sustained rear-push stepping behavior.

Action:

- added `--push_metric_mode {event,none}` to `scripts/rsl_rl/eval_track_adapter_static_push_step.py`;
  - `event`: injects `failure_push_*` metrics and opens Track Adapter recovery gates;
  - `none`: leaves push metrics inactive, closer to deploy where there is no explicit push event;
- added `scripts/rsl_rl/eval_track_adapter_sustained_rear_push_v11_v14.sh`;
- launched tmux `ta_sustained_rear_push_v11_v14`:
  - output `logs/sustained_rear_push_v11_v14_20260518_105948`;
  - v11 `model_6500` vs v14 `model_10000`;
  - body-frame `+x` trunk force;
  - weak sustained forces `30N` and `50N`;
  - duration `125` policy steps = `2.5s`;
  - modes `none` and `event`;
  - `32` envs.

Completed `30/50N x 2.5s` results:

| mode | candidate | force N | fall | base contact | clean | support step | clean support step | contacted step m | max speed m/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | v11 `model_6500` | 30 | 0.2500 | 0.2188 | 0.7500 | 0.9375 | 0.6875 | 0.1527 | 1.5748 |
| none | v14 `model_10000` | 30 | 0.1875 | 0.1250 | 0.8125 | 0.9688 | 0.7812 | 0.1598 | 1.4995 |
| none | v11 `model_6500` | 50 | 0.8438 | 0.8125 | 0.1562 | 1.0000 | 0.1562 | 0.2102 | 2.7613 |
| none | v14 `model_10000` | 50 | 0.9062 | 0.8125 | 0.0938 | 1.0000 | 0.0938 | 0.2232 | 2.8092 |
| event | v11 `model_6500` | 30 | 0.2500 | 0.2500 | 0.7500 | 0.9688 | 0.7188 | 0.1476 | 1.5837 |
| event | v14 `model_10000` | 30 | 0.1562 | 0.0625 | 0.8438 | 0.9688 | 0.8125 | 0.1655 | 1.4833 |
| event | v11 `model_6500` | 50 | 0.9062 | 0.8750 | 0.0938 | 1.0000 | 0.0938 | 0.2133 | 2.8336 |
| event | v14 `model_10000` | 50 | 0.8438 | 0.7188 | 0.1562 | 1.0000 | 0.1562 | 0.2415 | 2.7351 |

Interpretation:

- this sustained condition exposes a real weakness that the short `100/150N x 0.10s` eval hid;
- `30N x 2.5s` is already strong enough to accelerate the base to about `1.5m/s`, so it is likely harsher
  than the user's "weak hand push";
- `50N x 2.5s` is mostly outside the reliable envelope for both v11 and v14;
- v14 is not clearly worse than v11 in Isaac; the deploy-observed superiority of old `model_1100.onnx`
  still needs direct comparison against the old ONNX/source checkpoint;
- opening push metrics (`event`) does not fix the high-fall sustained-push issue, so the problem is not
  only gate detection. The learned behavior itself is not robust enough for sustained rear-push lean.

Follow-up launched:

- tmux `ta_sustained_rear_push_lowforce_v11_v14`;
- output `logs/sustained_rear_push_v11_v14_lowforce_20260518_110433`;
- deploy-like `mode=none` only;
- lower forces `10N` and `20N`, still `2.5s`, to better match weak manual pushing.

## 2026-05-18 11:35 UTC - old 1100 vs v11/v14 under deploy-like constant gate

Completed constant-gate sustained rear-push comparison:

- output: `logs/sustained_rear_push_constant_gate_compare_20260518_111155/summary.md`;
- candidates:
  - v7 `model_1100.pt`, likely source checkpoint for old `track_adapter_model_1100.onnx`;
  - v11 `model_6500.pt`;
  - v14 `model_10000.pt`;
- condition:
  - zero command;
  - body-frame `+x` trunk force;
  - `10/20/30N x 2.5s`;
  - no synthetic push metrics (`push_metric_mode=none`);
  - residual gate forced to `1.0`, matching constant-gate ONNX deployment and old no-gate ONNX behavior.

| candidate | force N | fall | base contact | clean support step | contacted step m | max speed m/s |
|---|---:|---:|---:|---:|---:|---:|
| v7 `model_1100` | 10 | 0.0000 | 0.0000 | 0.1250 | 0.0353 | 0.3242 |
| v7 `model_1100` | 20 | 0.0312 | 0.0312 | 0.6875 | 0.0998 | 0.8694 |
| v7 `model_1100` | 30 | 0.1875 | 0.1875 | 0.7812 | 0.1629 | 1.5284 |
| v11 `model_6500` | 10 | 0.0312 | 0.0312 | 0.1562 | 0.0513 | 0.4219 |
| v11 `model_6500` | 20 | 0.0000 | 0.0000 | 0.7812 | 0.1147 | 0.9160 |
| v11 `model_6500` | 30 | 0.0938 | 0.0938 | 0.9062 | 0.1770 | 1.4951 |
| v14 `model_10000` | 10 | 0.0000 | 0.0000 | 0.1875 | 0.0406 | 0.3388 |
| v14 `model_10000` | 20 | 0.0000 | 0.0000 | 0.7500 | 0.1012 | 0.7957 |
| v14 `model_10000` | 30 | 0.1875 | 0.1250 | 0.7812 | 0.1514 | 1.4441 |

Important interpretation:

- Isaac fall/contact does not reproduce the user's real-robot ranking; v7/old-1100 is not numerically dominant in this
  raw checkpoint benchmark.
- The key difference is residual usage:
  - v7 `model_1100`: residual norm at push start is about `0.0026`;
  - v11 `model_6500`: about `0.42`;
  - v14 `model_10000`: about `0.58`.
- This matches the deploy symptom better than the fall-rate table: v11/v14 can be constantly modifying the action at
  zero command, so sim2real gain/model mismatch can turn the learned recovery into ankle/toe bracing. The old 1100 is
  almost neutral and lets the frozen AMP policy do most of the standing/push response.

Implementation added for the next branch:

- extended `mdp.apply_external_wrench_pulse` with:
  - `fixed_direction_xy`;
  - `fixed_direction_probability`;
  - `fixed_direction_jitter_rad`;
  - `ramp_up_range_s`;
  - `start_command_norm_max` / `start_command_yaw_scale`;
- exposed these via `BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FIXED_DIRECTION_*` and
  `BOOSTER_TRACK_ADAPTER_FORCE_PUSH_RAMP_UP_*` / `BOOSTER_TRACK_ADAPTER_FORCE_PUSH_START_COMMAND_*`
  in the Track Adapter env config;
- added `recovery_directional_first_contact_step_exp`, a sparse reward for a fresh contacted support step in the push
  direction after air time;
- added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v22_quasistatic_rear_push.sh`.

v22 intent:

- new Stage 3 world-model pretrain with slow low-force sustained trunk pushes;
- body-frame rear-to-front `+x` direction is sampled with high probability (`0.92` by default);
- force pulses are started only when the command norm is near zero by default (`<=0.05`), matching the deploy failure;
- standing-command exposure is raised to `0.42` and failure mining is disabled for this first narrow branch;
- use `action_residual` rather than layerwise adapters;
- warm-start only the old v7 `model_1100` residual actor, not the old history encoder/world model;
- keep zero-command/stable residual close to zero, while allowing recovery residual only after the body state actually
  requires capture/support stepping.

Validation:

- initial validation before the first v22 launch passed for `mdp/events.py`, `run_amp/env_cfg.py`, and the v22 script.

Launch:

- first tmux launch `ta_v22_quasistatic_rear` was stopped during Stage 3 startup before useful training, because review
  found the standing/rear-push exposure was still too diluted;
- patched defaults before relaunch:
  - standing envs `0.42`;
  - hard envs `0.42`;
  - force activation probability `0.58`;
  - fixed body `+x` direction probability `0.92`;
  - start command norm cap `0.05`;
  - failure mining disabled;
  - first-contact directional step reward weight `2.60`.

Relaunch:

- tmux session: `ta_v22_quasistatic_rear`;
- run id: `track_adapter_81999_stage3_recovery_v22_quasistatic_rear_push_20260518_113137`;
- log: `logs/track_adapter_81999_stage3_recovery_v22_quasistatic_rear_push_20260518_113137.log`;
- Stage 3 pretrain directory:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-18_11-31-41_track_adapter_81999_stage3_recovery_v22_quasistatic_rear_push_20260518_113137_wm_pretrain_quasistatic_rear`.

Startup check:

- reward table includes `recovery_directional_first_contact_step_exp` with weight `2.6`;
- model reports `Adapter mode: action_residual`;
- Stage 3 pretrain reached iteration `4/600`;
- `wm_valid_fraction` is stable around `0.94`;
- GPU memory during Stage 3 is about `17.9GB / 32.6GB`.

## 2026-05-18 12:39 UTC - v22 stopped; v23 broad-force branch prepared

Correction:

- user correctly pointed out that v22 over-specialized the objective toward weak sustained rear push;
- main objective is not "step under weak hand push", but "do not fall under external force", while maintaining tracking;
- old `track_adapter_model_1100.onnx` is useful as a qualitative hint, but should not be treated as a target policy:
  the exact ONNX export/base pairing is not proven, and it may not correspond to the current frozen AMP
  `model_81999.pt` setup;
- stopped tmux `ta_v22_quasistatic_rear` during Stage 3 pretrain around iteration `421/600`.

Implemented:

- added `force_duration_mixture` support to `mdp.apply_external_wrench_pulse`;
- added `BOOSTER_TRACK_ADAPTER_FORCE_DURATION_MIXTURE` parser and env-config pass-through;
- added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v23_broad_force.sh`.

v23 plan:

- frozen base: AMP axis-v2 `model_81999.pt`;
- resume adapter from v15b `model_16499.pt`, not old 1100;
- rebuild Stage 3 world model from scratch on broad force disturbances;
- use coupled force/duration/ramp mixture:
  - `8-35N`, `1.00-3.00s`, ramp `0.15-0.65s`, weight `0.18`;
  - `35-90N`, `0.35-1.20s`, ramp `0.04-0.20s`, weight `0.22`;
  - `90-170N`, `0.08-0.16s`, ramp `0.00-0.03s`, weight `0.30`;
  - `170-250N`, `0.06-0.12s`, ramp `0.00-0.02s`, weight `0.22`;
  - `250-340N`, `0.05-0.10s`, ramp `0.00-0.01s`, weight `0.08`;
- standing envs are back to `0.12`, hard command envs `0.70`;
- first-contact step reward remains but with lower weight `1.40`, as a general recovery cue rather than the dominant
  weak-push objective.

Validation:

- `python -m py_compile` passed for `mdp/events.py`, `mdp/rewards.py`, and `run_amp/env_cfg.py`;
- `bash -n` passed for `run_track_adapter_stage3_recovery_v23_broad_force.sh`;
- checkpoint existence checks passed for AMP `model_81999.pt` and v15b `model_16499.pt`.

Launch:

- tmux session: `ta_v23_broad_force`;
- run id: `track_adapter_81999_stage3_recovery_v23_broad_force_20260518_124831`;
- log: `logs/track_adapter_81999_stage3_recovery_v23_broad_force_20260518_124831.log`;
- Stage 3 pretrain started with `H=79,N=20`, `800` iterations, `768` envs;
- startup check:
  - reward table includes `recovery_directional_first_contact_step_exp` at weight `1.4`;
  - model reports `Adapter mode: layerwise`;
  - Stage 3 reached iteration `2/800`;
  - `wm_valid_fraction` is around `0.94`.

## 2026-05-18 15:18 UTC - v23 Stage3 completed; PPO restarted after validation fix

Stage 3:

- `track_adapter_81999_stage3_recovery_v23_broad_force_20260518_124831` completed all `800` pretrain iterations;
- final checkpoint: `logs/rsl_rl/run_amp_track_adapter/2026-05-18_12-48-35_track_adapter_81999_stage3_recovery_v23_broad_force_20260518_124831_wm_pretrain_broad_force/model_799.pt`;
- final pretrain metrics were finite:
  - `wm=0.4131`;
  - `wm_valid_fraction=0.9278`.

Validation issue and fix:

- Stage3-to-PPO validation initially failed because both the shell script and runner still required
  `manifest.push_enabled`, even though v23 intentionally uses physical-force push coverage with root-velocity push
  disabled;
- updated validation in `scripts/rsl_rl/run_track_adapter_stage3_to_ppo.sh` and
  `rsl_rl/rsl_rl/runners/track_adapter_runner.py` to accept either root push or force push coverage.

PPO restart:

- tmux session: `ta_v23_broad_force`;
- run id: `track_adapter_81999_stage3_recovery_v23_broad_force_20260518_151650`;
- pretrain checkpoint reused: v23 `model_799.pt`;
- resume checkpoint: v15b `model_16499.pt`;
- early PPO status at iteration `16505/25499`:
  - `base_contact=0.1019`, inside current safety gate `0.14`;
  - `force_push_active=0.0625`;
  - `force_push_applied_force_max_n=159.3030`;
  - `error_vel_xy=0.2044`, `error_vel_yaw=0.2636`;
  - `adapter_scaled_residual_action_l2=0.0335`;
  - recovery directional/support rewards are non-zero, so force-recovery reward wiring is active;
  - GPU memory about `25.6GB / 32.6GB`.

## 2026-05-18 15:23 UTC - v23 PPO stopped due reward masking contact regression

Stopped tmux `ta_v23_broad_force`.

Reason:

- `Mean reward` rose as high as about `2684`, but this was not a clean improvement signal;
- the high return correlated with longer episode length (`~977` at the peak) and large survival/recovery-alive terms,
  not with lower contact;
- `Episode_Termination/base_contact` climbed and stayed around `0.24`, far above the intended safety envelope;
- latest observed values before stop:
  - `base_contact=0.2445`;
  - `time_out=0.7555`;
  - `force_push_applied_force_max_n=292.2506`;
  - `force_push_active=0.1305`;
  - `force_push_started_count=5.8438`.

Diagnosis:

- the reward scale itself is not necessarily invalid because RSL-RL logs episodic return, which grows with episode
  length;
- however, the v23 recovery/no-fall/no-base-contact/upright/height/support rewards are too large relative to the
  terminal/base-contact signal, so longer-but-contacting rollouts can look attractive;
- next branch should normalize or reduce survival-style recovery rewards and strengthen direct base-contact/fall
  penalties/gates before relaunching broad-force training.

## 2026-05-18 15:34 UTC - v24 stopped by contact guard; v24b launched with curriculum-scaled force mixture

v24 result:

- tmux/log: `ta_v24_contact_guard`,
  `logs/track_adapter_81999_stage3_recovery_v24_contact_guard_20260518_152358.log`;
- v24 entered PPO from v23 Stage 3 `model_799.pt` and v15b `model_16499.pt`;
- the reward masking fix worked in the sense that the safety gate detected the contact regression instead of allowing a
  high-return run to continue;
- v24 stopped with safety-gate strike `2/2`:
  - `base_contact=0.1446 > 0.1200`;
  - `force_push_active=0.2734`;
  - `force_push_force_max_n=340.0`;
  - `force_push_applied_force_max_n=213.3247`;
  - `adapter_scaled_residual_action_l2=0.0290`.

Diagnosis:

- `force_duration_mixture` coupled force and duration correctly, but it bypassed the effective force curriculum;
- the existing alpha affected scalar force ranges, probability, interval, and equivalent push metrics, but not the
  actual force sampled from `force_duration_mixture`;
- therefore v24 started with the full `8-340N` mixture and hit the stricter contact gate before it had a chance to
  adapt.

Code changes:

- `source/.../mdp/events.py`:
  - added `force_duration_mixture_scale_with_curriculum`;
  - added `force_duration_mixture_force_floor`;
  - when enabled, sampled mixture forces are scaled by `floor + (1-floor) * alpha`;
  - force metrics now report the scaled effective force range.
- `source/.../run_amp/env_cfg.py`:
  - added env pass-through for
    `BOOSTER_TRACK_ADAPTER_FORCE_DURATION_MIXTURE_SCALE_WITH_CURRICULUM`;
  - added env pass-through for `BOOSTER_TRACK_ADAPTER_FORCE_DURATION_MIXTURE_FORCE_FLOOR`.
- added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v24b_curriculum_guard.sh`.

Validation:

- `bash -n` passed for v24 and v24b scripts;
- `python -m py_compile` passed for modified `events.py` and `env_cfg.py`;
- checkpoint existence checks passed for v23 Stage 3 `model_799.pt` and v15b `model_16499.pt`.

Launch:

- tmux session: `ta_v24b_curriculum_guard`;
- run id: `track_adapter_81999_stage3_recovery_v24b_curriculum_guard_20260518_153217`;
- log: `logs/track_adapter_81999_stage3_recovery_v24b_curriculum_guard_20260518_153217.log`;
- early PPO check at iteration `16504/25499`:
  - `force_push_curriculum_alpha=0.2800`;
  - `force_push_force_max_n=168.6400`, confirming the mixture scaling is active;
  - `force_push_activation_probability=0.2280`;
  - `force_push_active=0.2031`;
  - `base_contact=0.0043`;
  - `error_vel_xy=0.1838`, `error_vel_yaw=0.2011`;
  - `adapter_scaled_residual_action_l2=0.0557`;
  - `wm_valid_fraction=0.8323`.

Follow-up after safety gate became active:

- iteration `16514/25499`;
- `base_contact=0.0145`, well below v24b safety limit `0.14`;
- `force_push_curriculum_alpha=0.2800`;
- `force_push_force_max_n=168.6400`;
- `force_push_applied_force_max_n=107.2128`;
- `force_push_active=0.0062` in completed-episode logs, with force-push timers still running in rollout metrics;
- `Mean episode length=1000.00`, `time_out=0.3849`;
- `Mean reward=576.56`, no longer in the v23 runaway reward band;
- `adapter_scaled_residual_action_l2=0.0260`;
- GPU memory during PPO: about `25.6GB / 32.6GB`.

## 2026-05-18 16:22 UTC - v24b still running; curriculum promoted to mid-strength

Status check:

- tmux session `ta_v24b_curriculum_guard` is still running;
- latest observed iteration: `16742/25499`;
- latest checkpoint: `model_16725.pt`;
- GPU memory during PPO: about `25.6GB / 32.6GB`.

Training metrics:

- `force_push_curriculum_alpha=0.5600`;
- effective `force_push_force_max_n=235.2800`;
- recent `force_push_applied_force_max_n` has been around `126-186N`, latest observed `157.7207N`;
- `base_contact` has stayed below the v24b safety threshold `0.14`; recent values were roughly `0.085-0.096`,
  latest observed `0.0853`;
- `Mean episode length` is around `932-952`;
- `Mean reward` is around `455-511`, not in the previous v23 runaway range;
- velocity metrics latest observed near `error_vel_xy=0.2358`, `error_vel_yaw=0.2827`.

Interpretation:

- the curriculum-scaled mixture fix is working: v24b is no longer starting at full `340N`;
- safety gate has not aborted, and no safety strike was observed in the latest log window;
- base contact is materially higher than the very early v24b value, but currently stable under stronger force and still
  inside the guard band;
- continue training unless `base_contact` trends toward `0.14` or alpha promotion stalls while applied force remains
  below the target evaluation range.

## 2026-05-18 16:33 UTC - v24b stable at alpha 0.56; no safety abort

Status check:

- tmux session `ta_v24b_curriculum_guard` is still running;
- latest observed iteration: `16794/25499`;
- latest checkpoints include `model_16775.pt`;
- GPU memory during PPO: about `25.6GB / 32.6GB`.

Training metrics:

- `force_push_curriculum_alpha=0.5600`;
- effective `force_push_force_max_n=235.2800`;
- latest `force_push_applied_force_max_n=150.8010N`;
- recent applied-force maxima were roughly `144-177N`;
- `base_contact` recent range is roughly `0.073-0.080`, latest `0.0788`;
- safety threshold remains `0.14`, and no safety-gate strike/error was observed;
- `Mean reward=477.75`;
- `Mean episode length=936.20`;
- `error_vel_xy=0.2426`, `error_vel_yaw=0.2974`;
- `adapter_scaled_residual_action_l2=0.0463`;
- `wm_valid_fraction=0.8259`.

Interpretation:

- the run is stable and no longer shows the v23 contact blow-up;
- adaptive curriculum has paused at `alpha=0.56` because promotion requires `base_contact <= 0.055`, while the current
  steady range is closer to `0.075-0.080`;
- this is acceptable for now because the policy is surviving mid-strength force exposure, but if alpha remains stuck
  for a long span, the next intervention should be either a small promotion-threshold relaxation or a separate eval at
  the current checkpoints before pushing toward full `340N`.

## 2026-05-19 04:14 UTC - v24b still running overnight; stable but alpha remains stuck

Status check:

- tmux session `ta_v24b_curriculum_guard` is still running;
- latest observed iteration: `20132/25499`;
- latest checkpoint found: `model_20125.pt`;
- log is actively updating as of `2026-05-19 04:14 UTC`;
- no `Safety gate` strike was found in the log;
- GPU memory during PPO remains about `25.6GB / 32.6GB`.

Training metrics:

- `force_push_curriculum_alpha=0.5600`;
- effective `force_push_force_max_n=235.2800`;
- latest `force_push_applied_force_max_n=167.8723N`;
- recent applied-force maxima are mostly around `150-177N`;
- recent `base_contact` improved from the earlier `0.08-0.09` band to roughly `0.071-0.084`;
- latest `base_contact=0.0709`;
- `Mean reward=490.15`;
- `Mean episode length=938.24`;
- `error_vel_xy=0.2384`, `error_vel_yaw=0.2755`;
- `adapter_scaled_residual_action_l2=0.0872`;
- `wm_valid_fraction=0.8223`.

Interpretation:

- v24b is not diverging and is much cleaner than v23 with respect to base-contact blow-up;
- however, the adaptive curriculum has stayed at `alpha=0.56` for a long span because promotion still requires
  `base_contact <= 0.055`;
- this means the current run is mostly training mid-strength recovery around an effective `235N` envelope, not the full
  `340N` final target;
- if the objective remains full stronger-push robustness, the next likely intervention is to stop/relaunch with a
  slightly more permissive promote threshold, for example around `base_contact <= 0.075`, while keeping the abort guard
  at `0.14`.

## 2026-05-19 05:44 UTC - v24b fixed weak sustained push evaluation

Action:

- stopped tmux training `ta_v24b_curriculum_guard` after checkpoint `model_20525.pt` was available;
- added `scripts/rsl_rl/eval_track_adapter_sustained_push_single.sh` for single-candidate sustained push sweeps;
- evaluated v24b `model_20525.pt` on zero-command sustained body-frame `+x` push, 2.5s duration, 32 envs.

Output:

- `logs/sustained_rear_push_v24b_model_20525_20260519_053945/summary.md`

Results:

| mode | force N | fall | base contact | clean | support step | clean support step | final speed m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| event | 10 | 0.0000 | 0.0000 | 1.0000 | 0.0938 | 0.0938 | 0.0258 |
| event | 20 | 0.0312 | 0.0312 | 0.9688 | 0.7500 | 0.7188 | 0.0271 |
| event | 30 | 0.1250 | 0.1250 | 0.8750 | 1.0000 | 0.8750 | 0.0291 |
| event | 50 | 0.8125 | 0.6250 | 0.1875 | 1.0000 | 0.1875 | 0.0277 |
| none | 10 | 0.0000 | 0.0000 | 1.0000 | 0.1562 | 0.1562 | 0.0276 |
| none | 20 | 0.0312 | 0.0312 | 0.9688 | 0.6250 | 0.5938 | 0.0268 |
| none | 30 | 0.1562 | 0.1250 | 0.8438 | 0.9688 | 0.8125 | 0.0225 |
| none | 50 | 0.7812 | 0.7188 | 0.2188 | 1.0000 | 0.2188 | 0.0271 |

Interpretation:

- weak sustained push is not catastrophically broken at `10-20N`;
- at `30N`, v24b is not perfect but is in the same or slightly better range than earlier v11/v14 fixed evaluations;
- `50N x 2.5s` remains a failure case and should not be considered solved;
- event/no-event results are close, so this specific weak-push result is not solely dependent on synthetic push-active
  metric injection;
- this supports the view that v24b learned some mid-strength recovery, but not enough for strong or long sustained
  external force robustness.

## 2026-05-19 07:07 UTC - v25 robot-contact branch prepared

Goal:

- target robot-to-robot scuffle robustness rather than human-kick robustness;
- prevent falls under light sustained shoves, short bumps, repeated contacts, and random horizontal contact directions;
- preserve final return-to-command behavior, but prioritize survival/upright/base-height/contact avoidance while the
  contact is active.

Implementation:

- added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v25_robot_contact.sh`;
- resume anchor: v24b `model_20525.pt`;
- world-model initialization: v23 Stage 3 `model_799.pt`;
- PPO continuation length: `6500` iterations from the resumed checkpoint;
- training distribution:
  - `8-30N` for `1.2-3.0s`;
  - `30-65N` for `1.2-2.8s`, targeting the failed `50N x 2.5s` sustained-push case;
  - `45-90N` for `0.45-1.3s`;
  - `80-150N` for `0.12-0.40s`;
  - `140-220N` for `0.06-0.16s`;
  - rare `220-280N` for `0.05-0.10s`;
- up to `3` force pulses per episode;
- mostly random horizontal directions, with a small rear-push bias through fixed body-frame `+x` probability;
- no explicit yaw torque injection in this branch;
- adaptive curriculum starts near v24b's stable point:
  - `alpha_initial=0.55`;
  - force floor `0.50`;
  - promote at `base_contact <= 0.08`;
  - demote at `base_contact >= 0.14`;
- residual authority increased slightly with `residual_scale=0.18`;
- direct base-contact penalty strengthened to `-105`.

Validation:

- `bash -n scripts/rsl_rl/run_track_adapter_stage3_recovery_v25_robot_contact.sh` passed;
- `python -m py_compile` passed for the force-event/env config files;
- checkpoint existence checks passed for v23 Stage 3 `model_799.pt` and v24b `model_20525.pt`.

## 2026-05-19 07:11 UTC - v25 aborted manually; v25b staged branch prepared

v25 early result:

- launched `ta_v25_robot_contact` from v24b `model_20525.pt`;
- PPO entered successfully, but the initial robot-contact distribution was too aggressive;
- by iteration `20535/27025`:
  - `force_push_active=0.4870`;
  - `force_push_applied_force_max_n=157.4932N`;
  - `force_push_curriculum_alpha=0.4500`;
  - `force_push_force_max_n=203.0000`;
  - `base_contact=0.4164`;
  - `adapter_scaled_residual_action_l2=0.5339`;
- stopped the tmux run manually before waiting for safety-gate abort.

Diagnosis:

- the target distribution is relevant, but v25 introduced repeated/sustained contacts too abruptly;
- `max_pulses=3`, high active fraction, and long-duration components caused contact to dominate before the adapter
  could adapt.

v25b changes:

- added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v25b_robot_contact_staged.sh`;
- keeps the robot-contact objective but stages it more slowly:
  - `max_pulses=2`;
  - lower activation probability `0.14 -> 0.26`;
  - longer intervals `4.0-6.5s -> 2.6-4.2s`;
  - lower final force envelope `240N`;
  - lower initial adaptive alpha `0.35`;
  - lower force floor `0.38`;
  - safety gate active after `8` relative iterations with `base_contact <= 0.16`, patience `2`;
  - residual scale reduced back to `0.17`;
  - direct base-contact penalty strengthened to `-125`.

Validation:

- `bash -n` passed for v25 and v25b scripts;
- checkpoint existence checks passed.

## 2026-05-19 07:18 UTC - v25b staged robot-contact PPO launched

Run:

- tmux session: `ta_v25b_robot_contact`;
- log: `logs/track_adapter_81999_stage3_recovery_v25b_robot_contact_staged_20260519_071213.log`;
- run directory:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-19_07-12-18_track_adapter_81999_stage3_recovery_v25b_robot_contact_staged_20260519_071213_ppo_robot_contact_staged`;
- resume checkpoint: v24b `model_20525.pt`;
- world-model initialization: v23 Stage 3 `model_799.pt`.

Early status:

- v25b has passed the early window that killed v25, but is not yet a promotion candidate;
- at iteration `20539/27025`:
  - `force_push_active=0.3359`;
  - `force_push_applied_force_max_n=109.1737N`;
  - `force_push_curriculum_alpha=0.3500`;
  - `force_push_force_max_n=143.2800N`;
  - `base_contact=0.0807`;
  - `adapter_scaled_residual_action_l2=0.1824`;
  - `error_vel_xy=0.2679`;
  - `error_vel_yaw=0.2871`;
  - `wm_valid_fraction=0.8325`;
- this is much healthier than v25's `base_contact=0.4164`, but the base-contact rate is just above the intended
  promote threshold, so the adaptive curriculum should hold rather than advance.

Current decision:

- keep v25b running while `base_contact` remains below the safety band;
- if the adaptive curriculum climbs while `base_contact` is still above roughly `0.08-0.10`, stop and make v25c more
  conservative;
- if `base_contact >= 0.13-0.16` persists, stop and restart with a gentler staged curriculum:
  - start with one pulse per episode;
  - reduce initial activation probability;
  - cap early effective force around `100-120N`;
  - reintroduce repeated contacts only after sustained/rear/random single-push survival improves.

## 2026-05-19 07:24 UTC - v25b continues; auto eval watcher installed

Training status:

- v25b is now clearly better than the aborted v25 startup;
- by iteration `20577/27025`:
  - `force_push_curriculum_alpha=0.3860`;
  - `force_push_force_max_n=148.6368N`;
  - recent `force_push_applied_force_max_n=68-117N`;
  - `base_contact=0.0866`;
  - recent low point was `base_contact=0.0727`, which allowed alpha promotion from `0.3500`;
  - `wm_valid_fraction=0.8283`;
  - `adapter_scaled_residual_action_l2=0.1385`;
- decision: keep training.  Base contact is not low enough for promotion as a finished policy, but it is below demote
  and safety thresholds, and the adaptive curriculum is behaving as designed.

Implementation:

- added `scripts/rsl_rl/eval_track_adapter_robot_contact_suite.sh`;
  - zero-command no-push stillness;
  - sustained body-frame `+x` pushes at `10/20/30/50N`;
  - repeated random horizontal bump sweep at `50/80/120/160N`;
- added `scripts/rsl_rl/watch_track_adapter_v25b_robot_contact_auto_eval.sh`;
  - watches the active v25b training PID;
  - after training finishes, evaluates the last two v25b checkpoints plus the v24b anchor;
  - output root:
    `logs/track_adapter_81999_stage3_recovery_v25b_robot_contact_staged_20260519_071213_robot_contact_auto_eval_20260519_072335`;
- launched tmux watcher: `ta_v25b_robot_contact_auto_eval`.

Validation:

- `bash -n scripts/rsl_rl/eval_track_adapter_robot_contact_suite.sh` passed;
- `bash -n scripts/rsl_rl/watch_track_adapter_v25b_robot_contact_auto_eval.sh` passed;
- watcher log confirms training is alive and latest watched checkpoints include `model_20550.pt` and `model_20575.pt`.

Known limitation:

- v25b uses trunk force pulses, not actual opponent-body collision/contact geometry;
- the current Track Adapter push gate still uses `failure_push_active` with elapsed time from push start, while the event
  separately logs `force_push_active`; for v25b's max `3.0s` contact duration and `4.4s` push-active window this is
  acceptable, but if sustained contacts remain weak after fixed eval, the next code change should make the command
  gate explicitly aware of active physical force or force-end time.

## 2026-05-19 07:30 UTC - v25b base-contact trend check

Observation:

- user pointed out that `base_contact` may not actually be decreasing;
- checked the current log around iterations `20590-20606`;
- after alpha promotion from `0.3500` to `0.3860`, `base_contact` climbed and then plateaued:
  - last 80 logged values mean: `0.0982`;
  - last 20 logged values mean: `0.1068`;
  - latest value: `0.1053`;
  - current `force_push_force_max_n=148.6368N`;
  - recent applied max force is roughly `78-107N`;
  - `wm_valid_fraction` remains healthy around `0.81-0.83`;
  - scaled residual is not exploding.

Decision:

- the run is still much healthier than v25 and below the configured demote/safety bands, so it should not be killed
  immediately;
- however, this is not a good final trend: `base_contact` is not decreasing inside v25b yet;
- if `base_contact` remains above roughly `0.10` for another short window while alpha stays at `0.386`, prepare/launch
  v25c with stricter contact curriculum:
  - lower promote threshold to around `0.055-0.065`;
  - lower demote threshold to around `0.10-0.11`;
  - start with one pulse per episode;
  - keep sustained weak/medium pushes, but delay repeated-contact pressure until base contact drops;
  - strengthen base-contact avoidance without increasing high reward magnitudes.

## 2026-05-19 07:34 UTC - Any2Track push/disturbance setting note

Checked the Any2Track paper and the current OpenTrack public code.

Paper findings:

- Any2Track Table III lists external-force disturbance as:
  - interval range: `U(5.0, 10.0)`;
  - velocity magnitude range: `U(0.1, 1.0)`;
- the experiment text says random magnitude/direction disturbance is applied to the robot torso during motion tracking;
- the paper does not provide a Newton force, force duration, impulse, contact point, yaw torque, or repeated-contact
  curriculum for push recovery;
- real-world external force is reported as an external constraint via a rope attached to the robot back, not a quantified
  kick/push force.

OpenTrack public-code findings:

- `push_config` uses `interval_range=[5.0, 10.0]` and `magnitude_range=[0.1, 1.0]`;
- the implementation samples a random horizontal direction and magnitude;
- it applies the disturbance as an instantaneous root horizontal velocity addition:
  `qvel[:2] = qvel[:2] + push * push_magnitude`;
- therefore the public setting is closer to root-velocity impulse/domain-randomization than finite physical force in N.

Implication for this project:

- do not directly translate Any2Track's `0.1-1.0` setting into `N`;
- our root-velocity-push branches are closer to the paper's external-force implementation than v25b's finite-force
  robot-contact branch;
- v25b is stricter/more physical for robot-contact robustness, but Any2Track itself gives only weak guidance for
  force duration, contact location, or sustained scuffle settings.

## 2026-05-19 07:55 UTC - v26 hybrid-robust branch prepared

Reason:

- the current v25b training is healthier than v25, but it is not clearly improving contact:
  - latest checked log around iteration `20696/27025`;
  - `base_contact=0.1011`;
  - `force_push_force_max_n=148.6368N`;
  - `force_push_active=0.0856`;
  - `adapter_scaled_residual_action_l2=0.1545`;
  - `wm_valid_fraction=0.8177`;
- this means the branch is below the hard safety gate but already above the desired robustness band;
- continuing v25b would spend long wall-clock time at a permissive curriculum point instead of pushing contact down.

Implementation:

- added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v26_hybrid_robust.sh`;
- added `scripts/rsl_rl/watch_track_adapter_v26_hybrid_robust_auto_eval.sh`;
- v26 resumes from:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-18_15-32-22_track_adapter_81999_stage3_recovery_v24b_curriculum_guard_20260518_153217_ppo_curriculum_guard/model_20525.pt`;
- v26 reuses Stage 3:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-18_12-48-35_track_adapter_81999_stage3_recovery_v23_broad_force_20260518_124831_wm_pretrain_broad_force/model_799.pt`;
- v26 changes the objective from tracking improvement to robustness-first recovery:
  - `task_weight=0.35`;
  - `style_weight=0.035`;
  - `recovery_weight=7.20`;
  - immediate push tracking remains disabled;
  - delayed return-to-command stays weak and delayed;
- v26 adds hybrid disturbance training:
  - initial plan used root-velocity impulse push `push_xy 0.10 -> 0.65`, yaw `0.00 -> 0.12`;
  - this was later corrected before the active v26b run to `push_xy 0.08 -> 0.50`, yaw `0.00 -> 0.08`;
  - physical force push enabled with one pulse per episode;
  - force mixture covers weak sustained pushes, medium shoves, and rare short high-force bumps;
- v26 makes physical-force curriculum stricter:
  - adaptive alpha starts at `0.28`, min `0.18`;
  - promote base-contact gate `0.060`;
  - demote base-contact gate `0.100`;
  - step up `0.007`, step down `0.105`;
- v26 reduces per-iteration cost:
  - rollout length `48` instead of `64`;
  - WM replay updates `4` instead of `8`;
  - keeps `3072` envs and `524288` WM replay size.

Validation:

- `bash -n scripts/rsl_rl/run_track_adapter_stage3_recovery_v26_hybrid_robust.sh` passed;
- `bash -n scripts/rsl_rl/watch_track_adapter_v26_hybrid_robust_auto_eval.sh` passed;
- both scripts were made executable.

Next action:

- stop v25b and its old auto-eval watcher before launching v26, because the GPU should be dedicated to the new branch;
- launch v26 in tmux;
- launch the v26 auto-eval watcher with the actual run directory and training PID.

## 2026-05-19 08:01 UTC - v26 aborted, v26b launched after root-push gate fix

v25b stop:

- stopped tmux session `ta_v25b_robot_contact`;
- stopped tmux session `ta_v25b_robot_contact_auto_eval`;
- final checked v25b log had worsened contact:
  - `base_contact=0.1180`;
  - `force_push_force_max_n=148.6368N`;
  - `adapter_scaled_residual_action_l2=0.1530`;
  - `wm_valid_fraction=0.8256`.

v26 first attempt:

- launched `track_adapter_81999_stage3_recovery_v26_hybrid_robust_20260519_075419`;
- stopped it within the first few iterations;
- failure mode was not base contact, but an always-on recovery gate:
  - root-push interval was too short relative to `failure_push_window_s`;
  - `failure_push_active=1.0000`;
  - `adapter_residual_action_gate=0.9306`;
  - `adapter_scaled_residual_action_l2=2.3551`;
- this would have trained an always-on residual controller, not a recovery adapter, so the run was killed.

Fix:

- kept hybrid training, but made root impulses sparse:
  - root-push interval `6.5-10.0s`;
  - root-push final `xy=0.50`, final yaw `0.08`;
  - `failure_push_window_s=3.2`;
  - failure-push mining sample probability `0.55`;
- reduced residual pressure during recovery:
  - `residual_scale=0.160`;
  - recovery activation target norm `0.055`;
  - recovery activation weight `0.015`;
- reduced physical force activation slightly:
  - force interval `4.2-7.2s`, final `3.4-5.8s`;
  - force activation probability `0.12 -> 0.24`.

v26b launch:

- tmux: `ta_v26b_hybrid_robust`;
- run id: `track_adapter_81999_stage3_recovery_v26b_hybrid_robust_20260519_075714`;
- run dir:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-19_07-57-18_track_adapter_81999_stage3_recovery_v26b_hybrid_robust_20260519_075714_ppo_hybrid_robust`;
- log:
  `logs/track_adapter_81999_stage3_recovery_v26b_hybrid_robust_20260519_075714.log`;
- training PID: `689104`.

Early v26b status:

- iteration time is now about `9.0-9.7s`, down from v25b's roughly `12.6-12.9s`;
- `wm_updates=4`;
- early `base_contact` is low:
  - around iteration `20530`: `base_contact=0.0009`, `adapter_scaled_residual_action_l2=0.0979`;
  - around iteration `20539`: `base_contact=0.0430`, `adapter_scaled_residual_action_l2=0.1229`;
- force/root recovery exposure is active but no longer producing immediate residual explosion:
  - latest checked `failure_push_active=0.7708`;
  - `force_push_active=0.4583`;
  - applied physical max around `82.8N`;
  - curriculum alpha remains around `0.28`.

Auto eval:

- launched tmux `ta_v26b_hybrid_robust_auto_eval`;
- initial watcher launch used output root:
  `logs/track_adapter_81999_stage3_recovery_v26b_hybrid_robust_20260519_075714_hybrid_robust_auto_eval_20260519_080005`;
- watcher will evaluate the tail checkpoints and the v24b anchor after training exits.

## 2026-05-19 08:05 UTC - v26b watcher fix and trend check

Review finding:

- the v26 watcher created only `OUT_ROOT`, but `tee` wrote into each candidate subdirectory;
- fixed `scripts/rsl_rl/watch_track_adapter_v26_hybrid_robust_auto_eval.sh` by adding `mkdir -p "$out"` before eval;
- `bash -n scripts/rsl_rl/watch_track_adapter_v26_hybrid_robust_auto_eval.sh` passed.

Watcher restart:

- stopped the old `ta_v26b_hybrid_robust_auto_eval`;
- restarted the watcher with the fixed script;
- active watcher output root:
  `logs/track_adapter_81999_stage3_recovery_v26b_hybrid_robust_20260519_075714_hybrid_robust_auto_eval_20260519_080451`.

Updated v26b status around iteration `20559/29525`:

- iteration time around `9.1s`;
- `wm_updates=4`;
- `wm_valid_fraction=0.7736`;
- `adapter_scaled_residual_action_l2=0.1879`;
- `failure_push_active=0.3542` in the latest checked iteration, down from the transient `0.7708` early spike;
- `force_push_active=0.0000` in the latest checked iteration, with active-count metrics still showing force exposure in the rollout;
- `force_push_force_max_n=168.64N`;
- `force_push_applied_force_max_n=77.90N`;
- `base_contact=0.0557`.

Interpretation:

- v26b is not showing the v26 always-on residual failure mode;
- contact is close to the `0.060` promote band but still below the `0.100` demote band;
- continue, but watch for `base_contact` drifting above `0.08-0.10` or scaled residual rising toward `0.72`.

## 2026-05-19 08:12 UTC - v26b stopped, v26c force-dominant branch launched

v26b stop reason:

- v26b started acceptably, but then residual usage climbed into the safety band;
- around iteration `20584/29525`:
  - `adapter_scaled_residual_action_l2=0.7349`;
  - safety gate strike `1/4` because the configured limit is `0.7200`;
  - `adapter_residual_action_gate=0.6014`;
  - `failure_push_active=0.7396`;
  - `base_contact=0.0693`;
  - `force_push_force_max_n=178.6360N`;
- this indicates root-impulse/proxy recovery was still too active and was encouraging broad residual use before physical
  force robustness had cleanly improved.

v26c changes:

- kept the robustness-first objective and physical force curriculum;
- made root-velocity impulse only a rare auxiliary:
  - root-push interval `12.0-18.0s`;
  - root-push `xy=0.05 -> 0.35`;
  - root yaw `0.00 -> 0.05`;
  - push-active failure window `2.2s`;
- reduced action-side adapter authority:
  - `residual_scale=0.135`;
  - push gate minimum `0.92`;
  - late push gate minimum `0.35`;
  - high-score push gate minimum `0.96`.

v26c launch:

- tmux: `ta_v26c_force_dominant`;
- run id: `track_adapter_81999_stage3_recovery_v26c_force_dominant_20260519_080749`;
- run dir:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-19_08-07-54_track_adapter_81999_stage3_recovery_v26c_force_dominant_20260519_080749_ppo_hybrid_robust`;
- log:
  `logs/track_adapter_81999_stage3_recovery_v26c_force_dominant_20260519_080749.log`;
- training PID: `690005`.

Early v26c status:

- around iteration `20532`:
  - `base_contact=0.0062`;
  - `adapter_scaled_residual_action_l2=0.0550`;
  - `wm_valid_fraction=0.7772`;
  - `force_push_active=0.1250`;
  - applied force max around `69N`;
- around iteration `20547`:
  - `base_contact=0.0244`;
  - `adapter_scaled_residual_action_l2=0.0039`;
  - `adapter_residual_action_gate=0.0510`;
  - `failure_push_active=0.0747`;
  - `force_push_active=0.0120`;
  - `force_push_force_max_n=168.64N`;
  - applied force max around `48N`;
  - iteration time around `8.95s`.

Interpretation:

- v26c is the first v26-family run that keeps contact low and avoids residual explosion after the early window;
- force exposure is lower than v25/v26b, so the key question is whether adaptive curriculum can increase force without
  reintroducing contact;
- continue v26c and monitor for alpha promotion plus `base_contact < 0.08` and residual below the safety band.

Auto eval:

- launched tmux `ta_v26c_force_dominant_auto_eval`;
- it uses the fixed watcher and training PID `690005`;
- it will evaluate tail checkpoints plus the v24b anchor after training exits.

## 2026-05-19 08:56 UTC - v26c live status check

Status:

- tmux training session `ta_v26c_force_dominant` is alive;
- tmux watcher `ta_v26c_force_dominant_auto_eval` is alive;
- latest checked training iteration: `20838/29525`;
- latest checkpoint sequence in watcher includes `model_20775.pt`, `model_20800.pt`, and `model_20825.pt`.

Metrics at latest check:

- iteration time: about `9.36s`;
- `wm_updates=4`;
- `wm_valid_fraction=0.7514`;
- `force_push_curriculum_alpha=0.4760`;
- `force_push_force_max_n=215.2880N`;
- `force_push_applied_force_max_n=130.7537N`;
- `force_push_active=0.0625`;
- `failure_push_active=0.1840`;
- `adapter_scaled_residual_action_l2=0.0370`;
- `adapter_residual_action_gate=0.1359`;
- `base_contact=0.0645`;
- no safety-gate strike was observed in the latest tail.

Interpretation:

- v26c is currently healthier than v26/v26b because curriculum force has increased while residual remains low;
- `base_contact` around `0.064` is not perfect, but it is below the `0.10` demotion/safety concern band;
- continue the run unless base contact trends above `0.08-0.10` or residual rises toward the `0.72` safety band.

## 2026-05-19 10:06 UTC - v26c live status check

Status:

- tmux training session `ta_v26c_force_dominant` is alive;
- tmux watcher `ta_v26c_force_dominant_auto_eval` is alive;
- latest checked training iteration: `21291/29525`;
- latest saved checkpoint observed: `model_21275.pt`;
- watcher latest checkpoint list includes `model_21225.pt`, `model_21250.pt`, and `model_21275.pt`.

Metrics at latest check:

- iteration time: about `9.66s`;
- `wm_updates=4`;
- `wm_valid_fraction=0.7694`;
- `force_push_curriculum_alpha=0.4970`;
- `force_push_force_max_n=220.2860N`;
- `force_push_applied_force_max_n=138.0374N`;
- `force_push_active=0.0347`;
- `failure_push_active=0.1264`;
- `adapter_scaled_residual_action_l2=0.0816`;
- recent residual tail is roughly `0.065-0.082`;
- `base_contact=0.0720`;
- recent base-contact tail is roughly `0.069-0.072`;
- no safety-gate strike was observed in the latest grep/tail.

Interpretation:

- v26c remains stable and much faster than v25b;
- curriculum has increased from `alpha=0.476` to `0.497`, and force max from about `215N` to `220N`;
- base contact has crept up from the previous `0.064` band to around `0.07`, so this is now a yellow band, not a clean pass;
- continue for now because residual remains low and there is no safety strike, but stop/revise if contact rises above
  `0.08-0.10` or if the curriculum stops progressing while contact stays elevated.

## 2026-05-19 10:22 UTC - strict stop decision: v26c plateau, v27 failed

v26c strict decision:

- v26c was stopped manually instead of continuing to the end;
- reason: it had a stable but weak-looking plateau, not a credible path to the target:
  - latest checked iteration: about `21343/29525`;
  - `force_push_curriculum_alpha=0.4970`;
  - `force_push_force_max_n=220.2860N`;
  - recent applied max force: roughly `110-155N`;
  - `base_contact` drifted from about `0.071` to `0.077-0.080`;
  - `adapter_scaled_residual_action_l2` stayed low around `0.065-0.083`;
- interpretation: residual stayed controlled, but the curriculum was stuck below the stronger-force target and contact was
  trending the wrong way. Continuing would likely spend time rather than produce a stronger push-recovery policy.

v27 experiment:

- added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v27_force_gated_intervention.sh`;
- v27 resumed from v26c `model_21325.pt`;
- root-velocity proxy push was disabled completely;
- physical-force exposure and force-gated adapter authority were increased;
- safety was tightened:
  - `BOOSTER_TRACK_ADAPTER_SAFETY_MAX_BASE_CONTACT=0.095`;
  - `BOOSTER_TRACK_ADAPTER_SAFETY_MAX_SCALED_RESIDUAL_ACTION_L2=0.55`.

v27 outcome:

- launched tmux `ta_v27_force_gated`;
- stopped it within the first few minutes;
- it immediately moved in the wrong direction:
  - `base_contact=0.1411` then `0.1441`;
  - `force_push_force_max_n=199.4240N`;
  - `force_push_active` was high around `0.34-0.41`;
  - `adapter_scaled_residual_action_l2=0.2223` at the latest checked point;
- interpretation: denser physical force and higher recovery authority created too much contact before useful recovery
  behavior emerged. v27 is rejected.

Current action:

- no training is currently running;
- started strict fixed evaluation instead of another blind training continuation;
- tmux: `ta_v26c_strict_eval`;
- output root:
  `logs/v26c_strict_eval_20260519_102200`;
- candidates:
  - v26c `model_20825.pt`;
  - v26c `model_21275.pt`;
  - v26c `model_21325.pt`;
  - v24b anchor `model_20525.pt`;
- evaluation forces:
  - sustained: `30/50/80N`;
  - repeated bumps: `80/120/160N`.

Next decision:

- if none of these beat the v24b anchor on robot-contact fixed eval, stop this reward/curriculum branch and redesign the
  recovery mechanism instead of continuing PPO;
- if one v26c checkpoint clearly beats the anchor without unacceptable contact, promote that checkpoint and stop training.

## 2026-05-19 10:25 UTC - strict eval partial result: v26c_20825 is not promotable

Current tmux state:

- no PPO training is running;
- strict fixed evaluation is still running in tmux `ta_v26c_strict_eval`;
- output root: `logs/v26c_strict_eval_20260519_102200`.

Partial result for `v26c_20825`:

- zero-command/no-push stillness is clean:
  - `fall_rate_mean=0.0`;
  - `base_contact_rate_mean=0.0`;
  - final local drift is small (`x=0.0409m`, `y=0.0057m`).
- sustained zero-command trunk push for `2.5s` is not acceptable:
  - `30N`: fall `0.0938-0.1250`, clean `0.8750-0.9062`;
  - `50N`: fall `0.7188-0.8125`, clean `0.1875-0.2812`;
  - `80N`: fall `1.0000`, clean `0.0000`.

Strict interpretation:

- `v26c_20825` is not a promotion candidate;
- if the later v26c checkpoints and the v24b anchor show the same pattern, this branch should not be continued;
- the current evidence points away from more blind PPO time and toward a recovery-mechanism change.

Additional partial result for `v26c_21275`:

- sustained push, no extra event gate:
  - `30N`: fall `0.1250`, clean `0.8750`;
  - `50N`: fall `0.8750`, clean `0.1250`;
  - `80N`: fall `1.0000`, clean `0.0000`.

Interpretation:

- v26c did not improve later in training on the sustained-push failure mode;
- the branch is trending worse on the exact robot-contact scenario that matters.

## 2026-05-19 10:34 UTC - strict eval sustained-push comparison

Sustained zero-command trunk push, body-frame `+x`, `2.5s`, `32` envs:

| candidate | mode | 30N fall | 50N fall | 80N fall | 50N clean | 80N clean |
|---|---|---:|---:|---:|---:|---:|
| v24b_20525 | event | 0.1250 | 0.8125 | 1.0000 | 0.1875 | 0.0000 |
| v24b_20525 | none | 0.1562 | 0.7812 | 1.0000 | 0.2188 | 0.0000 |
| v26c_20825 | event | 0.1250 | 0.8125 | 1.0000 | 0.1875 | 0.0000 |
| v26c_20825 | none | 0.0938 | 0.7188 | 1.0000 | 0.2812 | 0.0000 |
| v26c_21275 | event | 0.1250 | 0.8438 | 1.0000 | 0.1562 | 0.0000 |
| v26c_21275 | none | 0.1250 | 0.8750 | 1.0000 | 0.1250 | 0.0000 |
| v26c_21325 | event | 0.1875 | 0.8125 | 1.0000 | 0.1875 | 0.0000 |
| v26c_21325 | none | 0.1875 | 0.7812 | 1.0000 | 0.2188 | 0.0000 |

Strict decision:

- none of the v26c checkpoints is promotable for robot-contact robustness;
- v26c does not materially beat the v24b anchor on sustained push;
- the branch should not be continued as-is.

Likely cause:

- the v26c training mixture is too short-bump-heavy for the fixed sustained-push target:
  - the only long-duration component is `8-35N` for `1.2-3.2s`;
  - the `35-75N` component lasts only `0.65-1.8s`;
  - therefore `50N x 2.5s` is largely out of distribution;
- v26c also used `failure_mining_push_active_window_s=2.2s` while force durations can reach `3.2s`, so the recovery
  gate/recovery metrics can expire before long force exposure ends.

## 2026-05-19 10:39 UTC - v28 sustained-scuffle branch prepared

Added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v28_sustained_scuffle.sh`.

Purpose:

- do not continue v26c as-is;
- resume from the least-bad sustained-push checkpoint `v26c model_20825.pt`;
- train the failed target condition directly:
  - sustained low/medium trunk force is now the main disturbance;
  - `35-65N` for `1.8-3.5s` has the largest weight;
  - `55-90N` for `1.2-2.8s` is included;
  - short `80-280N` bumps are retained but secondary;
- disable root-velocity proxy push;
- keep zero/no-push residual closed;
- make recovery active long enough for sustained force:
  - `BOOSTER_TRACK_ADAPTER_FAILURE_PUSH_WINDOW_S=5.0`;
  - command/recovery gate taper goes out to `5.0s`;
- reduce command/style pressure during force windows and increase survival/contact recovery weights.

Validation:

- `bash -n scripts/rsl_rl/run_track_adapter_stage3_recovery_v28_sustained_scuffle.sh` passes.

Launch:

- tmux: `ta_v28_sustained_scuffle`;
- run id: `track_adapter_81999_stage3_recovery_v28_sustained_scuffle_20260519_103753`;
- run dir:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-19_10-37-58_track_adapter_81999_stage3_recovery_v28_sustained_scuffle_20260519_103753_ppo_sustained_scuffle`;
- resumed from:
  `2026-05-19_08-07-54_track_adapter_81999_stage3_recovery_v26c_force_dominant_20260519_080749_ppo_hybrid_robust/model_20825.pt`;
- startup confirmed:
  - `force_duration_mixture_force_floor=0.75`;
  - `failure_mining_push_active_window_s=5.0`;
  - `push_command_gate_full_s=3.5`;
  - `push_command_gate_taper_s=5.0`;
  - root push disabled;
  - physical force push enabled.

Early sanity:

- first logged iteration after resume had a transient force alpha display, then stabilized at `force_push_curriculum_alpha=0.4500`;
- active force envelope became `force_push_force_max_n=241.5000N`;
- early `base_contact` remained very low (`0.0003-0.0028`), so the run is not immediately collapsing.

Early stop:

- stopped tmux `ta_v28_sustained_scuffle`;
- reason: once sustained-force episodes became active, exposure was too dense/strong:
  - `failure_push_active` rose to about `1.0`;
  - `force_push_active` stayed around `0.52-0.81`;
  - `base_contact` rose through `0.0796 -> 0.1051 -> 0.1314 -> 0.1475`;
- this is the same failure class as v27: the target scenario is correct, but the first curriculum step is too harsh.

Correction for next run:

- keep the sustained-scuffle target;
- reduce initial force scale and high-force tail;
- lower force activation probability and increase pulse interval;
- keep the longer `5.0s` recovery window.

## 2026-05-19 10:42 UTC - v28b sustained-scuffle ramp launched

Added and launched `scripts/rsl_rl/run_track_adapter_stage3_recovery_v28b_sustained_scuffle_ramp.sh`.

Changes from v28:

- lower initial force curriculum:
  - `adaptive_alpha_initial=0.32`;
  - `force_duration_mixture_force_floor=0.55`;
  - max force tail capped at `240N`;
- lower exposure density:
  - activation probability `0.06 -> 0.14`;
  - interval `6.5-10.0s`, final `5.5-8.5s`;
- keep target coverage:
  - `42-72N` for `1.6-3.2s`;
  - `18-42N` for `1.8-3.4s`;
  - short `70-240N` bump tail remains secondary;
- retain `failure_push_window_s=5.0`.

Launch:

- tmux: `ta_v28b_sustained_scuffle_ramp`;
- run id: `track_adapter_81999_stage3_recovery_v28b_sustained_scuffle_ramp_20260519_104202`.

Early status:

- v28b is materially healthier than v28;
- force active windows are now training the target without immediate collapse:
  - `force_push_curriculum_alpha=0.3200`;
  - `force_push_force_max_n=166.5600N`;
  - `failure_push_active` reaches about `1.0`;
  - `force_push_active` varies around `0.27-1.0`;
  - `adapter_scaled_residual_action_l2` remains bounded (`0.012-0.079` after push starts);
  - `wm_valid_fraction` stays around `0.77-0.78`;
  - `base_contact` rose gradually but remains below the safety gate:
    `0.0067 -> 0.0141 -> 0.0266 -> 0.0357 -> 0.0415 -> 0.0498 -> 0.0557 -> 0.0624`.

Strict interpretation:

- unlike v28, v28b has a plausible continuation path;
- continue only while base contact stays below roughly `0.09-0.10` or the adaptive curriculum demotes before contact
  becomes unsafe;
- if base contact crosses the safety gate, stop and do not restart this exact schedule.

Latest check:

- tmux `ta_v28b_sustained_scuffle_ramp` is still running;
- by iteration `20848/27825`, `base_contact` had risen to about `0.069` and then stopped rising while force became inactive;
- residual usage dropped back down (`adapter_scaled_residual_action_l2` around `0.004-0.010`);
- current state is still acceptable for continuation, but it is a yellow band, not a pass.

## 2026-05-19 latest - v28b live status

tmux `ta_v28b_sustained_scuffle_ramp` is still running.

Latest observed range around iterations `21021-21038/27825`:

- `force_push_curriculum_alpha=0.3200`;
- `force_push_force_max_n=166.5600N`;
- applied force max is typically `68-107N`;
- `force_push_active` varies by rollout, mostly `0.04-0.34` recently;
- `failure_push_active` varies, occasionally up to about `0.51`;
- `base_contact` stabilized around `0.070-0.077` after peaking near `0.0768`;
- `adapter_scaled_residual_action_l2` remains bounded around `0.010-0.048`;
- `wm_valid_fraction` is mostly `0.74-0.78`.

Strict interpretation:

- this is much better than v28 and should continue for now;
- it is not yet proof of improved recovery because curriculum alpha has not promoted and base contact is still elevated;
- stop/revise if `base_contact` climbs back above `0.09-0.10` without demotion or if residual starts growing without contact improvement.

## 2026-05-19 11:47 UTC - strict continuation gate

v28b live state:

- tmux `ta_v28b_sustained_scuffle_ramp` is still running;
- latest checked iterations were around `21220-21242/27825`;
- `base_contact` improved from the earlier `0.070-0.077` band to about `0.062-0.066`;
- `force_push_curriculum_alpha` remains `0.3200`;
- `force_push_force_max_n=166.5600N`;
- applied force max is usually around `66-101N`;
- `force_push_activation_probability=0.0856`;
- `adapter_scaled_residual_action_l2` is bounded around `0.015-0.043`;
- `wm_valid_fraction` is around `0.756-0.776`.

Strict decision:

- continue v28b for now because contact is no longer rising and has improved slightly;
- do not treat this as success yet because alpha has not promoted and the current force curriculum is still below the
  desired strong-scuffle target;
- no open-ended training: started fixed sustained-push gate eval for saved checkpoint `model_21225.pt`.

Gate eval:

- tmux: `ta_v28b_gate_eval_21225`;
- output root: `logs/v28b_gate_eval_21225_20260519_114727`;
- checkpoint:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-19_10-42-07_track_adapter_81999_stage3_recovery_v28b_sustained_scuffle_ramp_20260519_104202_ppo_sustained_scuffle_ramp/model_21225.pt`;
- test: sustained zero-command body-frame `+x` push at `30/50/80N`, modes `none/event`, `32` envs.

Next hard gate:

- if `50N x 2.5s` is not materially better than v26c/v24b, stop this branch;
- if `50N` improves but `80N` remains bad, continue only long enough to promote curriculum and re-evaluate;
- if live `base_contact` rises back above `0.09-0.10`, stop without waiting for the full run.

## 2026-05-19 11:52 UTC - v28b gate eval failed; training stopped

Fixed sustained-push eval completed for `v28b_21225`.

Output:

- `logs/v28b_gate_eval_21225_20260519_114727/summary.md`

Result:

| candidate | mode | 30N fall | 50N fall | 80N fall | 50N clean | 80N clean |
|---|---|---:|---:|---:|---:|---:|
| v28b_21225 | event | 0.1250 | 0.8125 | 1.0000 | 0.1875 | 0.0000 |
| v28b_21225 | none | 0.1562 | 0.8125 | 1.0000 | 0.1875 | 0.0000 |

Comparison to previous gate:

- v24b event at `50N`: fall `0.8125`, clean `0.1875`;
- v26c_20825 event at `50N`: fall `0.8125`, clean `0.1875`;
- v28b_21225 event at `50N`: fall `0.8125`, clean `0.1875`;
- v28b therefore shows no material fixed-eval improvement despite live training contact looking acceptable.

Decision:

- stopped tmux `ta_v28b_sustained_scuffle_ramp`;
- this reward/curriculum branch should not be extended;
- the remaining problem is not solved by more PPO time on this setup.

Interpretation:

- live `base_contact` around `0.06-0.07` under mild sampled force was misleading;
- the fixed `50N x 2.5s` shove remains out of learned recovery capability;
- the adapter is not learning the required sustained capture/recovery behavior from this reward schedule.

Postmortem detail:

- this was not a simple "adapter gate stayed closed" failure;
- in `event` mode at `50N`, the fixed eval had:
  - `push_active` mean about `0.8889`;
  - `gate_push_active_mean` about `0.8889`;
  - `gate_residual_action_mean` about `0.6466`, max `0.7604`;
  - `residual_norm_mean` about `0.5906`, max `1.0282`;
- therefore the adapter was allowed to intervene and did intervene;
- the failure is that the learned residual did not produce a useful capture/recovery step.  It mostly changed action
  locally while the body was accelerated to about `1.9m/s` max speed under `50N`, and fall/contact remained unchanged.

Implication:

- this is not fixed by simply running longer with the same PPO reward/curriculum;
- AnyAdapter-style architecture is present, but the recovery behavior itself has not been learned;
- the next design must create a recoverable stepping behavior explicitly, then distill or constrain it into the adapter.

## 2026-05-19 12:20 UTC - v29 teacher warm-start implementation started

Decision:

- implement option 2: add a training-only recovery teacher/warm-start signal, then launch PPO;
- do not continue v28b as-is because fixed `50N x 2.5s` sustained push remained unchanged from v24b/v26c;
- use v26c `model_20825.pt` as the practical recovery anchor and the verified v23 Stage 3 world-model pretrain.

Implementation:

- added `recovery_teacher_cfg` to `TrackAdapterAlgorithmCfg`;
- added env vars:
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER`;
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_COEF`;
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_GATE_THRESHOLD`;
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_SCORE_THRESHOLD`;
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_VELOCITY_GAIN`;
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_TILT_GAIN`;
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_ANG_VEL_GAIN`;
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_MAX_DELTA`;
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_NON_LEG_WEIGHT`;
- added a PPO-time recovery teacher loss in `TrackAdapterAMPPPO`;
- the loss acts only on recovery/high-gate minibatch samples and supervises the final ungated scaled residual action;
- the teacher uses inverse-normalized actor/critic batches when normalizers are available;
- K1 action indices use the IsaacLab policy/action order:
  `L/R hip pitch=3/4`, `L/R hip roll=8/9`, `L/R hip yaw=12/13`,
  `L/R knee=16/17`, `L/R ankle pitch=18/19`, `L/R ankle roll=20/21`;
- attached runner normalizers to the algorithm so the teacher can recover raw-ish velocities from normalized storage.

Training script:

- added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v29_teacher_warmstart.sh`;
- defaults:
  - resume: `2026-05-19_08-07-54_track_adapter_81999_stage3_recovery_v26c_force_dominant_20260519_080749_ppo_hybrid_robust/model_20825.pt`;
  - Stage 3 pretrain: `2026-05-18_12-48-35_track_adapter_81999_stage3_recovery_v23_broad_force_20260518_124831_wm_pretrain_broad_force/model_799.pt`;
  - physical sustained scuffle force mixture with adaptive alpha initial `0.25`;
  - WM replay/alternating updates remain enabled;
  - recovery teacher enabled with coefficient `8.0` and max residual target delta `0.14`.

Next:

- run syntax/compile checks;
- launch v29 in tmux through the standard script path, which sources `/workspace/Isaac_uv_template/.venv/bin/activate`.

## 2026-05-19 13:03 UTC - v29 teacher warm-start launched

Verification before launch:

- `py_compile` passed for:
  - `rsl_rl/rsl_rl/algorithms/track_adapter_amp_ppo.py`;
  - `rsl_rl/rsl_rl/runners/track_adapter_runner.py`;
  - `source/booster_rl_tasks/booster_rl_tasks/tasks/manager_based/beyond_mimic/robots/k1/run_amp/ppo_cfg.py`;
- `bash -n` passed for:
  - `scripts/rsl_rl/run_track_adapter_stage3_recovery_v29_teacher_warmstart.sh`;
  - `scripts/rsl_rl/run_track_adapter_stage3_recovery_v26_hybrid_robust.sh`;
  - `scripts/rsl_rl/run_track_adapter_stage3_to_ppo.sh`;
- local tensor smoke for `_recovery_teacher_target()` passed with K1 action order and bounded residual target.

Launch:

- tmux session: `ta_v29_teacher_warmstart`;
- run id: `track_adapter_81999_stage3_recovery_v29_teacher_warmstart_20260519_130234`;
- log: `logs/track_adapter_81999_stage3_recovery_v29_teacher_warmstart_20260519_130234.log`;
- run dir: `logs/rsl_rl/run_amp_track_adapter/2026-05-19_13-02-39_track_adapter_81999_stage3_recovery_v29_teacher_warmstart_20260519_130234_ppo_teacher_warmstart`;
- iteration range after resume: `20825 -> 28825`;
- Stage 3 pretrain validation passed:
  - checkpoint `model_799.pt`;
  - `wm=0.413089`;
  - `wm_valid_fraction=0.928`;
  - `psi_s` target dim `81`;
  - autoregressive horizon `20`.

Early runtime check:

- PPO started and reached at least iteration `20826`;
- teacher metrics are logging:
  - `Mean recovery_teacher loss`;
  - `Mean recovery_teacher_mask_fraction`;
  - `Mean recovery_teacher_target_l2`;
- WM replay is active:
  - `wm_replay_inserted=147456`;
  - `wm_replay_size` filled to `524288`;
  - `wm_updates=4`;
- early force-push episode metrics are still zero because the first logged episodes have not yet reached push windows/timeouts. Recheck after several minutes before judging curriculum behavior.

## 2026-05-19 13:06 UTC - v29 early push/teacher check passed

At iteration `20838-20839`:

- `force_push_active` became non-zero (`0.27-0.79`);
- `force_push_curriculum_alpha=0.2500`, matching the v29 initial curriculum;
- effective current force envelope:
  - `force_push_force_min_n=10.5750`;
  - `force_push_force_max_n=164.5000`;
  - applied max force about `88-108N`;
- `base_contact` is `0.0540-0.0614`, below the v29 safety threshold `0.115`;
- `failure_base_contact=0.0` and `failure_terminated=0.0` in the logged episode windows;
- `recovery_teacher` is active:
  - loss about `0.013`;
  - mask fraction about `0.057`;
  - target residual L2 about `0.014`;
- WM replay remains active with `wm_updates=4`, `wm_valid_fraction` about `0.776`.

Current decision:

- keep v29 running;
- do not judge success from live reward;
- first hard evidence should be a fixed sustained-push eval once a meaningful checkpoint is saved after the teacher branch has adapted.

## 2026-05-19 14:03 UTC - v29 one-hour status

Training is still running in tmux session `ta_v29_teacher_warmstart`.

Latest observed state:

- iteration: about `21216/28825`;
- elapsed: about `01:00:12`;
- latest saved checkpoint observed: `model_21200.pt`;
- `base_contact`: about `0.062`;
- `time_out`: about `0.938`;
- `failure_base_contact=0.0`;
- `failure_terminated=0.0`;
- `force_push_curriculum_alpha=0.1800`;
- current force envelope:
  - min about `9.9N`;
  - max about `153.7N`;
  - applied max about `87-97N`;
- `force_push_active`: about `0.047-0.058` in the latest episode logs;
- `recovery_teacher` remains active:
  - loss about `0.0098`;
  - mask fraction about `0.0556`;
  - target L2 about `0.0151`;
- WM replay remains active:
  - `wm=0.095`;
  - `wm_valid_fraction=0.7756`;
  - `wm_updates=4`;
  - replay size at `524288`;
- adapter residual usage is controlled:
  - `adapter_residual_action_gate` about `0.10`;
  - `adapter_scaled_residual_action_l2` about `0.029`;
  - `adapter_ungated_scaled_residual_action_l2` about `0.083`.

Interpretation:

- positive: training is alive, teacher loss is being applied, WM replay is stable, and base-contact is below the safety
  threshold `0.115`;
- negative: adaptive force curriculum has fallen to its lower bound `0.18`, so the learner is not yet strong enough to
  promote force intensity;
- status: continue for now, but success is not proven. Need fixed sustained-push eval once there is more teacher-adapted
  training, preferably around `model_21500` or `model_22000` if base-contact stays in the current band.

## 2026-05-19 14:19 UTC - v29 stopped; v30 privileged teacher launched

User correction accepted:

- the v29 heuristic teacher is not enough for the stated target;
- a small hand-coded residual direction loss should not be expected to produce `300N`-class push recovery;
- the proper route is a training-only privileged recovery teacher, then adapter distillation.

Implementation added:

- `TrackAdapterActorCritic` now supports an optional `privileged_teacher_residual_actor`;
- teacher actor input is `[actor_obs, critic_obs, frozen_base_action]`;
- teacher action is still residual-on-frozen-base:
  `teacher_action = base_action + residual_gate * teacher_residual_scale * teacher_residual`;
- `TrackAdapterAMPPPO` now supports `privileged_teacher_cfg`:
  - when `train=True`, PPO actions/log-probs are produced by the privileged teacher actor;
  - optimizer updates teacher residual actor, critic, and action noise only;
  - frozen AMP base stays frozen;
- old adapter checkpoints without teacher keys can be strict-loaded into teacher-enabled policy; missing teacher keys are initialized locally;
- added adapter distillation support:
  - `privileged_teacher_distill_cfg`;
  - `BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL`;
  - the adapter can later train against the privileged teacher's action target in recovery/high-score minibatches.

Scripts added:

- `scripts/rsl_rl/run_track_adapter_stage3_recovery_v30_privileged_teacher.sh`;
- `scripts/rsl_rl/run_track_adapter_stage3_recovery_v31_teacher_distill.sh`.

Verification:

- `py_compile` passed for modified module/algorithm/config files;
- `bash -n` passed for v30/v31 scripts;
- strict load smoke passed:
  - base: AMP axis-v2 `model_81999.pt`;
  - resume: v26c `model_20825.pt`;
  - teacher-enabled policy loaded the checkpoint and produced `[4, 22]` teacher actions.

Training transition:

- stopped tmux `ta_v29_teacher_warmstart`;
- launched tmux `ta_v30_privileged_teacher`;
- run id: `track_adapter_81999_stage3_recovery_v30_privileged_teacher_20260519_141921`;
- log: `logs/track_adapter_81999_stage3_recovery_v30_privileged_teacher_20260519_141921.log`;
- resume: v26c `model_20825.pt`;
- teacher residual scale: `0.42`;
- force curriculum initial alpha: `0.30`;
- current configured force max reaches `312N` at the initial curriculum point and can promote toward the final `480N` envelope.

Early v30 check:

- `Privileged Teacher Residual MLP` is printed in the runtime model;
- Stage 3 pretrain loaded successfully;
- PPO resume loaded without optimizer;
- early iterations started around `20826/29825`;
- initial `force_push_curriculum_alpha=0.3000`;
- initial current force envelope reports `force_push_force_max_n=312.0000`;
- early episode push metrics are still zero until episodes reach push windows;
- first logged `base_contact` is near zero, so no immediate collapse at startup.

Next:

- wait until force push is active in logged episodes;
- judge the teacher by fixed sustained-push/kick eval, not reward;
- only after teacher passes the hard force tests should v31 adapter distillation be launched.

## 2026-05-19 14:33 UTC - v30 stopped; v30b correction prepared

v30 was stopped before spending more time:

- tmux `ta_v30_privileged_teacher` was killed;
- latest live status had `base_contact` around `0.21`;
- force curriculum had already demoted to alpha `0.22`;
- current force max was still about `293N`;
- teacher/adapter residual diagnostics were still effectively `0.0000`.

Conclusion:

- this was not a path to `300N` recovery;
- the teacher head was being exposed to hard force before it had learned any useful recovery action;
- the teacher optimizer/exploration settings were too conservative because they inherited adapter-preservation values;
- the run was also wasting per-iteration time on world-model updates that the privileged teacher does not use.

Implementation changes:

- added explicit actor-source diagnostics in `TrackAdapterActorCritic`;
- when the privileged teacher path is used, logs can now expose `privileged_teacher_active` and teacher residual L2;
- added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v30b_privileged_teacher.sh`.

v30b settings:

- resume: v26c `model_20825.pt`;
- privileged teacher residual scale: `0.70`;
- teacher output init scale: `0.012`;
- action std: `0.14`;
- PPO LR: `1.5e-5`;
- PPO-time WM loss/updates: disabled;
- physical force mixture final max: `520N`;
- curriculum initial alpha/floor: `0.10` / `0.12`, giving an initial current max near `110N`;
- curriculum promotes toward `300N+` only after low base-contact and push-failure rates;
- stable/no-push residual gate remains closed.

Verification:

- `bash -n scripts/rsl_rl/run_track_adapter_stage3_recovery_v30b_privileged_teacher.sh` passed;
- `py_compile` passed for the modified Track Adapter module, algorithm, and PPO config.

Training:

- launched tmux `ta_v30b_privileged_teacher`;
- run id: `track_adapter_81999_stage3_recovery_v30b_privileged_teacher_20260519_143316`;
- log: `logs/track_adapter_81999_stage3_recovery_v30b_privileged_teacher_20260519_143316.log`;
- early runtime confirms:
  - action std override is `0.14`;
  - learning time is about `0.4s` because PPO-time WM updates are disabled;
  - total iteration time is about `6.7s`;
  - `adapter_privileged_teacher_active=1.0000`;
  - `adapter_ungated_scaled_residual_action_l2` and
    `adapter_privileged_teacher_ungated_scaled_residual_action_l2` are non-zero (`~0.0050`) from the start;
  - `recovery_residual_activation` is active (`~0.0044`).

Next checks:

- wait for enough episode completions to make force/base-contact metrics meaningful;
- if `base_contact` stays high after force metrics become non-zero, stop and lower initial force or adjust reward;
- if residual remains non-zero and base-contact stays controlled, let curriculum promote toward the `300N+` region.

Early force-active check:

- by iteration `20832/25825`, force metrics are non-zero;
- current force envelope is `4.16N - 108.16N`;
- sampled/applied max force reached about `104N`;
- `base_contact` remains very low (`0.0008`);
- teacher residual remains non-zero:
  - `adapter_privileged_teacher_ungated_scaled_residual_action_l2 ~0.0079`;
  - `adapter_scaled_residual_action_l2 ~0.0013`;
- this is the intended bootstrap behavior: survive the first physical-force band, then allow adaptive promotion.

## 2026-05-19 15:22 UTC - v32/v33 complete recovery-teacher pipeline implemented

The three missing pieces from the recovery-teacher plan were implemented.

1. Explicit privileged force/contact input:
   - added `track_adapter_recovery_privileged_state`;
   - when `BOOSTER_TRACK_ADAPTER_PRIVILEGED_RECOVERY_OBS=1`, the critic/teacher observation receives:
     - physical wrench resultant, norm, elapsed/remaining/duration;
     - force-push command metrics;
     - left/right foot contact and trunk contact;
   - actor/deploy observation remains unchanged.
2. Teacher trajectory replay + BC/DAgger:
   - added `PrivilegedTeacherDistillReplayBuffer`;
   - current rollout recovery samples are filtered by recovery gate/score and inserted into persistent replay;
   - replay BC updates run after PPO minibatches, so PPO old log-probs are not invalidated;
   - added env controls for replay size, batch size, warm-start updates, per-iteration updates, and minimum samples.
3. Failure reset curriculum:
   - runner now records rolling pre-failure root/joint/command/force states;
   - on some reset episodes, it restores a cached pre-failure state and optionally replays the cached force;
   - observations are refreshed after this overwrite so the next rollout state matches the injected recovery state.

Safety/runtime corrections made before launch:

- strict shape loading now skips incompatible auxiliary tensors when critic/teacher input size changes;
- contact body IDs are resolved inside the privileged obs function if Isaac has not pre-resolved the `SceneEntityCfg`;
- after failure-reset overwrite, critic/privileged obs now comes from refreshed observations, while terminal AMP state still comes from the step output.

Scripts added:

- `scripts/rsl_rl/run_track_adapter_stage3_recovery_v32_privileged_teacher_reset.sh`;
- `scripts/rsl_rl/run_track_adapter_stage3_recovery_v33_teacher_dagger.sh`.

Verification:

- `bash -n` passed for v32/v33 scripts;
- `py_compile` passed for modified Track Adapter algorithm, module, runner, obs, env config, and PPO config files.

Training transition:

- stopped the incomplete teacher-only tmux branch `ta_v30b_privileged_teacher`;
- next launch is `v32`, the privileged teacher + failure-reset branch;
- after v32 produces a viable teacher checkpoint, launch `v33` for deployable adapter distillation.

Startup fix:

- first v32 launch failed while loading the resumed `privileged_obs_normalizer`;
- cause: privileged critic obs increased from `78` to `96` dims after adding recovery privileged state;
- fixed `TrackAdapterRunner.load()` to skip checkpoint obs/privileged normalizers on shape mismatch when
  `BOOSTER_TRACK_ADAPTER_STRICT_BASE_AUXILIARY_STATE=0`;
- re-ran `py_compile` for the runner and privileged observation module.

Training:

- launched tmux `ta_v32_privileged_teacher_reset`;
- run id: `track_adapter_81999_stage3_recovery_v32_privileged_teacher_reset_20260519_152525`;
- log: `logs/track_adapter_81999_stage3_recovery_v32_privileged_teacher_reset_20260519_152525.log`;
- run directory:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-19_15-25-30_track_adapter_81999_stage3_recovery_v32_privileged_teacher_reset_20260519_152525_ppo_privileged_teacher_reset`;
- first live iteration reached `20825/27825`;
- runtime confirms:
  - privileged teacher MLP input is `193`, reflecting actor obs + expanded critic obs + base action;
  - critic input is `160`, reflecting expanded privileged obs;
  - v26c checkpoint actor/critic mismatched tensors were skipped;
  - checkpoint privileged normalizer mismatch was skipped;
  - `adapter_privileged_teacher_active=1.0000`;
  - teacher residual is non-zero at startup, though still very small;
  - initial episode metrics are mostly zero because episode completions have not yet populated force/reward summaries.

Review fixes before final relaunch:

- stopped the first successful v32 process at about `20830/27825`;
- reason: it proved the teacher branch runs, but it did not yet export file-backed teacher replay for v33 warm-start;
- fixed two v33/runtime issues:
  - recovery-score mask now passes `action_dim`, so v33 replay insertion will not crash;
  - privileged recovery obs now uses `env.device`/robot device instead of default `cuda:0`;
- added file-backed privileged-teacher replay:
  - v32 captures non-terminal recovery-window teacher mean actions into replay;
  - v32 saves replay to `logs/<run_id>_teacher_distill_replay.pt`;
  - v33 auto-loads the latest v32 teacher replay unless an explicit load path is supplied;
  - v33 runs replay BC warm-start before the first PPO rollout/update, not after the PPO minibatches;
  - runner checkpoints include the replay path when replay is saved;
- re-ran `py_compile` and `bash -n` for the modified files/scripts.

Final v32 relaunch:

- launched tmux `ta_v32_privileged_teacher_reset`;
- run id: `track_adapter_81999_stage3_recovery_v32_privileged_teacher_reset_20260519_153124`;
- log: `logs/track_adapter_81999_stage3_recovery_v32_privileged_teacher_reset_20260519_153124.log`;
- run directory:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-19_15-31-29_track_adapter_81999_stage3_recovery_v32_privileged_teacher_reset_20260519_153124_ppo_privileged_teacher_reset`;
- replay path:
  `logs/track_adapter_81999_stage3_recovery_v32_privileged_teacher_reset_20260519_153124_teacher_distill_replay.pt`;
- early confirmation:
  - tmux is running;
  - `adapter_privileged_teacher_active=1.0000`;
  - `privileged_teacher_capture_replay_inserted=17151`;
  - replay buffer is full at `16384` samples;
  - replay file exists at about `246M`;
  - standalone replay loader successfully loaded `16384` samples.

Automation:

- added `scripts/rsl_rl/watch_v32_launch_v33_teacher_dagger.sh`;
- launched tmux `ta_v32_to_v33_chain`;
- it polls v32 and launches tmux `ta_v33_teacher_dagger` only if `model_27800.pt` exists;
- it passes the explicit v32 run, `model_27800.pt`, and latest teacher replay path into v33;
- if v32 stops before producing `model_27800.pt`, v33 is not launched automatically.

Early force-active status:

- by iteration `20838/27825`, force metrics are populated;
- force curriculum alpha is `0.10`;
- current force max is `116.48N`;
- applied force max is roughly `47-85N` in recent logged episodes;
- `base_contact` is still low, around `0.002-0.010`;
- privileged teacher residual is active and non-zero:
  - `adapter_privileged_teacher_active=1.0000`;
  - `adapter_privileged_teacher_ungated_scaled_residual_action_l2 ~0.011`;
- teacher replay capture continues inserting recovery-window samples every iteration.

Live status at `20857/27825`:

- tmux `ta_v32_privileged_teacher_reset` is still running;
- tmux `ta_v32_to_v33_chain` is still waiting for `model_27800.pt`;
- latest saved checkpoint is `model_20850.pt`;
- teacher replay file is present and still about `246M`;
- force curriculum is slowly promoting:
  - alpha `0.1111`;
  - current force max `121.95N`;
  - applied force max about `89.91N`;
  - force active fraction `0.1750`;
- `base_contact=0.0208`, still below the v32 promote threshold and far below the safety stop band;
- teacher remains active:
  - `adapter_privileged_teacher_active=1.0000`;
  - teacher ungated scaled residual L2 about `0.0115`;
  - actual gated scaled residual L2 about `0.0023`;
- velocity errors in the latest completed episodes are moderate:
  - xy `0.1397`;
  - yaw `0.1556`;
- strict read: this is healthy for the early 120N band, but it has not yet proven 300N-class recovery.
## 2026-05-19 16:02 UTC - v32 stall detected and restarted

- tmux pane stopped updating because the process stalled, not just because the screen was stale;
- last logged iteration was `20857/27825`;
- log mtime stopped at `2026-05-19 15:37:04 UTC`;
- process was still alive and GPU was busy, but no new iteration/checkpoint appeared for over 20 minutes;
- latest safe checkpoint is `logs/rsl_rl/run_amp_track_adapter/2026-05-19_15-31-29_track_adapter_81999_stage3_recovery_v32_privileged_teacher_reset_20260519_153124_ppo_privileged_teacher_reset/model_20850.pt`;
- decision: kill the stalled v32 process and continue from `model_20850.pt` in a new v32 run.


Correction: the first restart command did not propagate resume env into the tmux server, so it briefly relaunched from v26c `model_20825.pt`.
Stopped that process and relaunched with explicit in-pane exports from v32 `2026-05-19_15-31-29_track_adapter_81999_stage3_recovery_v32_privileged_teacher_reset_20260519_153124_ppo_privileged_teacher_reset/model_20850.pt`.

Restart confirmation:

- new log: `logs/track_adapter_81999_stage3_recovery_v32_privileged_teacher_reset_20260519_160359.log`;
- resume path confirmed in log:
  `2026-05-19_15-31-29_track_adapter_81999_stage3_recovery_v32_privileged_teacher_reset_20260519_153124_ppo_privileged_teacher_reset/model_20850.pt`;
- iteration is advancing again:
  - `20850/27850`;
  - `20851/27850`;
  - `20852/27850`;
  - `20853/27850`;
  - `20854/27850`;
- log mtime updated at `2026-05-19 16:05:03 UTC`;
- early force metrics are zero immediately after restart because episode summaries have not yet reached force windows.

## 2026-05-19 16:14 UTC - v32 repeated stall root cause fix

- v32 advanced to `20873/27850`, then log stopped at `2026-05-19 16:08:10 UTC`;
- the stop point matches the 25th update after resume, where file-backed teacher replay save would run before logging the next iteration;
- replay files are `246M` and the filesystem is `97%` used, so repeated large replay saves are a bad runtime path;
- patched replay saving so `replay_save_interval <= 0` means disabled, and added `replay_save_on_checkpoint`;
- v32 script now disables periodic replay save and checkpoint replay save; existing teacher replay remains available for v33;
- restarting again from `2026-05-19_16-04-04_track_adapter_81999_stage3_recovery_v32_privileged_teacher_reset_20260519_160359_ppo_privileged_teacher_reset/model_20850.pt`.

## 2026-05-19 16:30 UTC - switched v32 to chunked supervisor

- the run stalled again at `20881/27850` with no log update after `2026-05-19 16:21:32 UTC`;
- this is no longer only replay-save related; the Isaac/PPO process can hang after tens of iterations;
- added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v32_chunked_teacher_reset.sh`;
- the supervisor runs short PPO chunks, saves every 5 iterations, and resumes from the latest v32 checkpoint;
- fixed the supervisor checkpoint parser and confirmed latest checkpoint is `model_20875.pt`;
- replacing the long-running v32 tmux process with the chunked supervisor.

Chunked supervisor confirmation:

- tmux `ta_v32_privileged_teacher_reset` now runs the chunked supervisor, not a single long PPO process;
- chunk0 resumed from `model_20875.pt`;
- chunk0 advanced and saved:
  - `model_20880.pt`;
  - `model_20885.pt`;
  - `model_20890.pt`;
  - final `model_20894.pt`;
- chunk1 started automatically from `model_20894.pt`;
- this confirms the supervisor can cross the previous stall point and carry progress forward by checkpoint.

## 2026-05-19 16:40 UTC - chunking policy corrected

- chunking is acceptable only as a crash-containment strategy, not as a learning algorithm change;
- found an important flaw: v32 script forced `BOOSTER_TRACK_ADAPTER_LOAD_OPTIMIZER=0`, so chunk restarts were dropping Adam state;
- patched v32 script to respect external `BOOSTER_TRACK_ADAPTER_LOAD_OPTIMIZER`;
- patched chunked supervisor to set `BOOSTER_TRACK_ADAPTER_LOAD_OPTIMIZER=1` for same-branch resumes;
- latest checkpoint before restart is `model_20905.pt`;
- restarting chunked supervisor so subsequent chunks preserve optimizer state.

## 2026-05-19 16:43 UTC - optimizer-state restart verified

- found a second overwrite path: v32 inherits v26 defaults, and `run_track_adapter_stage3_recovery_v26_hybrid_robust.sh` still forced `BOOSTER_TRACK_ADAPTER_LOAD_OPTIMIZER=0`;
- patched v26, v31, and v33 wrappers to respect external `BOOSTER_TRACK_ADAPTER_LOAD_OPTIMIZER` while keeping the default at `0` for branch-changing resumes;
- stopped the chunk that had started with `[ppo] load_optimizer=0`;
- latest saved checkpoint before clean restart was `model_20920.pt`;
- restarted tmux `ta_v32_privileged_teacher_reset` with:
  - `BOOSTER_TRACK_ADAPTER_LOAD_OPTIMIZER=1`;
  - `CHUNK_ITERATIONS=20`;
  - `CHUNK_SAVE_INTERVAL=5`;
  - `TARGET_ITER=27800`;
- restart verification:
  - log: `logs/track_adapter_81999_stage3_recovery_v32_chunked_teacher_reset_20260519_164900_optstate_from20920_chunk0_20260519_164022.log`;
  - `[ppo] resume_checkpoint=model_20920.pt`;
  - `[ppo] load_optimizer=1`;
  - learning advanced from `20920/20940` to at least `20922/20940`.

Interpretation: chunked training is now acceptable as an operational workaround for the Isaac/PPO hang because same-branch restarts preserve optimizer state.  It is still not a root-cause fix for the hang itself.

## 2026-05-19 16:53 UTC - switched back from chunked to single v32 run

- user correctly flagged that chunking resets the displayed `Time elapsed` every chunk and is not a clean single training run;
- stopped the chunked supervisor after it reached `model_20977.pt`;
- verified the apparent privileged normalizer warning was from the frozen AMP base checkpoint, not the resumed v32 checkpoint:
  - frozen AMP base privileged normalizer shape: `(1, 78)`;
  - current v32 critic obs / privileged normalizer shape: `(1, 96)`;
  - resumed v32 checkpoint `model_20958.pt` privileged normalizer shape: `(1, 96)`;
- patched base privileged normalizer loading so this expected base mismatch is quiet unless `BOOSTER_TRACK_ADAPTER_VERBOSE_NORMALIZER_MISMATCH=1`;
- restarted as a single tmux training run from:
  `2026-05-19_16-47-39_track_adapter_81999_stage3_recovery_v32_chunked_teacher_reset_20260519_164900_optstate_from20958_chunk2_20260519_164734_ppo_privileged_teacher_reset/model_20977.pt`;
- new single-run log:
  `logs/track_adapter_81999_stage3_recovery_v32_single_from20977_20260519_165300.log`;
- restart verification:
  - `[ppo] resume_checkpoint=model_20977.pt`;
  - `[ppo] load_optimizer=1`;
  - first printed iteration: `20977/27801`;
  - no privileged normalizer mismatch message in the new log.

## 2026-05-19 17:10 UTC - v32 stall root cause isolated

- the single v32 run from `model_20977.pt` stalled after printing `Learning iteration 21009/27801`;
- the process stayed alive with GPU utilization at `100%`, no Python exception, and no new log/checkpoint for several minutes;
- last safe checkpoint from that run was:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-19_16-52-16_track_adapter_81999_stage3_recovery_v32_single_from20977_20260519_165300_ppo_privileged_teacher_reset/model_21000.pt`;
- external attach with `py-spy` was blocked by container ptrace restrictions, so stack dumping has to be installed inside the process before launch;
- added `BOOSTER_TRAIN_DEBUG_SIGNALS=1` support to `scripts/rsl_rl/train.py`, so `SIGUSR1` can dump Python stacks in future diagnostic runs;
- added `BOOSTER_TRACK_ADAPTER_PHASE_TRACE=1` runner/PPO phase breadcrumbs to locate any future hang boundary;
- code review of the stall path found the highest-risk new v32 feature is failure reset:
  - it writes root/joint state directly after IsaacLab's normal `env.step()` reset pipeline has already run;
  - replayed force modifies sidecar push state but does not synchronously call the articulation external-wrench API or update command push state;
  - this can leave PhysX, sensors, manager state, command state, and privileged push observations inconsistent;
- restarted from `model_21000.pt` with:
  - `BOOSTER_TRACK_ADAPTER_FAILURE_RESET_CURRICULUM=0`;
  - `BOOSTER_TRACK_ADAPTER_FAILURE_RESET_REPLAY_FORCE=0`;
  - physical force push, privileged teacher training, and teacher replay capture still enabled;
  - debug signals and phase tracing enabled;
- new run:
  `logs/track_adapter_81999_stage3_recovery_v32_no_failure_reset_from21000_20260519_171000.log`;
- confirmation so far:
  - resumed with `load_optimizer=1`;
  - passed the previous failure point (`21009`) and reached at least `21021`;
  - base-contact remains low in the observed window, around `0.0035-0.0178`;
  - no repeated `Time elapsed` reset from chunking.

Operational decision: keep failure-reset disabled for the current full training.  Reintroduce it only after it is moved into IsaacLab's reset/event pipeline or made strictly guarded with same-episode validation and synchronized external-wrench/command state updates.

## 2026-05-19 17:13 UTC - no-failure-reset v32 status

- current run remains a single tmux process:
  `track_adapter_81999_stage3_recovery_v32_no_failure_reset_from21000_20260519_171000`;
- failure reset is disabled, but physical force push, privileged teacher training, and teacher replay capture remain enabled;
- the run passed the previous stall point and reached at least `21046/27801`;
- checkpoint save boundary `21025` completed successfully;
- latest observed training band:
  - force curriculum alpha about `0.118`;
  - force max about `125N`;
  - applied force max about `96N` in the latest pane;
  - base_contact about `0.022-0.026`;
  - scaled residual L2 remains small, about `0.003-0.006`;
  - velocity errors vary by episode mix, roughly `xy 0.13-0.20`, `yaw 0.21-0.26` in recent logs.

Interpretation: operationally stable so far and still learning under physical force pushes.  This has not yet proven high-force recovery; it is still early in the force curriculum.

## 2026-05-19 17:40 UTC - no-failure-reset v32 30-minute status

- run is still active in tmux `ta_v32_privileged_teacher_reset`;
- elapsed training time in the active log is about `00:34:04`;
- iteration progressed from resume `21000` to about `21205`, with no stall recurrence;
- checkpoint cadence is healthy:
  - `model_21025.pt`;
  - `model_21050.pt`;
  - `model_21075.pt`;
  - `model_21100.pt`;
  - `model_21125.pt`;
  - `model_21150.pt`;
  - `model_21175.pt`;
  - `model_21200.pt`;
- current force curriculum is still early:
  - alpha about `0.148`;
  - force max about `140N`;
  - recent applied force max about `70-110N`;
- recent base contact rose from the early `0.02` band to about `0.058-0.061`;
- recent velocity errors are still reasonable for a recovery-heavy run:
  - xy roughly `0.11-0.16`;
  - yaw roughly `0.19-0.26`;
- residual use remains controlled:
  - scaled residual L2 roughly `0.002-0.006`;
  - privileged teacher ungated scaled residual L2 roughly `0.010-0.020`.

Interpretation: operationally stable and still usable, but not cleanly “excellent” because base_contact is creeping up and the force curriculum appears to be holding around the `140N` band.  Continue for now; intervene if base_contact trends toward `0.08-0.10` or curriculum stops improving for a long window.

## 2026-05-19 18:02 UTC - no-failure-reset v32 near-60-minute status

- run is still active as a single tmux process, with no stall recurrence;
- elapsed training time in the active log is about `00:56:21`;
- iteration reached about `21338/27801`;
- checkpoint cadence remains healthy up to at least:
  - `model_21225.pt`;
  - `model_21250.pt`;
  - `model_21275.pt`;
  - `model_21300.pt`;
  - `model_21325.pt`;
- current force curriculum:
  - alpha remains about `0.148`;
  - force max remains about `140N`;
  - recent applied force max is mostly `80-105N`;
- current stability:
  - base_contact is about `0.052-0.056`, slightly better than the previous `0.060` peak but still higher than the early `0.02` band;
  - no recent error/exception in the log;
  - residual remains controlled, with scaled residual L2 mostly around `0.003-0.008`;
- current tracking:
  - xy error roughly `0.11-0.15`;
  - yaw error roughly `0.21-0.24`.

Interpretation: operationally stable and not degrading catastrophically.  The concern is curriculum stagnation near `140N`; if it does not promote beyond this band after a longer window, evaluate a checkpoint and consider relaxing the adaptive promotion gate or reducing the contact penalty pressure rather than waiting blindly.

## 2026-05-19 18:45 UTC - no-failure-reset v32 near-100-minute status

- run is still active as a single tmux process, with no stall recurrence;
- elapsed training time in the active log is about `01:39:28`;
- iteration reached about `21594/27801`;
- checkpoint cadence remains healthy up to at least:
  - `model_21350.pt`;
  - `model_21375.pt`;
  - `model_21400.pt`;
  - `model_21425.pt`;
  - `model_21450.pt`;
  - `model_21475.pt`;
  - `model_21500.pt`;
  - `model_21525.pt`;
  - `model_21550.pt`;
  - `model_21575.pt`;
- current force curriculum is stagnant:
  - alpha remains about `0.148`;
  - force max remains about `140N`;
  - recent applied force max is mostly `80-110N`;
- current stability is acceptable:
  - base_contact improved/stabilized around `0.049-0.054`;
  - no recent error/exception in the log;
  - residual remains controlled, with scaled residual L2 mostly around `0.006-0.012`;
- current tracking:
  - xy error roughly `0.10-0.13`;
  - yaw error roughly `0.19-0.23`.

Interpretation: operationally stable, but not making the desired force-curriculum progress.  Continue only as a short further observation window; if force max is still pinned at `140N` around `21600-21650`, run fixed evaluation and adjust the adaptive promotion/contact thresholds rather than spending the rest of the run blindly.

## 2026-05-19 19:00 UTC - stopped stagnant v32 and prepared v34 force-escape teacher

- stopped tmux `ta_v32_privileged_teacher_reset` and the `ta_v32_to_v33_chain` watcher;
- latest saved v32 checkpoint before stop:
  - `logs/rsl_rl/run_amp_track_adapter/2026-05-19_17-05-22_track_adapter_81999_stage3_recovery_v32_no_failure_reset_from21000_20260519_171000_ppo_privileged_teacher_reset/model_21650.pt`;
- stop reason:
  - curriculum remained pinned at `force_push_curriculum_alpha ~= 0.148`;
  - exposed force max stayed around `140.13N`;
  - base-contact stayed around `0.05`, which was too high for the old promote gate `0.035` but not high enough to prove the teacher was learning harder recovery;
  - privileged-teacher residual remained small (`adapter_scaled_residual_action_l2 ~= 0.01`, ungated around `0.019`), so the run was not producing a strong recovery teacher;
- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v34_teacher_force_escape.sh`;
- v34 changes:
  - resume from v32 `model_21650.pt` with a fresh optimizer;
  - keep failure-reset disabled;
  - keep weak sustained pushes in the force-duration mixture;
  - expose the teacher to 200N+ force samples immediately by starting adaptive alpha at `0.38` with a `0.50` force floor;
  - relax promote/demote thresholds so alpha can advance under realistic non-zero contact (`promote_base_contact=0.075`, demote at `0.145`);
  - increase teacher exploration, teacher residual scale, and recovery residual activation pressure so PPO cannot remain a near-zero residual policy while collecting upright rewards.

## 2026-05-19 19:01 UTC - v34 restarted with replay persistence

- first v34 launch was stopped after only a few startup iterations because the inherited v32 defaults would not persist teacher replay;
- patched `scripts/rsl_rl/run_track_adapter_stage3_recovery_v34_teacher_force_escape.sh` to:
  - save privileged-teacher distill replay every `25` iterations and on checkpoint;
  - increase replay capacity/saved samples to `65536`;
  - reduce teacher-stage stable residual suppression (`stable_weight=0.35`, `zero_command_weight=1.50`, `ungated_stable_weight=0.08`);
- restarted tmux `ta_v34_teacher_force_escape` as:
  - run id `track_adapter_81999_stage3_recovery_v34_teacher_force_escape_replay_20260519_190041`;
  - log `logs/track_adapter_81999_stage3_recovery_v34_teacher_force_escape_replay_20260519_190041.log`;
  - resume `2026-05-19_17-05-22_track_adapter_81999_stage3_recovery_v32_no_failure_reset_from21000_20260519_171000_ppo_privileged_teacher_reset/model_21650.pt`;
  - fresh optimizer (`BOOSTER_TRACK_ADAPTER_LOAD_OPTIMIZER=0`).

Early v34 signal at iteration `21656`:

- `force_push_curriculum_alpha=0.3800`;
- exposed `force_push_force_max_n=427.8N`;
- observed `force_push_applied_force_max_n=303.1N`;
- `force_push_active=0.3909`;
- `Episode_Termination/base_contact=0.0328`;
- `adapter_ungated_scaled_residual_action_l2=0.1554`;
- `adapter_scaled_residual_action_l2=0.0700`;
- teacher replay size reached `65536`.

Interpretation: the force curriculum is no longer stuck in the `140N` band, and the privileged teacher is no longer a near-zero residual actor.  Continue monitoring whether base contact remains controlled as episode statistics mature.

## 2026-05-19 19:04 UTC - stopped over-aggressive v34 and created v34b

- current replay-enabled v34 did solve the curriculum-stall symptom but was too aggressive:
  - force max stayed around `403N`;
  - applied max was around `285-308N`;
  - `base_contact` jumped to roughly `0.25-0.29` within a few iterations;
- stopped tmux `ta_v34_teacher_force_escape` rather than letting a bad high-contact teacher run continue;
- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v34b_teacher_force_escape_balanced.sh`;
- v34b keeps initial exposure above the old v32 band but backs off from the immediate 400N class:
  - initial alpha `0.26`, minimum `0.22`;
  - force floor `0.45`;
  - mixture tail reduced to `380-520N` with low probability;
  - expected initial force max is around the `250-300N` class, with promotion toward `400N+` if contact is controlled;
  - replay persistence from v34 remains enabled.

Early v34b signal at iteration `21656`:

- run id `track_adapter_81999_stage3_recovery_v34b_teacher_force_escape_balanced_20260519_190436`;
- `force_push_curriculum_alpha=0.2600`;
- exposed `force_push_force_max_n=308.36N`;
- observed `force_push_applied_force_max_n=228.30N`;
- `force_push_active=0.3863`;
- `Episode_Termination/base_contact=0.0168`;
- `adapter_ungated_scaled_residual_action_l2=0.1029`;
- `adapter_scaled_residual_action_l2=0.0551`;
- teacher replay size is `65536`.

Interpretation: v34b is currently in the intended band: stronger than v32's stagnant `140N` curriculum, but not immediately destroying episodes like v34.

Later v34b signal at iteration `21678-21679`:

- checkpoint `model_21675.pt` was saved;
- force max had demoted to about `296.9N`, with applied max around `195-197N`;
- `Episode_Termination/base_contact` nevertheless rose to about `0.3645`;
- stopped tmux `ta_v34b_teacher_force_escape` instead of continuing.

Interpretation: even the `~300N` initial envelope is too high when combined with the current teacher exploration/reward setup.  Treat v34/v34b as failed escalation probes, not promotion candidates.

## 2026-05-19 19:11 UTC - prepared v34c stable ramp

- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v34c_teacher_force_stable_ramp.sh`;
- v34c starts above the old v32 `140N` stall but below the v34b failure band:
  - expected initial force max roughly `200-230N`;
  - mixture keeps weak sustained pushes and rare short stronger tail up to `440N`;
  - lower action std/LR and smaller recovery activation target than v34b;
  - stricter base-contact demotion/stop gates;
  - still saves teacher replay for distillation.

Early v34c signal at iteration `21657`:

- run id `track_adapter_81999_stage3_recovery_v34c_teacher_force_stable_ramp_20260519_191104`;
- `force_push_curriculum_alpha=0.1800`;
- exposed `force_push_force_max_n=223.52N`;
- observed `force_push_applied_force_max_n=167.52N`;
- `force_push_active=0.3163`;
- `Episode_Termination/base_contact=0.0267`;
- `adapter_ungated_scaled_residual_action_l2=0.0655`;
- `adapter_scaled_residual_action_l2=0.0470`;
- teacher replay size is `65536`.

Interpretation: v34c is currently above the old `140N` exposure but not immediately unstable.  Let it reach the first new checkpoint and continue only if base contact stays bounded.

Later v34c signal:

- stopped tmux `ta_v34c_teacher_force_ramp`;
- no remaining `track_adapter_81999_stage3_recovery_v34c` / `train.py` process after stop;
- stop reason:
  - v34c initially looked learnable, but later `base_contact` rose to about `0.206`;
  - this repeated the v34/v34b pattern: raising force alone makes high-contact teacher data before producing reliable recovery;
  - continuing would train a worse privileged teacher instead of a stronger one.

## 2026-05-19 19:25 UTC - implemented v34d teacher overlay on existing adapter base

- changed `TrackAdapterActorCritic` so privileged teacher action can use the already-trained adapter action as its base:
  - new config/env flag: `privileged_teacher_use_adapter_base` / `BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_USE_ADAPTER_BASE`;
  - when enabled, teacher action becomes `adapter_action_mean(obs, history, gate) + teacher_residual_overlay`;
  - the adapter base is evaluated under `torch.no_grad()`, so teacher PPO still updates only the privileged teacher head, critic, and action std;
- threaded `history` through all privileged-teacher paths:
  - rollout action;
  - PPO minibatch distribution reconstruction;
  - online teacher target query for distillation;
  - replay teacher target insertion;
- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v34d_teacher_adapter_base_force_ramp.sh`;
- v34d design:
  - resume from v32 `model_21650.pt`;
  - load optimizer state (`BOOSTER_TRACK_ADAPTER_LOAD_OPTIMIZER=1`);
  - enable `BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_USE_ADAPTER_BASE=1`;
  - keep failure-reset disabled;
  - save teacher replay every `25` iterations and on checkpoint;
  - start above the old `~140N` curriculum stall but below the failed v34c startup band:
    - force floor `0.34`;
    - alpha initial `0.15`;
    - force max `440N`;
    - expected first displayed max is roughly `175-190N`, then it promotes if contact is controlled;
  - promote/demote thresholds:
    - promote base-contact `0.060`;
    - demote base-contact `0.115`;
    - safety stop base-contact `0.135` after `50` relative iterations.

Verification before launch:

- `bash -n` passed for v34d/v34/v32 scripts;
- `py_compile` passed for modified actor/algorithm/config files;
- tensor smoke passed for `privileged_teacher_use_adapter_base=True` with `privileged_teacher_act`, `privileged_teacher_action_target`, and normal adapter `act`.

v34d launch and stop:

- launched tmux `ta_v34d_teacher_adapter_base_force_ramp`;
- run id `track_adapter_81999_stage3_recovery_v34d_teacher_adapter_base_force_ramp_20260519_192347`;
- startup confirmed:
  - resume `v32/model_21650.pt`;
  - `load_optimizer=1`;
  - `adapter_privileged_teacher_active=1.0`;
  - teacher capture replay reached `65536` samples;
  - initial force max was above v32's stalled band: `force_push_force_max_n=193.16N`, applied max up to about `154N`;
- stopped manually before the safety gate waited out its full relative window:
  - `base_contact` rose from `0.043` to `0.062`, then `0.075`, then about `0.16-0.18`;
  - adaptive alpha demoted from `0.15` to `0.12`;
  - displayed force max was only `~184-193N`, so this was not an acceptable strong-force teacher;
  - continuing would have trained another high-contact teacher.

Diagnosis:

- the architectural change was necessary but not sufficient;
- the v32 privileged teacher head was trained with `base_mean = frozen AMP actor(obs)` as its third input;
- v34d changed that third input to `base_mean = adapter actor(obs, history, gate)` but reused both teacher weights and optimizer moments;
- this makes the teacher residual actor start from a mismatched input distribution.

## 2026-05-19 19:35 UTC - prepared v34e fresh teacher head

- added `TrackAdapterActorCritic.reset_privileged_teacher_residual_actor(output_init_scale=...)`;
- added runner env flag `BOOSTER_TRACK_ADAPTER_RESET_PRIVILEGED_TEACHER_ON_RESUME=1`;
  - when enabled, the privileged teacher residual actor is reset after checkpoint load;
  - optimizer loading is skipped if the teacher is reset, even if `load_optimizer` was requested;
- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v34e_teacher_adapter_base_fresh_head.sh`;
- v34e changes from v34d:
  - `BOOSTER_TRACK_ADAPTER_LOAD_OPTIMIZER=0`;
  - `BOOSTER_TRACK_ADAPTER_RESET_PRIVILEGED_TEACHER_ON_RESUME=1`;
  - `BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_USE_ADAPTER_BASE=1`;
  - teacher residual scale reduced to `0.50`;
  - teacher head output init `0.010`;
  - initial force step reduced but still above the old v32 stall:
    - force max `420N`;
    - force floor `0.32`;
    - alpha initial `0.12`;
    - expected first displayed max is around `165-170N`;
  - demotion/safety tightened:
    - demote base-contact `0.095`;
    - safety base-contact `0.120`.

Verification:

- `bash -n` passed for v34e/v34d scripts;
- `py_compile` passed for modified actor/algorithm/runner/config files;
- tensor smoke passed for reset + teacher action target shape.

v34e launch and stop:

- launched tmux `ta_v34e_teacher_adapter_base_fresh_head`;
- run id `track_adapter_81999_stage3_recovery_v34e_teacher_adapter_base_fresh_head_20260519_193104`;
- startup confirmed:
  - fresh optimizer (`Loading Track Adapter checkpoint without optimizer state`);
  - privileged teacher reset after resume;
  - `adapter_privileged_teacher_active=1.0`;
  - teacher capture replay filled to `65536`;
- early v34e was better than v34d:
  - displayed force max `168.672N`;
  - `base_contact` initially around `0.043-0.051`;
- stopped manually after the signal degraded:
  - curriculum demoted to alpha `0.100`;
  - displayed force max dropped to `162.960N`;
  - `base_contact` reached `0.1206`;
  - teacher residual stayed nearly zero (`adapter_scaled_residual_action_l2` about `0.0005-0.0013`);
  - this is not a stronger recovery teacher; it is the adapter base surviving until contact rises.

## 2026-05-19 19:42 UTC - prepared v34f capture-step prior for privileged teacher

- changed `TrackAdapterAMPPPO` so the existing recovery teacher loss can optionally apply during privileged teacher PPO;
- new env/config flag:
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_APPLY_TO_PRIVILEGED=1`;
- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v34f_teacher_capture_step_prior.sh`;
- v34f keeps:
  - fresh privileged teacher head;
  - fresh optimizer;
  - adapter-base teacher overlay;
  - mild force start above the old `~140N` stall;
- v34f adds:
  - recovery teacher loss during privileged teacher training;
  - stronger leg/capture-step target (`coef=12`, `max_delta=0.18`);
  - lower posture-survival dense weights and stronger directional step/support/first-contact terms;
  - stronger base-contact penalty.

Verification:

- `bash -n` passed for v34f/v34e scripts;
- `py_compile` passed for modified actor/algorithm/runner/config files;
- tensor smoke passed for privileged recovery-teacher loss while `train_privileged_teacher=True`.

## 2026-05-19 19:48 UTC - v34f/v34g diagnosis and v34h force ladder

v34f was stopped after confirming the capture-step prior was active but too weak:

- `Mean recovery_teacher loss` appeared during privileged teacher PPO;
- `recovery_teacher_target_l2` stayed around `0.013-0.018`;
- privileged/adapted residual remained near zero;
- this showed the prior was wired, but it did not know about the physical force vector.

Implemented v34g force-aware recovery teacher target:

- parsed appended privileged recovery features from critic observations:
  - force `xy`;
  - force norm;
  - equivalent push-delta `xy`;
  - push-delta norm;
- added force and push-delta terms to `_recovery_teacher_target(...)`;
- logged `recovery_teacher_force_norm` and `recovery_teacher_push_delta_norm`;
- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v34g_teacher_force_aware_prior.sh`;
- synthetic tensor smoke produced a non-trivial target (`target_l2 ~= 0.15`) when force features were present.

v34g launch:

- tmux: `ta_v34g_teacher_force_aware_prior`;
- run id: `track_adapter_81999_stage3_recovery_v34g_teacher_force_aware_prior_20260519_194752`;
- resume: v32 `model_21650.pt`;
- early logs confirmed:
  - `recovery_teacher_force_norm` and `recovery_teacher_push_delta_norm` are non-zero when force samples appear;
  - `force_push_force_max_n` is still only `162.96N` because alpha is pinned at `0.10`;
  - base-contact rose into the `0.04-0.07` range as episodes lengthened;
  - residual is still small (`adapter_scaled_residual_action_l2` roughly `0.001-0.002`).

Decision:

- v34g is useful as a wiring check, not as the final curriculum;
- the next run must not wait at the old mild band if it is acceptable;
- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v34h_teacher_force_ladder.sh`.

v34h changes:

- starts the effective force ceiling around the next band above v34g:
  - force mixture max `520N`;
  - force floor `0.48`;
  - alpha initial/min `0.18`;
  - expected initial displayed max is about `250N`;
- promotes faster when stable:
  - step up `0.025`;
  - promote base-contact `0.065`;
  - promote episode length `420`;
  - min force-active `0.070`;
- demotes only when clearly bad:
  - demote base-contact `0.125`;
  - step down `0.035`;
- strengthens the force-aware capture prior:
  - teacher coef `24`;
  - force gain `2.60`;
  - push-delta gain `1.85`;
  - max delta `0.30`;
  - residual activation target `0.24`.

Verification:

- `bash -n scripts/rsl_rl/run_track_adapter_stage3_recovery_v34h_teacher_force_ladder.sh` passed.

v34g stop / v34h launch:

- v34g produced `model_21675.pt`;
- last checked v34g signal:
  - force max stayed `162.96N`;
  - `base_contact=0.0979`;
  - `adapter_scaled_residual_action_l2=0.0004`;
  - this is not a satisfactory recovery policy, but it is enough evidence that the v34g force-aware prior is wired;
- stopped v34g and launched v34h in tmux:
  - tmux: `ta_v34h_teacher_force_ladder`;
  - run id: `track_adapter_81999_stage3_recovery_v34h_teacher_force_ladder_20260519_195400`;
  - resume run: `2026-05-19_19-47-57_track_adapter_81999_stage3_recovery_v34g_teacher_force_aware_prior_20260519_194752_ppo_teacher_force_aware_prior`;
  - checkpoint: `model_21675.pt`;
  - command used the required venv: `source /workspace/Isaac_uv_template/.venv/bin/activate`.
- first live v34h force check:
  - after command metrics populated, `force_push_curriculum_alpha=0.18`;
  - displayed `force_push_force_max_n=298.272N`;
  - applied max force about `224.6N`;
  - `base_contact=0.0182`;
  - `recovery_teacher_target_l2=0.0482`, higher than v34g's force-active windows;
  - `adapter_scaled_residual_action_l2=0.0045`, still small but no longer completely closed.

v34h stop:

- stopped v34h after the stronger band proved too aggressive:
  - `force_push_force_max_n=298.272N`;
  - applied max about `200-220N`;
  - `base_contact` jumped to `0.3724`, then `0.3877`;
  - residual remained too small (`adapter_scaled_residual_action_l2 ~= 0.004-0.005`);
- this confirms the user criticism: external force should be increased only after the current band is actually stable.

v34i correction:

- added separate privileged-teacher actor grad clipping:
  - env flag `BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_SEPARATE_ACTOR_GRAD_CLIP`;
  - env flag `BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_ACTOR_MAX_GRAD_NORM`;
  - teacher actor gradients can now avoid being clipped together with large critic/value gradients;
- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v34i_teacher_actor_clip_ladder.sh`;
- v34i starts around `~220-230N` displayed max instead of `~300N`;
- v34i uses stronger recovery teacher loss (`coef=180`) and separate teacher actor clip (`2.5`);
- promotion is gated by `base_contact <= 0.040`, push-failure <= `0.045`, and episode length >= `500`.

Verification:

- `bash -n scripts/rsl_rl/run_track_adapter_stage3_recovery_v34i_teacher_actor_clip_ladder.sh` passed;
- `python -m py_compile` passed for:
  - `rsl_rl/rsl_rl/algorithms/track_adapter_amp_ppo.py`;
  - `source/booster_rl_tasks/booster_rl_tasks/tasks/manager_based/beyond_mimic/robots/k1/run_amp/ppo_cfg.py`.

v34i launch:

- tmux: `ta_v34i_teacher_actor_clip_ladder`;
- run id: `track_adapter_81999_stage3_recovery_v34i_teacher_actor_clip_ladder_20260519_200500`;
- resume run: `2026-05-19_19-47-57_track_adapter_81999_stage3_recovery_v34g_teacher_force_aware_prior_20260519_194752_ppo_teacher_force_aware_prior`;
- checkpoint: `model_21675.pt`;
- command used required venv: `source /workspace/Isaac_uv_template/.venv/bin/activate`.

v34i early signal:

- force metrics populated correctly:
  - `force_push_curriculum_alpha=0.1400`;
  - `force_push_force_max_n=224.8480N`;
  - applied max `~171-180N`;
- separate actor clip + higher coef increased the recovery teacher signal:
  - `recovery_teacher loss ~= 0.26-0.29`;
  - `recovery_teacher_target_l2 ~= 0.066-0.069`;
  - `adapter_scaled_residual_action_l2 ~= 0.0077-0.0083`;
- base-contact is not clean yet:
  - `0.0338` then `0.0649`;
  - continue only while it stays below the demotion band; stop if it settles above `0.10`.

v34i stop / v34j launch:

- stopped v34i after full-length episodes showed the band was still too hard:
  - curriculum demoted to alpha `0.12`;
  - displayed force max `217.984N`;
  - `base_contact` reached `0.264-0.267`;
  - teacher/residual signal was stronger than v34g, but not enough to make the `~220N` band safe;
- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v34j_teacher_actor_clip_stabilize.sh`;
- v34j explicitly follows the curriculum rule:
  - start near `~155-170N`;
  - do not promote unless full-length episodes have `base_contact <= 0.030`;
  - demote if `base_contact >= 0.080`;
  - keep separate teacher actor grad clipping from v34i;
- verification:
  - `bash -n scripts/rsl_rl/run_track_adapter_stage3_recovery_v34j_teacher_actor_clip_stabilize.sh` passed;
  - `python -m py_compile` passed for modified PPO/config files;
- launched v34j in tmux:
  - tmux: `ta_v34j_teacher_actor_clip_stabilize`;
  - run id: `track_adapter_81999_stage3_recovery_v34j_teacher_actor_clip_stabilize_20260519_201300`;
  - resume: v34g `model_21675.pt`;
  - required venv sourced.

v34j early signal:

- force max is now in the intended stabilization band:
  - `force_push_curriculum_alpha=0.1000`;
  - `force_push_force_max_n=155.4000N`;
  - applied max about `90-94N` in the displayed windows;
- base-contact is controlled in early/mid episodes:
  - `base_contact=0.0287-0.0312`;
- not promoted yet:
  - episode length is still only `~350-365`;
  - promotion requires full-length/near-full-length episodes, so this is only a "continue watching" signal.

v34j near-full episode signal:

- current run remains active in tmux;
- after near-full/full episodes:
  - curriculum demoted to alpha `0.0800`;
  - displayed force max `149.5200N`;
  - applied max roughly `82-106N`;
  - episode length `~686-723`, with timeout fraction around `0.92`;
  - `base_contact=0.0755-0.0786`;
- decision:
  - this is much better than the v34h/v34i jumps, but it is not a pass;
  - do not strengthen the curriculum yet;
  - continue v34j at the `~150N` band and require base-contact to drop below the `0.030` promote gate before increasing force.

## 2026-05-20 03:05 UTC - v34j 7h status

- v34j is still running:
  - tmux: `ta_v34j_teacher_actor_clip_stabilize`;
  - process elapsed: about `7h`;
  - latest observed iteration: `24024/24875`;
  - latest checkpoint present: `model_24000.pt`;
  - ETA: about `2h31m`;
- process health is normal:
  - iteration time about `10.6-10.7s`;
  - GPU process alive;
  - checkpoints continue to save every 25 iterations;
- learning status is plateaued, not promoted:
  - `force_push_curriculum_alpha=0.0800`;
  - `force_push_force_max_n=149.5200N`;
  - applied max roughly `90-100N`;
  - near-full episode length `~940-990`;
  - `base_contact` is stable around `0.070-0.076`;
  - residual remains small but nonzero (`adapter_scaled_residual_action_l2 ~= 0.0055-0.0070`);
- decision:
  - this is safer than v34h/v34i but not strong enough and not improving toward the `0.030` promotion gate;
  - do not raise force from this run;
  - if no late improvement appears by completion, run fixed evaluation on `model_24000.pt`/final and prepare the next branch rather than extending v34j blindly.

## 2026-05-20 03:12 UTC - v34j stopped as plateau

- user asked whether the teacher branch had grown relative to the pre-teacher approach;
- inspected the live run at about `7h`;
- latest observed state:
  - iteration around `24039/24875`;
  - latest checkpoint present: `model_24025.pt`;
  - `force_push_curriculum_alpha=0.0800`;
  - `force_push_force_max_n=149.5200N`;
  - applied max usually `~90-100N`;
  - near-full episode length `~950-990`;
  - `base_contact` plateaued around `0.074-0.077`;
  - `adapter_scaled_residual_action_l2 ~= 0.0055-0.0070`;
- verdict:
  - the teacher branch changed internal supervision and made residual nonzero;
  - it did not produce material push-recovery growth;
  - it did not unlock promotion beyond the mild `~150N` band;
  - v34j is not worth extending;
- stopped tmux run `ta_v34j_teacher_actor_clip_stabilize`.

## 2026-05-20 03:25 UTC - v35 Any2Track adapt-only branch prepared

- user clarified the desired Any2Track interpretation:
  - frozen AMP owns nominal locomotion and velocity tracking;
  - adapter should learn external-disturbance adaptation, not a new locomotion policy;
  - teacher/distillation are unnecessary if we follow that split directly;
- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v35_any2track_adapt_only.sh`;
- v35 removes the failed teacher stack:
  - `BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER=0`;
  - `BOOSTER_TRACK_ADAPTER_TRAIN_PRIVILEGED_TEACHER=0`;
  - `BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL=0`;
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER=0`;
  - `BOOSTER_TRACK_ADAPTER_FAILURE_RESET_CURRICULUM=0`;
- v35 keeps the Any2Track sequence:
  - Stage 3 world-model/history pretrain with frozen base and zero residual;
  - PPO fine-tune of the deployable adapter/critic/noise with persistent WM replay and alternating WM updates;
- adapter PPO reward is now adaptation-only:
  - task weight `0.0`;
  - AMP style weight `0.0`;
  - recovery weight `22.0`;
  - reg weight `0.015`, with recovery reg scale `0.0`;
  - push-time command/style gates are `0.0`;
  - stable/no-push residual gates stay closed;
- force curriculum now mixes weak sustained contact-like pushes and short stronger bumps:
  - broad mixture from `12-55N` sustained to rare `460-620N` short pulses;
  - adaptive alpha starts at `0.16`, promotes only under low base contact/push failure, and demotes on bad contact;
  - pushes can start under standing and moving commands, but tracking reward is not used to train the adapter.

## 2026-05-20 03:26 UTC - v35 launched in tmux

- verification:
  - `bash -n scripts/rsl_rl/run_track_adapter_stage3_recovery_v35_any2track_adapt_only.sh` passed;
  - frozen base checkpoint exists:
    `logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt`;
- launched tmux:
  - session: `ta_v35_any2track_adapt_only`;
  - run id: `track_adapter_81999_stage3_recovery_v35_any2track_adapt_only_20260520_032600`;
  - script: `scripts/rsl_rl/run_track_adapter_stage3_recovery_v35_any2track_adapt_only.sh`;
  - required venv is sourced by `run_track_adapter_stage3_to_ppo.sh`;
- initial Stage 3 signal:
  - pretrain started cleanly;
  - iteration `0/650`: `wm=0.5302`, `wm_valid_fraction=0.9258`;
  - iteration `1/650`: `wm=0.5104`, `wm_valid_fraction=0.9250`;
  - no teacher/distill/recovery-teacher path is active in this branch.

## 2026-05-20 03:35 UTC - v35b reused validated Stage 3 pretrain

- user clarified that "fine-tune" meant a possible second blending phase after recovery-only learning, and asked whether the Stage 3 pretrain can be reused;
- inspected the existing v23 pretrain checkpoint:
  - path: `logs/rsl_rl/run_amp_track_adapter/2026-05-18_12-48-35_track_adapter_81999_stage3_recovery_v23_broad_force_20260518_124831_wm_pretrain_broad_force/model_799.pt`;
  - base checkpoint matches current v35 base `model_81999.pt`;
  - `history_length=79`;
  - target metadata matches `source=psi_s`, `target_dim=81`, `horizon=20`;
  - collection policy is `base_residual_zero`;
  - final metrics are finite: `wm=0.413089`, `wm_valid_fraction=0.928`;
- decision:
  - v23 pretrain is compatible and good enough as a warm-start;
  - v35's exact disturbance distribution is broader (`12-620N` vs v23 `8-340N`), but PPO-time persistent WM replay will keep updating the history/world-model on v35 rollouts;
  - because time matters, stopped the fresh v35 Stage 3 run instead of spending roughly another full pretrain cycle;
- launched v35b:
  - tmux: `ta_v35b_any2track_reuse_wm`;
  - run id: `track_adapter_81999_stage3_recovery_v35b_any2track_adapt_only_reusewm_20260520_033600`;
  - `BOOSTER_TRACK_ADAPTER_SKIP_STAGE3=1`;
  - `BOOSTER_TRACK_ADAPTER_PRETRAINED_WM` points to the v23 `model_799.pt`;
- validation and PPO start:
  - Stage 3 checkpoint validation passed;
  - runner printed `Loaded world-model pretrain`;
  - PPO reached learning iteration `0/12000`;
  - WM replay inserted `196608` samples and ran `3` WM updates;
  - task/style branch remains disabled: `task/style/recovery/reg=0.0/0.0/22.0/0.015`;
  - iteration `0` episode rewards/metrics are still mostly zero because no completed episodes are available in the initial log.

## 2026-05-20 03:44 UTC - v35c strict survival branch

- user asked whether the active run truly trains only the anti-fall/recovery objective and not AMP/style signals;
- confirmed v35b had no AMP/style contribution, but it still kept tiny extra terms:
  - delayed return-to-command weight `0.04`;
  - recovery return-to-command weight `0.004`;
  - reward-manager regularization group weight `0.015`;
- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v35c_any2track_strict_survival.sh`;
- v35c strict settings:
  - `BOOSTER_TRACK_ADAPTER_TASK_WEIGHT=0.0`;
  - `BOOSTER_TRACK_ADAPTER_STYLE_WEIGHT=0.0`;
  - `BOOSTER_TRACK_ADAPTER_REG_WEIGHT=0.0`;
  - `BOOSTER_TRACK_ADAPTER_PUSH_RECOVERY_TRACK_WEIGHT=0.0`;
  - `BOOSTER_TRACK_ADAPTER_PUSH_RECOVERY_DELAYED_TRACK_WEIGHT=0.0`;
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_RETURN_CMD_WEIGHT=0.0`;
  - `BOOSTER_TRACK_ADAPTER_SUDDEN_STOP_ENVS=0.0`;
  - teacher/distill/failure-reset remain disabled;
- fixed wrapper launch bug:
  - direct `exec` failed because the base v35 script is not executable;
  - wrapper now uses `exec bash ...`;
- stopped v35b and launched v35c:
  - tmux: `ta_v35c_strict_survival`;
  - run id: `track_adapter_81999_stage3_recovery_v35c_any2track_strict_survival_20260520_034600`;
  - reused validated v23 Stage 3 pretrain `model_799.pt`;
- initial v35c PPO check:
  - Stage 3 validation passed;
  - PPO reached iteration `1/12000`;
  - `Mean amp loss=0.0000`, `Mean amp_grad_pen=0.0000`, `Mean amp_policy_pred=0.0000`, `Mean amp_expert_pred=0.0000`;
  - tracking terms still appear in `Episode_Reward/...` because IsaacLab logs raw environment reward terms, but TrackAdapterRunner applies group weights before the PPO reward; with task/style/reg weights zero, those raw task/style/reg logs do not contribute to adapter PPO reward;
  - recovery return-to-command logs are `0.0000`, delayed tracking logs are `0.0000`;
  - remaining non-PPO-reward learning signals are the world-model auxiliary loss and residual discipline losses.

## 2026-05-20 05:57 UTC - v35c two-hour status

- v35c is still running:
  - tmux: `ta_v35c_strict_survival`;
  - latest observed iteration: `687/12000`;
  - elapsed time: `02:12:24`;
  - iteration time: about `11.5-11.6s`;
- process health:
  - no crash or stall;
  - WM replay is full at `524288` samples;
  - WM updates continue at `3` per PPO iteration;
  - `wm` is finite and improved to about `0.336-0.341`;
  - `wm_valid_fraction` is about `0.766-0.774`;
- strict reward check:
  - `Mean amp loss`, `amp_grad_pen`, `amp_policy_pred`, and `amp_expert_pred` remain `0.0000`;
  - `recovery_return_to_command_exp` and delayed tracking terms remain `0.0000`;
  - raw tracking terms are still printed by IsaacLab but remain outside the weighted PPO reward due to `task/style/reg=0`;
- recovery status:
  - `force_push_curriculum_alpha=0.1000`, pinned at the minimum;
  - displayed force band is about `3.144-162.44N`;
  - applied max is roughly `96-119N` in latest windows;
  - `base_contact` is about `0.0758-0.0761`;
  - timeout fraction is about `0.924`;
  - mean episode length is near full (`~973-979`);
- adapter usage:
  - residual action gate is open around `0.44`;
  - scaled residual action L2 is still tiny (`~0.0002`);
  - ungated scaled residual L2 is also tiny (`~0.0004`);
- verdict:
  - this is stable enough to keep running mechanically, but it is not yet learning a strong recovery adapter;
  - the branch is pinned at the low force band and the adapter is barely intervening;
  - if the goal is fast push-recovery improvement, the next correction should increase recovery-only residual activation/authority and/or make the survival reward depend more directly on force-active windows.

## 2026-05-20 06:07 UTC - v36 strict-survival intervention restart

- stopped v35c because it was not acceptable for the recovery target:
  - the objective was strict survival-only, but the adapter stayed nearly zero;
  - `force_push_curriculum_alpha` stayed pinned at `0.1000`;
  - `adapter_scaled_residual_action_l2` was only about `0.0002`;
- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v36_strict_survival_intervention.sh`;
- v36 keeps the Any2Track-style strict-survival objective:
  - no AMP/style PPO loss;
  - no command-tracking reward group;
  - no delayed return-to-command reward;
  - no reward-manager regularization group;
- v36 deliberately forces the adapter out of the near-zero solution:
  - resumes from v35c `model_700.pt`;
  - loads weights without optimizer state;
  - perturbs resumed adapter output layers with `weight_std=0.0025`;
  - raises residual authority to `residual_scale=0.24`;
  - raises policy action std to `0.16`;
  - strengthens recovery-only residual activation to `1.60 @ target_norm=0.18`;
  - lowers the activation gate threshold to `0.10`;
- launch details:
  - tmux: `ta_v36_strict_intervention`;
  - run id: `track_adapter_81999_stage3_recovery_v36_strict_survival_intervention_20260520_060500`;
  - run dir: `logs/rsl_rl/run_amp_track_adapter/2026-05-20_06-05-31_track_adapter_81999_stage3_recovery_v36_strict_survival_intervention_20260520_060500_ppo_intervention`;
  - reused Stage 3 pretrain: `logs/rsl_rl/run_amp_track_adapter/2026-05-18_12-48-35_track_adapter_81999_stage3_recovery_v23_broad_force_20260518_124831_wm_pretrain_broad_force/model_799.pt`;
- early status at iteration `706/12700`:
  - process is alive; iteration time about `11.4-11.5s`;
  - `Mean amp loss`, `amp_grad_pen`, `amp_policy_pred`, and `amp_expert_pred` remain `0.0000`;
  - delayed and return-to-command recovery rewards remain `0.0000`;
  - `wm` is finite around `0.37`, with `wm_valid_fraction` around `0.78`;
  - `force_push_curriculum_alpha` is still at the minimum `0.1000`;
  - displayed force band is about `3.144-162.44N`;
  - early `base_contact` is low but still warming up (`0.005-0.013` in the latest printed iterations);
  - residual is no longer fully dead but still too small:
    - `adapter_ungated_scaled_residual_action_l2` about `0.0029`;
    - `adapter_scaled_residual_action_l2` about `0.0010-0.0015`;
    - `recovery_residual_activation` about `0.024`;
- next hard gate:
  - within roughly 50-100 PPO iterations, residual must continue rising while base contact stays controlled;
  - if residual stays near `0.001` or base contact climbs back toward v35c levels, v36 should be stopped and the survival objective/code should be tightened again rather than wasting a full run.

## 2026-05-20 06:16 UTC - v36 stopped, v37 recovery-weighted PPO launched

- v36 was stopped at the 50-iteration gate after saving `model_750.pt`;
- stop reason:
  - `Episode_Termination/base_contact` returned to about `0.067-0.074`, similar to v35c;
  - `adapter_scaled_residual_action_l2` only reached about `0.003-0.005`;
  - `force_push_curriculum_alpha` stayed pinned at `0.1000`;
  - reward scale was too large (`Mean reward` often above `10k`) and value loss stayed high, so the branch was not producing a useful recovery actor;
- implemented a PPO-side correction in `rsl_rl/rsl_rl/algorithms/track_adapter_amp_ppo.py`:
  - added optional recovery-gated actor surrogate weighting;
  - the weighting is driven by `residual_action_gate_batch`;
  - weights are normalized to mean `1.0` per minibatch so global PPO scale remains controlled;
  - added `Loss/recovery_ppo_sample_weight_max` logging when enabled;
- exposed the new knobs in `ppo_cfg.py`:
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_PPO_SAMPLE_WEIGHT`;
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_PPO_SAMPLE_GATE_THRESHOLD`;
  - `BOOSTER_TRACK_ADAPTER_RECOVERY_PPO_SAMPLE_POWER`;
- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v37_strict_survival_weighted_ppo.sh`;
- v37 changes:
  - keeps strict survival objective: task/style/reg/delayed-return all `0.0`;
  - resumes from v36 `model_750.pt` without optimizer state;
  - uses stronger residual authority/exploration: `residual_scale=0.32`, action std `0.22`;
  - enables recovery-gated PPO sample weighting: weight `8.0`, gate threshold `0.08`;
  - reduces reward scale to avoid v36's huge value-loss regime: recovery group weight `7.0`, base-contact penalty `-150`;
  - increases recovery-specific shaping terms for capture/step/support while keeping command tracking disabled;
- validation:
  - `python -m py_compile` passed for `track_adapter_amp_ppo.py` and `ppo_cfg.py`;
  - `bash -n` passed for the v37 launcher;
- launched v37:
  - tmux: `ta_v37_weighted_ppo`;
  - run id: `track_adapter_81999_stage3_recovery_v37_strict_survival_weighted_ppo_20260520_061559`;
  - resume source: `2026-05-20_06-05-31_track_adapter_81999_stage3_recovery_v36_strict_survival_intervention_20260520_060500_ppo_intervention/model_750.pt`;
  - reused v23 Stage 3 world-model pretrain;
- next gate:
  - first 25-50 resumed iterations must show `Loss/recovery_ppo_sample_weight_max` present;
  - `adapter_scaled_residual_action_l2` should rise materially beyond v36's `0.005` without base-contact exploding;
  - if residual still stays effectively closed, the remaining issue is not reward weighting but adapter architecture/initialization or missing privileged recovery state.

## 2026-05-20 06:18 UTC - v37 initial PPO check

- v37 reached learning iteration `756/12750`;
- strict objective check:
  - `Mean amp loss`, `amp_grad_pen`, `amp_policy_pred`, `amp_expert_pred` are all `0.0000`;
  - `recovery_push_velocity_track_exp`, `recovery_delayed_velocity_track_exp`, and `recovery_return_to_command_exp` remain `0.0000`;
- recovery-weighted PPO is active:
  - `Mean recovery_ppo_sample_weight_max loss: 1.7350`;
- reward/value scale improved compared with v36:
  - `Mean reward` about `502.83` instead of v36's `10k+`;
  - `Mean value_function loss` about `761.54` instead of tens of thousands;
- adapter intervention improved immediately:
  - `adapter_ungated_scaled_residual_action_l2`: `0.0497`;
  - `adapter_scaled_residual_action_l2`: `0.0254`;
  - this is materially above v36's `0.003-0.005` plateau;
- early safety:
  - `Episode_Termination/base_contact`: `0.0147`;
  - force curriculum still at alpha `0.1000`, as expected this early;
- verdict:
  - this is the first branch in this strict-survival reset that clearly leaves the near-zero residual basin;
  - do not call it successful yet; next check is whether residual can improve push survival without base-contact climbing back toward `0.07+`.

## 2026-05-20 06:21 UTC - v37 25-iteration gate

- v37 saved `model_775.pt`;
- latest observed iteration: `775/12750`;
- strict objective remains intact:
  - AMP losses remain `0.0000`;
  - delayed/return tracking rewards remain `0.0000`;
- recovery-weighted PPO remains active:
  - `recovery_ppo_sample_weight_max`: `2.1675`;
- adapter is no longer closed:
  - `adapter_ungated_scaled_residual_action_l2`: `0.0608`;
  - `adapter_scaled_residual_action_l2`: `0.0226`;
- stability:
  - `Episode_Termination/base_contact`: `0.0633`;
  - `Episode_Termination/time_out`: `0.9367`;
  - `force_push_curriculum_alpha`: still `0.1000`;
- verdict:
  - v37 is better than v36 on the main blocker because residual intervention is real and reward/value scale is sane;
  - it is not yet a completed recovery policy because base-contact is still above the `0.05` promotion target and curriculum has not advanced;
  - continue training, but do not promote until fixed push evaluation improves and alpha can rise without base-contact increasing.

## 2026-05-20 06:28 UTC - v37 stopped at model_800, v38 consolidation launched

- monitored v37 through `model_800.pt`;
- v37 final short-run verdict:
  - residual stayed alive and increased compared with v36 (`adapter_scaled_residual_action_l2` roughly `0.02-0.03`);
  - reward/value scale remained sane compared with v36;
  - however `Episode_Termination/base_contact` returned to about `0.070-0.072`;
  - `force_push_curriculum_alpha` stayed at `0.1000`;
  - this means v37 solved the "dead adapter" issue but did not yet produce higher-quality recovery;
- stopped v37:
  - tmux: `ta_v37_weighted_ppo`;
  - last checkpoint retained: `model_800.pt`;
- created `scripts/rsl_rl/run_track_adapter_stage3_recovery_v38_strict_survival_consolidate.sh`;
- v38 goal:
  - consolidate the non-zero residual from v37 with lower stochasticity;
  - reduce exploration-driven base contacts;
  - keep strict survival-only objective;
- v38 settings:
  - resume from v37 `model_800.pt`;
  - action std lowered from `0.22` to `0.13`;
  - no additional residual output perturbation;
  - recovery PPO sample weighting remains enabled at `6.0`;
  - recovery activation target reduced to `0.18` so it preserves residual instead of forcing more random magnitude;
  - base-contact penalty strengthened to `-220`, recovery group weight `8.0`;
  - safety gate tightened to `max_base_contact=0.125`;
- launched v38:
  - tmux: `ta_v38_consolidate`;
  - run id: `track_adapter_81999_stage3_recovery_v38_strict_survival_consolidate_20260520_062758`;
  - runner loaded v37 `model_800.pt`;
  - runner overrode resumed action std to `0.1300`;
- next gate:
  - early base-contact must fall below the v37 `~0.07` band;
  - residual should remain non-zero but not grow noisily;
  - if v38 still holds at `~0.07` base-contact, the remaining issue is not exploration noise and will require changing the recovery action target/objective, not another scalar retune.

## 2026-05-20 06:31 UTC - v38 early consolidation check

- v38 reached the first few PPO iterations after resume;
- strict objective remains intact:
  - AMP losses are `0.0000`;
  - delayed/return tracking rewards are `0.0000`;
- intended consolidation behavior is visible:
  - action std is `0.13`;
  - recovery PPO sample weighting remains active (`recovery_ppo_sample_weight_max` around `1.88`);
  - residual stayed non-zero:
    - `adapter_ungated_scaled_residual_action_l2` around `0.0808`;
    - `adapter_scaled_residual_action_l2` around `0.0355`;
- base-contact improved versus v37:
  - early `base_contact` snapshots dropped to about `0.005-0.024`;
  - this is clearly below v37's `~0.07` band;
- verdict:
  - v38 is currently the best short-run direction: it keeps the useful residual from v37 while reducing contact;
  - keep it running and only promote after longer fixed evaluation, because the first episodes are still warming up.

## 2026-05-20 08:34 UTC - v38 stopped by safety gate

- tmux session `ta_v38_consolidate` is no longer running;
- run directory:
  - `logs/rsl_rl/run_amp_track_adapter/2026-05-20_06-28-02_track_adapter_81999_stage3_recovery_v38_strict_survival_consolidate_20260520_062758_ppo_consolidate`;
- final saved checkpoint:
  - `model_1410_aborted.pt`;
- safety gate failure:
  - `Episode_Termination/base_contact`: `0.1262`;
  - configured safety limit: `0.1250`;
  - failure message: `Track Adapter safety gate failed; training stopped before promotion to full run.`;
- final v38 metrics at the abort:
  - `Episode_Termination/time_out`: `0.8738`;
  - `adapter_scaled_residual_action_l2`: `0.6596`;
  - `adapter_ungated_scaled_residual_action_l2`: `1.3825`;
  - `adapter_residual_action_l2`: `13.5010`;
  - `force_push_curriculum_alpha`: `0.1000`;
  - `force_push_force_max_n`: `162.4400`;
  - `force_push_applied_force_max_n`: `112.6534`;
  - AMP/style and delayed/return tracking rewards remain disabled for this strict-survival branch;
- trend extracted from the v38 event file:
  - `model_850.pt`: `base_contact=0.0635`, `timeout=0.9365`, `scaled_residual=0.0377`;
  - `model_1000.pt`: `base_contact=0.0675`, `timeout=0.9325`, `scaled_residual=0.0559`;
  - `model_1100.pt`: `base_contact=0.0664`, `timeout=0.9336`, `scaled_residual=0.0863`;
  - `model_1200.pt`: `base_contact=0.0754`, `timeout=0.9246`, `scaled_residual=0.1531`;
  - `model_1300.pt`: `base_contact=0.0870`, `timeout=0.9130`, `scaled_residual=0.2762`;
  - `model_1400.pt`: `base_contact=0.1224`, `timeout=0.8776`, `scaled_residual=0.6033`;
- verdict:
  - v38 did not solve recovery;
  - the early low-contact snapshots were not stable;
  - residual intervention is now alive, but it grows into noisy over-intervention instead of coordinated recovery;
  - the force curriculum never advanced past `alpha=0.1000`, so this run did not become a strong-push policy;
  - the best training-metric candidate in this run is likely around `model_850.pt` to `model_1100.pt`, not the aborted final checkpoint;
  - next action is fixed sustained-push evaluation of the early candidates before deciding whether to preserve one or move to a new objective/action prior.

## 2026-05-20 08:57 UTC - v38 candidate fixed evaluation launched

- created `scripts/rsl_rl/eval_track_adapter_v38_candidates.sh`;
- launched tmux session:
  - `eval_v38_candidates`;
- evaluation scope:
  - `v38_model_850`: early low-contact candidate;
  - `v38_model_1100`: stronger residual candidate before overgrowth;
  - zero command sustained push;
  - forces: `50N`, `100N`, `120N`;
  - modes: `none` and `event`;
  - `NUM_ENVS=16`;
  - `RESIDUAL_SCALE=0.32`;
- purpose:
  - decide whether any early v38 checkpoint is worth keeping;
  - confirm whether v38's training-metric improvement transfers to fixed push recovery;
  - avoid promoting the aborted final checkpoint by reward or residual magnitude alone.

## 2026-05-20 09:03 UTC - v38 candidate fixed evaluation result

- tmux `eval_v38_candidates` completed;
- summary:
  - `logs/eval_track_adapter_v38_candidates_20260520_085725/summary.md`;
- evaluation result:
  - `v38_model_850`:
    - `50N none`: `fall=0.7500`, `base_contact=0.7500`, `clean=0.2500`;
    - `100N none`: `fall=1.0000`, `base_contact=1.0000`, `clean=0.0000`;
    - `120N none`: `fall=1.0000`, `base_contact=1.0000`, `clean=0.0000`;
    - `50N event`: `fall=0.7500`, `base_contact=0.7500`, `clean=0.2500`;
    - `100N event`: `fall=1.0000`, `base_contact=1.0000`, `clean=0.0000`;
    - `120N event`: `fall=1.0000`, `base_contact=1.0000`, `clean=0.0000`;
  - `v38_model_1100`:
    - `50N none`: `fall=0.6875`, `base_contact=0.6875`, `clean=0.3125`;
    - `100N none`: `fall=1.0000`, `base_contact=1.0000`, `clean=0.0000`;
    - `120N none`: `fall=1.0000`, `base_contact=1.0000`, `clean=0.0000`;
    - `50N event`: `fall=0.7500`, `base_contact=0.6250`, `clean=0.2500`;
    - `100N event`: `fall=1.0000`, `base_contact=1.0000`, `clean=0.0000`;
    - `120N event`: `fall=1.0000`, `base_contact=1.0000`, `clean=0.0000`;
- verdict:
  - neither early v38 candidate is acceptable;
  - event-mode gating did not rescue the policy;
  - v38's apparent training improvement did not transfer to fixed sustained-push recovery;
  - next branch must change the recovery representation/objective, not only retune residual magnitude, noise, or scalar reward weights.

## 2026-05-20 09:05 UTC - v39 bounded-recovery implementation

- added residual activation band support:
  - `rsl_rl/rsl_rl/algorithms/track_adapter_amp_ppo.py`;
  - new env/config knobs:
    - `BOOSTER_TRACK_ADAPTER_RESIDUAL_RECOVERY_ACTIVATION_MAX_NORM`;
    - `BOOSTER_TRACK_ADAPTER_RESIDUAL_RECOVERY_ACTIVATION_OVERGROWTH_WEIGHT`;
  - behavior:
    - old activation loss only penalized residual below a target;
    - new band loss can also penalize residual above a configured max norm;
- wired new knobs into:
  - `source/booster_rl_tasks/booster_rl_tasks/tasks/manager_based/beyond_mimic/robots/k1/run_amp/ppo_cfg.py`;
- created v39 training script:
  - `scripts/rsl_rl/run_track_adapter_stage3_recovery_v39_bounded_recovery.sh`;
- v39 changes from v38:
  - resumes from v38 `model_1100.pt`, not the aborted final checkpoint;
  - disables optimizer resume;
  - lowers residual scale/action std/lr;
  - reduces recovery PPO sample weighting from v38's aggressive setting;
  - adds residual overgrowth penalty;
  - shortens push residual authority window;
  - restores small task/style/reg guardrails so the adapter does not become a noisy second locomotion policy;
  - keeps teacher/distill/failure-reset disabled;
- verification:
  - `python -m py_compile rsl_rl/rsl_rl/algorithms/track_adapter_amp_ppo.py source/booster_rl_tasks/booster_rl_tasks/tasks/manager_based/beyond_mimic/robots/k1/run_amp/ppo_cfg.py` passed.

## 2026-05-20 09:06 UTC - v39 bounded-recovery training launched

- launched tmux:
  - `ta_v39_bounded`;
- run id:
  - `track_adapter_81999_stage3_recovery_v39_bounded_recovery_20260520_090610`;
- run directory:
  - `logs/rsl_rl/run_amp_track_adapter/2026-05-20_09-06-15_track_adapter_81999_stage3_recovery_v39_bounded_recovery_20260520_090610_ppo_bounded_recovery`;
- resume source:
  - v38 `model_1100.pt`;
  - optimizer reload disabled;
- first observed PPO metrics:
  - `Mean recovery_residual_activation loss`: `0.0065`;
  - `Mean recovery_ppo_sample_weight_max loss`: `2.1645`;
  - `adapter_ungated_scaled_residual_action_l2`: `0.1046`;
  - `adapter_scaled_residual_action_l2`: `0.0255`;
  - `force_push_curriculum_alpha`: `0.1000`;
  - `force_push_force_max_n`: `136.6400`;
  - `Episode_Termination/base_contact`: `0.0002`;
- interpretation:
  - this is only startup and not proof of recovery;
  - the immediate check is whether residual remains bounded while force-active windows begin producing non-zero applied-force metrics;
  - if fixed push does not improve, stop v39 and change the action/recovery representation again.

## 2026-05-20 09:08 UTC - v39 early live check

- v39 reached iteration `1107/4300`;
- early trend:
  - `adapter_scaled_residual_action_l2`: stayed in `0.0255-0.0369`;
  - `force_push_curriculum_alpha`: `0.1000`;
  - `force_push_applied_force_max_n`: became non-zero, reaching `97.7634N` then `68.5059N`;
  - `Episode_Termination/base_contact`: increased slowly from `0.0002` to `0.0054`;
- interpretation:
  - the force pipeline is active;
  - residual is bounded so far, unlike late v38;
  - still too early to judge push-recovery quality because the run has not reached sustained checkpoints or fixed evaluation yet.

## 2026-05-20 10:38 UTC - v39 model_1550 fixed evaluation launched

- v39 reached checkpoint:
  - `model_1550.pt`;
- training metrics at the pause:
  - `force_push_curriculum_alpha`: `0.1520`;
  - `force_push_force_max_n`: `161.1008`;
  - `force_push_applied_force_max_n`: around `90-114N` in recent logs;
  - `Episode_Termination/time_out`: around `0.94`;
  - `Episode_Termination/base_contact`: around `0.060`;
  - `adapter_scaled_residual_action_l2`: around `0.038-0.041`;
  - `adapter_ungated_scaled_residual_action_l2`: around `0.13`;
- interpretation before fixed eval:
  - v39 is much more bounded than late v38;
  - force curriculum has promoted beyond the `0.10` floor;
  - training metrics alone are not sufficient because v38 failed fixed push despite acceptable-looking checkpoints;
- action:
  - stopped tmux `ta_v39_bounded` after `model_1550.pt` was saved;
  - launched fixed evaluation tmux `eval_v39_1550`;
  - scope:
    - zero-command sustained push;
    - forces `50N`, `100N`, `120N`;
    - modes `none` and `event`;
    - `NUM_ENVS=16`;
    - `RESIDUAL_SCALE=0.24`;
  - checkpoint:
    - `logs/rsl_rl/run_amp_track_adapter/2026-05-20_09-06-15_track_adapter_81999_stage3_recovery_v39_bounded_recovery_20260520_090610_ppo_bounded_recovery/model_1550.pt`.

## 2026-05-20 10:48 UTC - v39 fixed evaluation failed, v40 direct sustained-push branch prepared

- v39 `model_1550.pt` fixed sustained-push result:
  - `50N none`: `fall=0.7500`, `base_contact=0.6875`, `clean=0.2500`;
  - `100N none`: `fall=1.0000`, `base_contact=1.0000`, `clean=0.0000`;
  - `120N none`: `fall=1.0000`, `base_contact=1.0000`, `clean=0.0000`;
  - `50N event`: `fall=0.7500`, `base_contact=0.6250`, `clean=0.2500`;
  - `100N event`: `fall=1.0000`, `base_contact=1.0000`, `clean=0.0000`;
  - `120N event`: `fall=1.0000`, `base_contact=1.0000`, `clean=0.0000`;
- diagnosis from F100 event rows:
  - residual/gate did activate;
  - residual norm rose to roughly `0.4`;
  - speed reached roughly `3m/s` while force was still active;
  - first fall/contact began around eval step `108`;
  - the policy is not converting sustained shove into coordinated stepping/running recovery;
- decision:
  - do not resume v39;
  - train the exact failed distribution directly instead of hoping broad curriculum transfers;
- implementation:
  - changed `scripts/rsl_rl/run_track_adapter_stage3_recovery_v35_any2track_adapt_only.sh` so `BOOSTER_TRACK_ADAPTER_FORCE_DURATION_MIXTURE_SCALE_WITH_CURRICULUM` can be overridden;
  - added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v40_sustained_push_direct.sh`;
- v40 intent:
  - resume from v39 `model_1550.pt`;
  - train zero/near-zero-command sustained pushes;
  - force mixture concentrated on `35-130N` for roughly `2.0-2.9s`;
  - disable force scaling curriculum for this branch;
  - remove velocity-cancel and return-command rewards during push;
  - emphasize no-fall, no-base-contact, upright, capture/support stepping;
  - allow more recovery authority than v39 but keep overgrowth capped.

## 2026-05-20 10:58 UTC - v40 sustained-push branch verified live

- active tmux:
  - `ta_v40_sustained`;
- run directory:
  - `logs/rsl_rl/run_amp_track_adapter/2026-05-20_10-51-47_track_adapter_81999_stage3_recovery_v40_sustained_push_direct_20260520_105142_ppo_sustained_push_direct`;
- verified live configuration at iteration `1565/4150`:
  - `force_push_curriculum_alpha`: `1.0000`;
  - `force_push_force_min_n`: `35.0000`;
  - `force_push_force_max_n`: `170.0000`;
  - `force_push_duration_min_s`: `2.0000`;
  - `force_push_duration_max_s`: `2.9000`;
  - `force_push_activation_probability`: `0.6500`;
  - `force_push_applied_force_max_n`: `169.4759N`;
  - `force_push_applied_force_n`: `102.6956N`;
  - `force_push_active`: `0.9930`;
- current learning metrics:
  - `Episode_Termination/time_out`: `0.0000`;
  - `Episode_Termination/base_contact`: `0.9202`;
  - `adapter_scaled_residual_action_l2`: `0.0840`;
  - `adapter_ungated_scaled_residual_action_l2`: `0.1839`;
  - `wm_valid_fraction`: `0.7619`;
- interpretation:
  - the previous v40 launch issue was script override behavior; the current run is now applying the intended strong sustained-push distribution;
  - the policy is currently failing this distribution very badly;
  - this is not a successful trend yet, and the run should only continue if `base_contact` drops sharply during the early safety window;
  - if it stays above roughly `0.5`, stop the branch and move to a less all-or-nothing recovery representation instead of spending hours on noisy failure rollouts.

## 2026-05-20 11:08 UTC - v40 stopped, v41 recovery-command override implemented

- v40 was stopped because early metrics worsened:
  - `Episode_Termination/base_contact`: rose to `0.9464`;
  - `Episode_Termination/time_out`: stayed `0.0000`;
  - sustained-push force settings were correct, so the failure was not a launch/config issue;
- diagnosis:
  - residual intervention was active, but zero-command frozen AMP still had no reliable path into stepping/running recovery;
  - continuing v40 would mostly collect failure rollouts from an all-or-nothing distribution;
- implementation:
  - added runner-side `recovery_command_override`;
  - during active zero-command push recovery, only the policy-observation command slice `[6:9]` is replaced with a temporary push/velocity-aligned command;
  - the real environment command remains zero, so no-push standing and reward command semantics are unchanged;
  - added env-config plumbing for `BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE*`;
  - added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v41_recovery_command_override.sh`;
- verification:
  - `python -m py_compile rsl_rl/rsl_rl/runners/track_adapter_runner.py source/booster_rl_tasks/booster_rl_tasks/tasks/manager_based/beyond_mimic/robots/k1/run_amp/ppo_cfg.py`;
- v41 intent:
  - resume from v39 `model_1550.pt`;
  - train sustained zero/near-zero-command pushes with a less all-or-nothing force ramp;
  - use the frozen AMP gait generator for recovery stepping instead of requiring the adapter residual alone to invent the gait from a static command.

## 2026-05-20 11:10 UTC - v41 launch corrected to bootstrap force range

- first v41 launch accidentally used immediate final force settings:
  - `30-185N`, `1.5-3.0s`, activation probability `0.55`;
  - command override was active and reduced failure relative to v40, but `base_contact` still rose to about `0.489`;
  - stopped this launch because it was too hard for a bootstrap phase;
- patched v41 defaults so final force/duration/probability match the bootstrap range unless explicitly overridden;
- relaunched tmux:
  - session: `ta_v41_recovery_cmd`;
  - run id: `track_adapter_81999_stage3_recovery_v41_recovery_command_override_bootstrap_20260520_110556`;
- verified active settings:
  - `force_push_force_min_n`: `20.0000`;
  - `force_push_force_max_n`: `145.0000`;
  - `force_push_duration_min_s`: `1.2000`;
  - `force_push_duration_max_s`: `2.6000`;
  - `force_push_activation_probability`: `0.3800`;
  - `force_push_applied_force_max_n`: about `144N`;
- early metrics:
  - `base_contact`: `0.0481` at iter `1553`, `0.1750` at iter `1554`, `0.3640` at the next inspected checkpoint;
  - `adapter_scaled_residual_action_l2`: around `0.071`;
  - TensorBoard confirms recovery-command override is active:
    - `RecoveryCommand/active_mean`: nonzero during push windows;
    - `RecoveryCommand/blend_mean`: nonzero during push windows;
- interpretation:
  - v41 bootstrap is not solved yet, but it is no longer the v40 all-fail regime;
  - continue under safety gate and stop if `base_contact` remains above `0.4` after the relative safety window.

## 2026-05-20 11:12 UTC - v41 easy bootstrap selected

- the `20-145N` bootstrap still drifted too high:
  - latest inspected `base_contact`: `0.4717`;
  - stopped before spending the full safety window;
- patched v41 default bootstrap lower:
  - force: `20-100N`;
  - duration: `1.2-2.0s`;
  - activation probability: `0.28`;
  - recovery-command max speed: `1.15m/s`;
  - recovery-command push/velocity speed gains reduced;
- relaunched tmux:
  - session: `ta_v41_recovery_cmd`;
  - run id: `track_adapter_81999_stage3_recovery_v41_recovery_command_override_easy_20260520_110856`;
- early verified metrics:
  - `force_push_force_min_n`: `20.0000`;
  - `force_push_force_max_n`: `100.0000`;
  - `force_push_duration_min_s`: `1.2000`;
  - `force_push_duration_max_s`: `2.0000`;
  - `force_push_activation_probability`: `0.2800`;
  - `force_push_applied_force_max_n`: about `99N`;
  - `base_contact`: `0.1988 -> 0.2467` over the latest inspected iterations;
  - `adapter_scaled_residual_action_l2`: around `0.062`;
- interpretation:
  - this is the first v41 setting that is clearly outside the all-fail regime while still applying sustained physical force;
  - continue this run and promote only after fixed evaluation confirms improved zero-command sustained-push recovery.

## 2026-05-20 11:34 UTC - v41 easy bootstrap stopped by safety gate, v42 prepared

- v41 easy bootstrap stopped by the safety gate, not by a process crash:
  - final checkpoint set: `model_1550.pt`, `model_1575.pt`, `model_1600.pt`, `model_1625.pt`, `model_1650.pt`, `model_1652_aborted.pt`;
  - stop reason: `base_contact=0.5727 > 0.4000`, strike `3/3`;
  - `time_out` improved to about `0.427`, so the policy survived longer but accepted too many trunk contacts;
  - `RecoveryCommand/*` TensorBoard scalars confirmed the command override was active;
- diagnosis:
  - v41 learned a longer-survival/contact-accepting solution;
  - the existing recovery contact penalty was not strong enough relative to positive alive/upright/support rewards;
- implementation:
  - added runner-level `terminal_penalty`;
  - base-contact terminations can now receive an immediate PPO reward penalty via `BOOSTER_TRACK_ADAPTER_TERMINAL_BASE_CONTACT_PENALTY`;
  - added config plumbing in `run_amp/ppo_cfg.py`;
  - added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v42_terminal_contact_bootstrap.sh`;
- v42 bootstrap settings:
  - force: `12-75N`;
  - duration: `0.75-1.35s`;
  - activation probability: `0.22`;
  - recovery-command speed cap: `0.85m/s`;
  - terminal base-contact penalty: `-320`;
  - recovery base-contact penalty weight: `-950`;
  - safety base-contact gate: `0.25`.

## 2026-05-20 11:54 UTC - v42 terminal penalty bug fixed and relaunched

- first v42 launch was stopped manually:
  - force settings were correct and `base_contact` stayed below the v41 failure band;
  - however `Mean reward` became excessively negative and `value_function loss` rose above `2e6`;
- root cause:
  - terminal penalty used `termination_manager.get_term("base_contact")`;
  - in Isaac Lab this is an episodic termination-stat buffer, not a clean current-step-only signal;
  - the penalty was therefore applied repeatedly after contact, not just on the terminal step;
- fix:
  - for `base_contact`, terminal penalty now uses the current `termination_manager.terminated` tensor;
  - default terminal base-contact penalty reduced from `-320` to `-220`;
- verification:
  - `python -m py_compile rsl_rl/rsl_rl/runners/track_adapter_runner.py source/booster_rl_tasks/booster_rl_tasks/tasks/manager_based/beyond_mimic/robots/k1/run_amp/ppo_cfg.py`;
  - `bash -n scripts/rsl_rl/run_track_adapter_stage3_recovery_v42_terminal_contact_bootstrap.sh`.
- relaunched tmux:
  - session: `ta_v42_terminal_contact`;
  - run id: `track_adapter_81999_stage3_recovery_v42_terminal_contact_bootstrap_fix_20260520_115132`;
- corrected early metrics:
  - force: `12-75N`, duration `0.75-1.35s`, activation probability `0.22`;
  - `base_contact`: `0.0064 -> 0.0386` in the first force-active iterations;
  - `TerminalPenalty/base_contact_mean`: around `0.0002-0.0007`, confirming current-step-only penalty;
  - `TerminalPenalty/mean`: around `-0.05` to `-0.16`, no longer the repeated large penalty;
  - `value_function loss`: about `17k` at the latest inspected iteration, much lower than the aborted `2e6+` failure;
- status:
  - v42 fixed run is currently active and should be monitored until the safety window passes.

## 2026-05-20 12:12 UTC - v42 fixed stopped by safety gate, v43 hard terminal penalty prepared

- v42 fixed stopped by safety gate:
  - stop reason: `base_contact=0.2954 > 0.2500`, strike `3/3`;
  - last saved checkpoints include `model_1550.pt`, `model_1575.pt`, `model_1600.pt`, `model_1625.pt`, `model_1632_aborted.pt`;
- v42 result summary:
  - `time_out` improved to roughly `0.70`;
  - `base_contact` plateaued around `0.27-0.30`;
  - this is much better than v41's `base_contact~=0.57`, but still not acceptable;
- diagnosis:
  - terminal penalty now applies correctly, but `-220` is too small relative to the accumulated survival/upright/support reward over long episodes;
  - a late trunk-contact termination remains economically tolerable to PPO;
- implementation:
  - added `scripts/rsl_rl/run_track_adapter_stage3_recovery_v43_terminal_contact_hard.sh`;
  - v43 resumes v42 `model_1625.pt`;
  - terminal base-contact penalty increased to `-3200`;
  - recovery base-contact penalty weight increased to `-1400`;
  - positive alive/no-contact weights reduced to keep returns numerically reasonable;
  - safety gate loosened to `0.42` during this short hard-penalty correction so the run does not stop before the new value estimates settle.

## 2026-05-20 12:38 UTC - v43 hard terminal penalty running, not crashed

- checked after the user reported the run looked dead:
  - tmux session `ta_v43_terminal_hard` is alive;
  - Python process is running with the expected resume command from v42 `model_1625.pt`;
  - log file is updating under `logs/track_adapter_81999_stage3_recovery_v43_terminal_contact_hard_20260520_123426.log`;
- current v43 status around iteration `1638`:
  - force: `12-75N`, duration `0.75-1.35s`, activation probability `0.22`;
  - `base_contact`: rising from `0.0404` to `0.1977`;
  - `time_out`: still `0.0` in early resumed iterations;
  - `Mean reward`: `-4834 -> -3559`, improving but still strongly negative;
  - `value_function loss`: high, roughly `3.9e4-1.5e5`, caused by the larger one-shot terminal penalty;
  - `TerminalPenalty/base_contact_mean`: small step fraction (`0.0-0.0017`), confirming the penalty is current-step-only and not the previous repeated-penalty bug;
- interpretation:
  - v43 is not dead;
  - the run is in a hard correction phase where the critic is absorbing a large terminal-contact penalty;
  - continue only while `base_contact` stays below the temporary `0.42` safety gate and reward/value loss do not diverge further.

## 2026-05-20 12:42 UTC - v43 short monitor result

- continuous monitoring from iterations `1642-1659`:
  - `base_contact`: `0.2496 -> 0.2965 -> 0.2959`;
  - `time_out`: `0.0 -> 0.7209 -> 0.7041`;
  - `Mean reward`: recovered from large negative returns to around `-266` to `+119`, depending on recent terminations;
  - `value_function loss`: still high but no longer monotonically exploding;
- interpretation:
  - v43 is alive and currently below the temporary `0.42` safety gate;
  - the run has mostly returned to the v42 survival/contact plateau (`time_out~=0.70`, `base_contact~=0.30`);
  - this is not a success yet: the hard terminal penalty has not clearly pushed base-contact below the v42 level;
  - continue only as a short test; if it remains around `0.30`, the next change should alter the recovery objective/exposure rather than only increasing terminal penalty further.

## 2026-05-20 14:23 UTC - v43 1h49m status

- tmux session `ta_v43_terminal_hard` remains alive;
- latest inspected iteration: `2153`;
- latest metrics:
  - `base_contact`: `0.2732`;
  - `time_out`: `0.7268`;
  - `Mean reward`: about `1350.7`;
  - `value_function loss`: about `5.8e4`;
  - force curriculum is at full configured range: `12-75N`, `0.75-1.35s`, probability `0.22`;
- trend:
  - over the latest 12 points, `base_contact` is almost flat around `0.269-0.273`;
  - `time_out` is almost flat around `0.727-0.731`;
  - value loss is high but not exploding;
- interpretation:
  - the run is operationally stable;
  - v43 is slightly better than the v42 stop point (`base_contact~=0.295`), but not enough to call the recovery objective solved;
  - if this plateau persists, continuing alone is unlikely to produce a large push-recovery jump.

## 2026-05-20 14:25 UTC - v43 stopped on request

- user requested stopping the run;
- sent Ctrl-C to tmux session `ta_v43_terminal_hard`;
- confirmed no remaining v43 `train.py` or wrapper process;
- tmux session `ta_v43_terminal_hard` is gone;
- latest saved checkpoint:
  - `logs/rsl_rl/run_amp_track_adapter/2026-05-20_12-34-31_track_adapter_81999_stage3_recovery_v43_terminal_contact_hard_20260520_123426_ppo_recovery_command_override/model_2150.pt`;
- final observed metrics before stop:
  - `base_contact`: about `0.2764`;
  - `time_out`: about `0.7236`;
  - force: `12-75N`, duration `0.75-1.35s`, probability `0.22`;
- status:
  - run stopped cleanly by operator request, not by crash or safety gate.

## 2026-05-20 14:40 UTC - current robust checkpoint ranking

- no training process is currently running;
- latest v43 checkpoint exists at:
  `logs/rsl_rl/run_amp_track_adapter/2026-05-20_12-34-31_track_adapter_81999_stage3_recovery_v43_terminal_contact_hard_20260520_123426_ppo_recovery_command_override/model_2150.pt`;
- v43 is not a promotion candidate yet:
  - it has no fixed push-recovery evaluation;
  - its train-time plateau is only `time_out~=0.72`, `base_contact~=0.27`;
  - its force exposure was mild (`12-75N`, `0.75-1.35s`), so it cannot supersede older fixed-eval anchors;
- best fixed-evaluated aggregate physical-force candidate remains:
  `v14 model_10000.pt`
  (`logs/rsl_rl/run_amp_track_adapter/2026-05-15_10-17-58_track_adapter_81999_stage3_recovery_v14_capture_step_20260515_081803_ppo_capture_step/model_10000.pt`);
- close/narrow alternative:
  `v15b model_16499.pt`
  (`logs/rsl_rl/run_amp_track_adapter/2026-05-16_11-44-40_track_adapter_81999_stage3_recovery_v15b_force_ladder_20260516_114435_ppo_force_ladder/model_16499.pt`);
- rationale:
  - v14 remains the most defensible aggregate across fixed `100/150/175/200/225N` pure-horizontal physical force while keeping no-push tracking clean;
  - v15b is competitive or better in some `100-175N` fall-rate slices, but not clearly better across the whole envelope;
  - v24b/v26c/v28b/v38/v39/v43 did not produce a fixed-eval improvement;
  - old v7/model_1100 is a sim2real qualitative clue for static hand-push stepping, but Isaac fixed eval did not show it as the best robust policy and it belongs to an older adapter/base setup.

## 2026-05-20 15:07 UTC - deploy slow-push failure note for v11/v14

- user reported an important real-robot deployment observation:
  - both `track_adapter_v11_model_6500.onnx` and `track_adapter_v14_model_10000.onnx` failed under a slow manual push on the real robot;
  - observed behavior was not a useful recovery step;
  - the robot mostly moved/braced with the toes, then fell;
- this must be treated as a promotion-blocking sim2real finding even though v14 remains the best fixed-evaluated Isaac aggregate checkpoint;
- implication:
  - v11/v14 are not acceptable as final robust deploy policies for slow sustained human/robot contact;
  - future promotion must include a real-robot-style slow-push gate, specifically checking whether the policy takes a meaningful support/capture step instead of toe-only bracing;
  - Isaac fixed evaluation alone is insufficient until it reproduces this deploy failure mode.

## 2026-05-20 15:35 UTC - Any2Track OCR reread: gap review

- reread `docs/output-adapter/any2track.md` against the current Track Adapter state;
- parts that are implemented or substantially aligned:
  - frozen base policy plus trainable adapter;
  - layer-wise adapter with zero-initialized adapter outputs;
  - `H=79` history and `N=20` autoregressive world-model prediction;
  - history is built from state/action pairs;
  - persistent WM replay and per-PPO-iteration replay updates before PPO minibatch updates;
  - frozen base actor is not updated during adapter training;
- important mismatches against the paper:
  - Any2Track's base `AnyTracker` is a general full-body motion tracker trained on diverse dynamic/contact-rich references; our base is an AMP/velocity locomotion policy, not a reference-motion tracker with an explicit capture-step/recovery repertoire;
  - Any2Track fine-tunes the adapter with the same tracking rewards under dynamics variance; many of our recent branches disabled tracking/style and used survival-only custom rewards, which is not paper-faithful;
  - Any2Track's Table III external-force setting is an interval `U(5,10)s` and velocity-magnitude disturbance `U(0.1,1.0)`, closer to root-velocity impulse/domain randomization than our finite-N slow sustained trunk push;
  - Any2Track trains against multiple dynamics variations at once: terrain, friction, mass, CoM, armature, default-pose jitter, and external disturbance; our recent loops focused mostly on physical push;
  - Any2Track does not rely on explicit push-event gates at deployment; our implementation introduced residual gates/recovery gates to protect no-push behavior, which can either under-open under slow real pushes or over-intervene under constant gate;
- conclusion:
  - the current failure should not be called an inherent adapter-limit proof;
  - it is more likely a target/base/reward/disturbance mismatch: zero-command slow sustained push asks the adapter to create recovery stepping that the frozen base is not being commanded or rewarded to express;
  - a paper-faithful sanity run should first validate AnyAdapter on root-velocity disturbances with tracking/style rewards enabled before judging the architecture;
  - solving the real slow-push deploy failure likely requires either a base policy/recovery reference that can step under disturbance or a principled recovery-command/reference mechanism, not just more scalar survival reward on the existing adapter.

## 2026-05-20 15:50 UTC - objective clarification: AMP robustness, not paper reproduction

- user clarified the actual objective:
  - not to reproduce Any2Track for its own sake;
  - keep the AMP-base locomotion policy because it gives human-like walking/running without a reward-heavy locomotion design;
  - use the adapter idea to add robustness against collisions, pushes, and robot-to-robot contact;
  - target behavior is: do not fall when pushed, recover after contact, and preserve usable velocity tracking and nominal gait when not disturbed;
- roadmap updated to reflect this:
  - Any2Track is a diagnostic/design reference, not the target specification;
  - final evaluation must match deployment disturbances: short impacts, slow sustained pushes, lateral/frontal/back contact, no-push walking, and zero-command stillness;
  - if scalar survival rewards plus residual gates do not produce useful recovery steps, add a recovery reference/command mechanism while keeping the frozen AMP base as nominal locomotion.

## 2026-05-20 16:05 UTC - new recovery-controller design note

- created `docs/amp_recovery_adapter_controller_design.md`;
- the document reframes the next branch as an AMP Recovery Adapter Controller, not an Any2Track reproduction:
  - frozen AMP remains the nominal human-like locomotion generator;
  - adapter becomes a phase-aware recovery controller;
  - controller phases are `nominal`, `recovery`, and `return`;
  - recovery mode can temporarily override the policy-observation command and open the adapter gate;
  - return mode decays the command/residual back to the user command and frozen AMP gait;
- documented why previous branches were insufficient:
  - v11/v14 reacted but produced toe-only bracing under real slow push;
  - survival-only branches lacked a clear "where to step" signal;
  - teacher/distillation branches did not produce a strong recovery teacher;
  - v41 command override was useful but incomplete without phase state, hold/decay, deploy-safe detection, and phase-aware rewards;
- documented next implementation checklist:
  - per-env recovery state machine;
  - capture-signal recovery command;
  - deploy-safe disturbance score;
  - phase-specific reward weighting;
  - fixed eval for no-push, slow sustained push, walking push, and short impacts.

## 2026-05-21 09:45 UTC - v44 fixed eval and deploy status

- v44 recovery-controller training stopped at PPO iteration `2475/5200` by the runner safety gate:
  - checkpoint:
    `logs/rsl_rl/run_amp_track_adapter/2026-05-20_18-04-18_track_adapter_81999_stage3_recovery_v44_recovery_controller_20260520_162052_ppo_recovery_controller/model_2475.pt`;
  - stop reason: scaled residual action norm exceeded the configured safety threshold;
  - this is not a full promotion checkpoint, but it is the current recovery-controller artifact.
- fixed evaluation output:
  - `logs/recovery_controller_gates_20260521_043812/summary.md`;
  - compared v14 `model_10000.pt`, v15b `model_16499.pt`, v44 `model_2475.pt`, and old v7/model_1100.
- fixed-eval result:
  - v44 passed no-push and short-impact gates;
  - v44 still failed zero-command slow sustained push and walking push gates;
  - v44 was not promoted by Isaac fixed-eval criteria.
- deployment support:
  - exported v44 to `booster_k1_locomotion` as:
    `/workspace/booster_k1_locomotion/assets/track_adapter_v44_model_2475.onnx`;
  - ONNX ABI:
    `obs[1,75]`, `history[1,79,97]`, `residual_gate[1,1]`, `reset[1,1]`
    -> `action[1,22]`, `next_history[1,79,97]`;
  - `booster_k1_locomotion` C++ node now supports `track_adapter_gate_mode:=recovery_controller`.

## 2026-05-21 10:10 UTC - v44 real-robot deployment note

- user deployed v44 recovery-controller on the real robot;
- qualitative result:
  - v44 moved better than the previous v11 and v14 deployed checkpoints;
  - under standing external pushes, v44 looked like a real improvement over the toe-only bracing behavior seen with v11/v14;
  - this is an important sim2real positive signal even though v44 did not pass all fixed Isaac gates.
- observed issue:
  - when the policy is first started, small actuator oscillations can appear;
  - when a foot/leg ends up too far toward the inside of the body, small actuator oscillations can also appear;
  - the oscillation was not severe and tended to settle after a short time, but it is real and should be treated as a deployment issue.
- likely next debug targets:
  - add a startup ramp/settle gate for recovery command and residual gate in deploy;
  - add an inner-foot or cross-leg posture guard to the recovery-controller score/gate;
  - add fixed sim checks that start from slightly crossed or narrow-stance states;
  - avoid declaring v44 final until this startup/cross-leg oscillation is understood.

## 2026-05-21 10:20 UTC - local log cleanup policy

- keep logs/checkpoints needed for current ranking, deployment, and reproducibility:
  - v7/model_1100 qualitative static-push clue;
  - v11/model_6500 historical deploy and fixed-eval anchor;
  - v14/model_10000 current fixed-eval aggregate anchor;
  - v15b/model_16499 narrow robustness comparison;
  - v44/model_2475 current deploy-positive recovery-controller artifact;
  - `recovery_controller_gates_20260521_043812` fixed gate summary.
- safe to delete raw training logs for branches already rejected in the worklog when their summarized results are preserved here:
  - v18, v24b, v26c, v28b, v34j, v35c, v38, v39, v40, v41, v43;
  - older failed v8/v10b/v12b/v13b raw runs may also be removed if more disk is needed, but keep v7/v11/v14/v15b/v44.
- cleanup performed:
  - deleted the rejected raw training directories for v18, v24b, v26c, v28b, v34j, v35c, v38, v39, v40, v41-easy, and v43;
  - kept v7, v11, v14, v15b, v44, and the 2026-05-21 fixed recovery-controller gate summary;
  - `logs/` size after cleanup is about `42G`.

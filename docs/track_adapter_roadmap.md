# Track Adapter Implementation Roadmap

Date: 2026-05-09
Last updated: 2026-05-19

## Goal

Implement the Track Adapter described in `idea-adapter.md`.

Status correction, 2026-05-10: the first action-residual implementation is not a complete implementation of this roadmap. A full training run was started too early and was stopped after `base_contact` climbed to roughly `0.9`. Full training is blocked until the acceptance gates in this roadmap pass.

- keep the AMP locomotion base policy frozen,
- add a history-informed residual adapter,
- train only the adapter, critic, and exploration noise,
- use Recovery-Gated AMP so style reward protects nominal gait while recovery rewards dominate disturbed states.

The original frozen base checkpoint requested for the first Track Adapter branch was:

`logs/rsl_rl/run_amp_y/2026-05-04_19-10-18_resume56_yawlow_zero_stand/model_70000.pt`

Active branch update, 2026-05-12: after AMP axis fine-tune v2, the Track Adapter branch now uses the stronger-tracking base:

`logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt`

Reason: `model_81999.pt` is clearly better on the user-priority no-push single-axis velocity tracking grid. Its high-speed jerk/contact regression is now treated as Track Adapter's recovery correction target rather than a reason to fall back to the safer-but-weaker `model_70100.pt`.

## Constraints

- Do not overwrite `Booster-Run-AMP-v0`; add a dedicated Track Adapter task.
- Keep the policy observation compatible with the frozen checkpoint: 75 actor dims, 78 critic dims, 22 actions.
- Load and freeze the base AMP actor from the checkpoint.
- Prefer a narrow implementation over broad changes to the existing AMP path.
- Treat `/workspace/forked_booster_amp_lab` as reference only. Its Recovery-Gated AMP artifacts failed MuJoCo/SDK zero-command stability, so trained artifacts and broad runner changes are not adopted.
- Training must be launched via tmux after implementation, with:

```bash
source /workspace/Isaac_uv_template/.venv/bin/activate
```

## Architecture

### 1. Frozen Base Policy

Add a `TrackAdapterActorCritic` module:

- internal frozen `ActorCritic` loads the Stage-1 AMP checkpoint,
- `base_mean = frozen_actor(obs)` under no-grad,
- all base parameters have `requires_grad=False`,
- trainable critic remains separate for adapter PPO.

### 2. History Adapter

Maintain a runner-side history buffer of normalized policy observations and executed actions.

Current design:

- `history_length = 79`,
- history input shape: `[batch, history_length, num_actor_obs + num_actions]`,
- GRU or MLP encoder outputs a compact dynamics embedding,
- residual head receives `[obs, embedding, previous_action]`,
- final residual layer is zero-initialized.

Action mean:

```text
action_mean = base_mean + residual_scale * residual_mean
```

### 3. PPO/Storage Path

Add Track Adapter-specific storage and algorithm rather than changing standard `AMPPPO` behavior.

The algorithm must:

- pass history to policy `act`, `evaluate`, and update-time log-prob recomputation,
- optimize only trainable policy parameters,
- support `freeze_discriminator=True` by excluding discriminator params and skipping discriminator updates,
- retain AMP reward inference through the discriminator.

### 4. Recovery-Gated AMP

Compute disturbance score from:

- base roll/pitch tilt,
- roll/pitch angular velocity,
- command tracking error,
- base height error,
- optional capture/contact/slip terms.

Then:

```text
style_gate = exp(-beta * disturbance_score)
recovery_gate = 1 - style_gate
command_gate = min_command_weight + (1 - min_command_weight) * style_gate
```

Reward composition:

```text
r = command_gate * r_task
  + style_gate * r_amp
  + recovery_gate * r_recovery
  + reg_gate * r_regularization
  - residual_penalty_gate * ||delta_action||^2
```

### 5. Task Registration

Add:

- `Booster-Run-AMP-TrackAdapter-v0`
- `TrackAdapterRoughWoStateEstimationEnvCfg`
- `TrackAdapterPPORunnerCfg`
- `TrackAdapterRunner`

### 6. World Model Auxiliary Loss

Train a short-horizon future-observation predictor from the adapter latent and current action:

- storage records normalized `next_observations`,
- PPO batches expose `N=20` horizon targets and validity masks,
- world-model loss updates trainable adapter/history modules only,
- metrics log `wm`, `wm_abs_error`, and `wm_valid_fraction`.

### 7. Stage 3 World-Model Pretraining

Before full PPO training, collect disturbance rollouts with the frozen AMP base and train the history encoder plus world-model predictor:

- base actor and discriminator stay frozen;
- residual output is kept effectively zero or ignored during pretrain;
- optimizer updates only `history_encoder` and `world_model_predictor`;
- checkpoints store base checkpoint metadata, history length, world-model target layout, and normalizer state;
- PPO training can explicitly load this pretrain checkpoint before adapter updates.

### 8. Rich World-Model Targets

The default target can remain normalized actor observation for compatibility, but the full Track Adapter path should support a structured `psi_s(s)` target:

- base angular velocity,
- projected gravity,
- command,
- joint position,
- joint velocity,
- previous action,
- optional base velocity/height/contact features when available from env extras.

### 9. Failure-Aware Curriculum

Random disturbance is not enough. Track Adapter full training should bias future disturbances toward recent failures:

- record failure bins by command plus push direction/magnitude/duration/location when base contact or termination occurs;
- decay old bins;
- sample a configurable fraction of pushes/commands from failed bins;
- keep smoke mode conservative and deterministic enough for gating.

### 10. Verification

Lightweight tests must not require Isaac Sim:

- base checkpoint loads and freezes,
- zero-init adapter action equals base action,
- optimizer excludes frozen base and frozen discriminator,
- storage returns history batches,
- reward gate math is monotonic and finite.

Isaac-dependent smoke:

- config registration,
- zero-iteration or one-iteration headless run,
- tmux training launch with the requested Python environment.

## Phases

1. Documentation and worklog setup.
2. Policy module and exports.
3. Track Adapter storage and PPO algorithm.
4. Track Adapter runner with history, frozen checkpoint normalizers, recovery-gated rewards, residual penalty.
5. Recovery reward/gate config and task registration.
6. World-model auxiliary loss and pretraining/update path.
7. Observation plus action history support.
8. Push, dynamics-randomization, latency/noise, and sudden-stop curriculum.
9. Failure-aware mining hooks and evaluation metrics.
10. Stage 3 pretrain script/runner.
11. Unit/static verification.
12. Bounded world-model pretrain smoke in tmux.
13. Bounded adapter PPO smoke in tmux.
14. Abrupt all-zero command evaluation.
15. Longer training run setup only after the smoke gates pass.

## Current Implementation Status

- Implemented: frozen base actor load/freeze, observation plus action history, GRU/MLP history encoder, zero-initialized residual actor, trainable critic/noise, frozen discriminator option, strict base normalizer/AMP auxiliary loading, explicit Stage 3 world-model pretrain, structured `psi_s(s)` targets, PPO-time world-model replay/alternating updates, recovery-gated rewards, sudden-stop rewards, contact-aware disturbance gate, push-active recovery gate, failure-aware command/push mining, low-speed straight-walk command stratification, stable-only zero-command residual suppression, smoke/full-training guards, Track Adapter play/eval path.
- Verified: focused non-Isaac tests, py_compile, diff whitespace check, strict Stage 3 pretrain smoke, pretrain-loaded PPO smoke, abrupt all-zero command evaluation, fixed velocity grid evaluation, and high-speed jerk/push evaluation support.
- Stage 3 done: a non-smoke Stage 3 push-coverage pretrain completed and produced a validated `world_model_pretrain` checkpoint with `psi_s` target dim `81`, finite loss, base-only collection, and push/material/mass/CoM/sudden-stop coverage.
- First full PPO attempt: started from that checkpoint and initially stayed healthy, but was intentionally stopped by the safety gate at iteration `128/12000` after `Episode_Termination/base_contact` exceeded `0.20` for two consecutive checks. The proximate issue was fast residual growth, not world-model divergence.
- Hardening added after the stop: smoke-iteration clamp, complete pretrain-weight coverage checks, PPO checkpoint pretrain-metadata copy, lower residual scale/learning rate/noise/entropy, stronger residual penalty, and a residual-norm safety gate.
- Correction: the first Stage 3/PPO attempt used `H=20,N=1`, which is too small for the Any2Track-style Track Adapter described in `idea-adapter.md`. The default configuration and chained runner have been updated to `H=79,N=20`; with the current 50 Hz policy step this is a `1.58 s` history window and `0.40 s` prediction window. The old `H=20,N=1` pretrain checkpoint is no longer compatible with the corrected PPO config.
- Correction: the H79/N20 world model is now autoregressive instead of direct-horizon prediction. Checkpoints must carry `world_model_metadata.prediction_mode=autoregressive`, so old direct H79/N20 checkpoints are rejected before PPO.
- Fixed-grid evaluation found that the frozen base policy itself under-tracks `vx=0.2` (`actual vx` about `0.065` without push), while `vx=0.4/0.8` are much closer. The Track Adapter selected checkpoint is safe in the grid but uses high residual at zero/low speeds and slightly worsens higher-speed tracking, so the next training branch should emphasize low-speed straight commands and zero-command residual discipline without sacrificing push recovery.
- The push-robust/low-speed/jerk v2 branch stopped at iteration `1884` because `base_contact` exceeded the safety gate as residual usage rose.
- The safer v3 resume branch from `model_1700.pt` ran to iteration `6570/13700` and stopped by the residual safety gate, not by base contact: `adapter_scaled_residual_action_l2` reached about `1.20`, while `base_contact` was about `0.024`, `wm loss` was about `0.225`, `wm_valid_fraction` about `0.774`, and `failure_during_sudden_stop` stayed `0.0`.
- Latest regular v3 checkpoint for fixed-grid evaluation is `logs/rsl_rl/run_amp_track_adapter/2026-05-10_16-15-05_track_adapter_pushrobust_lowspd_jerk_v3_resume1700_20260510_161501_ppo/model_6500.pt`; do not promote or resume beyond it without push/jerk/low-speed/zero-command evaluation.
- Fixed evaluation on 2026-05-11 confirmed that the v3 branch regressed normal behavior. `model_6500.pt` has no-push zero-command drift (`vx/vy/wz = 0.076/0.103/0.198`), high residual (`~1.62` at zero command), `vx=2.0` no-push fall/base-contact about `0.071/0.066`, diagonal `vx=1.0,vy=1.0` no-push fall/base-contact about `0.246/0.211`, and high-speed jerk failures that are worse than the frozen base. Do not promote `model_6500.pt`; treat v3 as an over-intervention branch.
- v4 direction: an action-side residual gate is now implemented and must be used for the next branch. Stable no-push locomotion caps the actual residual action, stable zero-command can close it completely, and push-active/sudden-stop windows reopen it. PPO stores and reuses the rollout-time gate for log-prob recomputation, so this is an actual behavior constraint rather than only a reward penalty. Low-gate samples also carry a differentiable ungated residual auxiliary loss, so hidden residual growth is penalized even when the behavior gate is closed. Start v4 from the validated Stage 3 pretrain, not from v3 PPO checkpoints.
- v4 `model_600.pt` fixed eval: zero-command no-push is essentially stationary, high-speed jerk stop from `vx=2.0`, `vx=1.0`, `vy=1.0`, and `vx=1.0,vy=1.0` passes with fall/contact `0.0` even with the explicit stop-time push, and normal no-push tracking is stable except for weak `vx=0.2` under-tracking and a small diagonal `vx=1.0,vy=1.0` ever-fall/contact rate of about `0.0158`. Residual norms remain effectively zero, so do not yet claim learned adapter recovery; continue training and evaluate later checkpoints for whether residual recovery emerges without reintroducing v3-style over-intervention.
- v4 `model_1199.pt` completed the 1200-iteration run safely, but fixed eval shows it is not strictly better than `model_600.pt`: low-speed and `vx=2.0` tracking improve slightly, but diagonal `vx=1.0,vy=1.0` worsens to about `0.0205` ever-fall/contact in the no-push grid and `0.0078` fall/contact in high-speed jerk before the stop. Treat `model_600.pt` as the safer promotion candidate until a later branch improves diagonal stability and produces meaningful residual recovery.
- v5 direction: resume from v4 `model_600.pt`, not `model_1199.pt`. Add explicit diagonal command coverage, low-speed-only tracking reward, manual-stop recovery detection, command norm cap, zero-command-safe failure mining, recovery-window residual activation, and push/recovery diagnostics. The goal is not to loosen stable-state residual authority; it is to make push/stop recovery windows learnable while preserving v4 `model_600` zero-command quietness.
- v5 result: the first v5 branch stayed safe through `model_725.pt`, but was stopped before promotion because push-active state was being cleared by ordinary command resampling and low-speed/diagonal command labels were not clean enough for long training.
- v5b direction: restart from v4 `model_600.pt` after fixing push-window persistence, push-time command attribution, command-stratum exclusivity, low-speed reward masking, push failure-mining command-norm filtering, and opt-in residual output-layer perturbation on resume. Treat v5b as the branch to evaluate for push robustness plus diagonal/low-speed tracking.
- v5b outcome: v5b2 completed safely and learned non-zero residual usage, but fixed evaluation did not beat v4 `model_600.pt`. v5b2 final worsened high-speed jerk failures (`max 0.0156`), while intermediate `model700`/`model825` only matched v4's max failure rate (`0.0078`) with failures spread across more scenarios. Do not promote v5b2; keep v4 `model_600.pt` as the current safest candidate.
- v6 direction: residual emergence must be more recovery-specific. The next branch should reduce stable no-push residual growth, avoid over-weighting diagonal command exposure before push recovery improves, and add/weight objectives that explicitly reward survival plus return-to-command immediately after external pushes.
- v6 objective refinement: the promotion target is not generic reward improvement. It is external-push recovery without falls/base contact plus command tracking at the user-critical speeds: `vx in {0.2, 0.5, 1.0, 1.5, 2.0}` with `vy=0,wz=0`, and `vy in {0.2, 0.4, 1.0}` with `vx=0,wz=0`. For `vx`-only low speeds, straightness must be measured explicitly with cross-axis body velocity, yaw-rate, integrated lateral displacement, and yaw drift; mean forward velocity alone is insufficient.
- v6 branch policy: keep v4 `model_600.pt` as the current safe anchor. Before another long PPO run, run a frozen-base-only fixed evaluation on the same command grid. If the frozen AMP policy itself cannot produce low-speed straight walking at `vx=0.2/0.5`, treat that as a base-locomotion coverage gap and start a base AMP fine-tune branch in parallel with adapter work. The adapter can compensate by using more residual authority, but that becomes a residual locomotion controller rather than a small corrective adapter.
- v6 base-eval evidence: the frozen AMP base checkpoint under-tracks `vx=0.2` (`actual_vx` about `0.060`) and shows substantial lateral/yaw drift in straight `vx` commands once measured by integrated displacement. It passes the fixed high-speed jerk stop/push suite, but interval pushes at `vx=2.0` produce non-trivial failures. Treat low-speed straight tracking and push-active recovery as separate objectives.
- v6 v4-anchor evidence: v4 `model_600.pt` is effectively base-only on the user-critical velocity grid because residual norms remain near zero. It does not improve `vx=0.2` tracking, slows `vx=1.5/2.0`, and only slightly changes drift metrics. It slightly reduces interval-push `vx=2.0` failures but regresses `vx=1.0` interval push and fixed jerk has rare failures where the base-only evaluation did not. Do not rely on the current v4 adapter to solve velocity tracking; use it only as a conservative safety anchor.
- v6 benchmark priority update: high-speed jerk is no longer a primary optimization target. It is only an OOD/smoke check for "does not fall under a bad command transition" and should not drive reward design if it hurts tracking. Primary promotion metrics are single-axis command tracking, ordered by priority: `vx-only`, then `vy-only`, then `vyaw-only`. For `vx-only`, lateral `vy` and yaw crosstalk must be suppressed; for `vy-only`, forward `vx` and yaw crosstalk must be suppressed; for `vyaw-only`, translational drift must be suppressed. Secondary metrics are `vx+vyaw` and `vy+vyaw`. Random mixed `vx+vy` or `vx+vy+vyaw` commands are low priority and should not dominate the curriculum.
- AMP axis-fine-tune implementation: `BOOSTER_AMP_FT_ENABLE=1` now enables exact hard command points for zero, `vx-only`, `vy-only`, `vyaw-only` up to `|vyaw|=1.5`, and secondary `vx+vyaw`/`vy+vyaw` points. It also adds single-axis tracking rewards, crosstalk penalties, zero-command no-push stillness penalties, and push-active return-to-command reward. This is the intended next base-policy branch before rebuilding Track Adapter world-model pretraining on a new base checkpoint.
- AMP axis-fine-tune v1 was stopped early because push-active state was not connected to the command term. v2 is now active in tmux session `amp_axis_ft_v2_20260511_174702`, run directory `logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702`, resumed from `model_70000.pt` with a fresh optimizer. Early v2 logging confirms non-zero `failure_push_active`, non-zero `push_recovery_velocity_track_exp`, active hard command bins, and `base_contact` below `0.02`.
- AMP axis-fine-tune v2 completed through `model_81999.pt`. The final checkpoint improves single-axis tracking but regresses high-speed/no-push and jerk-smoke safety, so it should not be promoted. Fixed evaluation currently favors `model_70100.pt` as the conservative candidate: it largely fixes `vx=0.2` tracking while limiting jerk-smoke regression to one rare `vx=1.0,vy=1.0 -> 0` no-push failure. Later checkpoints such as `model_73000.pt` and `model_81999.pt` improve tracking further but introduce broader safety failures.
- Track Adapter base selection update: despite the safety regression, the active adapter branch uses AMP axis fine-tune `model_81999.pt` because it is the best tracking base and the adapter is explicitly responsible for correcting push/jerk/contact failures.
- v10/v10b result: v10 failed early under physical kicks because the zero-initialized adapter did not intervene before contact dominated. v10b warm-started from the v9 layerwise adapter and reached iteration `1573`, but base contact climbed with stronger kicks and the safety gate stopped the run. Do not promote v10b; use its earlier `model_750.pt` only as a recovery-capable warm resume point.
- v11 direction: recovery-first adaptive force-kick branch. It adds termination-time failure mining, short training-only trunk-contact grace, success-rate-adaptive physical-kick curriculum, resume-relative safety gates, local-frame force pulses, strict no-push residual suppression, and cleaned force-kick eval telemetry. The branch resumes PPO from v10b `model_750.pt`, reuses the verified v10 Stage3 `model_699.pt`, and must be judged by fixed recovery/tracking evaluations after training.
- v6 adapter direction: start from v4 `model_600.pt`, not v5b2. Oversample exact straight-command bins and explicit push windows, reduce early diagonal pressure, strengthen post-push survival/return-to-command rewards, and keep stable no-push residual suppression. Residual authority should open during external-push, command-jerk, or large velocity-error recovery windows, not leak broadly into stable no-push locomotion.
- 2026-05-12 branch change: for the next real Stage3/PPO chain, use AMP axis fine-tune `model_81999.pt` as frozen base. Track Adapter training now explicitly oversamples hard benchmark commands (`vx`, `vy`, `vyaw`, and secondary axis+yaw), widens Track Adapter yaw coverage to `|vyaw| <= 1.5`, adds task-side single-axis tracking/crosstalk rewards, adds no-push zero-command stillness penalties, and adds a recovery-group push return-to-command reward. The adapter's job in this branch is to preserve `81999`'s tracking while correcting push/high-speed/stop instability.
- 2026-05-13 v8 direction: resume from the safer v7 `model_3400.pt` and keep residual authority available during continuous push, but add push-stable residual shaping so residual is costly only when push is active and the disturbance score already indicates a stable body state. This preserves the user's intended push-recovery behavior while reducing broad residual growth that did not improve interval-push outcomes in v7 `model_4000`.
- 2026-05-13 v9 direction: move closer to the Any2Track paper by adding `adapter_mode=layerwise`. The action-residual path remains available for compatibility, but the new branch injects zero-initialized adapters after each frozen base actor linear layer. Existing residual gates still bound the adapter contribution, and diagnostics continue to report final action deviation from the frozen base.
- 2026-05-14 v9 outcome: layer-wise Track Adapter `model_4999.pt` completed `5000` PPO iterations and is the best current promotion candidate. Fixed no-push 12-command eval has max fall/base-contact `0.0/0.0`, average abs velocity errors `0.0700/0.0815/0.1747`, and near-static zero command. Fixed interval-push eval has average fall/base-contact `0.0031/0.0027` and max `0.0326/0.0326` at `vx=2.0`, substantially better than the v8 final branch. High-speed jerk no-push is clean, and push-at-stop jerk has rare max fall/base-contact `0.0156/0.0156`. Remaining work is targeted reduction of rare high-speed push failures without degrading no-push tracking.
- 2026-05-14 push envelope: the external push is implemented as an instantaneous root-velocity delta. Current v9 `model_4999.pt` is reliable for the evaluated `push_xy=0.60`, `push_yaw=0.10`, `1 s` interval condition. With harsher yaw disturbance (`push_yaw=0.22`) and `1 s` interval, `push_xy=0.80` already causes notable high-speed failures (`vx=2.0` fall/base-contact `0.1834/0.1326`), while `push_xy=1.00` and `1.20` are clearly outside the reliable envelope. Treat robust operation as about `0.6 m/s` repeated horizontal velocity disturbance today, not `1.0 m/s+`.
- 2026-05-14 v10 direction: the next branch should explicitly target AnyAdapter-style kicked-but-not-fallen behavior rather than only preserving no-push tracking. Do not merely extend v9. Use v9 as a warm start, but increase recovery-only adapter authority, train the world-model/history encoder on stronger push transitions, add a push-strength curriculum plus failure-mined hard bins, and relax velocity tracking during the first recovery steps while rewarding upright survival, foot replacement, and return-to-command after recovery. Keep stable/no-push residual suppression and zero-command stillness, but remove most residual penalties during push-active recovery windows.
- 2026-05-14 v10 implementation plan: run a fresh H79/N20 Stage3 pretrain on frozen AMP axis-v2 `model_81999.pt`, then PPO with persistent WM replay and alternating WM/PPO updates. The v10 branch must train with two disturbance families: root-velocity push curriculum (`push_xy 0.40 -> 1.00`, `push_yaw 0.10 -> 0.30`) and finite-duration physical trunk force/torque pulses sampled in world frame and applied through the link-frame IsaacLab API (`160-420 N -> 320-900 N`, yaw torque `24 -> 70 Nm`). Adapter authority is recovery-first: stable/no-push residual gates remain tight, while push-active/high-error/tilt/base-height-drop windows open the adapter gate, greatly reduce residual penalties, and include a small activation target so the adapter actually intervenes during recovery. Evaluation must keep velocity-jump push and physical kick results separate and force-kick eval must be run with an explicit v10 `CHECKPOINT`.
- 2026-05-14 v10 correction: the first v10 PPO failed safety gate at iteration `123` because physical kicks were too hard/frequent for a zero-initialized adapter. The corrected v10b branch reuses the verified v10 Stage3 checkpoint, warm-starts layerwise adapter keys from v9 `model_4999.pt`, and ramps physical kick probability/duration/interval as well as force magnitude. A successful recovery-first branch should show adapter intervention from iteration 0, base-contact below the relaxed early safety threshold, and later fixed force-kick eval improvements before promotion.
- 2026-05-15 v12b outcome: v12b avoided the immediate collapse of v12 and reached `model_8050.pt`, but fixed 500N physical-kick evaluation at `model_8000.pt` still showed broad failure: `ever_fall_rate_mean=0.9905-1.0000` and `ever_base_contact_rate_mean=0.9060-0.9653` across the tested commands. Do not promote or extend v12b as-is. The residual gate was open during evaluation, so the missing capability is recovery behavior, not adapter activation.
- 2026-05-15 v13 direction: switch from dense repeated-hit training to a single/low-frequency strong-kick recovery curriculum. The policy should first survive and recover posture, height, angular velocity, foot placement, and no trunk contact, then return to command after a short delay. New reward hooks `push_recovery_delayed_velocity_track_exp` and `recovery_no_base_contact` are available for this branch. v13 resumes from the good no-push v11 `model_6500.pt`, reuses the Stage3 checkpoint, tightens stable/no-push adapter gates, increases recovery-only adapter authority, and targets `480-640N` kicks at `3.2-5.0s` final intervals before re-testing stricter repeated-kick conditions.
- 2026-05-15 v13b horizontal-force checkpoint: v13b `model_7050.pt` is not yet promotion-ready. In the yaw-free/off-center-free horizontal XY repeated-force sweep, `150N x 0.10s` is clean only at zero command, while `vx=2.0` already has `ever_fall/base_contact=0.372/0.339`. At `250N+`, even zero command has large failures. Continue treating human-kick-level robustness as unsolved; use this sweep as the current lower-bound comparison for later recovery branches.
- 2026-05-15 recovery diagnosis: v13b is not mainly blocked by a closed adapter gate. The gate opens and residual actions are non-trivial, but the intervention has not become a coordinated capture-step/recovery controller. The next recovery branch should use competence-gated physical-force curriculum, hard single-kick pass gates before repeated-kick gates, stronger base-contact failure handling, capture-point/directional stepping rewards, and push-heavy Stage3 world-model pretraining. Simply extending v13b is unlikely to reach human-kick robustness if base-contact remains in the `~0.08` train-time band.
- 2026-05-15 v14 direction: implement and train a capture-step branch rather than extending v13b. v14 starts from the v11 `model_6500.pt` adapter anchor, rebuilds Stage3 under single physical kicks, stores continuous push direction for rewards, limits force pulses to one per episode, promotes force strength with push-failure-aware adaptive alpha, and adds explicit directional-step/capture-point/body-frame-stance/base-contact-penalty rewards. The first promotion gate is single-kick horizontal-force recovery; repeated-kick and yaw/off-center kicks remain later gates.
- 2026-05-15 robust-priority ranking correction: v9 layer-wise `model_4999.pt` remains the best evaluated root-velocity-push checkpoint, but it is not the best checkpoint under comparable physical-force push recovery. In the `100N x 0.10s`, `3.0s` interval pure-horizontal physical-force comparison, v9 had avg ever fall/contact `0.3419/0.2773` and worst `0.9189/0.8439`, while v11 `model_6500.pt` had avg `0.0039/0.0039` and worst `0.0176/0.0176`. For physical-force robustness around the current survivable envelope, treat v11 `model_6500.pt` as the best balanced evaluated candidate, with v10b `model_750.pt` close behind. No current checkpoint has yet satisfied the human-kick-like high-force recovery target.
- 2026-05-15 deployment update: v11 `model_6500.pt` is now exported for C++ ONNX deployment at `/workspace/booster_k1_locomotion/assets/track_adapter_v11_model_6500.onnx`. The deploy ABI is `obs[1,75]`, `history[1,79,97]`, `residual_gate[1,1]`, `reset[1,1]` -> `action[1,22]`, `next_history[1,79,97]`; observation normalization is baked into the graph. `booster_k1_locomotion` C++ launch defaults now point to this ONNX and support constant or heuristic adapter gating. `nomadz_deploy` remains a Python/raw-checkpoint path, but its runtime can now load layer-wise v11 checkpoints with `residual_scale=0.16`.
- 2026-05-15 nomadz deployment update: `nomadz_deploy` now also supports the exported v11 Track Adapter ONNX. The `k1_run_amp_track_adapter_latest` task points to `/workspace/booster_k1_locomotion/assets/track_adapter_v11_model_6500.onnx`, keeps the same `residual_gate/reset/history` ABI, and exposes `track_adapter_gate_mode`, `track_adapter_residual_gate`, and `track_adapter_stable_gate`. The process is still the nomadz Python controller, but policy inference uses ONNXRuntime when the checkpoint path is `.onnx`.
- 2026-05-19 v29 direction: v28b proved that the adapter gate and residual were active but the residual did not become a useful capture/recovery step. The next branch adds a training-only recovery teacher loss inside `TrackAdapterAMPPPO`: it supervises the adapter's final ungated scaled residual action during high-gate disturbed minibatches, using inverse-normalized critic/base velocity, projected gravity, angular velocity, and previous action phase to bias a mild leg capture-step residual. This is a warm-start/regularizer for PPO only; deployed inference remains the frozen AMP base plus adapter with no teacher input.
- 2026-05-19 v30/v31 correction: the v29 heuristic teacher is not strong enough as the main route to `300N`-class recovery. The main route is now two-stage: first train a privileged recovery teacher residual actor with PPO (`BOOSTER_TRACK_ADAPTER_TRAIN_PRIVILEGED_TEACHER=1`) using critic observations and larger recovery-only authority; then distill that teacher into the deployable Track Adapter with `BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL=1`. The teacher head is training-only and is not part of the final ONNX/deploy ABI.
- 2026-05-15 nomadz SDK mjviser update: `scripts/sim2sim_booster_sdk_mjviser.py` can now run the v11 Track Adapter ONNX task through the Booster SDK LowState/LowCmd split loop. Use `--task k1_run_amp_track_adapter_latest --track-adapter-backend onnx --track-adapter-gate-mode constant --track-adapter-residual-gate 1.0`; a short smoke reached Viser startup and stopped cleanly.
- 2026-05-16 v14 fixed-eval outcome: v14 improves moderate pure-horizontal physical-force recovery over v11 at `150N x 0.10s`, but the final checkpoint is not best. `model_10000.pt` has the best `150N` contact profile (`avg/max contact 0.0497/0.1274`), while `model_11000.pt` has the best `150N` fall profile (`avg/max fall 0.0613/0.1857`). No-push fall/contact stays clean for all tested checkpoints, and no-push tracking is essentially unchanged from v11. Treat v14 `model_10000.pt` as the conservative balanced promotion candidate and `model_11000.pt` as a close robustness alternative; do not promote final `model_13499.pt` by default.
- 2026-05-16 v15 direction: push recovery remains below the human-kick target, so the next branch resumes from v14 `model_10000.pt` and trains a stronger pure-horizontal physical-force ladder. v15 keeps the recovery reward gate strong during the push window but decouples the actual residual action gate with a phase-aware floor: high authority for the first recovery phase, taper after posture/velocity recover, and reopen when the disturbance score stays high. The new early reward `recovery_push_velocity_cancel_exp` directly rewards cancelling velocity that continues along the push direction before returning to command tracking. First fixed-eval targets are `150N`, `175N`, `200N`, and `250N` pure-horizontal single/repeated force sweeps; human-kick-like claims remain blocked until those pass with low fall/contact.

## Full-Training Gates

Do not start a long Track Adapter training run unless all gates pass:

- implementation gate: frozen base, frozen or stable-segment discriminator handling, observation plus action history, Stage 3 world-model pretrain, PPO-time world-model auxiliary loss, recovery-gated AMP, residual penalty, curriculum, and metrics are all present;
- pretrain gate: bounded frozen-base world-model pretrain completes and saves a checkpoint with finite loss and compatible metadata;
- smoke gate: a bounded headless run completes without startup errors;
- contact gate: non-foot/base contact termination is below the configured threshold during smoke;
- residual gate: stable-walking residual action norm remains small relative to base action norm;
- style gate: AMP/style reward remains active in stable states and is suppressed only during disturbed/recovery states;
- stop gate: sudden all-zero command events occur in smoke and do not dominate fall/contact terminations.

Default numeric gates:

- early contact pass: `Episode_Termination/base_contact < 0.02` during the first smoke window and the first few hundred PPO iterations;
- contact caution: sustained `base_contact >= 0.02` before the adapter has stabilized requires closer monitoring;
- contact stop/restart: `base_contact >= 0.05` early in full PPO, or any sharp increasing contact trend paired with residual growth;
- initial residual pass: `Mean adapter_scaled_residual_action_l2 <= 0.10` during the first 50-100 iterations of a fresh or resumed run;
- residual intervention: manually stop or restart if `adapter_scaled_residual_action_l2 >= 0.80` before iteration 500, and hard-stop if it approaches or exceeds `1.0` early;
- gated residual pass: in v4, stable/no-push `Gate/residual_action_mean` should stay near the configured stable floor/max, while `Gate/residual_action_max` should only approach recovery minimums during push-active or sudden-stop windows;
- ungated residual pass: `AdapterRollout/ungated_scaled_residual_action_l2` and `Loss/ungated_residual_policy_penalty` should not grow monotonically in stable/no-push windows; if ungated residual rises while the gated residual stays small, increase the ungated policy auxiliary before promoting the checkpoint;
- sudden-stop pass: `Metrics/base_velocity/failure_during_sudden_stop == 0.0` in smoke and monitored early full-PPO windows, with sudden-stop samples present in at least one monitored window;
- push-robustness pass: `Metrics/base_velocity/failure_push_active` must be non-zero in monitored windows, `Gate/push_active_mean` must be logged, and `Episode_Termination/base_contact` should remain below `0.02` while interval pushes are active;
- high-speed jerk eval pass: fixed `vx=2.0 -> 0`, `vx=1.0 -> 0`, `vy=1.0 -> 0`, and `vx=1.0,vy=1.0 -> 0` should have `fall_rate == 0.0` and `base_contact_rate == 0.0` before promoting a checkpoint;
- low-speed command coverage pass: `Metrics/base_velocity/low_speed_sampled` should be present and non-zero in full training, with fixed-grid checks at `vx=0.2` and `vx=0.4`;
- zero-command pass: fixed `vx=0, vy=0, yaw=0` evaluation without push should keep mean base speed near zero and should not require large residual action growth;
- WM replay pass: `Mean wm_valid_fraction >= 0.70` during PPO replay updates;
- WM stability pass: `Mean wm loss` must stay finite, with no NaN/Inf and no sustained explosive jump relative to the recent rolling window.

Code-side enforcement:

- default Track Adapter config runs smoke mode only;
- full training requires the single shared unlock predicate:
  `BOOSTER_TRACK_ADAPTER_FULL_TRAINING=1`,
  `BOOSTER_TRACK_ADAPTER_SMOKE_PASSED=1`, and
  `BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING=1`;
- `scripts/rsl_rl/train.py` refuses Track Adapter runs above the smoke limit when the unlock is absent;
- `TrackAdapterRunner` also refuses long runs and aborts smoke if the configured `base_contact` safety gate is violated.

## Open Risks

- The base checkpoint was trained with empirical observation normalization; adapter training must load and freeze that normalizer for base-consistent actions.
- Recovery reward grouping uses reward-manager internals in the fork; the implementation should be defensive if those fields change.
- Freezing the discriminator makes style reward stable but may become stale if AMP observation distribution drifts.
- Abrupt-stop evaluation passed on the smoke checkpoint, but the checkpoint is only a 16-iteration smoke artifact; it should not be treated as a trained adapter.
- Contact-gate and failure-mining contact lookup must fail fast if the contact sensor/body mapping cannot be resolved.
- Low-speed straight walking is partly limited by the frozen base policy, so the adapter may need targeted low-speed samples and reward balance; do not assume a generic full-range uniform command distribution is enough for `vx=0.2`.
- Push recovery remains the primary objective. Zero-command residual suppression must remain gated by stability/style and must be disabled during sudden-stop or push-active recovery so it does not prevent corrective steps during external disturbances.
- Push-active recovery gating intentionally weakens the strict tracker interpretation during the post-push window. This is acceptable for the current objective, but checkpoints must be evaluated for no-push zero-command quietness separately.

## Current v16 Plan

Goal: move from mild `100-150N` physical-force robustness toward human-kick-like recovery without losing the axis-command tracking improvements from the AMP axis-v2 base.

Implementation gates:

- use frozen AMP axis-v2 `model_81999.pt`;
- rebuild Stage 3 with `H=79,N=20` under a force mixture covering `60-380N`, not only `150-250N`;
- during PPO resume, keep the freshly rebuilt Stage 3 history encoder/world model instead of silently restoring the old PPO checkpoint's WM weights;
- use explicit force-mixture sampling rather than adaptive promotion for this branch, because v15b's adaptive curriculum never left the mild rung;
- keep a weak root-velocity proxy push enabled for Stage 3 manifest compatibility, but make finite physical-force pulses the dominant disturbance;
- keep stable/no-push residual gates tight and push-active recovery gates high.

Training/eval gates:

- early training: `Episode_Termination/base_contact < 0.11`, finite WM loss, `wm_valid_fraction >= 0.70`, logged force envelope max `380N`, and non-zero `force_push_applied_force_n` samples from the higher force bins;
- first fixed eval: no-push command grid must remain near v15b, especially zero command and `vx`/`vy` single-axis crosstalk;
- recovery eval: compare against v15b on pure-horizontal physical force at `100, 150, 175, 200, 250N`; prioritize lower fall/contact at `175N+`;
- do not claim human-kick-level success until `200N` is clearly improved and `250N` is materially better than v14/v15b without severe no-push regression.

## Current v19 Plan

v18 completed but did not improve fixed physical-force recovery. It increased residual intervention under force without lowering fall/contact versus v14/v15b. v19 therefore changes the reward priority rather than extending v18:

- restart from v14 `model_10000.pt`, the best aggregate fixed-eval anchor;
- do not use v18 checkpoints as parents;
- during push-active recovery, suppress command tracking almost completely and prioritize survival/recovery:
  - avoid fall and trunk/base contact;
  - keep torso upright and base height recoverable;
  - damp roll/pitch angular velocity;
  - place a capture/support foot in the push direction;
  - keep a usable stance width;
- return-to-command is delayed until about `1.1s` after the push and is also gated by stability;
- residual authority remains available early in the push window, but residual penalties are stronger than v18 so the branch cannot win only by outputting near-maximum residuals;
- force exposure starts with a mostly `80-180N` mixture and only a small `180-260N` tail.

Training/eval gates:

- training `base_contact` should stay below `0.11`, with caution above `0.07`;
- `wm_valid_fraction >= 0.70` and finite `wm loss`;
- force-eval residual norm should not climb toward `1.0` without corresponding fall/contact gains;
- first fixed-eval comparison: v19 checkpoints vs v14 `model_10000.pt` and v15b `model_16499.pt` on `100/150/175/200N` pure-horizontal physical force and no-push command tracking.

## Current v20 Plan

v19 was stopped early because `push_command_gate_min` was only a floor. The default `command_min_weight=0.55` meant push-active task tracking still stayed high. v20 fixes this by adding an explicit `push_command_gate_max` to the runner and capping command/task tracking during physical-push recovery.

Implementation:

- runner supports `BOOSTER_TRACK_ADAPTER_PUSH_COMMAND_GATE_MAX`;
- v20 keeps normal no-push tracking with `command_min_weight=0.55`;
- during push-active recovery, command gate is clamped to `0.025-0.060`;
- recovery gate remains fully open with `push_recovery_gate_min=1.00`;
- style gate is capped to `0.045`;
- immediate push velocity tracking remains disabled, delayed tracking starts later (`1.35s`) and has lower weight;
- recovery/no-fall/no-base-contact/upright/height/angular-damping/capture-step rewards are stronger than v19.

Training/eval gates:

- continue only if `base_contact` stays below `0.11` after the relative safety gate starts;
- caution if `base_contact` sustains above `0.07`;
- fixed eval must use the v20 runtime gate, especially `push_command_gate_max=0.06`;
- compare last v20 checkpoints against v14 `model_10000.pt` and v15b `model_16499.pt` on `100/150/175/200/225N` pure-horizontal physical force plus no-push velocity grid.

Outcome:

- v20 stopped by the safety gate at `model_10503_aborted.pt` after `base_contact` exceeded `0.11`;
- fixed eval of `model_10425/10450/10475/10500` did not clearly beat v14/v15b;
- v20 reduced force-eval residual usage substantially, but lower residual did not improve fall/base-contact enough;
- keep v14 `model_10000.pt` as the aggregate physical-force benchmark and v15b `model_16499.pt` as the `100-175N` fall-rate anchor;
- do not promote v20.

Next direction:

- keep the `push_command_gate_max` fix;
- do not continue v20 as-is;
- the next branch should improve recovery behavior directly, likely by training/evaluating earlier recovery phases with explicit success labels or harder capture-step shaping, not just by suppressing speed tracking further.

## Current v21 Plan

v21 targets the remaining failure after v20: the adapter can gate and intervene, but it has not learned a sufficiently useful capture/support behavior.

Implementation changes:

- keep the v20 `push_command_gate_max` fix;
- make push command gate phase-aware:
  - early push window: command/task gate capped near `0.065`;
  - after `0.65-1.55s`: cap tapers toward `0.55` so the policy can return to command before the full push-active window expires;
- add `recovery_directional_support_contact_exp`, a reward that only pays when a contacted support foot is placed in the push direction;
- keep immediate push velocity tracking disabled;
- keep delayed return-to-command reward, but allow task tracking to re-enter during the later recovery phase.

Training branch:

- parent: v15b `model_16499.pt`, because v20 became too low-intervention and v15b remains the strongest evaluated `100-175N` fall-rate anchor;
- v14 `model_10000.pt` remains the aggregate benchmark in fixed eval;
- force mixture focuses on `70-210N` with a small `210-250N` tail;
- safety threshold is relaxed to `base_contact < 0.13` with patience `5` so the branch can learn through harder recovery samples without running unchecked.

Promotion gate:

- compare v21 checkpoints against v14 `model_10000.pt` and v15b `model_16499.pt`;
- primary fixed eval: `100/150/175/200/225N` pure-horizontal physical force plus no-push velocity grid;
- v21 must improve `175/200N` fall+base-contact against v14 without no-push regression;
- do not claim progress based only on lower residual use.

Sim2real observation and added gate:

- user sim2real testing found that `/workspace/booster_k1_locomotion/assets/track_adapter_model_1100.onnx`
  produces a better static zero-command rear-push response than
  `track_adapter_v11_model_6500.onnx` and `track_adapter_v14_model_10000.onnx`: the old
  `model_1100` steps a foot, while v11/v14 tend to handle the push with toe/ankle bracing;
- later real-robot deployment confirmed the v11/v14 failure mode more strongly: under a slow manual push,
  both v11 and v14 mostly braced/moved with the toes instead of taking a useful recovery step, then fell;
- treat this as a real promotion-gate gap, not just a visualization issue: existing fixed evals
  average fall/base-contact over command grids and random/pure-horizontal forces, but do not
  score "static rear push -> contacted capture/support step";
- also separate deployment tuning from learned behavior: `model_1100.onnx` has only
  `obs/history` inputs, while v11/v14 have explicit `residual_gate/reset` inputs. If v11/v14 are
  run with a low heuristic gate or mismatched real-robot PD/action gain, they can appear much worse
  even if the learned policy is usable;
- next fixed eval must include a zero-command static rear-push stepping benchmark that reports
  fall/contact, foot step projection in push direction, contacted support-foot placement,
  base velocity arrest, and residual/gate level. Compare `model_1100.onnx`, v11 `model_6500`,
  v14 `model_10000`, v15b, and v21 checkpoints before promoting another branch.

## Superseded v22 Plan

v22 is a correction for the deploy-observed slow rear-push failure. The previous v10-v21 branches mostly optimized
short physical kicks and higher-force impulse recovery. The real-robot failure is different: zero command, weak
rear-to-front force applied over several seconds, and the newer layerwise adapters appear to brace with toes/ankles
instead of taking an early support step.

Status: stopped. This branch was too narrow for the main objective. The weak sustained push case remains an evaluation
gate, but the policy must not be specialized around it. Old `track_adapter_model_1100.onnx` is also not a reliable
training target because its exact frozen AMP base/export pairing is not proven to match the current `model_81999.pt`
base.

Implementation changes:

- external force event now supports a fixed body-frame direction with probability and jitter;
- external force event now supports ramp-up time, so pushes can look like a hand shove rather than an impulse;
- external force event can restrict pulse starts to near-zero velocity commands, so the weak rear-push distribution is
  trained as a standing recovery problem rather than mixed into normal tracking;
- reward set includes a first-contact directional step term, which only pays when a foot newly lands in the push
  direction after air time. This targets the real-robot failure where newer adapters brace with toes/ankles instead of
  stepping.
- v22 trains on low-force sustained pushes:
  - `6-45N`;
  - `1.2-3.2s`;
  - `0.25-0.90s` ramp-up;
  - body-frame `+x` selected with about `0.92` probability;
  - start command norm capped at `0.05`;
- v22 increases standing-command exposure to about `0.42` and disables failure mining for this first narrow branch,
  so the replay distribution is not diluted away from the manual-push failure mode;
- v22 uses `action_residual` instead of `layerwise`;
- v22 warm-starts only the old v7 `model_1100` residual actor and retrains Stage 3 world model for this disturbance
  distribution.

Why this branch:

- old 1100's Isaac fall/contact result is not clearly better than v11/v14, but its residual norm at zero-command
  sustained rear-push onset is about `0.0026`, while v11/v14 are around `0.42-0.58`;
- that is consistent with the sim2real symptom: newer adapters are over-intervening under deployment constant gate;
- the next policy should preserve the old low-intervention behavior when stable, then step only when the slow push
  produces real velocity/tilt/capture demand.

Promotion gates:

- zero-command sustained rear-push fixed eval must include `10/20/30N x 2.5s` and report fall/contact plus contacted
  support-step projection;
- compare against old v7/1100, v11, v14, v15b, and latest v21;
- no-push zero command must keep residual near zero and not drift;
- velocity tracking grid must not regress badly on the priority commands: `vx=0.2/0.5/1.0/1.5/2.0`,
  `vy=0.2/0.4/1.0`, and yaw-only up to `1.5rad/s`;
- if deploy still uses constant gate, promotion must consider constant-gate residual magnitude, not only Isaac fall rate.

## Superseded v23/v24 Plan

v23 restores the main objective: robust recovery under broad external force. The weak sustained hand-push case is
included, but it is not the training target by itself.

Implementation changes:

- external force event supports `force_duration_mixture`, sampled as coupled
  `force_min:force_max:duration_min:duration_max:ramp_min:ramp_max:weight` components;
- this avoids bad independent samples like `250N x 3s`, while still mixing:
  - weak sustained shove: `8-35N`, `1.0-3.0s`, ramped;
  - moderate shove: `35-90N`, `0.35-1.2s`;
  - normal kick: `90-170N`, `0.08-0.16s`;
  - strong kick: `170-250N`, `0.06-0.12s`;
  - rare hard kick: `250-340N`, `0.05-0.10s`;
- resume from v15b `model_16499.pt`, the best confirmed `100-150N` robust anchor, not old `model_1100`;
- rebuild Stage 3 world model from scratch for this broad disturbance distribution on frozen AMP `model_81999.pt`;
- keep layerwise adapter mode and residual scale `0.17`, because native higher authority worsened v15b force eval;
- keep a small first-contact step reward, but use it as a general recovery cue rather than a weak-push specialization;
- standing exposure is normal (`0.12`), hard command exposure remains high (`0.70`), and force directions remain mostly
  random with only a small rear-push bias.

Promotion gates:

- weak sustained static push: `10/20/30N x 2.5s`, no fall/base-contact, no drift, and support step if needed;
- strong physical kick: `100/150/175/200/250N x 0.08-0.10s`, improve v15b/v14 at least at `175N+`;
- no-push velocity grid must remain close to v15b/v14;
- deploy gate sweep must not show constant high residual at zero command.

Status:

- v23 Stage 3 pretrain completed and remains the active world-model checkpoint;
- v23 PPO was stopped because high episodic reward masked base-contact regression around `0.24`;
- v24 reduced survival/recovery reward masking and raised base-contact penalty, but failed the early safety gate because
  the broad `force_duration_mixture` was still sampled at full strength from the first PPO iterations.

## Superseded v24b Plan

v24b keeps the broad-force recovery objective and the v23 Stage 3 checkpoint, but fixes the force curriculum itself.
When `force_duration_mixture` is active, force magnitudes can now be scaled by the same adaptive curriculum alpha used
for force probability and scalar force ranges. This prevents the run from starting with full `340N` mixture samples
while keeping the final target distribution unchanged.

Implementation changes:

- `mdp.apply_external_wrench_pulse` now accepts
  `force_duration_mixture_scale_with_curriculum` and `force_duration_mixture_force_floor`;
- `run_amp/env_cfg.py` exposes those parameters through
  `BOOSTER_TRACK_ADAPTER_FORCE_DURATION_MIXTURE_SCALE_WITH_CURRICULUM` and
  `BOOSTER_TRACK_ADAPTER_FORCE_DURATION_MIXTURE_FORCE_FLOOR`;
- `run_track_adapter_stage3_recovery_v24b_curriculum_guard.sh` reuses the v23 world-model checkpoint and v15b robust
  adapter anchor, starts the mixture around `168N` max effective force, and only promotes toward `340N` when
  `base_contact`, push-failure, and episode-length gates are acceptable;
- contact masking remains guarded by a stronger base-contact penalty and safety gate, so high return alone is not a
  promotion signal.

Promotion gates:

- during training, `Episode_Termination/base_contact` must not trend into the v23 failure band (`~0.20+`);
- adaptive force alpha should rise only when `base_contact <= 0.055`, push failure is low, and episode length recovers;
- fixed evaluation must compare v15b/v11/v14/latest on physical force sweeps, including `100/150/175/200/250N`
  kick-style pushes and weak sustained zero-command pushes;
- velocity tracking grid must remain acceptable on the priority commands:
  `vx=0.2/0.5/1.0/1.5/2.0`, `vy=0.2/0.4/1.0`, and yaw-only up to `1.5rad/s`.

Status:

- v24b reached `model_20525.pt` and stayed stable around effective `235N` max training force;
- fixed weak sustained push eval showed `10-20N x 2.5s` is mostly clean, `30N` is improved but imperfect, and
  `50N x 2.5s` still fails badly;
- adaptive force alpha stalled at `0.56`, so v24b did not finish the stronger-force curriculum.

## Superseded v25 Plan

v25 changes the target from "human kick" to robot-contact robustness: light shoving, short bumps, repeated minor
contacts, and recovery during walking or standing.  The goal is not to win against arbitrary large impulses; it is to
avoid falling during small robot-to-robot scuffles and return to command afterward.

Implementation changes:

- resume from v24b `model_20525.pt`, the best current midpoint checkpoint;
- reuse v23 Stage 3 world-model pretrain and keep alternating replay-buffer world-model updates during PPO;
- train with up to three force pulses per episode instead of one;
- use a robot-contact mixture:
  - `8-30N` for `1.2-3.0s`;
  - `30-65N` for `1.2-2.8s`, directly targeting the failed `50N x 2.5s` sustained-push case;
  - `45-90N` for `0.45-1.3s`;
  - `80-150N` for `0.12-0.40s`;
  - `140-220N` for `0.06-0.16s`;
  - rare `220-280N` for `0.05-0.10s`;
- keep horizontal trunk forces without extra yaw torque, so yaw disturbance comes from the robot dynamics rather than
  injected torque;
- start near the v24b operating point and let adaptive curriculum promote toward `280N` using
  `promote_base_contact=0.08`, `demote_base_contact=0.14`;
- raise recovery authority slightly with `residual_scale=0.18`, but keep zero-command and stable residual penalties;
- reduce command/style pressure during active contact and strengthen upright, base-height, capture-point, directional
  step/support, and direct base-contact penalties.

Promotion gates:

- zero-command sustained body-frame push:
  - `10/20N x 2.5s`: clean rate should stay near `1.0`;
  - `30N x 2.5s`: improve over v24b `clean=0.8438-0.8750`;
  - `50N x 2.5s`: must improve materially over v24b `clean=0.1875-0.2188`;
- repeated bump eval: `2-3` contacts per episode in the `80-180N` range should not produce a high fall/contact rate;
- random horizontal direction evaluation should include forward, rearward, lateral, and diagonal contacts;
- velocity grid should not regress badly on priority commands after contact training.

Status:

- v25 was stopped early because the first robot-contact distribution was too abrupt;
- by iteration `20535/27025`, `base_contact=0.4164` under `force_push_applied_force_max_n=157.4932N`, so it was not a
  useful continuation point.

## Current v25b Plan

v25b keeps the robot-contact objective but stages it gradually enough that the adapter can learn the contact regime
instead of immediately falling into base-contact failures.

Implementation changes relative to v25:

- resume from v24b `model_20525.pt`;
- reuse the v23 Stage 3 world-model checkpoint;
- reduce initial contact pressure:
  - `max_pulses=2`;
  - activation probability starts at `0.14`;
  - initial adaptive curriculum alpha starts at `0.35`;
  - force-duration mixture is curriculum-scaled with force floor `0.38`;
  - early effective force envelope starts around `143N` max instead of the v25 `203N` early envelope;
- keep the final target broad enough for robot-contact robustness:
  - weak sustained pushes in the `8-55N` range;
  - medium bumps around `45-130N`;
  - rarer short bumps up to `190-240N`;
- hold curriculum promotion unless base contact is low enough:
  - promote around `base_contact <= 0.075`;
  - demote around `base_contact >= 0.13`;
  - hard safety band around `0.16` after the initial warmup;
- use `residual_scale=0.17`, not the more aggressive v25 `0.18`;
- keep direct base-contact penalty stronger at `-125`.

Live gates:

- healthy start: `base_contact < 0.10` while force max is around `100-145N`;
- stop/revise if `base_contact >= 0.13-0.16` persists;
- do not promote the run solely from reward; fixed evaluation must improve:
  - zero-command sustained `30N x 2.5s`;
  - zero-command sustained `50N x 2.5s`;
  - repeated random horizontal bumps;
  - velocity grid on priority `vx/vy/yaw` commands.

Current status:

- v25b is running in tmux session `ta_v25b_robot_contact`;
- v25b survived the early window that killed v25;
- the adaptive curriculum has started to promote after `base_contact` briefly dropped below the `0.075` promote gate;
- current force envelope is still moderate, around `145-149N` max, so this is the adaptation stage rather than final
  proof of robustness;
- auto-evaluation is armed in tmux session `ta_v25b_robot_contact_auto_eval`.

Post-training fixed eval:

- `scripts/rsl_rl/eval_track_adapter_robot_contact_suite.sh` is the current promotion suite;
- it evaluates:
  - zero-command no-push stillness;
  - sustained body-frame `+x` pushes: `10/20/30/50N`;
  - repeated random horizontal bumps: `50/80/120/160N`;
- the watcher will evaluate the last two v25b checkpoints and the v24b anchor after the training process exits.

## Current v26 Plan

v26 switches the adapter objective from tracking improvement to robustness-first recovery.  The frozen AMP base policy is
treated as the command-tracking locomotion policy; the adapter should mostly stay closed in stable/no-push states and spend
its capacity on external-disturbance recovery.

Implementation changes relative to v25b:

- resume from the v24b stable anchor `model_20525.pt`, not from the v25b contact-plateau branch;
- reuse the v23 broad-force Stage 3 checkpoint and keep PPO-time alternating WM replay updates;
- reduce per-iteration cost:
  - `num_steps_per_env=48` instead of `64`;
  - `world_model_updates_per_iteration=4` instead of `8`;
- enable a hybrid disturbance distribution:
  - Any2Track-style root-velocity impulses are reintroduced as a cheap recovery-state generator;
  - physical trunk force remains enabled and remains the promotion/evaluation target;
- downweight normal velocity/style pressure during push windows:
  - task weight around `0.35`;
  - style weight around `0.035`;
  - recovery weight around `7.20`;
  - immediate push tracking stays disabled and return-to-command is delayed;
- keep stable/no-push residual authority nearly closed while opening recovery authority:
  - stable residual gate max `0.002`;
  - zero-command gate `0.0`;
  - push recovery gate `1.0`;
- make the force curriculum contact-strict:
  - start around alpha `0.28`, floor `0.30`;
  - promote only near `base_contact <= 0.060`;
  - demote near `base_contact >= 0.100`;
  - one physical force pulse per episode at first;
  - short high-force tail is present but rare.

Live gates:

- kill/revise if `base_contact` stays above `0.10` after the early warmup instead of demoting the curriculum;
- kill/revise if scaled residual grows above roughly `0.72` without lowering contact;
- healthy signal is not high reward by itself, but lower `base_contact` under non-zero `force_push_active` plus non-zero
  recovery residual usage;
- final promotion still requires fixed robot-contact eval, not just training logs.

## Current Strict Decision Gate

Do not continue a PPO branch just because the live reward or force curriculum appears to move.  For the current
robustness-first phase, continuation requires fixed-eval evidence:

- zero-command/no-push stillness remains clean;
- sustained zero-command trunk push does not collapse at `30N`;
- `50N x 2.5s` sustained push has a usable clean rate;
- random horizontal bumps improve against the v24b anchor;
- base-contact/fall rates improve without relying on excessive residual action.

As of the first strict-eval partial result, `v26c_20825` fails this gate: it is clean at no-push stillness but falls in
most `50N` sustained-push trials and all `80N` sustained-push trials.  If the remaining v26c checkpoints do not beat the
v24b anchor, stop this reward/curriculum line and redesign the recovery mechanism rather than spending more PPO time.

The sustained-push comparison now confirms that v26c is not the next branch to extend.  The next robustness branch must
explicitly train the target condition instead of relying on short-bump transfer:

- add a sustained scuffle component around `35-85N` for `1.8-3.5s`;
- keep `80-180N` short bump components, but treat them as secondary;
- make the push-active/recovery window at least as long as the maximum sustained-force duration plus recovery margin;
- do not promote on reward or curriculum alpha unless fixed `30/50/80N x 2.5s` sustained-push clean rates improve;
- keep zero-command no-push residual closed.

Operational correction:

- the first v26 launch was stopped because root-push intervals made `failure_push_active=1.0` and opened the residual gate
  almost continuously;
- v26b keeps the same robustness-first plan but uses sparse root impulses, lower residual scale, a smaller recovery
  activation target, and slightly lower physical-force activation;
- v26b was also stopped after `adapter_scaled_residual_action_l2` exceeded the configured safety band (`0.7349 > 0.72`);
- the active run is now v26c, tmux `ta_v26c_force_dominant`;
- the active auto-eval watcher is tmux `ta_v26c_force_dominant_auto_eval`;
- v26c keeps physical-force recovery as the main training target and uses root-velocity impulses only as a rare auxiliary;
- early v26c is acceptable while scaled residual remains bounded and `base_contact` stays below the stricter `0.10`
  demotion band.

## Current v30b Correction Plan

v30 proved the architecture hook works, but it was not a viable 300N-class teacher run:

- force exposure started too high for a zero-initialized teacher head;
- PPO learning rate and exploration were still adapter-preservation settings, not teacher-from-scratch settings;
- PPO-time world-model updates consumed time although the privileged teacher does not use the deployable history encoder;
- logged residual remained effectively zero while `base_contact` rose above `0.21`.

v30b replaces that with a real teacher-training schedule:

- train only the privileged teacher residual actor, critic, and action noise;
- disable PPO-time world-model loss/updates during teacher training;
- raise teacher LR to `1.5e-5`, action std to `0.14`, and residual scale to `0.70`;
- start the physical-force curriculum near `100-110N` max, not `300N`;
- promote toward `300N+` only when base contact and push-failure metrics are already low;
- keep weak/slow sustained pushes in the mixture, but keep short high-force components so the policy does not specialize to slow leaning;
- keep stable/no-push residual closed and use explicit recovery activation only during recovery-gated samples.

Continue v30b only if these early signals appear within roughly the first `100-200` relative iterations:

- `adapter_privileged_teacher_active = 1.0`;
- teacher ungated/scaled residual L2 is non-zero and rising during push windows;
- `base_contact` is below the safety band after the initial curriculum settles;
- force curriculum alpha promotes instead of pinning at the lower bound;
- fixed sustained-push eval beats v26c before launching v31 distillation.

## v32/v33 Completion Plan: Teacher, Replay, Failure Reset

The previous teacher-only branch was still incomplete relative to the recovery-teacher plan.  The complete sequence is now:

1. `v32`: train a privileged recovery teacher.
   - critic/teacher privileged input includes explicit physical force state and foot/trunk contacts;
   - actor policy observation remains unchanged, so the deployable adapter interface is preserved;
   - failure reset curriculum records pre-fall states and restarts some reset episodes from those states;
   - PPO-time world-model updates remain disabled because the teacher does not use the deployable history encoder.
2. `v33`: distill the successful v32 recovery teacher into the deployable Track Adapter.
   - resume from the latest v32 teacher checkpoint;
   - load the file-backed v32 teacher replay before training starts;
   - keep the frozen AMP base policy;
   - train the adapter with PPO plus online privileged-teacher targets;
   - insert recovery-window teacher targets into persistent replay;
   - run replay BC warm-start before the first PPO rollout/update, then run DAgger-style replay updates each iteration;
   - keep the failure reset curriculum active so the student sees the same pre-failure recovery states.

Promotion gate:

- do not consider the branch complete until fixed sustained-push and robot-contact evals beat the v26c/v30b anchors;
- judge recovery primarily by fall/base-contact and walk-back-to-command after force, not by high training reward;
- require no-push/zero-command residual to remain closed;
- if v32 cannot learn non-zero teacher residual with controlled base-contact, stop teacher training before v33;
- if v33 replay insertion or BC updates stay at zero after recovery samples exist, stop and fix distillation.

Operational rule for v32/v33 continuation:

- short chunking is allowed only to contain the Isaac/PPO stall;
- same-branch chunk resumes must use `BOOSTER_TRACK_ADAPTER_LOAD_OPTIMIZER=1`;
- branch-changing resumes may still default to `0` when checkpoint shapes or optimizer parameter groups differ;
- if chunks stop making checkpoint progress, do not keep relaunching blindly; reduce the runtime surface or isolate the reset/replay code path that is hanging.

Current operational choice:

- v32 has been moved back to a single tmux training process from `model_20977.pt`;
- chunking should not be used again unless the single process demonstrably stalls;
- an expected base-checkpoint privileged normalizer shape difference must not be treated as a v32 resume failure, because the v32 checkpoint normalizer shape matches the active critic observation shape.

Root-cause correction after the v32 stall:

- out-of-band failure-reset state injection is no longer allowed in the current full training path;
- keep physical force push and privileged teacher learning active, but keep `BOOSTER_TRACK_ADAPTER_FAILURE_RESET_CURRICULUM=0`;
- if failure-reset is needed again, implement it as a proper IsaacLab reset/event path, or add all of these guards before enabling:
  - same-episode validation for stored pre-failure samples;
  - finite/root-height/quaternion/joint-state validation;
  - synchronized `asset.set_external_force_and_torque()` plus command `record_external_push(...)` when replaying force;
  - scene/sensor forward/update before observations are read.

v34 correction after v32 curriculum stagnation:

- do not continue a teacher run that stays pinned at `~140N` while residual remains near zero;
- keep both force families in training:
  - weak sustained pushes of roughly `10-90N` over `1.2-3.6s`;
  - short stronger bumps/kicks from `180N` upward, with rare `420-620N` samples;
- direct force escalation probes showed that immediate `~220-400N` startup exposure creates high-contact teacher data before reliable recovery:
  - v34 reached `~403-428N` displayed force max and quickly rose to `base_contact ~= 0.25-0.29`;
  - v34b reached `~296-308N` displayed force max and later rose to `base_contact ~= 0.36`;
  - v34c started around `~223N` displayed force max and later rose to `base_contact ~= 0.206`;
- the next correction is v34d:
  - privileged teacher no longer has to solve recovery from frozen AMP alone;
  - with `BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_USE_ADAPTER_BASE=1`, teacher PPO overlays recovery residual on the resumed deployable adapter action;
  - adapter base is evaluated without gradients, preserving the staged teacher-then-distill structure;
  - force curriculum starts above the old mild band, but in the `~175-190N` displayed max range instead of the failed `220N+` startup band;
  - promotion gate is realistic (`base_contact <= 0.060`) and demotion/safety gates are strict enough to avoid training on high-contact failure data;
- v34d still failed as a continuation branch:
  - displayed force max was only `~184-193N`, but `base_contact` rose to about `0.16-0.18`;
  - alpha demoted to the minimum instead of promoting;
  - root cause: the v32 teacher head and optimizer were reused after changing the teacher base input from frozen AMP action to adapter action;
- v34e therefore keeps the adapter-base teacher architecture but resets the privileged teacher residual head and uses a fresh optimizer:
  - `BOOSTER_TRACK_ADAPTER_RESET_PRIVILEGED_TEACHER_ON_RESUME=1`;
  - `BOOSTER_TRACK_ADAPTER_LOAD_OPTIMIZER=0`;
  - initial force max target is reduced to the `~165-170N` displayed range, still above the old `~140N` stall;
  - promotion is allowed only if base-contact stays below `0.055`, with safety stop at `0.120`;
- v34e fixed the input/optimizer mismatch but still did not create a useful teacher:
  - force max demoted to `~163N`;
  - `base_contact` reached `0.1206`;
  - privileged teacher residual stayed nearly zero;
- v34f adds a training-only capture-step prior to privileged teacher PPO:
  - enable `BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_APPLY_TO_PRIVILEGED=1`;
  - use recovery teacher loss only during training, not deploy;
  - rebalance recovery rewards away from dense posture survival and toward directional step/support/first-contact;
  - stop if residual still stays near zero or base-contact rises at the mild `~160N` band;
- v34g extends that prior with explicit physical-force and push-delta features from privileged recovery observations:
  - this fixes the v34f issue where the target only reacted to velocity/tilt and produced tiny residual targets;
  - continue only if force/push-delta metrics are non-zero in force-active windows;
- v34h is the next curriculum step if the `~160N` band is acceptable:
  - start near the next force band rather than waiting at alpha `0.10`;
  - expected initial displayed max is roughly `250N`;
  - promote by controlled base-contact and push-active metrics, with a shorter `420` step episode-length gate;
  - keep weak sustained force and short stronger bump components in the same mixture;
  - strengthen the capture-step prior so the adapter/teacher actually uses non-zero residual during recovery;
- v34h demonstrated the danger of jumping too far:
  - displayed force max reached `~298N`;
  - base-contact rose to `~0.38`;
  - residual stayed small, so this was not a valid curriculum promotion;
- v34i is the corrected next step:
  - start near `~220-230N`, not `~300N`;
  - promote only when current-band base-contact is controlled;
  - use separate privileged-teacher actor gradient clipping so recovery supervision is not drowned by critic/value gradients;
  - stop if residual remains tiny while contact rises.
- teacher success is not high reward alone.  It must show non-trivial recovery residual use, higher force exposure, controlled base-contact, and later fixed weak-sustained plus strong-bump recovery checks.

## v35 Direction: Any2Track Adapt-Only Reset

The teacher/distillation branch did not materially improve push recovery. The next branch returns to the core Any2Track split:

- frozen AMP base policy keeps nominal walking/running and command tracking;
- Stage 3 learns only the history/world-model dynamics embedding from frozen-base rollouts under dynamics variance;
- PPO fine-tunes only the deployable adapter/critic/noise;
- no privileged teacher, no recovery-teacher hand-designed target, no DAgger/BC replay, and no failure-reset state injection.

Operationally, v35 treats adapter PPO as a disturbance-adaptation task:

- set global task and AMP-style weights to `0.0` for adapter training;
- set recovery reward as the dominant objective;
- set push-active command/style gates to `0.0`;
- keep delayed return-to-command near zero so command tracking returns through the frozen base after recovery, not through a learned adapter locomotion policy;
- keep stable/no-push adapter residual closed with strong residual penalties and zero residual gate;
- open adapter authority only under push/recovery/tilt/contact/height disturbance signals;
- keep persistent WM replay and alternating WM/PPO updates, matching the intended AnyAdapter update order.

Promotion should be judged by fixed recovery evaluations, not training reward:

- weak sustained push should not cause slow collapse;
- short stronger bumps should not cause base contact/fall;
- after recovery, the policy should return to the frozen AMP gait without requiring constant adapter residual;
- no-push/zero-command residual must remain near zero.

### v36 Intervention Gate

v35c proved that survival-only reward is not enough if the adapter remains in a near-zero residual basin. The recovery branch must now pass an intervention gate before it is allowed to consume a full training run:

- strict survival-only reward remains required: no AMP/style reward, no command-tracking reward group, no delayed return-to-command reward, no reward-manager regularization group;
- resume from the best strict-survival checkpoint only if output-layer perturbation and fresh optimizer are used;
- recovery-active residual must become measurable early, not just gated open:
  - target early `adapter_scaled_residual_action_l2` should rise materially above the v35c `~0.0002` level;
  - if it stays near `0.001` after roughly 50-100 PPO iterations, stop and revise the activation/architecture;
- force curriculum must not promote by reward alone:
  - promote only when force-active base-contact/fall metrics are controlled;
  - demote quickly when base contact climbs;
- a branch is not a candidate unless it improves fixed weak-sustained and stronger-bump push recovery over v11/v14 anchors while preserving no-push residual closure.

### v37 Recovery-Weighted PPO Correction

If v36 fails the intervention gate, the next branch should not simply raise reward weights again. It should concentrate actor learning on recovery-active samples while keeping the strict survival objective:

- keep task/style/reg/delayed-return rewards disabled;
- weight PPO actor surrogate terms by recovery/residual gate, normalized to mean `1.0` per minibatch;
- log `recovery_ppo_sample_weight_max` so the weighting is visible in training logs;
- reduce the absolute recovery reward scale to avoid huge value losses;
- raise residual authority/exploration only enough to leave the near-zero basin;
- continue to judge progress by fixed push-recovery evaluation, not training reward.

Early continuation criteria:

- `adapter_scaled_residual_action_l2` must exceed the v36 plateau (`~0.003-0.005`) within the first 25-50 resumed iterations;
- `base_contact` must not climb past the safety band while residual opens;
- curriculum promotion is allowed only after recovery/fall metrics improve, not because mean reward increases.

### v38 Consolidation Rule

If recovery-weighted PPO opens the adapter but base contact stays in the old failed band, treat it as a consolidation problem rather than adding more exploration:

- resume from the non-zero adapter checkpoint;
- lower action std and entropy;
- do not perturb output layers again;
- keep recovery-weighted PPO active at a moderate level;
- preserve non-zero residual with a weaker activation target;
- strengthen base-contact avoidance enough to test whether contact was exploration-driven.

Pass criteria:

- base-contact must fall below the v37 `~0.07` band before increasing force curriculum;
- residual must stay non-zero but bounded;
- if contact does not fall, stop retuning scalar noise/penalty values and change the recovery action objective/representation.

v38 result:

- the adapter opened, but residual did not stay bounded;
- contact initially improved, then degraded as residual grew;
- final safety gate stopped at iteration `1410` with `base_contact=0.1262`;
- force curriculum remained at `alpha=0.1000`;
- v38 therefore failed as a recovery policy and should not be promoted from the final checkpoint.

Immediate next step:

- fixed-evaluate early v38 checkpoints (`model_850.pt`, `model_1000.pt`, `model_1100.pt`) against sustained push before choosing a retained candidate;
- if fixed evaluation does not show a clear push-recovery gain, move to a representation/objective change instead of another scalar retune.

Fixed evaluation result:

- `model_850.pt` and `model_1100.pt` both fail zero-command sustained push:
  - `100N` and `120N` produce `fall=1.0000`, `base_contact=1.0000`;
  - even `50N` still fails most environments;
  - `event` mode does not materially rescue recovery;
- v38 is therefore rejected as a candidate branch.

Next branch requirement:

- keep the Any2Track frozen-base adapter structure;
- stop treating non-zero residual as success;
- make the recovery target explicitly produce coordinated support/capture behavior during push;
- preserve strict no-push residual closure;
- require fixed sustained-push improvement before any long run is allowed to continue.

## v39 Direction: Bounded Recovery Adapter

v39 changes the failure mode directly:

- keep teacher/distill/failure-reset disabled;
- resume from a non-overgrown adapter checkpoint (`v38 model_1100.pt`);
- keep small task/style/reg guardrails so the frozen AMP policy remains the locomotion prior;
- reduce push-window sample weighting and action noise;
- shorten push residual authority instead of keeping the adapter fully open for several seconds;
- add a residual band:
  - encourage non-zero residual only up to the useful range;
  - penalize residual overgrowth above the configured max norm;
- evaluate by fixed sustained push, not training reward.

Initial gates:

- reject if `base_contact` rises above `0.115` after the early relative safety window;
- reject if `scaled_residual_action_l2` grows toward v38's `0.6+` band without a fixed push improvement;
- continue only if 50N sustained-push clean rate improves materially before increasing force curriculum.

v39 result:

- residual stayed bounded during training, and curriculum rose to `alpha=0.152`;
- fixed sustained-push evaluation still failed:
  - `100N` and `120N` remained `fall=1.0000`, `base_contact=1.0000`;
  - `50N` clean rate stayed around `0.25`;
- v39 is rejected and should not be resumed.

## v40 Direction: Direct Sustained Push

The next branch targets the actual failed benchmark distribution:

- zero/near-zero command;
- sustained `35-170N` body-frame pushes for about `2.0-2.9s`;
- no force scaling curriculum in the first phase;
- no velocity-cancel reward while force is active, because continuous shove cannot always be cancelled in place;
- reward survival, upright torso, base height, support/capture stepping, and no base contact;
- keep Track Adapter structure and frozen AMP base, but allow enough adapter authority to turn a stationary gait into recovery stepping.

Promotion gate:

- evaluate early checkpoints on the same `50/100/120N` sustained push suite;
- only resume/extend if `50N clean` clearly exceeds v39 and `100N` is no longer complete failure.

Initial v40 live result:

- configuration is correctly active: `35-170N`, `2.0-2.9s`, `alpha=1.0`;
- the early live run is failing badly at iteration `1565`, with `base_contact=0.9202` and `time_out=0.0000`;
- continue only through the short safety window; if `base_contact` does not collapse quickly, do not extend this branch.

v40 final decision:

- stopped after `base_contact` worsened to `0.9464` and `time_out` remained `0.0000`;
- reject direct high-force sustained-push training without changing the recovery representation.

## v41 Direction: Recovery Command Override

v41 keeps the Track Adapter/frozen AMP structure but changes how recovery enters the frozen base gait:

- during active zero-command push recovery, replace only the policy-observation command slice with a temporary push/velocity-aligned command;
- keep the real environment command at zero, so stable standing still means zero command and no-push residual remains closed;
- allow the frozen AMP actor to produce stepping/running recovery instead of forcing adapter residuals to create the whole recovery gait from a static command;
- train on sustained pushes with a ramped `20-185N`, `1.2-3.0s` distribution rather than the v40 all-at-once `35-170N`, `2.0-2.9s` distribution.

Promotion gate:

- early training must show `RecoveryCommand/active_mean > 0` during force-active windows;
- `base_contact` must fall below v40 quickly, not hover above `0.4`;
- fixed evaluation must beat v39/v40 on zero-command sustained `50/100/120N` pushes before extending.

Bootstrap adjustment:

- first v41 launch proved the override path works but `30-185N`, `1.5-3.0s` was still too hard at startup;
- `20-145N`, `1.2-2.6s`, activation probability `0.38` was also too high for a clean bootstrap;
- current easy bootstrap run uses `20-100N`, `1.2-2.0s`, activation probability `0.28`, with recovery-command speed capped at `1.15m/s`;
- next promotion is to resume the best easy-bootstrap checkpoint into `20-145N`, then `30-185N`, only after fixed `50/100/120N` improves.

v41 easy result:

- stopped by safety gate at iteration `1652`;
- `time_out` rose to about `0.427`, but `base_contact` also rose to about `0.573`;
- reject v41 easy as a contact-accepting solution.

## v42 Direction: Terminal Contact Bootstrap

v42 keeps the recovery-command override and adds direct terminal shaping:

- apply a runner-level negative reward on `base_contact` terminations;
- increase recovery base-contact penalty;
- restart from an easier physical-force bootstrap (`12-75N`, `0.75-1.35s`, probability `0.22`);
- promote only if base-contact stays below `0.25` while timeout/survival improves.

Implementation correction:

- terminal penalty must use the current `termination_manager.terminated` signal for `base_contact`;
- do not use `termination_manager.get_term("base_contact")` for step reward shaping, because it reflects episodic termination stats and can repeatedly penalize after contact.

v42 fixed result:

- fixed terminal penalty stabilized the run and improved survival to about `time_out=0.70`;
- base contact plateaued around `0.27-0.30`, which is improved but still too high;
- next correction is v43: resume v42 `model_1625.pt`, increase one-shot base-contact terminal penalty to `-3200`, and reduce positive survival reward scale.

v43 early status:

- tmux session `ta_v43_terminal_hard` is running from v42 `model_1625.pt`;
- early base contact rose to about `0.20`, below the temporary `0.42` gate but still trending upward;
- value loss is high because of the `-3200` one-shot terminal penalty, so this run must be monitored for critic instability;
- if base contact crosses the gate or value loss diverges, reduce penalty magnitude and/or lower force exposure rather than accepting contact-heavy recovery.

### Any2Track OCR Gap Review

The 2026-05-20 reread of `docs/output-adapter/any2track.md` says the current failures are not yet proof of an inherent adapter limit.  The core Track Adapter mechanics are present: frozen base actor, layer-wise zero-initialized adapter, `H=79/N=20` history/world-model setup, state-action history, persistent WM replay, and replay-buffer WM updates before PPO.

Project objective clarification: the goal is not Any2Track paper reproduction.  The goal is to keep the human-like AMP locomotion base and add robustness so the robot does not fall when it collides with something, is pushed, or is involved in robot-to-robot contact.  Any2Track is useful as an adapter concept, not as the final target specification.

The remaining gap is mostly a mismatch between the paper's assumptions and our harder deployment target:

- Any2Track's base `AnyTracker` is a full-body motion tracker trained from diverse, dynamic, contact-rich reference motions; our frozen base is an AMP velocity locomotion policy and may not contain a clean zero-command capture-step/recovery repertoire for the adapter to expose.
- The paper fine-tunes the adapter with the same tracking rewards under dynamics variance.  Recent v35+ branches deliberately removed task/style/tracking and used survival-only rewards, which is useful as a stress test but not paper-faithful.
- Table III's external disturbance is specified by interval and velocity magnitude, closer to root-velocity impulse/domain perturbation than our slow sustained finite-Newton trunk push.  Human/robot contact over seconds is a different and harder recovery problem.
- The paper trains disturbance adaptation across terrain, floor friction, DoF friction, armature, torso CoM/mass, default-pose jitter, and external disturbances together.  Our later loops focused mostly on force pushes.
- The paper does not depend on explicit deployment push-event gates.  Our residual/recovery gates can under-open under slow real pushes or over-intervene and produce toe-only bracing.
- Zero-command slow push has no explicit reference motion in our setup; asking the adapter to invent a capture step from scalar survival reward is under-specified compared with tracking a reference motion through dynamics variance.

Before calling this an adapter architecture limit, run one paper-faithful sanity branch only as a diagnostic, not as the final goal: frozen base plus layer-wise adapter, tracking/style rewards enabled as in the base task, full dynamics randomization including root-velocity disturbances, and fixed eval on paper-like perturbations.  If that works but slow sustained push still fails, the missing piece is likely a base/recovery reference or command mechanism for capture stepping, not more scalar survival reward.

The production direction should prioritize AMP-base robustness:

- keep the frozen AMP base as the nominal human-like locomotion source;
- make adapter authority conditional on physical instability, contact, and history-inferred disturbance, not on reproducing Any2Track's exact disturbance setup;
- train/evaluate against mixed disturbances that match deployment: short impacts, slow sustained pushes, lateral contact, frontal/back pushes, no-push walking, and zero-command stand;
- require recovery without corrupting nominal gait, especially no-push velocity tracking and zero-command stillness;
- add reference/command support for capture stepping if scalar rewards and residual gates cannot produce a useful support step from the frozen AMP base.

The detailed next-branch design is now tracked in `docs/amp_recovery_adapter_controller_design.md`.  The key change is to treat the adapter as a phase-aware recovery controller with `nominal`, `recovery`, and `return` modes, instead of only tuning global force curriculum, residual scale, or scalar survival reward.

### v44 Deployment Result

v44 implemented the recovery-controller direction and was exported/deployed through `booster_k1_locomotion` as `track_adapter_v44_model_2475.onnx`.  Isaac fixed eval did not promote it because slow sustained push and walking push still failed, but real-robot deployment showed a meaningful qualitative improvement over v11/v14: standing push behavior was better and no longer looked like the same toe-only bracing failure.

Do not treat v44 as final.  The current deploy issue is light actuator oscillation:

- at policy startup;
- when the feet/legs drift too far toward the body centerline.

Next branch priority should therefore be deploy-aligned stabilization rather than only stronger force curriculum:

- startup ramp for residual gate and recovery command;
- stance/cross-leg posture guard in recovery score/gate;
- sim initial-state tests with narrow stance or inward foot placement;
- deploy-aligned controller inputs based on IMU/proprioception, with InEKF velocity added only if it is reliable on hardware.

### Optional Post-Recovery Blending Fine-Tune

"Adapter fine-tune" in the Any2Track sense is already the frozen-base adapter PPO stage. A separate second fine-tune is only optional and should happen after recovery-only learning passes fixed push-recovery gates.

If used, it should be a short resume from the best recovery-only adapter:

- keep frozen AMP fixed;
- keep no-push/stable residual closed;
- keep recovery reward dominant;
- re-enable only small task/style weights after stable recovery, not during active push;
- restore command tracking through the frozen base gait, not by letting the adapter become a second locomotion policy;
- stop immediately if weak/strong push recovery regresses or stable residual grows.

Do not run this blending stage before the recovery-only adapter demonstrates better fixed push-recovery than the current anchors.

### Stage 3 Reuse Rule

Stage 3 world-model pretrain may be reused when all compatibility metadata match:

- same frozen base checkpoint;
- same observation/history layout;
- `H=79`, `N=20`;
- `source=psi_s`, `target_dim=81`;
- `collection_policy=base_residual_zero`;
- finite `wm` and positive `wm_valid_fraction`.

The existing v23 pretrain `model_799.pt` satisfies those checks for v35/v35b. Its disturbance distribution is milder than v35b's full `12-620N` mix, but PPO-time persistent WM replay updates the world model on current v35b rollouts, so reuse is acceptable when training time matters.

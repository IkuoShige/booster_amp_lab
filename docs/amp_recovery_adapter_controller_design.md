# AMP Recovery Adapter Controller Design

Date: 2026-05-20

## Purpose

This project is not trying to reproduce Any2Track for its own sake.

The goal is to keep the frozen AMP locomotion policy as the nominal human-like walking/running source, then add a deployable adapter that prevents falls under real disturbances:

- collision with objects;
- pushes from humans or other robots;
- short impacts;
- slow sustained contact;
- lateral, frontal, and rear disturbances;
- recovery back to normal velocity tracking after the disturbance.

Any2Track is useful as a design reference because it separates a frozen base policy from a dynamics adapter.  The final target, however, is AMP-base robustness, not paper-faithful tracking reproduction.

## Design Thesis

Treat the adapter as a recovery controller, not as a small always-on correction.

Nominal behavior:

```text
command -> frozen AMP base -> human-like locomotion
adapter gate ~= 0
```

Recovery behavior:

```text
disturbance detected
  -> temporary recovery command is generated
  -> adapter gate opens
  -> adapter residual corrects posture, foot placement, and support
  -> robot survives and takes a useful support/capture step if needed
  -> command and residual decay back to nominal AMP behavior
```

The adapter should not learn a second locomotion policy for normal walking.  It should learn when and how to override the base policy during instability.

## Deployment Note: v44

After the first v44 recovery-controller deployment, the real robot moved better than the previous v11/v14 checkpoints under standing push tests.  This is the first positive deployment signal for the recovery-controller direction.

The remaining deploy issue is small, transient actuator oscillation:

- at policy startup;
- when a foot/leg is displaced too far toward the body centerline.

This suggests the next iteration should add startup and posture guards around the recovery controller instead of only increasing disturbance strength:

- ramp residual/recovery-command authority after policy start;
- detect narrow/crossed stance from joint state and suppress abrupt recovery commands;
- add fixed simulation starts from inward/narrow foot placements;
- align the deploy controller inputs with the training controller inputs.

## What Was Tried Before

### v11/v14: Track Adapter with push recovery

Implemented:

- frozen AMP base;
- Track Adapter with history/world-model;
- layer-wise adapter;
- physical push curriculum;
- recovery rewards and residual gates.

Outcome:

- v14 `model_10000.pt` remains the best fixed-evaluated Isaac aggregate checkpoint so far;
- real robot deployment showed a critical failure mode: slow manual push caused toe-only bracing instead of a useful recovery step;
- this indicates that the adapter could react, but did not become a coordinated capture-step controller.

### v35+ survival-only branches

Implemented:

- disabled most task/style/tracking rewards;
- made survival/no-contact rewards dominant;
- strengthened residual gates, terminal penalties, and recovery sampling.

Outcome:

- base contact remained high or plateaued;
- the adapter often stayed too small or entered local solutions;
- scalar survival reward alone did not reliably teach "where to put the foot".

### Teacher/distillation/failure-reset branches

Implemented:

- privileged recovery teacher variants;
- capture-step residual priors;
- attempted failure-reset replay.

Outcome:

- the teacher did not become a strong recovery policy;
- some runs were unstable or stalled;
- the approach did not produce a robust deployable checkpoint.

### v41/v42 recovery command override

Implemented:

- during active recovery, replace the policy-observation command slice with a temporary push/velocity-aligned command;
- keep the real environment command and reward command unchanged;
- allow the frozen AMP base to enter a stepping gait during zero-command disturbance.

Outcome:

- useful idea, but still incomplete;
- it is a command override hook, not a full recovery state machine;
- it does not yet solve detection, hold/decay timing, phase-specific rewards, and deploy-like evaluation as one system.

## Current Missing Piece

The missing capability is not just "stronger force curriculum".

The missing capability is:

```text
detect physical instability
  -> choose a useful recovery direction
  -> temporarily command/allow a support step
  -> use residual action to shape the step and body posture
  -> return to AMP gait after stability is recovered
```

Without this structure, the policy can converge to toe bracing, joint stiffening, or tiny residuals because those are easier local optima than coordinated stepping.

## Proposed Controller Structure

### 1. Nominal Mode

Active when the robot is stable and no disturbance is inferred.

Behavior:

- policy observation uses the user command unchanged;
- adapter residual gate is closed or near zero;
- stable zero-command should remain still;
- velocity tracking and AMP style remain the responsibility of the frozen base;
- residual penalties are strong.

Nominal metrics:

- no-push velocity error;
- zero-command drift;
- adapter residual norm;
- base contact/fall rate;
- gait/style degradation.

### 2. Recovery Mode

Active when instability is detected.

Possible triggers:

- large body-frame base velocity error;
- large projected gravity XY tilt;
- roll/pitch angular velocity;
- base height drop;
- unexpected trunk/base contact;
- foot slip or abnormal foot contact;
- history/world-model prediction error;
- training-only external force/push labels;
- deploy-time heuristic disturbance score from proprioception history.

Important rule:

Training may use privileged force labels for reward analysis and curriculum, but deployed inference must not require privileged force input.  The controller must work from observation/history and runtime proprioception.

Recovery outputs:

- temporary recovery command for the frozen AMP base;
- high adapter residual gate;
- lower residual penalty during recovery;
- recovery-specific reward weights.

### 3. Return Mode

Active after the body is upright and velocity/height have recovered.

Behavior:

- decay recovery command back to the user command;
- decay residual gate;
- restore velocity tracking;
- prevent oscillation or repeated over-correction;
- keep AMP gait from being permanently corrupted.

Return metrics:

- time to stable;
- post-push command tracking error;
- residual decay time;
- no renewed fall/contact within the return window.

## Recovery Command

The recovery command should be a temporary command visible to the frozen AMP base through the policy observation.  It should not permanently change the user's command.

Current v41 already does the simplest form:

```text
recovery_direction = push_direction_gain * push_delta_xy
                   + velocity_direction_gain * base_velocity_xy

recovery_speed = clamp(push_speed + velocity_speed, min_speed, max_speed)
policy_obs.command_xy = blend(user_command_xy, recovery_direction * recovery_speed)
```

The next version should make this more principled and phase-aware.

Recommended body-frame recovery direction:

```text
capture_signal_xy =
    k_vel  * base_lin_vel_xy
  + k_tilt * projected_gravity_xy
  + k_ang  * roll_pitch_ang_vel_projected
  + k_wm   * world_model_prediction_error_xy
  + k_push * privileged_push_or_force_xy   # training/curriculum only
```

Then:

```text
dir_xy = normalize(capture_signal_xy)
speed  = clamp(k_norm * ||capture_signal_xy||, min_recovery_speed, max_recovery_speed)
cmd_xy = dir_xy * speed
cmd_yaw = clamp(-k_yaw * base_ang_vel_z, -max_yaw, max_yaw)
```

Initial recommended ranges:

- `min_recovery_speed`: `0.20-0.35 m/s`;
- `max_recovery_speed`: `0.8-1.2 m/s` for the first branch;
- `max_abs_x`: `0.8-1.2 m/s`;
- `max_abs_y`: `0.6-1.0 m/s`;
- `max_yaw`: `0.0-0.6 rad/s` initially, keep yaw small until XY recovery works;
- ramp-in: `0.10-0.25 s`;
- hold: at least `0.40-1.20 s` after active disturbance;
- decay: `0.50-1.50 s` after stability returns.

Do not make this command always active.  It must be gated by recovery mode.

## Adapter Residual Control

The residual should complement the recovery command.  It should not carry normal locomotion.

Nominal:

- gate: `0.0-0.02`;
- strong residual penalty;
- zero-command residual should be nearly zero.

Recovery:

- gate: `0.6-1.0`;
- residual scale: start around `0.16-0.24`, only raise if residual is too weak;
- recovery residual penalty much lower than nominal;
- cap residual overgrowth to avoid destroying AMP gait.

Return:

- gate decays with stability score;
- residual penalty increases again;
- command tracking and AMP style are restored.

Key logs:

- `RecoveryMode/active_mean`;
- `RecoveryMode/phase_nominal_mean`;
- `RecoveryMode/phase_recovery_mean`;
- `RecoveryMode/phase_return_mean`;
- `RecoveryCommand/x_mean`;
- `RecoveryCommand/y_mean`;
- `RecoveryCommand/speed_mean`;
- `Gate/recovery_residual_mean`;
- `adapter_scaled_residual_action_l2`;
- `adapter_ungated_residual_action_l2`;
- `post_recovery_residual_decay_s`.

## Reward Design

Use phase-aware rewards instead of one global reward mixture.

### Nominal reward

Purpose: keep the frozen AMP behavior intact.

Terms:

- velocity tracking for the user command;
- AMP/style reward if available and stable;
- zero-command stillness;
- no base contact;
- residual penalty;
- smooth action/no excessive action delta.

### Recovery reward

Purpose: survive and recover support.

Terms:

- no fall;
- no trunk/base contact;
- upright torso;
- base height;
- roll/pitch angular velocity damping;
- support polygon or capture-point improvement;
- foot clearance during step;
- new useful foot contact in the recovery direction;
- avoid toe-only bracing if measurable;
- temporary tracking toward recovery command, not the original command.

During early recovery, original velocity tracking should be weak or disabled.  The robot should first avoid falling, then return to command.

### Return reward

Purpose: recover normal locomotion.

Terms:

- return to original user command;
- residual decay;
- style/gait restoration;
- no secondary fall;
- no oscillatory command switching.

## Disturbance Curriculum

The curriculum should match deployment, not only high-N short kicks.

Use a mixture:

1. No-push nominal episodes
   - protects gait and zero-command stillness.

2. Weak sustained pushes
   - `10-80N`;
   - `1.0-4.0s`;
   - front/back/lateral;
   - important for human slow push and robot-to-robot leaning/contact.

3. Medium sustained pushes
   - `80-160N`;
   - `0.5-2.0s`;
   - should require support steps.

4. Short impacts
   - `150-350N`;
   - `0.05-0.15s`;
   - used for bump/kick-like robustness.

5. Root-velocity disturbances
   - `0.1-1.0 m/s`;
   - keeps compatibility with Any2Track-style dynamics adaptation and helps world-model/history learning.

6. Dynamics randomization
   - floor friction;
   - mass/CoM;
   - armature/DoF friction;
   - mild terrain/contact variations;
   - sensor/action noise.

Promotion must not be based on training reward alone.  Promote only if fixed eval improves.

## Training Plan

### Phase A: Implement recovery state machine

Add per-env phase state:

- `nominal`;
- `recovery`;
- `return`.

State transitions:

```text
nominal -> recovery:
  disturbance_score > enter_threshold

recovery -> return:
  disturbance inactive AND stability_score > stable_threshold for min_stable_time

return -> nominal:
  residual gate decayed AND command blend decayed

return -> recovery:
  disturbance_score rises again
```

The state machine should be deterministic and logged.  This makes debugging possible.

### Phase B: Recovery command controller

Replace the current simple v41 command override with:

- capture-signal based direction;
- ramp/hold/decay;
- deploy-safe disturbance score;
- optional privileged training signal only as an auxiliary term;
- no command override in nominal mode.

### Phase C: Phase-aware PPO training

Use frozen AMP base and train only:

- adapter;
- critic;
- action noise if needed;
- history/world-model updates as before.

Keep:

- persistent WM replay;
- `H=79/N=20`;
- replay WM update before PPO update.

Train with:

- nominal/recovery/return reward weights;
- mixed disturbance curriculum;
- recovery sample weighting for PPO;
- controlled residual opening.

### Phase D: Return-to-AMP blend

Only after recovery gates pass:

- restore more original velocity tracking;
- restore style/AMP reward in stable and return phases;
- keep recovery phase survival-dominant;
- reject if recovery regresses.

## Evaluation Gates

A checkpoint is not a promotion candidate unless it passes all core gates.

### No-push gates

- zero command: no drift, no stepping, no base contact;
- velocity grid: acceptable tracking for `vx=0.2,0.5,1.0,1.5,2.0`;
- lateral grid: acceptable tracking for `vy=0.2,0.4,1.0`;
- yaw grid: up to `vyaw=1.5`;
- adapter residual near zero in stable no-push states.

### Slow-push gates

Zero-command:

- rear push;
- front push;
- lateral push;
- weak sustained `10-80N`;
- medium sustained `80-160N`;
- duration up to several seconds.

Success:

- no fall;
- no trunk/base contact;
- if needed, useful support step occurs;
- no toe-only bracing collapse;
- return to stillness after push ends.

### Walking-push gates

Commands:

- `vx=0.5,1.0,1.5,2.0`;
- `vy=0.4,1.0`;
- `vyaw=0.8,1.5`;
- selected mixed `vx+vyaw`, `vy+vyaw`.

Disturbances:

- front/back/lateral sustained push;
- short impact;
- random root-velocity disturbance.

Success:

- no fall;
- no base contact;
- returns to command after recovery window;
- no permanent gait corruption.

### Robustness ranking

Primary:

1. fall/base-contact rate under slow sustained push;
2. fall/base-contact rate under short impact;
3. recovery time;
4. no-push behavior preservation.

Secondary:

1. velocity tracking error after recovery;
2. residual magnitude;
3. style degradation;
4. energy/action smoothness.

## Difference From Previous Branches

Previous branches mostly changed:

- force strength;
- residual scale;
- reward weights;
- teacher/distill priors;
- terminal penalties;
- simple command override.

The new branch changes the control structure:

- explicit phase state: nominal/recovery/return;
- recovery command as a controller output, not just an ad hoc zero-command override;
- residual gate tied to phase and stability;
- phase-specific rewards;
- deploy-safe disturbance detection;
- fixed eval designed around deployment failures.

This is the key conceptual shift:

```text
Old:
  AMP action + residual, trained by mixed rewards.

New:
  AMP base is nominal locomotion.
  Recovery controller decides when to temporarily command/shape a support step.
  Adapter residual handles the part that command alone cannot solve.
```

## Implementation Checklist

- Add recovery phase state to `TrackAdapterRunner`.
- Log phase ratios and transition counts.
- Refactor `_policy_obs_with_recovery_command()` into a phase-aware recovery command controller.
- Add deploy-safe disturbance score using proprioception/history signals.
- Keep privileged push/force signals training-only.
- Add hold/decay timers per env.
- Add phase masks to reward weighting.
- Add recovery command metrics to rollout/eval logs.
- Add fixed eval for zero-command slow push, walking push, and no-push preservation.
- Export any runtime knobs needed by C++/ONNX deployment.

## Initial Branch Recommendation

Start a new branch/run family rather than extending v43.

Recommended name:

```text
v44_recovery_controller
```

Starting point:

- frozen AMP base: latest selected AMP locomotion policy;
- adapter checkpoint: compare two starts:
  - zero-init layer-wise adapter from Stage3;
  - v14 `model_10000.pt` as a warm start;
- choose the start that passes no-push behavior and opens recovery residual without toe-only bracing.

First training target:

- not `300N` immediately;
- first solve mixed `10-160N` sustained push plus `150-250N` short impact;
- only promote to stronger impacts after slow-push and no-push gates pass.

Reason:

If weak sustained push still collapses, stronger kicks are not the bottleneck.  The controller has not learned recovery stepping.

## Open Risks

- The frozen AMP base may not contain enough stepping behavior at zero command.  Recovery command should mitigate this by temporarily invoking the base locomotion repertoire.
- If residual scale is too large, nominal gait will degrade.
- If residual scale is too small, recovery remains toe bracing.
- If detection depends on privileged force, deployment will fail.
- If reward magnitude is too large, critic instability can hide poor behavior behind high reward.
- If evaluation does not include slow sustained push, sim2real failure will be missed again.

## Decision Rule

Do not continue a long run just because reward rises.

Continue only if fixed eval or monitored metrics show:

- lower base contact under non-zero disturbance;
- non-zero useful recovery residual;
- recovery command activates only during instability;
- no-push behavior remains clean;
- slow sustained push does not collapse into toe-only bracing.

If these are not true, revise the controller structure rather than increasing force or reward scale.

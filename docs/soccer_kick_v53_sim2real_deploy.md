# Soccer Kick V5.3 Sim2Real Deploy Notes

This document describes the deploy contract for:

```text
logs/rsl_rl/soccer_kick_skill_amp/2026-05-23_13-28-28_kick_skill_v53_lvdrs_critic_split_from_2800_curriculum_restore/model_5400.pt
```

The checkpoint is an `EncoderActorCritic` policy:

```text
raw actor obs:          334
ball-history slice:     obs[84:334] = 250
history encoder:        250 -> 64
actor MLP input:        84 + 64 = 148
actions:                22
control dt:             0.02 s
physics dt in training: 0.005 s, decimation 4
```

The deploy ONNX wrapper embeds the empirical observation normalizer, the
history encoder, and the actor MLP. The runtime should feed raw 334-dim actor
observations and read raw 22-dim actions.

## Export

Export the deploy ONNX graph:

```bash
cd /workspace/adapter_booster_amp_lab
source /workspace/.venv/bin/activate

python scripts/rsl_rl/export_soccer_kick_v53_onnx.py \
  --checkpoint logs/rsl_rl/soccer_kick_skill_amp/2026-05-23_13-28-28_kick_skill_v53_lvdrs_critic_split_from_2800_curriculum_restore/model_5400.pt
```

Default output:

```text
logs/rsl_rl/soccer_kick_skill_amp/2026-05-23_13-28-28_kick_skill_v53_lvdrs_critic_split_from_2800_curriculum_restore/exported/model_5400_soccer_kick_v53_deploy.onnx
logs/rsl_rl/soccer_kick_skill_amp/2026-05-23_13-28-28_kick_skill_v53_lvdrs_critic_split_from_2800_curriculum_restore/exported/model_5400_soccer_kick_v53_deploy.json
```

The `.json` file records the observation layout and export metadata.

## ONNX Interface

Input:

```text
obs: float32[batch, 334]
```

Output:

```text
actions: float32[batch, 22]
```

The ONNX graph includes:

```text
(obs - obs_mean) / (obs_std + 1e-2)
encoder(obs[84:334])
actor(concat(obs[:84], latent))
```

Do not normalize observations again outside the ONNX graph unless the model is
exported with `--no_normalizer`.

## Observation Layout

All values are float32.

```text
0:3      base_ang_vel
3:6      projected_gravity
6:28     joint_pos
28:50    joint_vel
50:72    previous_actions
72:74    ball_pos_b_xy
74:75    ball_mask
75:76    last_seen_dt
76:78    target_dir_b
78:79    target_strength_norm
79:80    is_shoot
80:84    role_one_hot
84:334   ball_history
```

`role_one_hot` for this kick skill should be:

```text
kicker = [1, 0, 0, 0]
```

`is_shoot`:

```text
shoot = 1
pass  = 0
```

`target_strength_norm` is normalized, not m/s.

For V5.3:

```text
shoot target range: 7.0 to 10.0 m/s
pass target range:  3.0 to 6.0 m/s
```

Use:

```text
norm = (target_speed_mps - range_min) / (range_max - range_min)
```

clamped to `[0, 1]`.

## Ball Perception Contract

The policy does not consume images. The real vision stack must produce:

```text
ball_pos_b_xy: ball xy in robot body-yaw frame, meters
ball_mask:     1 if current detection is valid, else 0
last_seen_dt:  seconds since last valid detection
```

Training used `hold_last_on_miss=True`. Match that behavior on hardware:

```text
if ball is detected:
  ball_pos_b_xy = measured body-yaw xy
  ball_mask = 1
  last_seen_dt = 0
else:
  ball_pos_b_xy = last valid body-yaw xy
  ball_mask = 0
  last_seen_dt += 0.02
```

Clip `ball_pos_b_xy` and history values to the training observation clip range
`[-30, 30]`.

## Ball History

`ball_history` is a 50-frame ring buffer flattened oldest-to-newest.

Each frame stores:

```text
[ball_pos_b_x, ball_pos_b_y, ball_mask, last_seen_dt, ball_speed_b]
```

At each 0.02 s control step:

```text
prev_pos = previous history slot xy
prev_mask = previous history slot mask
cur_pos = current ball_pos_b_xy
cur_mask = current ball_mask

if prev_mask == 1 and cur_mask == 1:
  ball_speed_b = norm(cur_pos - prev_pos) / 0.02
else:
  ball_speed_b = 0

ball_speed_b = min(ball_speed_b, 20.0)
append [cur_x, cur_y, cur_mask, last_seen_dt, ball_speed_b]
drop oldest slot
```

On episode/startup, zero the full history buffer until detections arrive.

## Target Direction

`target_dir_b` is the desired ball travel direction in robot body-yaw frame.

For shoot, compute the goal center or intended shot point in world/field frame,
subtract the current ball world xy, then rotate the xy unit vector into robot
body-yaw frame.

For pass, do the same with the pass target position.

The vector should be unit length:

```text
target_dir_b = [cos(theta_body), sin(theta_body)]
```

## Action Contract

The action output is the same 22-dim action used by the existing adapter deploy
stack. Keep the same:

```text
joint order
action scaling
PD gains
control dt = 0.02 s
previous action feedback
```

The ONNX output is the raw policy action. Apply the same downstream action
post-processing as the existing RSL-RL/adapter deploy path.

## Training Domain Randomization

This checkpoint was not trained with zero DR. The V5.3 run used:

Physics/material startup randomization:

```text
robot material:
  static friction  0.6 to 1.2
  dynamic friction 0.6 to 1.2
  restitution      0.0 to 0.05

ball material:
  static friction  0.5 to 1.0
  dynamic friction 0.4 to 0.9
  restitution      0.2 to 0.5

ball mass:
  scale 0.85 to 1.15

trunk mass:
  add -0.2 to +0.8 kg

trunk COM:
  x -0.04 to +0.04 m
  y -0.04 to +0.04 m
  z -0.01 to +0.01 m
```

Actuator model:

```text
DelayedImplicitActuator on main joint groups
delay sampled every reset: 2 to 8 physics steps
```

Observation corruption on policy proprio/action terms:

```text
base_ang_vel       gaussian std 0.05
projected_gravity  gaussian std 0.025
joint_pos          gaussian std 0.01
joint_vel          gaussian std 0.01
previous_actions   gaussian std 0.01
```

Virtual perception in V5.3 used the repair/easy preset:

```text
camera pitch down:             40 deg
FOV:                           105.12 deg horizontal, 94.17 deg vertical
max detection range:           8.0 m
detection probability in FOV:  0.95 to 1.0
blind episode probability:     0.0
xy noise:                      noise_a=0.02, noise_b=0.03
noise multipliers:             0.8 to 1.2
latency mean:                  0.020 to 0.060 s
latency std:                   0.006 s
detector update rate mean:     30 to 45 Hz
detector update rate std:      1.0 Hz
hold last on miss:             true
```

Command/scene distribution:

```text
shoot probability:             0.70
ball spawn distance:           0.55 to 1.45 m
ball spawn angle:              -0.65 to +0.65 rad
shoot target strength:         7.0 to 10.0 m/s
pass target strength:          3.0 to 6.0 m/s
pass target x:                 -2.0 to 6.0 m
pass target y:                 -3.5 to 3.5 m
```

Interval push was configured:

```text
base velocity impulse x/y: -0.3 to +0.3 m/s
interval: 6 s to very large
```

In practice this is light robustness DR, not a full hard-contact recovery
curriculum. The real deploy risk is more likely perception/frame mismatch,
ball/ground material mismatch, and action/PD mismatch than policy graph export.

## Safety Bring-Up

Recommended first hardware order:

```text
1. Run ONNX with logged real observations but do not apply actions.
2. Compare ONNX actions against PyTorch on the same obs log.
3. Stand in place, no ball, low gain/low action limit.
4. Static visible ball close to nominal training range.
5. Soft ball, reduced action amplitude, emergency stop active.
6. Increase target_strength_norm only after stable contact and no falls.
```

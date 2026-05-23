# Soccer AMP V3 — Multi-agent kicker + receiver design

V1 / V2 are single-agent. V3 introduces a second K1 (the **receiver**) that
shares the scene with the kicker (now also a **passer**), traps incoming
passes, and trains a separate policy concurrently.

This doc captures the architecture before code lands. Pair with
`soccer_amp_roadmap.md`.

---

## Scope

| Mode | Kicker behaviour | Receiver behaviour |
|---|---|---|
| shoot | search → approach → kick to goal mouth | (still / nothing) |
| pass  | search → approach → kick to receiver at speed matching distance | trap the incoming ball |

Per episode, `is_shoot ∈ {0, 1}` is sampled (50/50) and conditions both
policies via their observation. The pass distance and angle to the receiver
are randomized per episode within the field.

## Scene

- 1 × ground plane (14 m × 9 m)
- 1 × goal posts (2.6 × 1.2 m, kinematic) — only used in shoot mode
- 1 × soccer ball (RigidObject, m=0.43, R=0.11)
- 1 × `robot` K1 (Articulation, full kinematics, full actuators) — kicker
- 1 × `receiver` K1 (Articulation, full kinematics, full actuators)

env_spacing 16 m (we already use this in V1) accommodates both robots and the
field with margin.

## Per-step quantities maintained by `SoccerKickTrapCommand` (new)

(in addition to V1 fields)

* `receiver_pos_w: (N, 3)` — receiver root position
* `receiver_yaw_quat: (N, 4)` — receiver yaw frame
* `pass_target_pos_w: (N, 2)` — pass landing target (≈ receiver feet)
* `ball_to_receiver: (N, 3)` — vec, world frame, used by trap reward
* `ball_pos_b_receiver: (N, 3)` — ball in receiver's body-yaw frame
* `ball_vel_b_receiver: (N, 3)` — ditto
* `trap_success_awarded: (N,)` — latch
* `kicker_perception` and `receiver_perception` — two `VirtualPerception`
  modules with different mounted-body indices (`Head_2` on each robot)

## Action / observation layout

Two ActionsCfg terms:
```python
joint_pos_kicker   = mdp.JointPositionActionCfg(asset_name="robot",    joint_names=[".*"], ...)
joint_pos_receiver = mdp.JointPositionActionCfg(asset_name="receiver", joint_names=[".*"], ...)
```
Combined action dim per env: 44.

Observation groups:
```
policy_kicker     (22-DOF proprio + perceived ball + target_dir_b + strength + is_shoot)
policy_receiver   (22-DOF proprio + perceived ball + ball_vel_b + kicker_dir_b + is_shoot)
critic_kicker     (privileged + GT ball / receiver state + kick latches)
critic_receiver   (privileged + GT ball state + ball_to_receiver vec + trap latch)
amp_observations  (shared discriminator obs — both robots feed in)
```

## Training architecture

Two separate actor/critic networks; one shared AMP discriminator; one shared
rollout step over both agents.

```
                                            ┌── kicker_actor   ────→ 22-dim action
shared scene (4096 envs, 2 K1, 1 ball) ─→ → ┤
                                            └── receiver_actor ────→ 22-dim action

                                            ┌── kicker_critic_goal / aux
                                          ──┤
                                            └── receiver_critic_goal / aux

                                            ┌── AMP discriminator (shared)
                                          ──┘
```

In the runner each control step:
1. Read both obs groups.
2. Call `kicker_actor.act(obs_kicker)` and `receiver_actor.act(obs_receiver)`.
3. Stack into a 44-dim action vector → `env.step(action)`.
4. Pull per-term rewards and AMP rewards for both agents.
5. Store transitions into TWO rollouts (one per agent), each multi-critic.
6. Update each actor / critic separately. AMP discriminator is updated once,
   on the union of (kicker, receiver) trajectories.

Conceptually: agent dimension is doubled. With 4096 envs we get 8192
rollouts per step from the policy-learning perspective.

## Reward decomposition

| Term | Group | Kicker weight | Receiver weight |
|---|---|---:|---:|
| `target_progress` (ball→target) | goal | 8 | 0 |
| `ball_approach_kicker` | goal | 4 | 0 |
| `kick_contact` | goal | 2 | 0 |
| `kick_success` | goal | 10 | 0 |
| `kick_strength_error` (pass) | goal | -1 | 0 |
| `kick_angle_error` | goal | -1 | 0 |
| `goal_scored` (shoot) | goal | 30 | 0 |
| `pass_landing` (pass) | goal | 30 | 0 |
| `ball_at_receiver_feet` (pass) | goal | 0 | +20 |
| `trap_success` (pass) | goal | 0 | +30 |
| `intercept_alignment` (face ball) | goal | 0 | +5 |
| `look_at_ball` | aux | -2 each | -2 each |
| `pelvis_orientation` | aux | -1 | -1 |
| `feet_proximity` | aux | -1 | -1 |
| `alive`/`terminated` | aux | +5 / -200 | +5 / -200 |
| `arm_deviation_l2` | aux | -100 | -100 |
| AMP style | aux | 0.3 | 0.3 |

`pass_landing` fires once when the ball xy crosses through the
`pass_target_pos_w` cylinder (radius 0.5 m) with the right speed. The
`trap_success` is the receiver's mirror: ball within 0.25 m of a receiver
foot AND ball_xy_speed < 0.3 m/s.

## Files we will add

```
robots/k1/soccer_kick_trap_amp/
├── __init__.py                # gym.register Booster-Soccer-KickTrap-AMP-v0
├── tracking_env_cfg.py        # SoccerKickTrapEnvCfg with 2 robots
├── env_cfg.py                 # FlatSoccerKickTrapEnvCfg(K1)
└── ppo_cfg.py                 # Two-agent multi-critic AMP PPO

mdp/
└── soccer_trap_commands.py    # SoccerKickTrapCommand extends SoccerKickCommand
   soccer_trap_observations.py # receiver_* obs helpers
   soccer_trap_rewards.py      # ball_at_feet, trap_success, etc.

rsl_rl/
└── runners/two_agent_amp_runner.py  # Two policies, shared discriminator
```

## Open questions

- [ ] **Receiver AMP corpus.** Reuse the same K1 motions (walk + kick + getup)?
  Or add an explicit "stand calmly" / "shift weight" clip?
- [ ] **Pass landing target.** Should it be the receiver's foot, the receiver's
  pelvis, or a 1m² area in front of the receiver (more permissive)?
- [ ] **Shared encoder?** Cheaper but might lock both policies into the same
  representational basin. Default: separate.
- [ ] **Curriculum.** Start with kicker only (shoot), then enable pass mode,
  then enable receiver training. Each stage warm-starts from the previous.
- [ ] **Concurrent terminations.** If kicker falls, do we end the receiver's
  episode too? Probably yes (shared physics step) — easier accounting.

---

_Status: design only; implementation deferred until V1.1 (perception) and V1.2
(multi-critic) are merged._

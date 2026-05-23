# Soccer AMP — Worklog

A running log of decisions, blockers, and observations. Newest entries on top.

---

## 2026-05-21 — Kickoff & scope locked

- Read `output-sc/lvdrs.md` (LVDRS, RoboCup'25 winner) end-to-end and `/workspace/mjlab/src/mjlab/tasks/kick/` for the existing MuJoCo kick implementation.
- Mapped `adapter_booster_amp_lab` env stack: `ManagerBasedRLEnv` ← `TrackingEnvCfg` ← per-task cfg (e.g. `walk2run_amp/tracking_env_cfg.py`).
- K1 has 22 DOF, joints ordered (CSV col 8–29): head (yaw, pitch), arms (L pitch/roll/elb pitch/elb yaw, R same), legs (L hip pitch/roll/yaw, knee, ankle pitch/roll, R same).
- Head pitch range: −20°…+49° (down). The robot can look at its own feet at +49°.
- AMP corpus already contains `kick.csv` (4585 frames @ 30 Hz). Walk/run/getup also present.
- Decided architecture (full detail in `soccer_amp_roadmap.md`):
  - Single env, 2 K1s (V3); start with 1 K1 (V1).
  - Multi-critic AMP PPO with `w_goal=2, w_aux=1`.
  - Virtual perception (no rendered RGB) on Head_2: FOV 105×94°, latency ~116 ms, distance-noise σ=0.05d+0.08.
  - Mode flag in obs: shoot vs pass.
  - Field 9×14 m, goal 2.6×1.2 m.

### Next up (M1)
- Build the soccer scene: ball + goal + field; spawn 2 K1 articulations (kicker + receiver).
- Wire scene under a new `soccer_kick_amp/` task module.

---

## 2026-05-21 (late) — V1.0 baseline smoke-test passes

- Implemented scene assets (`assets/objects/soccer.py`) — ball (R=0.11 m, m=0.43 kg), goal posts (2.6×1.2 m kinematic cylinders), field-bound constants.
- Implemented `mdp/soccer_commands.SoccerKickCommand` — pure-tensor command term that spawns robot + ball, samples target dir/strength, caches body-yaw-frame state, and latches kick-contact / kick-success / goal-scored.
- Implemented `mdp/soccer_observations.py` — actor (GT + mask=1 placeholder) and critic (full GT) obs.
- Implemented `mdp/soccer_rewards.py` — task rewards (target_progress, ball_approach, kick_contact/success, goal_scored, kick_angle/strength_error) + posture/alignment + standard regularizers, all in a single critic for V1.0.
- Implemented `mdp/soccer_terminations.py` — fall_height, fall_tilt, ball_out_of_field, goal_scored_done.
- Built `robots/k1/soccer_kick_amp/{__init__.py, tracking_env_cfg.py, env_cfg.py, ppo_cfg.py}` — registers `Booster-Soccer-Kick-AMP-v0`.
- Re-installed editable `rsl_rl-lib` and `booster_rl_tasks` to point at `adapter_booster_amp_lab` (state drift had pointed them at `forked_booster_amp_lab`).

### Bugs fixed during smoke
- `class_type=MISSING` in `SoccerKickCommandCfg` failed configclass validator → defined `SoccerKickCommand` first, then `class_type: type = SoccerKickCommand` directly.
- Goal post prim paths `{ENV_REGEX_NS}/Goal/...` need the `/Goal` parent → flattened to siblings `/GoalPostLeft`, `/GoalPostRight`, `/GoalCrossbar`.

### Smoke result (8 envs, 1 iteration, headless)
- Scene creation 2.23 s, iter time 1.23 s.
- All reward terms compute non-NaN values; metrics flow through.
- `ball_out`: 12 % per-step termination (expected for random reset).
- `kick_contact_rate`: 0 — no kicks yet on a random policy (expected).

### V1.0 baseline training launched
- `python scripts/rsl_rl/train.py --task Booster-Soccer-Kick-AMP-v0 --num_envs 4096 --max_iterations 4000 --headless` in tmux `soccer:v1_kicker`.
- Log: `/workspace/adapter_booster_amp_lab/logs/v1_kicker_train.log`.

### Next up (parallel)
- **M2** Virtual perception (head camera with FOV / latency / noise / dropout) → V1.1.
- **M9** Multi-critic AMP-PPO (goal vs aux critic split with weighted advantage) → V1.2.
- **V2** Shoot/pass mode flag distinction in command + obs once V1.0 baseline shows life.

### Observations after ~5 min of V1.0 training (4096 envs, 12 iter)
- Mean reward: 32 → 48 — learning is non-trivial.
- `ball_approach`: 0.94 (positive, robot moves toward ball).
- `pre_kick_yaw_align`: −0.59 — robot isn't physically rotating to face the kick target.
- `head_yaw_align`: −0.10 — head tracking the ball but not perfectly.
- `alive`: 0.45 / step (~75 % of steps alive). Timeouts dominate.
- `kick_contact_rate`: 0; `peak_kick_speed`: 0 — robot never touches the ball with enough impulse.

**Hypotheses for missing kicks:**
1. Kick-detection threshold (`ball_xy_speed > 1.5 m/s` after foot-ball distance `< 0.20 m`) is too strict — with a young policy the ball barely moves.
2. `ball_approach` is gated to zero after kick (good) but provides no incentive past ~1 m distance; the robot plateaus.
3. No reward for "foot near ball" before contact — only `support_foot_proximity` rewards the *non-kicking* foot.

**V1.1 reward-shaping plan:**
- Add `foot_ball_proximity` (weight ~+1.5, σ=0.3): `pre_kick · exp(-min_foot_ball_d² / σ²)`. Replaces the early-stage gap.
- Lower `kick_ball_speed_thresh` 1.5 → 1.0.
- Lower `kick_success_speed_thresh` 2.0 → 1.5.
- Bump `kick_contact_bonus` weight 2 → 4.

(All applied after the perception integration lands so we don't compound changes.)

### V3 design captured
- See `docs/soccer_amp_v3_design.md` — full architecture for the 2-K1 multi-agent setup (kicker + receiver), shared AMP discriminator, two actor/critics, role-conditioned obs. Implementation deferred until V1.1 / V1.2 land.

---

## 2026-05-21 (V1.1 + V3 scaffold)

### V1.1 launched
- Killed V1.0 training (~7 min, 0.5 % kick contact, plateaued).
- Added `foot_ball_proximity` reward (weight +3, σ=0.35) gated on `~kick_contact_awarded` — the missing early-stage incentive for first-touch.
- Lowered `kick_ball_speed_thresh` 1.5 → 1.0 and `kick_success_speed_thresh` 2.0 → 1.5 so a young policy's light taps still register as kicks.
- Bumped `kick_contact_bonus` weight 2 → 4.
- VirtualPerception is now attached to the kicker via `CommandsCfg.soccer_kick.perception`. Episodes are now noisy & 10 % blind.
- Smoke passed; full run in tmux `soccer:v1_1`, 4096 envs × 8000 iter, ETA ~2.8 h. Log: `logs/v1_1_perception_train.log`.

### V3 scaffolding landed
- New task `Booster-Soccer-KickTrap-AMP-v0` registered at `robots/k1/soccer_kick_trap_amp/`.
- Scene contains two K1 articulations: `robot` (kicker, controllable) and `receiver` (sibling K1 at `{ENV_REGEX_NS}/Receiver`, passive — no ActionsCfg term).
- Receiver stays upright via stiffness + a reset event (`mdp.soccer_trap_events.reset_receiver_to_default`) that writes default joint state and a randomized root pose each reset.
- Smoke passed (1 iter, 57 steps/s, no errors). Actor obs 80→22 (kicker only), critic 92→1.
- Implementation deferred for receiver policy / trap rewards.

### Multi-critic AMP-PPO landed
- New classes (no edits to existing): `MultiCriticActorCritic`, `MultiCriticRolloutStorage`, `MultiCriticAMPPPO`, `MultiCriticAmpOnPolicyRunner` under `rsl_rl/`.
- Reward partition mapping `RewardsCfgMultiCritic.CRITIC_GROUPS = {...}`. Runner reads it, uses `scatter_add_` to build per-group reward tensors, and folds AMP style reward into "aux".
- Default group weights `w_goal=2.0`, `w_aux=1.0` per LVDRS §3.2.
- New task `Booster-Soccer-Kick-AMP-MC-v0` (env inherits from V1.0/V1.1 cfg, so it picks up `foot_ball_proximity` + lower thresholds + perception cfg automatically).
- All three smoke tests PASS: `Booster-Soccer-Kick-AMP-v0`, `Booster-Soccer-Kick-AMP-MC-v0`, `Booster-Walk2Run-AMP-v0` (the last validates `AMPPPO` path is untouched).

### V1.2 launched
- Killed V1.1 (~12 min) and replaced with V1.2 (`Booster-Soccer-Kick-AMP-MC-v0`).
- 4096 envs × 8000 iter in tmux `soccer:v1_2`, log `logs/v1_2_multicritic_train.log`.
- Runner reports the partition at startup:
  - goal: `target_progress`, `ball_approach`, `foot_ball_proximity`, `kick_contact`, `kick_success`, `goal_scored`, `kick_angle_error`, `kick_strength_error`
  - aux: `pre_kick_yaw_align`, `head_*_align`, `support_foot`, `feet_proximity`, `pelvis_orientation`, `alive`, `terminated`, regularizers
- TensorBoard logs `value_function_goal` and `value_function_aux` separately so we can read out per-group value flow.
- GPU usage ~7.6 GB / 32 GB — plenty of headroom for parallel V3 runs later.

### Open next steps
- V2 (shoot/pass mode): randomize `is_shoot` per episode; pass mode uses a point target inside the field instead of the goal mouth.
- V3 (receiver policy): extend the V3 scaffold (`Booster-Soccer-KickTrap-AMP-v0`) with receiver ActionsCfg + receiver obs/rewards + two-agent runner.
- Curriculum: progressive ball spawn distance once V1.2 sustains positive kick rate.
- Debug-vis markers on `SoccerKickCommand`.

---

## 2026-05-21 (late late) — V1.2 learning kicks, V2 done, V3.1 launched

### V2 — shoot/pass mode landed
- `SoccerKickCommandCfg` gained `shoot_prob`, `pass_target_x/y_range`, `pass_landing_radius`, `pass_landing_speed_window`.
- `_resample_command` now Bernoulli-samples `is_shoot`. Shoot mode aims at a sampled point in the goal mouth (with 0.8× margin from posts). Pass mode samples a uniform xy in the field.
- New `pass_landing` reward (weight +30, one-shot via `_emit_once`) fires on rising edge of `pass_landing_awarded`. `goal_scored_reward` hardened with explicit `is_shoot` AND.
- Critic obs gained privileged `pass_target_dir_b` and `pass_target_dist` (zero in shoot mode).
- Smoke tests PASS for `Booster-Soccer-Kick-AMP-v0`, `Booster-Soccer-Kick-AMP-MC-v0`, `Booster-Soccer-KickTrap-AMP-v0`.

### V3.1 — pass target follows the receiver
- `SoccerKickCommandCfg` gained `receiver_name: str | None`. When set (V3 task does `receiver_name="receiver"`), the command term per-step overrides `pass_target_pos_w` with the receiver's xy and re-derives `target_dir_w`. Shoot-mode envs are unaffected.
- New privileged obs `receiver_pos_b` in the kicker's body-yaw frame (zero when no receiver configured).
- V3 task `Booster-Soccer-KickTrap-AMP-v0` now sets `shoot_prob=0.5`, so half the episodes are passes-to-receiver.

### Training observations
- V1.2 (multi-critic, shoot-only via legacy in-memory `_is_shoot=True`) at ~10 min:
  - `kick_contact_rate`: 16.7 % (V1.0 baseline ≈ 0.5 % at the same wall-clock; ≈ 33×).
  - `target_progress`: 0.47.
  - `peak_kick_speed`: 0.83 m/s.
  - `value_function_goal` and `value_function_aux` losses tracked separately as expected.
- V3.1 (`Booster-Soccer-KickTrap-AMP-v0`) launched in parallel at 2048 envs (V1.2 uses 4096). Mixed shoot/pass with pass target = receiver world pose.

### Open next steps
- V3.2: receiver policy — needs ActionsCfg for receiver + receiver obs group + trap rewards + two-policy training. Considering "shared-policy with role flag" (forward twice per step) vs "two separate policies".
- Curriculum: progressive ball spawn distance.
- Debug-vis markers on `SoccerKickCommand`.

---

## 2026-05-21 (V1.2 learning at 28min, V3.2 active-receiver landed, all 4 tasks pass smoke)

### V1.2 progression (multi-critic, shoot-only)
| t  | kick_contact | kick_success | peak_kick_speed | goal_scored |
|---:|-------------:|-------------:|----------------:|------------:|
| 6m | 3.6 %        | 0 %          | 0.11 m/s        | 0 %         |
| 10m| 16.7 %       | ~5 %         | 0.83 m/s        | 0 %         |
| 17m| 16.6 %       | 6.1 %        | 1.22 m/s        | 0 %         |
| 28m| 18.5 %       | 9.2 %        | 1.67 m/s        | 0.6 % (term)|

The kicker is producing real kicks; ball speed crosses the 1.5 m/s success threshold ~10 % of episodes. Goals scored (ball entering 2.6 × 1.2 m goal) starting to show.

### V3.2 landed (`Booster-Soccer-KickTrap-AMP-V2-v0`)
- Two-robot env. Both `robot` (kicker) and `receiver` get `JointPositionActionCfg`. Total action dim = 44.
- Single shared actor outputs 44-dim joint actions. Multi-critic still 2 groups.
- Receiver-side obs: proprio + ball pos in receiver's body-yaw frame + kicker direction in receiver frame + receiver foot proximity.
- New trap rewards in `mdp/soccer_trap_rewards.py`:
  - `receiver_ball_at_feet` (Gaussian on min receiver-foot↔ball xy, σ=0.30, +6)
  - `receiver_face_ball` (yaw alignment penalty, −0.4)
  - `trap_success` (one-shot: receiver foot within 0.3 m of ball ∧ ball speed < 0.4 ∧ pass mode, +30)
  - `receiver_alive` (+0.5)
  - `receiver_terminated` (−200)
- Receiver fall terminations added.
- Multi-critic partition adds trap rewards to the appropriate groups.
- AMP discriminator stays kicker-only (single-robot motion prior).
- Smoke PASS on all four tasks: V1, V2 multi-critic, V3.1 passive-receiver, V3.2 active-receiver.

### Debug-vis & curriculum landed
- Three markers on `SoccerKickCommand` (target dir, goal dir, pass target).
- `ball_distance_curriculum` linearly expands ball spawn range from (0.4, 1.2) → (1.0, 3.5) over 4000 × 4096 steps.

### V3.2 training launched
- 1024 envs × 8000 iter in tmux `soccer:v3_2_full`, log `logs/v3_2_kicktrap_train.log`.
- Running in parallel with V1.2 (4096 envs). GPU 32 GB has headroom.

### Open next steps (superseded — see V4 redesign below)
- After V1.2 finishes (~2 h more), evaluate kick performance; warm-start V3.2 from its checkpoint if useful.
- V3.3: virtual perception on receiver side (currently uses GT).
- AMP-on-receiver: optional, would require a separate AMP discriminator or shared one fed dual-stream.
- Curriculum on shoot vs pass mix.

---

## 2026-05-22 — V3.3 in-place fixes + V4 redesign accepted

After viser inspection of V3.2 we identified four issues that mean V3.2
is **not** the deployable shape, even though it does learn both behaviours:

1. **Camera mount was wrong** — `(0.00124, 0.04553, -0.01582)` is mjlab MJCF
   conventions (+Y forward). Isaac Lab K1 URDF is +X forward / +Z up
   (confirmed via Head_2 inertial CoM at (0.011, -0.001, 0.081)). Fixed to
   `(0.06, 0.0, 0.10)` (forehead position) with identity orientation.
2. **FOV viz** had a translucent yellow fill — switched to wireframe only.
3. **kick.txt is a mixed-motion compilation** (getup, low-stance walk,
   high-knee, vault), not a real kick. Rebuilt the AMP corpus from
   `/workspace/pkl-walk-kick-k1/*.pkl` (10 clips, 41 s, 56-col, K1 joint
   order). Kept `kick.txt` in the corpus too — its variety still helps
   recovery and unusual postures.
4. **44-DOF shared policy can't be deployed per-robot.** This is the
   structural problem that triggered the V4 redesign.

Also: V3.3 search rewards landed (search_yaw_velocity, last_seen_dt_penalty,
head_yaw_search; ball_mask gating on approach rewards), and a viser HUD
patch now shows SHOOT/PASS, ball-detected, last-seen-dt and the latches.

### V4 design accepted
- See `docs/soccer_amp_skill_library_design.md` for the full spec.
- **Move to a skill-library architecture**: separate kick / trap / defend
  skills, single-agent each, per-agent shared policy with a `role` one-hot.
- **Shoot mode** stays multi-attempt (time-weighted goal bonus + first-kick
  bonus); **pass mode** becomes single-shot (multi-kick penalty + landing
  bonus). Mode is in obs; same policy.
- **Defender** added: block / steal / lane-hold rewards; ball spawned moving
  toward own goal.
- **Temporal encoder** in two steps: Step A (history stacking, 50 extra
  obs dims) first; Step B (LVDRS-style encoder + decoder + aux loss) only
  if Step A trap performance plateaus below 20 %.
- V1.2 and V3.2 checkpoints preserved for ablation comparison.

Implementation order: Stage 1 (Kick Skill) → Stage 2 (Trap Skill) →
Stage 3 (Defend Skill) → Stage 4 (composition / pseudo self-play) →
Stage 5 (encoder upgrade, conditional).

---

## 2026-05-22 — V1.2 and V3.2 trainings complete

### V1.2 final (`Booster-Soccer-Kick-AMP-MC-v0`, 4096 envs × 8000 iter, 3h27m wall-clock)
- Checkpoint: `logs/rsl_rl/soccer_kick_amp_mc/2026-05-21_20-42-35/model_7999.pt` (16.9 MB).
- `kick_contact_rate` ≈ 69 %, `kick_success_rate` ≈ 50 %, `peak_kick_speed` ≈ 5.3 m/s, `goal_scored_done` ≈ 2.3 %.
- Pure shoot-mode (the running process loaded `SoccerKickCommand` before V2 added `shoot_prob`, so `_is_shoot` stayed True for every env).

### V3.2 final (`Booster-Soccer-KickTrap-AMP-V2-v0`, 2048 envs × 8000 iter, 3h38m wall-clock)
- Checkpoint: `logs/rsl_rl/soccer_kick_trap_amp_v2/2026-05-22_00-15-12/model_7999.pt` (18.8 MB).
- Shared 44-DOF policy across both K1 robots (kicker + receiver), multi-critic AMP-PPO, `shoot_prob=0.5`.
- `kick_contact_rate` ≈ 55 %, `kick_success_rate` ≈ 40 %, `peak_kick_speed` ≈ 3.87 m/s.
- `trap_success_rate` ≈ 8.6 %, `receiver_ball_at_feet` ≈ 0.086 (positive shaping flow).
- `goal_scored_done` ≈ 4.4 % (better than V1.2 — gives the policy two ways to "succeed").
- `receiver_fall_height` ≈ 8 % (receiver still falls sometimes; trap policy needs more polish).
- Curriculum reached `ball_distance` ≈ 0.82 (mid-expansion; given another 4 k iter it would saturate at (1.0, 3.5)).

### Final task list
1. `Booster-Soccer-Kick-AMP-v0` — V1.0 / V1.1 single-critic (deprecated, kept as smoke target).
2. `Booster-Soccer-Kick-AMP-MC-v0` — V1.2 multi-critic, shoot-only kicker. **Best kicker policy.**
3. `Booster-Soccer-KickTrap-AMP-v0` — V3.1 kicker passes to *passive* receiver (no receiver policy).
4. `Booster-Soccer-KickTrap-AMP-V2-v0` — V3.2 kicker + active receiver, single shared policy. **Both behaviors emerge.**

### Lessons learned
- The biggest reward-shaping win was `foot_ball_proximity` (V1.1) — without it the kicker never got close enough to actually contact the ball.
- Multi-critic (V1.2 onward) gave noticeably more stable updates than the single critic; `value_function_goal` and `value_function_aux` losses are separable as designed.
- A *single shared* policy for two K1s works (V3.2), but the 44-dim action space and double DOF doubles training wall-clock per env. Two separate policies (Option B in the V3 design doc) might converge faster — left as future work.
- Receiver fall rate (8 %) suggests the trap behavior is being learned in a high-variance way; stronger pose/posture regularizers on the receiver side would help.

### Suggested follow-ups
- Continue V3.2 for another 4 k iter (or warm-restart with smaller `init_noise_std`) to push trap_success > 20 %.
- Add a virtual perception module to the receiver side (V3.3) so it can also operate from realistic camera data.
- Implement curriculum on `shoot_prob` (start with 0.8 shoot, slowly ramp to 0.5 to let the kicker establish before introducing pass mode).
- Implement a play/eval script that loads either checkpoint and runs an interactive episode in the Isaac Sim viewer with the debug-vis markers enabled.

---

_(Append new dated entries below this line as work progresses.)_

---

## 2026-05-22 — V4 Stage 1–3 implementation landed (Stage 1 training)

### Shared infrastructure
- **`SoccerKickCommand`** extended (backward-compatible defaults):
  - New cfg fields: `ball_history_len: int = 10`, `role: str = "kicker"`.
  - New buffer: `_ball_history_buf: (history_len, num_envs, 5)`. Each slot
    stores `[perceived_ball_x_b, _y_b, ball_mask, last_seen_dt, ball_speed_b]`.
    Updated each step inside `_update_command`; zeroed for reset envs.
  - New public properties: `ball_history`, `ball_history_len`, `role`.
  - All existing V1/V2/V3.2 task smokes still pass (verified).
- **`mdp/soccer_role_obs.py`** (new): `role_one_hot(env, command_name) → (N, 4)`
  in order `(kicker, receiver, defender, idle)`. `ball_history_flat(env, command_name)
  → (N, history_len * 5)` returns the flattened Step A temporal-information obs.
  Both functions resolve via `env.command_manager.get_term(name)` so they work
  with any command term exposing the four properties `role`, `ball_history`,
  `num_envs`, `device` (Stage 2 and 3 commands also implement these).

### Stage 1 — Kick Skill (`Booster-Soccer-Kick-Skill-v0`)
- Task module `robots/k1/soccer_kick_skill_amp/`:
  - `__init__.py` registers the task; `tracking_env_cfg.py` derives from
    `soccer_kick_amp.SoccerKickEnvCfg` and overrides
    `ObservationsCfg.PolicyCfg/PrivilegedCfg` to append `role_one_hot` (4d)
    and `ball_history_flat` (50d), and overrides `RewardsCfg` with V4
    mode-conditional shaping (see below).
  - `env_cfg.py` defines `FlatSoccerKickSkillEnvCfg` with the multi-critic
    rewards split (`CRITIC_GROUPS`) and the standard K1 arm-scale tightening.
  - `ppo_cfg.py` selects the multi-critic AMP runner, `experiment_name =
    "soccer_kick_skill_amp"`, `max_iterations = 8000`, AMP corpus matches
    soccer_kick_amp_mc.
- New mode-conditional rewards in `mdp/soccer_rewards.py`:
  | Term | Weight | Group | Notes |
  |---|---:|---|---|
  | `goal_scored_quick` | 1.0 (magnitude is 30 + 50·time inside fn) | goal | One-shot, shoot-mode only, time-weighted. |
  | `time_to_goal_penalty` | -0.02 | goal | Per-step pressure to score fast. |
  | `kick_success_first_bonus` | 1.0 (10 first / 3 repeat) | goal | First kick worth more. |
  | `kick_strength_error_shoot` | -0.5 | goal | Weak |achieved - cmd| penalty. |
  | `kick_strength_error_pass` | -2.0 | goal | Strong; passes must be precise. |
  | `multi_kick_penalty` | -10.0 | goal | Pass mode — penalize 2nd+ touches. |
  | `approach_after_kick_penalty` | -3.0 | goal | Pass mode — don't chase the ball after release. |
  | `target_progress_pass_window` | 8.0 | goal | Pass-only, first 20 steps after kick. |
  | `ball_at_target_terminal` | 20.0 | goal | Pass-only, sparse terminal bonus. |
  The legacy `goal_scored` / `pass_landing` / `kick_strength_error` weights
  are zeroed in `RewardsCfg.__post_init__` to avoid double-counting.
- **Smoke** (8 envs × 1 iter, headless): PASS. Actor obs: 134d (= legacy 80 +
  role 4 + history 50); critic obs: 149d. Multi-critic actor-critic with 2
  value heads ('goal', 'aux') reports correct partition for all V4 terms.
- **Training launched** (4096 envs × 8000 iter, tmux `soccer:kick_skill_v0`):
  - Iteration time 3.3–3.6s (slower than the 1.2s baseline because the
    pre-existing `soccer:v3_3` window is still running at 7 GB / 96 % GPU).
    Wall-clock ETA ~7.5 h instead of 3.5 h.
  - Log: `logs/kick_skill_v0_train.log`.

### Stage 2 — Trap Skill (`Booster-Soccer-Trap-Skill-v0`)
- New `SoccerTrapCommand` / `SoccerTrapCommandCfg` (`mdp/soccer_trap_commands.py`):
  single K1 + ball thrown at the robot. On reset the ball spawns 1.5–3 m away
  in a forward cone and is given a 2–5 m/s velocity pointing at the receiver
  with ±20° angular noise (height 0.11–0.5 m). Trap-success latch: min foot↔ball
  distance < 0.30 m AND ball xy-speed < 0.4 m/s. Predicted-contact distance
  tracked (used by `trap_anticipation`).
- New observations (`mdp/soccer_trap_obs.py`):
  `trap_ball_pos_b` / `trap_ball_mask` / `trap_last_seen_dt` (actor);
  `trap_ball_pos_b_gt` / `trap_ball_vel_b_gt` / `trap_predicted_contact_distance`
  (critic). `role_one_hot` and `ball_history_flat` from
  `soccer_role_obs.py` are reused unchanged.
- New rewards appended in-place to `mdp/soccer_trap_rewards.py`:
  `trap_success_skill` (+30), `receiver_ball_at_feet_skill` (+6),
  `intercept_alignment_skill` (-1.0), `trap_anticipation` (+2),
  `receiver_idle_when_blind` (+0.3), plus standalone `trap_alive` /
  `trap_terminated` so the cfg does not depend on the kicker manager.
- Task module `robots/k1/soccer_trap_skill_amp/`: scene = K1 + ball (no goal
  posts), episode_length_s = 4.0, CRITIC_GROUPS = {goal: [receiver_ball_at_feet,
  trap_anticipation, trap_success], aux: everything else}.
- **Smoke** (8 envs × 1 iter): PASS. Reward partition exactly as specified.
- **Stage 1 / V1.2 / V3.2 regression smokes**: PASS.

### Stage 3 — Defend Skill (`Booster-Soccer-Defend-Skill-v0`)
- New `SoccerDefendCommand` (`mdp/soccer_defend_commands.py`): single K1 +
  ball + own-goal posts mirrored at `x = -GOAL_LINE_X`. On reset the defender
  is placed 1–4 m in front of own-goal; the ball spawns 4–8 m away and is given
  a 2–8 m/s velocity pointing at the own-goal center (±15° noise).
  - Detects: block (non-foot body within 0.30 m of ball — one-shot),
    steal (foot within 0.25 m AND ball xy-speed dropped > 0.5 m/s vs 5 frames
    back), is_in_lane (defender lies on the ball→own-goal line strip, ±0.8 m
    perpendicular tolerance), ball_pushed_away (post-contact ball_vel · away-from-
    goal > 0), own_goal_scored (ball crosses own-goal line, terminal).
  - Block bodies resolved at runtime as "all robot bodies − feet".
- New rewards (`mdp/soccer_defend_rewards.py`): `block_success` (+30),
  `steal_success` (+25), `defensive_line_hold` (+5/step), `ball_pushed_away`
  (+10), `own_goal_proximity_penalty` (-15, terminal).
- New observations (`mdp/soccer_defend_obs.py`): perceived ball pos/mask/dt
  (actor); GT ball pos/vel + own-goal dir + lane flag + block/steal latches
  (critic). `role_one_hot` + `ball_history_flat` reused.
- New asset helper `soccer_own_goal_assets()` in `assets/objects/soccer.py`
  (mirrors the existing goal at `-GOAL_LINE_X` with a distinct color).
- New termination `own_goal_scored` in `mdp/soccer_terminations.py`.
- Task module `robots/k1/soccer_defend_skill_amp/`: scene = K1 + ball +
  own-goal-posts, episode_length_s = 5.0.
- **Smoke** (8 envs × 1 iter): PASS. Reward partition exactly as specified.
- **12-iter × 64-envs diagnostic**: `peak_ball_speed = 5.4 m/s`,
  `lane_hold_frac = 0.31` — ball is moving toward own-goal and per-step lane
  reward fires.

### Stage 4 — Composition Validation (`Booster-Soccer-Composition-v0`)
- New task module `robots/k1/soccer_composition_amp/`. Scene = 2 K1 robots
  (kicker + receiver) + ball + goal posts. Dual command terms: the existing
  `SoccerKickCommand` for the kicker (pass-only via `shoot_prob=0`, role=`kicker`,
  receiver pose tracks the receiver asset) + a new
  `PassiveTrapCommand` for the receiver that mirrors `SoccerTrapCommand`
  except it does NOT respawn the ball (the kicker drives the ball). Both
  commands maintain their own ball-history buffers and perception.
- 44-dim concatenated action (kicker + receiver); two policy obs groups
  (`policy_kicker` 134-d, `policy_receiver` 130-d) layout-compatible with the
  Stage 1 / Stage 2 checkpoints. A `policy` alias points at `policy_kicker`
  so single-policy tooling (`play.py`) still resolves.
- New runner script `scripts/rsl_rl/play_multi.py`:
  - Args: `--task`, `--checkpoint_{kicker,receiver,defender}` (each optional),
    `--num_envs`, `--num_episodes`, `--max_steps`, `--headless`.
  - Per role, builds a `MultiCriticActorCritic` from the task's agent cfg
    dims, restores `model_state_dict` (non-strict) + `obs_norm_state_dict`
    into an `EmpiricalNormalization` head, and falls back to a random
    policy when no checkpoint is provided.
  - Reads metrics directly from `SoccerKickCommand` / `PassiveTrapCommand`
    properties so the eval loop is decoupled from the reward manager.
  - Emits a JSON summary: `pass_landing_rate`, `trap_success_rate`,
    `combined_success_rate`, `kicker_fall_rate`, `receiver_fall_rate`,
    `avg_kicks_per_episode`, `avg_episode_seconds`, plus checkpoint paths
    and wall-clock.
- **Smoke** (4 envs × 2 episodes, no checkpoints): SimulationApp launches,
  env builds, episodes run, JSON summary prints with every field
  populated. Shoot-vs-defender and 3-body compositions deferred (`--checkpoint_defender`
  flag is accepted but unused) — only the pass+trap composition is
  required by the §11 acceptance criteria.

### Stage 5 — Encoder Upgrade scaffolding (opt-in via runner_class_name)
- New module `rsl_rl/rsl_rl/modules/encoder_actor_critic.py` — `EncoderActorCritic`
  subclasses `MultiCriticActorCritic`. Internally slices a configurable
  segment of the actor obs (the `ball_history` band, default 50 dims) → 2-layer
  encoder MLP → 64-d latent → reinserted in place of the raw slice before the
  actor MLP. Critic mirrors the same encoder/concat structure on its slice
  of the critic obs. The encoder caches its most recent latent on
  `self.latest_latent` so the algorithm can compute a decoder MSE without
  re-running the encoder.
- New algorithm `rsl_rl/rsl_rl/algorithms/encoder_amp_ppo.py` —
  `EncoderMultiCriticAMPPPO` subclasses `MultiCriticAMPPPO`. In `update()`,
  after the value loss is computed, runs `decoder(latent)` against a
  configurable slice of the critic-obs (`decoder_target_slice`) and adds
  `decoder_loss_coef * MSE` to the total loss. Logs `mean_decoder_loss`.
- New runner `rsl_rl/rsl_rl/runners/encoder_amp_runner.py` —
  `EncoderMultiCriticAmpRunner` reads `train_cfg["encoder_cfg"]` and
  instantiates the encoder policy + algorithm.
- New base cfg `BaseEncoderMultiCriticAMPAgentCfg` in
  `agents/rsl_rl_ppo_cfg.py` — sets `runner_class_name =
  "EncoderMultiCriticAmpRunner"` and exposes the `EncoderCfg` block.
- `scripts/rsl_rl/train.py` — dispatch branch added for the encoder runner.
- **Unit smoke**: instantiate `EncoderActorCritic(num_actor_obs=134,
  num_critic_obs=149, history_slice=(84,134), latent_dim=64,
  decoder_target_dim=9)`; verify act/evaluate/decoder forward shapes. PASS.
- **Regression smokes**: Stage 1/2/3 1-iter trainings still pass (no
  existing task opts into the encoder; the new code is dormant).

### V3.3 retraining killed; Stage 2 + Stage 3 launched in parallel
- User confirmed V4 supersedes V3 (V4 single-skill tasks contain V3's
  kicker+receiver behaviour as separate trainings); the stuck `soccer:v3_3`
  retraining of `Booster-Soccer-KickTrap-AMP-V2-v0` (1h52m,
  `kick_contact_rate = 0` throughout) was killed to free the GPU.
- Stage 2 training launched in `tmux soccer:trap_skill_v0` (4096 envs × 8000
  iter); log `logs/trap_skill_v0_train.log`.
- Stage 3 training launched in `tmux soccer:defend_skill_v0` (4096 envs ×
  8000 iter); log `logs/defend_skill_v0_train.log`.
  - First launch attempt failed with a contact-sensor "No rigid bodies
    under '/World/envs/env_0/Robot'" error, almost certainly a startup race
    when two Isaac Sim processes were spawning simultaneously. Re-launched
    once trap was fully initialized; succeeded.
- Three concurrent trainings: GPU 22.7 GB / 32 GB, 96 % utilization. Iter
  times ~5.2 s each — adding the 3rd training pushed everything from
  3.5 s/iter to ~5 s/iter (compute-bound). Wall-clock ETAs:
  - Stage 1 Kick: ~6.8 h from start (lands around 16:30 local)
  - Stage 2 Trap: ~8.2 h from start (lands around 17:30 local)
  - Stage 3 Defend: ~11.3 h from start (lands around 20:30 local)

### Open / next
- Three V4 trainings in parallel; monitor for stalls / NaNs.
- After Stage 1 finishes, evaluate kick acceptance criteria
  (kick_success_rate > 50 %, goal_scored_done > 10 %, pass_landing > 30 %).
- After Stage 2 finishes, evaluate trap_success_rate. If < 20 %, activate
  Stage 5 EncoderActorCritic + retrain Stage 2.
- After Stage 3 finishes, evaluate block + steal > 30 %, own-goal < 30 %.
- Once all three stages complete, run `scripts/rsl_rl/play_multi.py` with
  the kick + trap checkpoints to validate Stage 4 composition.

### 2026-05-22 12:49 — Kick is scoring; goal_scored_rate metric is misleading

`Metrics/soccer_kick/goal_scored_rate` reads 0.0000 throughout training because
the metric is computed as a per-step average of the `goal_awarded` latch
(which is True for ~1 step out of ~250 per episode, then reset on
termination). The real signal is in `Episode_Termination/goal_scored_done`:

| Iter | goal_scored_done | kick_success | peak_kick_speed |
|---:|---:|---:|---:|
| 1450 | 0 % | 60 % | 5.4 m/s |
| 1920 | 0 % (not logged) | 77 % | 6.0 m/s |
| 2200 | **8.77 %** | 72 % | 6.2 m/s |

So the kicker IS scoring goals (~8.8 % of episodes), and the V4 mode-conditional
`goal_scored_quick` reward is firing (0.013 / step avg). At iter 2200 / 8000
(27 %) we're already approaching the §11.1 "10 % goal_scored_done" target,
on track to exceed it.

Followup: fix the metric so `goal_scored_rate` reports per-episode rather
than per-step. Recommend:
- In `SoccerKickCommand._update_metrics`, store `goal_awarded` only for
  episodes that just scored (one-time write on the rising edge of
  `goal_awarded`) instead of a continuous mirror.
Same fix is needed for `kick_success_rate`, `trap_success_rate`,
`block_success_rate`, etc.

### 2026-05-23 02:30 — V4 COMPLETE — Stage 4 composition eval passes

**Kick V4.4 finished** at iter 13499 (= V4.1@5500 + 8000 V4.4 iters):

| Final metric | V4.4 | V4.1 baseline | §11 target |
|---|---:|---:|---:|
| `goal_scored_done` | **14.1 %** | 13.5 % | ≥ 40 % single |
| `pass_landing_rate` (EMA) | **24.2 %** | 22 % | ≥ 50 % |
| `peak_kick_speed_shoot` (lifetime EMA) | **5.92** → 10.3 m/s when kicks | 5.3 mixed | — |
| `kick_contact_rate` (EMA) | **57.4 %** | 58 % | — |
| `ball_out` (term) | **11.6 %** | 27.5 % | (lower better) |

Final checkpoint:
`logs/rsl_rl/soccer_kick_skill_amp/2026-05-22_19-42-27/model_13499.pt`

The V4.4 improvements over V4.1 baseline:
* Aim precision 2.4× better (ball_out 27.5 → 11.6 %)
* Power +18 % (peak speed 8.5 → 10 m/s)
* Pass landing +10 % relative (22 → 24 %)
* Goal score modest +0.6 pp (13.5 → 14.1 %)

The goal_done plateau at ~14 % reflects a structural limit: time_out
70 % of episodes — even with peak 10 m/s the ball doesn't reach goal
from far spawns within 8 s. Pushing higher would need (a) shoot range
curriculum to (10, 15), (b) friction reduction, or (c) tighter
spawn distribution. The user chose to accept this plateau.

**Stage 4 composition validation** (pass+trap, 64 envs × 69 completed
episodes, real checkpoints):

| Metric | Value |
|---|---:|
| `pass_landing_rate` | 29.0 % |
| `trap_success_rate` | 15.9 % |
| `combined_success_rate` | 15.9 % |
| `kicker_fall_rate` | 5.8 % |
| `receiver_fall_rate` | 33.3 % ⚠️ |
| `avg_kicks_per_episode` | 0.39 |
| `avg_episode_seconds` | 5.74 |

Combined rate 15.9 % satisfies the trap acceptance criterion in the
end-to-end multi-agent pipeline. The receiver_fall_rate 33.3 % (vs 1 %
in solo Trap task) is the main composition-side defect — the receiver
hasn't seen "ball arriving from kicker" in training, only the
direct-injection synthetic case. Composition runner
(`scripts/rsl_rl/play_multi.py`) needed a patch: filter `critic*` keys
when loading checkpoints (training-time critic obs dim 149 doesn't
match the composition env's 134) — actor-only loading is fine for
inference.

### V4 final acceptance summary

| Stage | Target | Result | Status |
|---|---|---|---|
| 1 Kick / shoot | goal ≥ 40 % single OR < 8 s multi | 14.1 % per episode (~28 % per shoot) | ⚠️ partial |
| 1 Kick / pass | landing ≥ 50 % | 24.2 % | ⚠️ partial |
| 2 Trap | success ≥ 15 % | 93.5 % | ✅ 6.2× |
| 3 Defend | prevention ≥ 70 % | 99.1 % | ✅ 1.4× |
| 4 Composition (pass+trap) | functional | 15.9 % combined | ✅ |
| 5 Encoder | activate if trap < 20 % | trap = 93 % → not needed | ⏭️ |

The skill-library deliverable is complete. Trap and Defend pass §11
criteria by large margins. Kick partial — both shoot goal_rate and pass
landing fall short of §11 numerical targets, but the policies are
functional (positive goal rate, positive landing rate, both up vs
V4.0). Composition end-to-end works at 16 % combined pass+trap success.

### 2026-05-23 00:55 — DEFEND COMPLETE — Kick V4.4 alone on GPU

**Defend** finished its 10700-iter run (= V4.0 checkpoint @2700 + 8000
V4.1 iters) at 00:54.

| Metric | Final | §11 target |
|---|---:|---:|
| block_success_rate | **55.2%** | — |
| steal_success_rate | **35.8%** | — |
| block + steal combined | **91.0%** | 30% (3× over) |
| own_goal_scored_rate | **0.92%** | — |
| **prevention rate** (1 − own_goal) | **99.08%** | 70% (1.4× over) |
| lane_hold_frac | 63% | — |

Wall-clock 10h13m total (3h to V4.0 collapse + 7h13m V4.1 retraining
from V4.0@2700). Final checkpoint:
`logs/rsl_rl/soccer_defend_skill_amp/2026-05-22_19-39-XX/model_10699.pt`
(latest run dir, post-V4.1 warmstart).

The V4.1 redesign (defensive_line_hold weight 5 → 0.3, +
ball_intercept_proximity continuous shaping, + boosted block/steal
weights) was the decisive change — V4.0 plateaued at block+steal 5% from
the lane-park attractor; V4.1 reached block+steal 91% in the same iter
budget.

**Kick V4.4 now alone on GPU.** Iter time dropped 3.87 s → **1.48 s**
(2.6× speedup). 5h elapsed, peak_kick_speed_shoot EMA 5.95 (≈ 10.3 m/s
when policy kicks), goal_done 13.6%, ball_out 11.7%. Remaining
~4000-5000 iter at 1.5 s/iter ≈ 1.5-2 h to completion.

**All non-Kick stages complete:**
- Trap: ✅ 93.5% (target ≥ 15%, 6× over)
- Defend: ✅ 99.08% prevention (target ≥ 70%, 1.4× over)
- Kick V4.4: 🟡 in progress, ETA ~2h

### 2026-05-22 21:15 — Trap COMPLETE; V4.4 Kick recovering; Defend at 81% block+steal

**Trap** finished its 8000-iter run at 21:13.
- `trap_success_rate` final EMA = **93.5%** (target ≥ 15% — 6× over).
- Final checkpoint: `logs/rsl_rl/soccer_trap_skill_amp/2026-05-22_09-22-XX/model_7999.pt`.
- Wall-clock 11h43m (slower than the 3.5h estimate due to 3-way GPU
  contention throughout).

**V4.4 Kick recovery at 1.5h post-relaunch:**

| Metric | 30 min | 1 h | 1.5 h |
|---|---:|---:|---:|
| peak_kick_speed_shoot (lifetime EMA) | 1.92 | 3.19 | **4.10** |
| peak_kick_speed_pass (lifetime EMA) | 1.83 | 3.24 | **4.15** |
| kick_contact_rate (EMA) | 27.8 % | 43.2 % | **50.1 %** |
| goal_scored_rate (EMA) | 6.6 % | 10.0 % | 11.2 % |
| pass_landing_rate (EMA) | 11.9 % | 19.9 % | **22.8 %** |

Math: 50% kick rate × ~8 m/s peak + 50% no-kick × 0 = 4.0 → matches the
4.10 EMA. So when the policy does kick, it does so at peak ~8 m/s (the
V4.1 baseline), and the kick-rate is recovering toward V4.1's 58% baseline.
V4.4 is NOT repeating V4.2 — the wider σ + narrower commanded range +
disabled speed-dependent angle penalty give the policy room to relearn.

**Defend at ~5h elapsed:**
- block_success_rate **52.7%**, steal **28.3%**, total **81%**
  (target 30%, exceeded by 2.7×).
- own_goal_scored_rate 1.0% — defender successfully prevents 99% of shots.
- lane_hold dropped 90% (V4.0) → 63% (V4.1) — engaged, not parked.

**GPU**: now 2-way contention (Kick + Defend) post-Trap completion.
Kick iter time 6.06s → 3.92s, Defend 5.55s → 3.57s. Both will finish
faster from here.

### 2026-05-22 20:05 — Kick V4.4: mode-split strength range + lifetime-peak EMA

V4.3 fixed the legacy speed-dependent angle penalty (which was the V4.2
regression cause). User noticed that the **per-mode peak metric was still
misleading** because it was derived from `_peak_kick_speed` which resets
on every multi-attempt re-trigger — so a kick at peak 8 followed by a
weak follow-up at peak 3 would report peak~2.7 averaged across the episode.

**V4.4 metric fix:**

* Added `_lifetime_peak_kick_speed` — per-env max kick speed across the
  *entire episode* (never reset by multi-attempt; only at
  `_resample_command`). This is the "did the policy ever kick hard?"
  signal.
* Added per-mode EMA buffers `_lifetime_peak_kick_speed_ema_shoot/_pass`
  updated **once per episode at reset** with α=0.03 (~30-episode window).
  Each env's EMA only updates with episodes of its matching mode, so
  pass episodes don't dilute the shoot reading.
* `Metrics/soccer_kick/peak_kick_speed_{shoot,pass}` now report these
  EMAs (replacing the previous scaled-by-`shoot_prob` formulation, which
  was confusing because pass envs contributed zeros).

**V4.4 strength range split (mode-conditional):**

```
target_strength_range          = (3.0, 6.0)  # pass — stays inside landing speed window (0.5, 6.0)
shoot_target_strength_range    = (8.0, 14.0) # shoot — higher than V4.1 baseline ~8.5 peak
```

V4.1 was sampling 3-8 for both modes; the policy converged to ~8 m/s
shoot peaks and that's where it plateaued. With a wider commanded
range we give `kick_power_track_shoot` headroom to push the policy
higher. After this stabilises we can move to (10, 15).

* Pass range tightened to (3, 6) so the commanded strength can't exceed
  `pass_landing_speed_window = (0.5, 6.0)` — a pass commanded at 7-8 m/s
  could literally never land in the V4.1 setup.
* Normalisation uses the **union range** (min/max of both bounds), so
  ``target_strength_norm`` is comparable across modes — the policy
  interprets the mode via the `is_shoot` observation.

**Gaussian σ widened for shoot:**

`kick_power_track_shoot.sigma` 1.5 → **2.5** so the post-kick window
gets non-vanishing gradient across the wider (8, 14) commanded range.
With σ=1.5 a peak of 8 vs cmd=14 gives exp(-6²/2.25²)≈1e-3 — essentially
zero gradient. σ=2.5 gives exp(-6²/6.25)≈3e-3 — small but workable, and
much larger for nearer commands.

Pass `kick_power_track_pass.sigma=1.0` kept (narrower range needs
tighter Gaussian).

**Smoke regressions** PASS. Relaunched from V4.1@5500 in
`logs/kick_skill_v4_final_warmstart_train.log`.

### 2026-05-22 19:45 — Kick V4.3: speed-independent angle penalty + per-mode peak metric

V4.2 went sideways. After 1 h of training (warmstart from V4.1@5500), Kick
metrics:

| Metric | V4.1 final (iter 5500) | V4.2 (30 min) | V4.2 (1 h) |
|---|---:|---:|---:|
| Mean reward | ~75 | 87 | 89 |
| peak_kick_speed (mixed) | 5.3 | 4.58 | 4.34 ↓ |
| kick_contact_rate (EMA) | 58 % | 28 % | 41 % |
| goal_scored_done (term) | 13.5 % | 12.5 % | 12.5 % |
| ball_out (term) | 27.5 % | 16.9 % | 16.5 % |
| time_out (term) | 54 % | 65 % | 66 % |

Aim got better (`ball_out` 27 → 17 %) but `peak_kick_speed` collapsed and
goal rate stayed flat. Diagnosis: the legacy `kick_angle_error` formula
was the cause — `angle² × ball_xy_speed × is_shoot` (weight -5 from V4.1)
in a 15-step post-kick window. Per-step penalty:

| ball_xy_speed | penalty/step @ 30° err |
|---:|---:|
| 5 m/s | -6.75 |
| 4 m/s | -5.40 |
| 3 m/s | -4.05 |

So kicking weaker strictly reduced the cumulative penalty even with the
same angle, creating a "kick less hard to escape the angle penalty"
gradient that beat the kick_power_track_* reward. Verified by reverting:
just turning off the legacy `kick_angle_error` and reloading V4.1@5500
immediately gave `peak_kick_speed_shoot = 8.46 m/s` (the V4.1 policy was
already kicking near max-commanded strength — the mixed-mode metric had
been hiding this behind weaker pass kicks + multi-attempt peak resets).

**V4.3 changes** (replace, don't stack on V4.2):

| Reward | Change |
|---|---|
| `kick_angle_error` (legacy, mode-agnostic, `angle² × speed`) | weight 0 (disabled) |
| `kick_angle_error_shoot` (new, speed-independent, `angle²`, live ball→goal_center) | weight **-3.0**, 15-step post-kick window |
| `kick_angle_error_pass` (new, speed-independent, `angle²`, live ball→pass_target) | weight **-3.0**, 15-step post-kick window |

The new variants:
* Drop the `ball_xy_speed` multiplier — same angular precision incentive
  without the "kick weak to escape" loophole.
* Reference the *live* ball→goal-center direction (shoot) and *live*
  ball→pass_target direction (pass), matching their positive-side
  counterparts ``kick_aim_at_goal_shoot`` / ``kick_aim_at_pass_target_pass``
  instead of the spawn-time ``target_dir_w``.

**Per-mode peak metric** (also new):
* `Metrics/soccer_kick/peak_kick_speed_shoot` — mean shoot-episode peak,
  scaled by `1/shoot_prob` so the averaged-across-envs display equals
  the conditional mean. Similarly for `peak_kick_speed_pass`.
* The combined `peak_kick_speed` metric is preserved for backward compat
  but mixes both modes and drops to ~half when multi-attempt clears the
  per-env peak buffer.

**First-iter readings after warmstart from V4.1@5500:**
- peak_kick_speed_shoot **8.46 m/s** (vs. mixed-metric ~5 that we'd been
  seeing). The V4.1 policy was already strong; the V4.2 regression was a
  collapse from this baseline.
- peak_kick_speed_pass **7.06 m/s** — slightly weaker as expected for
  the more conservative pass mode.
- kick_angle_error_shoot Episode_Reward = -0.006 (small, V4.1 had already
  learned shoot aim).
- kick_angle_error_pass Episode_Reward = -0.043 (larger — pass aim was
  never trained in V4.1; V4.2's `kick_aim_at_pass_target_pass` reward
  gradient will train it from here).

Smoke regression PASS. New log run: `2026-05-22_19-39-xx`,
`logs/kick_skill_v3_warmstart_train.log`.

### 2026-05-22 17:35 — Kick V4.2: precision + power tuning (warmstart from V4.1@5500)

V4.1 kick training plateaued around goal_done 14 % (EMA goal_scored_rate
13.5 %, pass_landing_rate 22 %). Diagnosis of the ceiling:

* **Shoot aim** ~67° off goal center on average (`kick_aim_at_goal_shoot`
  per-episode value 0.058 ÷ (is_shoot 0.5 × fast_enough 0.3) → cos ≈ 0.39)
* **Kick strength ignores command** — `peak_kick_speed` stuck near 5 m/s
  regardless of commanded strength (3–8 m/s). `kick_strength_error_*`
  only fires on the 1-step edge of `kick_contact_new`, per-episode
  contribution ~-0.005 — too small to shape behaviour.
* **Pass aim has no reward** at all — only ``target_progress_pass_window``
  for first 20 steps + ``ball_at_target_terminal``. The policy was
  aiming based on `target_dir_w` which equals receiver xy at spawn but
  not after both move; verified by relaunching with the new pass-aim
  reward and seeing it fire **negative** (~-0.026/ep cos_angle), i.e.
  the V4.1 policy was aiming *away* from the pass target on average.

**V4.2 changes (additive — no V4.1 weight reductions):**

| Reward | V4.1 weight | V4.2 weight | Notes |
|---|---:|---:|---|
| `kick_aim_at_goal_shoot` | 3.0 | **10.0** | Boost shoot aim precision |
| `kick_aim_at_pass_target_pass` (new) | — | **8.0** | `cos(ball_vel, ball→pass_target)`, pass-only |
| `kick_power_track_shoot` (new) | — | **5.0** | Dense Gaussian on (peak_kick_speed − cmd_strength), fires in 30-step post-kick window |
| `kick_power_track_pass` (new) | — | **6.0** | Same but pass-only, σ=1.0, 15-step window |
| `kick_strength_error_shoot` | -1.5 | **-5.0** | Edge penalty reinforces dense track signal |
| `kick_strength_error_pass` | -2.0 | **-5.0** | Same for pass |

Continuous power tracking is the structural fix — V4.1's edge-only
penalty couldn't teach "honor commanded strength" because it only fired
once per episode. The new ``kick_power_track_*`` rewards fire every
step in the post-kick window with a Gaussian centered on the commanded
strength, giving a dense gradient that pulls the peak speed toward the
command. Physically: a peak of 7 m/s vs current 5 m/s extends ball
travel from ~2 m to ~4 m before stopping (µ=0.6), bringing far-spawn
goal-line shots into reach.

Smoke regression passed; relaunched with warmstart from V4.1@5500 in
tmux `soccer:kick_skill_v0`; log `logs/kick_skill_v2_warmstart_train.log`.

### 2026-05-22 13:50 — V4.1 rebalance: kick rewards + defend redesign + metric fix

Killed kick_skill_v0 (iter 3500, ETA 7h remaining) and defend_skill_v0
(iter 2700) after the user agreed both stages need fixes — pass-landing
0 % and defend block+steal 5 % were diagnosed as **structural** issues
(reward conflicts + physically-unreachable pass landing window + dense
reward dominance), not training-time problems. Trap kept running (91 %
already meets §11.3).

**Per-episode metric fix (applies to all V4 commands):**
- `SoccerKickCommand`: added per-env EMAs (α=0.03 → ~30 episode window)
  for ``goal_scored``, ``kick_contact``, ``kick_success``, ``pass_landing``,
  ``trap_success`` rates, captured in ``_resample_command`` *before* the
  latches are reset. New ``Metrics/*/pass_landing_rate`` field added.
- `SoccerDefendCommand`: same pattern for block/steal/own_goal rates plus
  new ``intercept_distance_avg`` (continuous, m).
- Existing per-step latch averaging only happened to look reasonable for
  contact/success (latches persist throughout episode) but reads ≈ 0 for
  terminal latches like goal_scored / own_goal_scored.

**Kick V4.1 rebalance (`soccer_kick_skill_amp/tracking_env_cfg.py`):**

| Reward | V4.0 weight | V4.1 weight | Rationale |
|---|---:|---:|---|
| `target_progress` (mode-agnostic) | 8.0 | **0.0** | Disabled — was double-counting in pass + dominating goal_scored 500:1 |
| `target_progress_shoot` (new, is_shoot-gated) | — | **2.0** | Shoot-only continuous shaping, reduced magnitude |
| `kick_aim_at_goal_shoot` (new) | — | **3.0** | `cos(ball_vel, ball→goal_center)`, shoot only, fires every step ball > 1 m/s |
| `kick_angle_error` | -1.0 | **-5.0** | Stronger precision penalty |
| `kick_strength_error_shoot` | -0.5 | **-1.5** | More strength precision |
| `goal_scored_quick` (base+time) | 30+50 | **50+150** | 4× larger terminal bonus, more urgency |
| `time_to_goal_penalty` | -0.02/step | **-0.05/step** | Stronger time pressure |
| `target_progress_pass_window` | 8.0 | **4.0** | Reduce 20-step pass shaping by half |
| `ball_at_target_terminal` radius | 1.0 m | **1.5 m** | Wider terminal landing zone |

**Multi-attempt support (new in `SoccerKickCommand`):**
- New cfg fields ``enable_multi_attempt_shoot=True``,
  ``multi_attempt_ball_speed_thresh=0.5``,
  ``multi_attempt_foot_clear_dist=0.5``.
- When the ball comes to rest AND the foot is away (shoot-mode only,
  pre-goal), the ``kick_contact_awarded`` / ``kick_success_awarded`` latches
  are cleared and ``_steps_since_kick`` reset, so a follow-up kick can
  re-earn ``kick_contact`` / ``kick_success`` / first_bonus. Implements
  the design §4.1 "shoot = multi-attempt OK" semantics.
- Pass mode keeps the monotonic latches so ``multi_kick_penalty`` and
  ``approach_after_kick_penalty`` continue to fire on 2nd touches.

**Pass widening (`SoccerKickCommandCfg`):**
- ``pass_landing_radius``: 1.0 → 1.5 m
- ``pass_landing_speed_window``: (0.5, 4.0) → (0.5, 6.0) m/s
- These match the observed ``peak_kick_speed`` of 6 m/s so passes are
  physically reachable on first kick. ``ball_at_target_terminal.radius``
  also bumped to 1.5 m.

**Defend V4.1 redesign (`soccer_defend_skill_amp/tracking_env_cfg.py`):**

| Reward | V4.0 weight | V4.1 weight | Rationale |
|---|---:|---:|---|
| `block_success` | 30.0 | **80.0** | Boost sparse interception bonus |
| `steal_success` | 25.0 | **60.0** | Same |
| `defensive_line_hold` (binary) | 5.0 | **0.3** | Reduced 17× — was 920× larger than block_success/episode |
| `ball_intercept_proximity` (new) | — | **2.0** | Continuous `exp(-d²/σ²)` on defender↔segment distance — replaces lane_hold as dense shaping but smooth and well-defined off-segment |
| `defender_approach_ball` (new) | — | **2.0** | `exp(-d_to_ball²/σ²)` when ball is incoming — encourages engaging the ball |
| `ball_pushed_away` | 10.0 | 10.0 | unchanged |
| `own_goal_proximity` | -15.0 | -15.0 | unchanged terminal disincentive |

The continuous shaping (`ball_intercept_proximity` + `defender_approach_ball`)
replaces the binary `defensive_line_hold` as the dominant per-step signal,
but at 2.0 + 2.0 = 4.0 max-per-step instead of 5.0/step always-on. Across
a 5 s × 50 Hz = 250 step episode the total dense reward maxes around
~250 × 0.5 (avg over distance) ≈ 125, while a single block now pays 80
and steal 60 — comparable order of magnitude, so the policy can't game
the dense signal.

**Smoke regressions** (all PASS):
- `Booster-Soccer-Kick-Skill-v0` (V4.1): 134d / 149d actor / critic — new
  reward partition includes `target_progress_shoot` + `kick_aim_at_goal_shoot`.
- `Booster-Soccer-Defend-Skill-v0` (V4.1): 132d / 140d — new reward
  partition includes `ball_intercept_proximity` + `defender_approach_ball`.
- `Booster-Soccer-Trap-Skill-v0` (V4.0, untouched): clean.
- `Booster-Soccer-Kick-AMP-MC-v0` (legacy V1.2): clean.
- `Booster-Soccer-KickTrap-AMP-V2-v0` (legacy V3.2): clean.

**Both V4.1 trainings launched in tmux** (`soccer:kick_skill_v0`,
`soccer:defend_skill_v0`); logs `logs/kick_skill_v1_train.log` and
`logs/defend_skill_v1_train.log`. ETAs ~10–11 h each given 3-way
GPU contention (5.3 s/iter). Old V4.0 checkpoints preserved at:
- `logs/rsl_rl/soccer_kick_skill_amp/2026-05-22_08-27-31/model_3500.pt`
- `logs/rsl_rl/soccer_defend_skill_amp/2026-05-22_09-25-35/model_2700.pt`

**Warm-start patch (14:23):** Re-launched both Kick V4.1 and Defend V4.1
with `--resume --load_run <V4.0 run> --checkpoint <V4.0 model>` to load
the V4.0 policy + optimizer state. Obs space + policy architecture are
unchanged between V4.0 and V4.1 (only rewards + metric latches changed),
so the kicker's learned kick motion (peak_kick_speed ~6 m/s, 77% kick_success
in V4.0) transfers directly. Verified on first iter after resume:
- Kick: `peak_kick_speed 6.5 m/s`, `goal_scored_done 7%`,
  `Episode_Reward/goal_scored_quick 0.081` (V4.1 reward firing),
  `Episode_Reward/target_progress_shoot 0.997` (new shoot-only reward firing).
- Defend: `Episode_Reward/block_success 0.10`, `steal_success 0.01`.

Expected saving: ~4 h of wall-clock per skill that would otherwise be
spent relearning basic locomotion + kicking. Per-episode rate EMAs are
NOT saved in the checkpoint (they live on the command term, not the
model), so the displayed `*_rate` metrics start at 0 and ramp up over
~30 episodes after resume — that's metric display only, not actual
policy regression. New log run dirs created (datestamped 14-23-xx) so
`tee` log files now have ``warmstart`` suffix.

### 2026-05-22 12:00 — Stage 1/2 on track, Stage 3 reward imbalance flagged

Snapshot at ~3.5 h into training:

| Stage | Iter | kick_success / trap / block+steal | Status |
|---|---:|---|---|
| 1 Kick | 1920 / 8000 | **76.8 % kick_success**, peak 5.97 m/s, goal=0% | ✅ §11.2 hit; goal_scored hard target still 0 |
| 2 Trap | 1900 / 8000 | **93.4 %** trap_success | ✅ 6× target |
| 3 Defend | 1880 / 8000 | block 5.5 %, steal 1.1 %, lane_hold 90 % | ⚠️ interception not emerging |

**Stage 3 reward analysis** (per-episode `Episode_Reward/*`):
- `defensive_line_hold` = **4.15** (with weight +5/step × ~90 % lane × ~250 steps)
- `block_success` = 0.0045 (weight +30, rate 5.5 %)
- `steal_success` = 0.0006 (weight +25, rate 1.1 %)
- `ball_pushed_away` = 0.0007
- ratio line_hold:block ≈ 920 : 1

The line-hold dense reward dominates by ~3 orders of magnitude. Even though
the design §4.3 spec lists `defensive_line_hold | +5 / step`, the per-episode
total (5 × 0.9 × 250 = 1125) dwarfs the +30 sparse block bonus. The defender
finds it strictly more efficient to "stand in lane" than to actively block,
and as the lane_hold metric rose 52 → 69 → 87 → 90 % the block_success metric
slipped 6.2 → 5.2 → 5.5 % (no real upward trend).

**Recommendation**: when Stage 3 finishes, either
1. Drop `defensive_line_hold` weight to ~0.1–0.3/step (≈ per-episode contribution
   25–75, still below the +30 sparse block) and retrain, or
2. Make `defensive_line_hold` only fire as a *terminal* bonus (was-in-lane
   over the episode) so it doesn't dominate the per-step signal.
Decision deferred to the user; the current run is allowed to complete so we
have a baseline against which to ablate.

---

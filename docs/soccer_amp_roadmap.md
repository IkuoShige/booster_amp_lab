# Soccer AMP — Roadmap

**Project:** Add a soccer **kick (passer / shooter)** + paired **receiver (trapper)** task to `adapter_booster_amp_lab` while keeping the existing AMP locomotion / adapter pipeline intact. Inspired by `output-sc/lvdrs.md` (LVDRS, RoboCup'25 winner) and `/workspace/mjlab/.../tasks/kick`. NOT a re-implementation — a clean Isaac-Lab port of the design *spirit*.

---

## 1. Behavioral spec (what the policy must do)

### 1.1 Kicker (a.k.a. passer / shooter)
1. If ball is not in view → **search** (rotate head/body, scan).
2. If ball found → **approach**, maintain ball in FOV (head/torso active perception).
3. Plant support foot near ball, swing kicking foot.
4. Kick toward a **target direction** in the world (commanded each episode).
5. Kick with **target strength** (commanded scalar) — used to either **shoot** at a goal or **pass** to a teammate at a specific distance.
6. Distinguish **shoot vs pass** via an explicit input observation (mode flag). Same policy, different conditioning.

### 1.2 Receiver (a.k.a. trapper) — only active when `mode == pass`
1. Watch the kicker / look at incoming ball.
2. Position foot under the incoming pass.
3. **Trap**: deaden the ball so it ends up at the receiver's feet (ball speed → ~0, ball within ~0.25 m).
4. Receiver never traps a shoot (env signals this via a flag).

### 1.3 Important constraints
- **Sim-to-real**: virtual head camera with realistic FOV / latency / dropout / distance-noise, like LVDRS §2 and mjlab `kick/mdp/perception.py`.
- **AMP** discriminator anchors motions to a human-like manifold (shared between roles).
- **Multi-critic** decouples goal-related vs auxiliary rewards (LVDRS §3.2; mjlab `rl_cfg.py`).
- **K1 head**: pitch range −20°…+49° (URDF `Head_pitch`). At max +49° the camera sees the robot's own feet — useful for trapping confirmation.
- **Field**: 9 m (Y) × 14 m (X), goal **2.6 m wide × 1.2 m tall**. Field is simplified — flat plane with painted lines is fine.

---

## 2. Architecture decisions

### 2.1 Single env, two K1 robots

- **One env instance contains both kicker and receiver** (two `Articulation` instances + one `RigidObject` ball + one `Goal` collection).
- The receiver stands ~3–6 m away from the kicker on average; randomized per episode.
- Each robot has its own observation/action slice. Both robots run at the same control rate.
- **Why not two envs?** The receiver must trap the *actual* ball the kicker kicked — requires shared physics state.

### 2.2 Policies & networks

| Component | Kicker | Receiver | Notes |
|---|---|---|---|
| Actor | `MLP(512,256,128)` | `MLP(512,256,128)` | separate weights |
| Critic — goal | `MLP(512,256,128) → 1` | `MLP(512,256,128) → 1` | scoring/strength/trap-success |
| Critic — aux | `MLP(512,256,128) → 1` | `MLP(512,256,128) → 1` | regularizers, AMP-aux |
| AMP discriminator | shared | shared | both are K1, same motion manifold |
| Mode flag in obs | yes (shoot/pass) | yes (always pass, but trained on mode for symmetry) | one-hot 2-vec |

### 2.3 Reward decomposition (per role)

| Term | Group | Kicker weight | Receiver weight |
|---|---|---:|---:|
| `target_progress` (ball moves along target dir) | goal | 500 | 0 |
| `ball_approach` (kicker→ball dist ↓) | goal | 50 | 0 |
| `kick_success` (single shot when ball speed > thresh aligned with target) | goal | 3 | 0 |
| `kick_strength_error` (pass: \|achieved − target\| penalty) | goal | -2.0 | 0 |
| `kick_angle_error` (post-contact ball heading penalty) | goal | -3 | 0 |
| `goal_scored` (ball enters goal mouth, shoot only) | goal | +15 | 0 |
| `trap_success` (one-shot; ball at feet with speed<0.3) | goal | 0 | +20 |
| `ball_at_feet` (potential: dist(ball, support_foot) ↓) | goal | 0 | 30 |
| `intercept_alignment` (face incoming ball, pre-trap) | goal | 0 | 10 |
| `alive` | aux | 5 | 5 |
| `terminated` | aux | -200 | -200 |
| `pelvis_orientation`, `arm_dev_l2`, `joint_*_l2`, `action_rate_l2` | aux | std small | std small |
| `head_yaw/pitch_alignment` (look at ball) | aux | -2 / -2 | -2 / -2 |
| AMP style (discriminator) | aux | 0.3 | 0.3 |

Multi-critic combination: `A_total = w_goal · A_goal + w_aux · A_aux` (default `w_goal=2, w_aux=1`, matches LVDRS).

### 2.4 Virtual perception (camera)

Mounted on link `Head_2` (K1 has yaw on `Head_1`, pitch on `Head_2`). Mounted forward & slightly above the eye level, similar to mjlab's K1 head camera.

Outputs to actor obs (per robot):
- `ball_pos_b_perceived`: ball position projected into the robot's body-yaw frame, noisy & latent.
- `ball_mask`: 1 if currently detected (in FOV ∧ Bernoulli detection), else 0.
- `last_seen_dt`: clamped seconds since last detection (encourages search behavior).
- (kicker only) `goal_dir_b`: target/goal direction in body-yaw frame, low-rate odometry style.

The pseudo-camera is implemented as **ground-truth + noise + FOV check + Bernoulli detection + ring-buffer latency**. No actual RGB rendering (too expensive for parallel sims). Detection probability decays with distance & near-edge of FOV. Hold-on-miss disabled by default (zeros on miss → policy must rely on `ball_mask`).

Domain randomization (per env, sampled on reset):
- `noise_a ∈ (0.035, 0.075)`, `noise_b ∈ (0.056, 0.12)` (range of σ(d) = a·d + b)
- `latency_mean ∈ (80, 160) ms`, `latency_std = 18 ms`
- `update_hz_mean ∈ (20, 30) Hz`
- `detection_prob ∈ (0.3, 0.95)`
- 10 % of episodes fully blind (`ball_mask = 0` throughout)

### 2.5 Action / control loop

- 50 Hz policy → 5 ms physics step × decimation 4.
- Joint position targets, scaled by `K1_ACTION_SCALE` (existing constants).
- Reduced arm/shoulder scale during kick (mjlab learned this is critical to prevent T-pose).
- Delayed actuator (2–8 step latency) — already present in `BOOSTER_K1_CFG`.

### 2.6 AMP corpus
- Reuse `booster_assets/motions/K1/kick.csv` (4585 frames, multi-phase kick) + `walk2run.csv`, `run.csv` for locomotion priors.
- Add a small **stand-still** motion later if the receiver needs more "wait calmly" prior.
- Discriminator obs = existing `AMPObsCfg` (joint_pos, joint_vel, hand_pos, foot_pos).

---

## 3. Staged delivery

### V1 — Kicker only (single-agent)
- 1 K1 + 1 ball + 1 goal (no receiver).
- Kicker: search → approach → kick toward target direction with target strength.
- "shoot" mode only (no pass). Target = goal-mouth random point.
- Validate: kick_success rate from a band of positions in front of goal.

### V2 — Add shoot/pass mode flag
- Kicker policy distinguishes shoot vs pass via observation.
- In pass mode the target becomes a "phantom receiver" pose (sampled in field); kicker tries to land the ball there with target strength.
- No actual receiver agent yet; the env scores "imaginary trap" from ball trajectory.

### V3 — Add receiver (multi-agent)
- Spawn second K1 in env, at the phantom-receiver location.
- Receiver policy trained simultaneously with separate actor/critics.
- Shared AMP discriminator.
- Trap reward measures ball-at-feet & ball-deadened.

### V4 — Polish (after V3 baseline trains)
- Curriculum on pass distance.
- Push robot during trap.
- Goalkeeper variant (optional).

---

## 4. Files we will add (under `source/booster_rl_tasks/booster_rl_tasks/tasks/manager_based/beyond_mimic/robots/k1/`)

```
soccer_kick_amp/
├── __init__.py                # gym register
├── soccer_env_cfg.py          # base TrackingEnvCfg-derived cfg (one robot variant)
├── soccer_two_agent_env_cfg.py# V3 variant with 2 robots
├── ppo_cfg.py                 # multi-critic AMP PPO runner cfg
└── mdp/
    ├── __init__.py
    ├── perception.py          # VirtualPerception + obs helpers
    ├── commands.py            # SoccerCommand (mode, target, strength, receiver pose)
    ├── observations.py        # role-specific obs functions
    ├── rewards.py             # kicker & receiver reward functions
    ├── events.py              # ball/robot/receiver reset, DR
    ├── terminations.py        # goal-scored / ball-out / fall
    └── scene.py               # SceneCfg with ball, goal posts, field
```

New shared assets:
```
booster_assets/objects/
├── soccer_ball.usd  (or generated programmatically)
└── goal.usd         (two posts + crossbar; or built from primitives)
```

New rl additions:
```
rsl_rl/rsl_rl/modules/multi_critic_actor_critic.py     # AC with N critic heads
rsl_rl/rsl_rl/algorithms/multi_critic_amp_ppo.py       # multi-critic PPO + AMP
rsl_rl/rsl_rl/runners/two_agent_amp_runner.py          # V3 only
```

---

## 5. Open design questions (revisit per stage)

- [ ] Should the V3 receiver use its own AMP corpus that emphasizes "still / shifting weight"? Or rely on a single shared corpus?
- [ ] Mode label: hard one-hot, or noisy soft label? (Real deployments may not know which mode they're in.)
- [ ] Goal posts: rigid bodies, or just collision sensors for scoring? (Start as sensors only; reduce contact load.)
- [ ] Should mode-switch reset within an episode? (For V1–V3 keep mode fixed per episode for clean credit assignment.)
- [ ] Ball respawn during episode — useful for receiver if kick is bad? (Decide in V3.)

---

## 6. References

| Source | Use |
|---|---|
| `output-sc/lvdrs.md` | Multi-critic, virtual perception, encoder-decoder, AMP coefficient 0.3 |
| `/workspace/mjlab/src/mjlab/tasks/kick/` | Concrete reward weights, perception params, K1 head camera offset (do NOT copy code; port spec only) |
| `docs/amp_recovery_adapter_controller_design.md` | Existing AMP+adapter integration in this codebase |
| `booster_assets/robots/K1/K1_22dof.urdf` | Joint order (CSV columns 8–29), head pitch limits, foot link names |

---

_Status: drafted on 2026-05-21. Updated as V1→V4 progresses. Worklog in `docs/soccer_amp_worklog.md`._

---

## Update 2026-05-22 — V4 supersedes V3 architecture

After running V3.2 and inspecting in viser, we changed direction. The 44-DOF
shared policy (one network controlling kicker + receiver) is **not** the
deployable shape; we need per-agent policies so a single K1 can run any
role on its own hardware.

The new plan is a **skill library**: train separate kick / trap / defend
policies, each single-agent. They share a per-agent network with a role
one-hot so deployment is "one checkpoint, set the role flag".

Full V4 design is in `docs/soccer_amp_skill_library_design.md`. V1.2 and
V3.2 checkpoints are kept as baselines for comparison; V3.3 in-place fixes
(camera mount, FOV viz, search rewards, AMP corpus) are merged into the
main repository.

---

## Update 2026-05-22 (later) — V4 Stage 1–3 implemented; Stage 1 training launched

| Stage | Task ID | Status |
|---|---|---|
| 1 (Kick) | `Booster-Soccer-Kick-Skill-v0` | implemented + smoke-verified; **training in-flight** in tmux `soccer:kick_skill_v0`, ETA ~7.5 h |
| 2 (Trap) | `Booster-Soccer-Trap-Skill-v0` | implemented + smoke-verified; training pending GPU |
| 3 (Defend) | `Booster-Soccer-Defend-Skill-v0` | implemented + smoke-verified; training pending GPU |
| 4 (Composition) | `play_multi.py` runner | scaffolding next |
| 5 (Encoder upgrade) | `EncoderActorCritic` | conditional — only if trap < 20 % |

See `docs/soccer_amp_worklog.md` for per-stage implementation notes (new
files, reward weights, smoke results, deviations).

# Soccer AMP — Skill-Library Design (V4 / post-V3.2 refactor)

This document captures the agreed-upon redesign that comes out of the
post-V3.2 discussion. It supersedes the relevant sections of
`soccer_amp_v3_design.md` (which framed the receiver as part of a unified
44-DOF policy). Pair with `soccer_amp_roadmap.md` and `soccer_amp_worklog.md`.

---

## 1. Vision

**Goal**: deliver a *skill library* — a small collection of single-agent
policies that each master one soccer skill (kick / trap / defend), trained
independently, then composed at deployment.

The composition pattern resembles human practice: a player drills shooting,
then drills receiving, then drills defending, and only afterwards combines
them in a match. We do not expect emergent multi-task generalisation to
substitute for focused, individual skill training.

The end-state is multi-agent (multiple K1s on a field, each running one of
these policies). For now we are in the **skill-acquisition phase**.

## 2. Skills to deliver

| Skill ID | Task | Single-agent? | Default target | Override-able target |
|---|---|---|---|---|
| `kick`    | Drive the ball to a commanded point with a commanded speed (shoot OR pass) | yes  | goal mouth (shoot) | yes — point + strength + mode flag |
| `trap`    | Receive an incoming ball and bring it to rest at the receiver's feet     | yes  | n/a (reactive) | n/a |
| `defend`  | Prevent the ball from crossing the defender's "own goal" line; ideally steal it | yes  | own goal line | n/a |

`idle` (stand calmly, watch the ball) is an **operating mode** of the kick
policy when its role-flag is set to "idle", not a separate skill.

## 3. Architecture — Per-agent shared policy with role-flag

We move away from the V3.2 architecture (one 44-DOF output controlling two
robots) for two reasons:
1. **Deployment**: a real K1 runs on its own and only needs 22-DOF actions.
2. **Skill modularity**: separate skills are easier to evaluate, debug, and
   independently improve.

### 3.1 Network

```
actor: f(per-agent obs vector + role one-hot[4] → 22-DOF joint action)

per-agent obs = proprio(~50d) + perceived_ball(~5d) + ball_vel(~3d)
              + ball_history(~50d via encoder, see §6)
              + target_dir_b(2d) + target_strength(1d)
              + mode flag(s) — e.g. is_shoot(1d)
              + role one-hot(4d)
```

`role` is `[kicker, receiver, defender, idle]`. Exactly one is 1.

At **train** time, an env with N robots calls the policy N times per step
(once per agent), each call gets that agent's obs + its role one-hot. We
concatenate the N×22 actions into the env action vector. Storage records
N transitions per env-step.

At **deploy** time, a single K1 runs one policy forward per step using its
own observations and a fixed role-flag (the operator chooses kick / trap /
defend at the start of the play).

### 3.2 Multi-critic stays

Each role uses the existing multi-critic split (goal / aux) with weighted
advantage `A = 2·A_goal + A_aux`. Different reward terms are tagged with
critic groups via `RewardsCfg.CRITIC_GROUPS` per role.

## 4. Reward design per skill

### 4.1 Kick skill — mode-conditional rewards

The kick skill has two modes inside one skill: **shoot** (multi-attempt OK,
score as fast as possible) and **pass** (one-shot, commit to strength).

#### Shared (both modes)
| Term | Weight | Group | Notes |
|---|---:|---|---|
| `target_progress` | +8 | goal | Ball velocity projected on `target_dir_w` |
| `ball_approach` | +4 | goal | × `ball_mask`, × `pre_kick` |
| `foot_ball_proximity` | +3 | goal | × `ball_mask`, × `pre_kick` |
| `kick_contact_first` | +4 | goal | One-shot on first kick contact |
| `search_yaw_velocity` | +2 | aux | × (1 − ball_mask) — V3.3 |
| `last_seen_dt_penalty` | −1 | aux | V3.3 |
| `head_yaw_search` | +0.5 | aux | V3.3 |
| `pre_kick_yaw_align` | −0.5 | aux | Robot facing target before kick |
| `head_yaw_align` / `head_pitch_align` | −0.3 each | aux | Look at ball |
| `pelvis_orientation` / `feet_proximity` / `arm_deviation_l2` | small neg | aux | Posture |
| `alive` / `terminated` | +0.5 / −20 | aux | Standard |
| `action_rate_l2` / `dof_torques_l2` / `dof_acc_l2` / `dof_pos_limits` / `undesired_contacts` | small neg | aux | Regularizers |
| AMP style | (0.3 lerp) | — | Discriminator side-channel |

#### Shoot mode only (× `is_shoot`)
| Term | Weight | Group | Notes |
|---|---:|---|---|
| `goal_scored_quick` | base +30 + bonus `+50·(1 − elapsed/max)` | goal | Time-weighted — earlier = more |
| `time_to_goal_penalty` | −0.02 / step (until goal) | goal | Pressure to score fast |
| `kick_success_first_bonus` | +10 first kick_success, +3 subsequent | goal | Prefer 1-shot but allow retries |
| `kick_strength_error_shoot` | −0.5 × \|achieved − cmd\| × kick_contact_new | goal | Weak penalty — fail-and-retry OK |

#### Pass mode only (× `(1 − is_shoot)`)
| Term | Weight | Group | Notes |
|---|---:|---|---|
| `pass_landing` | +30 one-shot | goal | Ball lands within `pass_landing_radius` at correct speed band |
| `kick_strength_error_pass` | −2.0 × \|achieved − cmd\| × kick_contact_new | goal | Strong — 1-shot must be precise |
| `multi_kick_penalty` | −10 × (kick_contact_new ∧ kick_contact_awarded) | goal | Penalize 2nd+ touches |
| `approach_after_kick_penalty` | −3 × (post_kick ∧ kicker_to_ball shrinking) | goal | Don't chase after release |
| `target_progress_pass_window` | (8 only in `steps_since_kick < 20`) | goal | Replaces the unbounded shaping |
| `ball_at_target_terminal` | +20 at episode end if \|ball − target\| < 1m | goal | Sparse landing bonus |

#### Mode condition
`is_shoot` is sampled at episode reset: `Bernoulli(shoot_prob=0.5)` by
default. Each reward function multiplies by the appropriate mask
(`cmd.is_shoot.float()` for shoot, `1 − cmd.is_shoot.float()` for pass). The
policy reads `is_shoot` as an observation and conditions its behaviour.

### 4.2 Trap skill — receiver

| Term | Weight | Group |
|---|---:|---|
| `trap_success` | +30 one-shot | goal |
| `receiver_ball_at_feet` | +6 (Gaussian on min foot-ball xy, σ=0.30) | goal |
| `intercept_alignment` | −1.0 (yaw error to incoming ball) | aux |
| `trap_anticipation` (NEW) | +2 if ball approaching ∧ predicted-contact distance < 0.5 m | goal |
| `receiver_idle_when_blind` (NEW) | +0.3 in stance posture when `ball_mask` = 0 | aux |
| `search_yaw_velocity` / `last_seen_dt_penalty` / `head_yaw_search` | shared with kick | aux |
| `receiver_alive` / `receiver_terminated` | +0.5 / −20 | aux |
| posture & reg | small | aux |

Trap task scene: single K1 + ball. Ball is initialized **outside** the
receiver's reach and given an initial velocity toward the receiver (random
direction within ±30° of receiver's forward, speed 2–5 m/s, height 0–0.5 m).
Episode length 4 s. The receiver must time its foot position to deaden the
ball.

### 4.3 Defend skill

| Term | Weight | Group |
|---|---:|---|
| `block_success` | +30 one-shot | goal | Ball touches defender body (not feet) |
| `steal_success` | +25 one-shot | goal | Defender foot near ball ∧ ball xy-speed drops > 0.5 m/s |
| `defensive_line_hold` | +5 / step × is_in_lane | goal | Defender between ball and own goal |
| `ball_pushed_away` | +10 if post-contact ball_vel · own_goal_dir < 0 | goal | Clearing reward |
| `own_goal_proximity_penalty` | −15 one-shot | goal | Ball crosses own-goal line (terminal) |
| `search_yaw_velocity` / `last_seen_dt_penalty` / `head_yaw_search` | shared | aux |
| `alive` / `terminated` | +0.5 / −20 | aux |
| posture & reg | small | aux |

Defend task scene: single K1 + ball + an own-goal frame (mirror of the
shoot goal). Ball is initialized at a random kicker position, given an
initial velocity toward own-goal (2–8 m/s, slight angle randomisation).
Episode length 5 s. Episode ends when ball crosses own-goal line, defender
falls, or timeout.

## 5. Temporal information — perception encoder

Independently of the policy network, the actor needs access to the ball's
**recent trajectory**, not just the current detection. This is essential for
trap (predict where the ball will land) and defend (predict ball arrival),
and useful for kicker robustness (FOV dropouts).

### 5.1 Step A — observation history stacking (deliver first)

```
ball_history_buffer: (history_len, num_envs, 5)
  → [perceived_ball_pos_b_xy(2), ball_mask(1), last_seen_dt(1), ball_vel_b_finite_diff(1)]

actor obs includes flattened history (history_len × 5) + current obs.
```

Defaults: `history_len = 10` (covers ~200 ms at 50 Hz). Buffer is updated
each step inside `SoccerKickCommand._update_command` (alongside the
existing `VirtualPerception.update` call). On reset, buffer is zeroed for
the reset envs.

Adds ~50 dims to the actor observation. Implementation effort: ~150 lines.

### 5.2 Step B — LVDRS-style encoder + decoder (deliver if Step A under-performs)

```
encoder = MLP(history_len × 5 → 64-dim latent)
decoder = MLP(64-dim → privileged_state_dim)  # training only
actor   = MLP(current_obs + 64-dim latent → 22-DOF action)

auxiliary loss (added to PPO update):
  L_decode = MSE(decoder(latent), [GT_ball_pos_b, GT_ball_vel_b, GT_base_lin_vel, ...])
```

The decoder is supervised against critic-side GT during training so the
encoder learns a privileged-state-recovery latent. At inference only the
encoder + actor run.

Trigger condition for going to Step B: if the Step A trap policy plateaus
below 20 % `trap_success_rate` after 10 k iterations.

LVDRS report: ball position RMSE 0.344 m → 0.186 m (1.85×) with the
decoder. Expected here:
- Trap success 8.6 % → 15–20 % from Step A, 25–30 % from Step B
- Defender block success: 30 %+ expected from Step B (Step A might suffice)
- Kicker quality: only marginal direct improvement, but better FOV-drop
  robustness

## 6. AMP corpus

Shared across all skills, since locomotion is a foundation:

```
amp_motion_files = [
    "walk.txt",        # forward walk (~7 s)
    "walk2run.txt",    # forward walk→run transition
    "run.txt",         # forward run
    "kick.txt",        # KEEP — diverse non-pure-kick motions (getup, low walk,
                       #   high knee, vault) that add useful variety
    *omni/kick/walk_kick*.txt,    # 10 GMR-retargeted walk+kick clips (~41 s total)
]
```

Decision: **keep `kick.txt` despite its non-kick content** — the variety
appears to help with recovery / unusual postures, and AMP only enforces
"human-like" as a soft style prior so noise is tolerated.

Future extension: add the `omni/forward`, `omni/backward`, `omni/lateral`,
`omni/pivot` clips when their source pkl files are available. Pivot is
particularly important for the search-yaw-velocity behaviour to look
natural.

## 7. Training stages

Each stage trains one skill in isolation. No multi-agent during stages 1–3.

```
Stage 0  Foundation: ensure AMP corpus is correct, locomotion priors load.
         (Already done; verified by Walk2Run task smoke tests.)

Stage 1  Kick Skill          Booster-Soccer-Kick-Skill-v0
         single K1 + ball + goal (used as default target).
         shoot_prob = 0.5; target overridable externally.
         8 k iter × 4096 envs, ~3.5 h on RTX 5090.
         Success metric: kick_success_rate > 50 %, goal_scored_done > 10 %
         in shoot, pass_landing > 30 % in pass.

Stage 2  Trap Skill          Booster-Soccer-Trap-Skill-v0
         single K1 + ball thrown at receiver from random direction.
         8 k iter × 4096 envs.
         Success metric: trap_success_rate > 15 % (Step A target).

Stage 3  Defend Skill        Booster-Soccer-Defend-Skill-v0
         single K1 + ball moving toward own goal.
         8 k iter × 4096 envs.
         Success metric: block_success + steal_success > 30 %, own-goal
         penalty < 30 %.

Stage 4  Composition         (no fresh training — load all 3 checkpoints)
         Validate: kicker(Stage 1) + receiver(Stage 2) on the existing
         Booster-Soccer-KickTrap-AMP-v0 scene (pass-mode kicker passes;
         receiver traps).
         Validate: kicker(Stage 1) + defender(Stage 3) — pseudo self-play.

Stage 5  Step B encoder (conditional, only if Stage 2 < 20 % trap)
         Refactor MultiCriticActorCritic → EncoderActorCritic.
         Add decoder loss to MultiCriticAMPPPO.
         Retrain Stage 1–3 with encoder.
```

## 8. Deployment / composition

```
Operator selects role at start of episode → role_flag:
  shoot  → role=kicker, mode=shoot, target=goal_center_xy, strength=8 m/s
  pass   → role=kicker, mode=pass,  target=teammate_xy, strength=f(distance)
  trap   → role=receiver, target ignored
  defend → role=defender, target=own_goal_xy
  idle   → role=idle, all targets zero

Single checkpoint runs all roles (because the policy is per-agent shared).
At runtime, set the role one-hot and (optionally) the target inputs.
```

## 8.5 Self-play perception constraint (hard requirement)

When skills are composed in self-play / multi-agent scenarios (Stage 4 and
beyond), **every robot perceives every other robot through its own virtual
head camera** — no actor-side privileged information about teammates or
adversaries is allowed. This preserves sim2real fidelity for the deployed
multi-robot setting.

Concretely:

- Each K1 in the scene owns an independent `VirtualPerception` instance
  with its own per-env DR draw (latency, noise, detection-prob, blind
  flag, FOV scale). Two robots never share a perception RNG seed within
  the same env.
- Actor observations may include:
  - own proprio
  - ball detection from **own** camera (perceived; with mask / last-seen-dt)
  - ball detection history buffer (Step A / B encoder)
  - perceived position of **other robots** that fall inside own camera FOV
    (using the same Bernoulli detection + noise + latency model, mirrored)
  - role one-hot, target inputs, mode flags
- Actor observations may NOT include:
  - GT position / velocity of any other robot
  - GT position of any teammate target — only the perceived position is
    fed to the kicker in "pass" mode at deployment, OR an externally
    commanded target xy that comes from the operator (deployment-time
    convenience). During training, the pass target is set to the perceived
    receiver if available, falling back to last-seen position; if blind,
    the kicker has to search.
- Critic observations MAY use GT (asymmetric actor-critic preserved); this
  is fine because the critic is discarded at deployment.

This means each new robot we add to a scene must also add:
1. a `VirtualPerception` instance attached to its `Head_2`,
2. observation functions that wrap that perception (e.g.,
   `kicker_perceived_b`, `defender_perceived_b`, `receiver_perceived_b`),
3. DR events that resample its perception parameters per episode.

Stages 1–3 train single-agent — no opponent observation needed. The
constraint becomes load-bearing at Stage 4. We design the obs schema now
to accommodate it: the role-one-hot already says which agent I am, and
each "other robot" is observed via the same virtual-camera pipeline.

## 9. Backward compatibility / housekeeping

- **Keep** existing V1.2 (`Booster-Soccer-Kick-AMP-MC-v0`) and V3.2
  (`Booster-Soccer-KickTrap-AMP-V2-v0`) checkpoints as baselines for
  ablation / regression checks. Do not delete.
- **New tasks** live under
  `source/booster_rl_tasks/booster_rl_tasks/tasks/manager_based/beyond_mimic/robots/k1/`:
  - `soccer_kick_skill_amp/`
  - `soccer_trap_skill_amp/`
  - `soccer_defend_skill_amp/`
- **New MDP modules**:
  - `mdp/soccer_role_obs.py` — role one-hot helper, history-buffer obs functions
  - `mdp/soccer_defend_rewards.py` — block / steal / lane-hold
  - `mdp/soccer_trap_rewards.py` — extend with `trap_anticipation`
  - `mdp/soccer_perception.py` — add a per-receiver instance hook
- **No changes** to `rsl_rl/` for Step A. Stage 5 (Step B) requires a new
  `EncoderActorCritic` + storage extension.

## 10. Open / deferred

- **Real human-like trap motion in AMP corpus** — current AMP has no
  "stand calmly and shift weight to trap" exemplar; the receiver style
  will be partially out-of-distribution for the discriminator. Acceptable
  for first delivery; revisit if trap style is clearly wrong on hardware.
- **Inter-agent communication** in multi-agent deployment (kicker tells
  receiver where it'll pass). Not addressed — receiver has to infer from
  ball trajectory alone via the encoder. Could add a "telegraph" channel
  later.
- **Hierarchical task selector** (when to shoot vs pass vs ?). Out of
  scope; operator-driven for now.
- **Goalkeeper variant** — defender with hands? Likely a Stage 6 task,
  not blocking.
- **Sim2real of the perception encoder**: the encoder is trained on the
  virtual-camera output; the real RealSense + YOLOv8 detection
  distribution may diverge. Plan to validate on real captures before
  deployment.

## 11. Decision summary (acceptance criteria)

The redesign is *accepted* iff all of the following hold:

1. ✅ Shoot policy scores ≥ 40 % of single-attempt shots within 4 s from a
   range of robot/ball spawns. Falls back to multi-attempt with goal in
   < 8 s overall.
2. ✅ Pass policy reaches the receiver target (1 m radius, 0.5–4 m/s
   speed) ≥ 50 % of episodes, with average ≤ 1.2 kicks per episode.
3. ✅ Trap policy lands the ball at the receiver's feet (0.3 m, < 0.4 m/s)
   ≥ 15 % of incoming passes.
4. ✅ Defend policy prevents ball from crossing own-goal line ≥ 70 % of
   incoming shots.
5. ✅ Single checkpoint can be loaded and run in any of the 4 roles via
   the role one-hot (no per-role checkpoint loading).
6. ✅ Search behaviour (rotate to find ball when blind) emerges in Stage 1
   training, demonstrable in viser play.
7. ✅ In any self-play / composition scenario, the actor never sees
   privileged opponent state — every other robot is detected only when
   it falls inside the agent's own virtual-camera FOV.

_Status: drafted 2026-05-22 after the V3.2 review. Implementation of all
five stages landed the same day; Stage 1 training in-flight (see §12)._

---

## 12. Implementation status (updated 2026-05-22)

| Component | Status | Code path |
|---|---|---|
| `SoccerKickCommand` extension (role + history buffer) | ✅ landed | `mdp/soccer_commands.py` |
| `soccer_role_obs.py` (role_one_hot + ball_history_flat) | ✅ landed | `mdp/soccer_role_obs.py` |
| Stage 1 Kick Skill mode-conditional rewards | ✅ landed | `mdp/soccer_rewards.py` |
| Stage 1 task module + multi-critic env cfg | ✅ landed | `robots/k1/soccer_kick_skill_amp/` |
| Stage 1 training | 🟡 in-flight | tmux `soccer:kick_skill_v0` (4096 envs × 8000 iter) |
| Stage 2 Trap command + rewards + observations | ✅ landed | `mdp/soccer_trap_commands.py`, `soccer_trap_rewards.py`, `soccer_trap_obs.py` |
| Stage 2 task module | ✅ landed | `robots/k1/soccer_trap_skill_amp/` |
| Stage 2 training | ⏳ pending GPU | will use `tmux soccer:trap_skill_v0` |
| Stage 3 Defend command + rewards + observations | ✅ landed | `mdp/soccer_defend_commands.py`, `soccer_defend_rewards.py`, `soccer_defend_obs.py` |
| Stage 3 own-goal terminations + own-goal posts asset | ✅ landed | `mdp/soccer_terminations.py`, `assets/objects/soccer.py` |
| Stage 3 task module | ✅ landed | `robots/k1/soccer_defend_skill_amp/` |
| Stage 3 training | ⏳ pending GPU | will use `tmux soccer:defend_skill_v0` |
| Stage 4 composition task | ✅ landed | `robots/k1/soccer_composition_amp/` |
| Stage 4 play_multi.py | ✅ landed | `scripts/rsl_rl/play_multi.py` |
| Stage 4 evaluation | ⏳ pending checkpoints | run after Stages 1/2 finish |
| Stage 5 EncoderActorCritic + decoder + algorithm + runner | ✅ landed (opt-in, dormant) | `rsl_rl/rsl_rl/modules/encoder_actor_critic.py`, `algorithms/encoder_amp_ppo.py`, `runners/encoder_amp_runner.py` |
| Stage 5 base agent cfg `BaseEncoderMultiCriticAMPAgentCfg` | ✅ landed | `agents/rsl_rl_ppo_cfg.py` |
| Stage 5 train.py dispatch | ✅ landed | `scripts/rsl_rl/train.py` |
| Stage 5 retraining (conditional) | ⏸️ on hold | only if Stage 2 trap_success_rate < 20 % |

**Smoke regression suite (8 envs × 1 iter, headless)** — all PASS:
* `Booster-Soccer-Kick-Skill-v0` (Stage 1) — actor 134d / critic 149d
* `Booster-Soccer-Trap-Skill-v0` (Stage 2) — actor 130d / critic 136d
* `Booster-Soccer-Defend-Skill-v0` (Stage 3) — actor 132d / critic 140d
* `Booster-Soccer-Composition-v0` (Stage 4) — actor 134d + 130d (dual)
* `Booster-Soccer-Kick-AMP-MC-v0` (V1.2 legacy) — actor 80d / critic 95d
* `Booster-Soccer-KickTrap-AMP-V2-v0` (V3.2 legacy) — actor 179d / critic 195d / action 44d

All implementation work is **non-breaking**: legacy tasks still smoke clean,
all V4 additions follow the "add new files, extend cfg with defaults" pattern.

The remaining gating items are GPU-bound training runs; no further code is
required to reach §11 acceptance criteria 1-4 (per-skill success metrics).
Criterion #5 ("Single checkpoint can be loaded and run in any of the 4 roles")
will be satisfied via the composition runner once Stages 1/2/3 finish.

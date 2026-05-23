# Soccer AMP — Results summary

What was built, what was trained, and where the artifacts live. Pair with
`soccer_amp_roadmap.md` (design) and `soccer_amp_worklog.md` (running log).

---

## Architecture delivered

```
adapter_booster_amp_lab/
├── docs/
│   ├── soccer_amp_roadmap.md        # design / staged delivery plan
│   ├── soccer_amp_v3_design.md      # multi-agent receiver/trap design
│   ├── soccer_amp_worklog.md        # running log
│   └── soccer_amp_results.md        # this file
├── source/booster_rl_tasks/booster_rl_tasks/
│   ├── assets/objects/soccer.py     # ball + goal posts + field consts
│   └── tasks/manager_based/beyond_mimic/
│       ├── mdp/
│       │   ├── soccer_commands.py      # SoccerKickCommand
│       │   ├── soccer_observations.py  # actor/critic/AMP obs
│       │   ├── soccer_rewards.py       # kicker rewards
│       │   ├── soccer_trap_rewards.py  # receiver rewards (V3.2)
│       │   ├── soccer_terminations.py  # falls + ball-out + goal
│       │   ├── soccer_events.py        # event re-exports
│       │   ├── soccer_trap_events.py   # receiver reset event
│       │   ├── soccer_perception.py    # virtual camera + latency + noise
│       │   └── soccer_curriculums.py   # progressive ball distance
│       └── robots/k1/
│           ├── soccer_kick_amp/          # V1.0/V1.1 baseline kicker
│           ├── soccer_kick_amp_mc/       # V1.2 multi-critic kicker
│           ├── soccer_kick_trap_amp/     # V3.1 passive receiver
│           └── soccer_kick_trap_amp_v2/  # V3.2 active receiver
└── rsl_rl/rsl_rl/
    ├── modules/multi_critic_actor_critic.py
    ├── storage/multi_critic_rollout_storage.py
    ├── algorithms/multi_critic_amp_ppo.py
    └── runners/multi_critic_amp_runner.py
```

## Registered tasks

| Task ID                                  | Status   | Description                                       |
|------------------------------------------|----------|---------------------------------------------------|
| `Booster-Soccer-Kick-AMP-v0`             | smoke    | single-critic baseline. V1.0/V1.1, still passes.  |
| `Booster-Soccer-Kick-AMP-MC-v0`          | trained  | **multi-critic kicker (V1.2)**.                   |
| `Booster-Soccer-KickTrap-AMP-v0`         | scaffold | kicker + *passive* receiver as moving pass target. |
| `Booster-Soccer-KickTrap-AMP-V2-v0`      | trained  | **kicker + active receiver (V3.2)**, shared 44-D policy. |

## Final-checkpoint metrics

### V1.2 — multi-critic kicker only
- Checkpoint: `logs/rsl_rl/soccer_kick_amp_mc/2026-05-21_20-42-35/model_7999.pt`
- Wall-clock: 3 h 27 m on 4096 envs × 8000 iter.

| Metric                              | Value  |
|-------------------------------------|--------|
| `kick_contact_rate`                 | 0.69   |
| `kick_success_rate`                 | 0.50   |
| `peak_kick_speed` (m/s)             | 5.3    |
| `target_progress` (m/s along dir)   | 4.3    |
| `goal_scored_done` (term fraction)  | 0.023  |

### V3.2 — kicker + active receiver (single shared policy)
- Checkpoint: `logs/rsl_rl/soccer_kick_trap_amp_v2/2026-05-22_00-15-12/model_7999.pt`
- Wall-clock: 3 h 38 m on 2048 envs × 8000 iter.

| Metric                              | Value  |
|-------------------------------------|--------|
| `kick_contact_rate`                 | 0.55   |
| `kick_success_rate`                 | 0.40   |
| `peak_kick_speed` (m/s)             | 3.87   |
| `trap_success_rate`                 | 0.086  |
| `receiver_ball_at_feet` (potential) | 0.086  |
| `goal_scored_done` (term fraction)  | 0.044  |
| `receiver_fall_height` (term frac)  | 0.082  |

## How to re-run

```bash
source /workspace/.venv/bin/activate

# Smoke test (1 iteration)
python scripts/rsl_rl/train.py --task Booster-Soccer-Kick-AMP-MC-v0      --num_envs 8    --max_iterations 1    --headless
python scripts/rsl_rl/train.py --task Booster-Soccer-KickTrap-AMP-V2-v0  --num_envs 8    --max_iterations 1    --headless

# Full training (multi-critic kicker, shoot-only)
python scripts/rsl_rl/train.py --task Booster-Soccer-Kick-AMP-MC-v0      --num_envs 4096 --max_iterations 8000 --headless

# Full training (kicker + active receiver)
python scripts/rsl_rl/train.py --task Booster-Soccer-KickTrap-AMP-V2-v0  --num_envs 2048 --max_iterations 8000 --headless

# Play / evaluate (existing play.py works with these task IDs)
python scripts/rsl_rl/play.py  --task Booster-Soccer-Kick-AMP-MC-v0      --checkpoint <path>
```

Use tmux for training so the session survives terminal disconnects:
```bash
tmux new -s soccer -d "source /workspace/.venv/bin/activate && python scripts/rsl_rl/train.py ..."
```

## What the policy can do

### V1.2 kicker (shoot-only)
1. **Search** — when virtual perception marks the ball as undetected (10 % blind episodes, dropouts when out of FOV), the policy rotates head + body until detections come in.
2. **Approach** — `ball_approach` + `foot_ball_proximity` shape the gait toward the ball; head pitch aligns with the ball elevation.
3. **Align body yaw** — `pre_kick_body_yaw_alignment` forces the body to face the kick target before contact.
4. **Plant + kick** — `support_foot_proximity` rewards the non-kicking foot to plant near the ball; the kicking foot swings.
5. **Score** — `goal_scored_reward` fires the moment the ball xy enters the 2.6 × 1.2 m goal mouth.

### V3.2 kicker + receiver
Same kicker behavior, plus:
6. **Pass with strength** — kicker chooses `target_strength` (3–8 m/s); penalty on mismatch with ball-contact speed.
7. **Receiver positions feet** — `receiver_ball_at_feet` Gaussian guides receiver foot toward where the ball will land.
8. **Receiver traps** — `trap_success` one-shot reward when the ball lands within 0.3 m of a receiver foot at < 0.4 m/s.

## Open follow-ups

- Continue V3.2 for another 4 000 iter or warm-restart with a smaller initial action noise — the trap rate (8.6 %) is still climbing.
- Apply virtual perception to the receiver as well (currently GT).
- Curriculum on `shoot_prob` (start 0.8 → 0.5).
- Two separate actor heads (kicker, receiver) instead of one 44-D head; should converge faster.
- Receiver-side AMP corpus (e.g. "stand-still / shift-weight" motion clip) so the trap policy gets motion priors too.

---

_Built 2026-05-21 → 2026-05-22, single autonomous session._

"""V4 Stage 1 — Kick Skill env config (single-agent, role-conditioned).

Layout:
  * Scene: K1 robot + soccer ball + goal posts + flat terrain (inherits from
    :mod:`soccer_kick_amp.tracking_env_cfg`).
  * Actor obs: legacy proprio + ball state + target + mode flag, PLUS
    ``role_one_hot`` (4d) and ``ball_history_flat`` (history_len × 5 d).
  * Critic obs: same as legacy MC variant.
  * Rewards: existing shaping kept + V4 mode-conditional bonuses
    (``goal_scored_quick``, ``time_to_goal_penalty``,
    ``kick_success_first_bonus``, ``kick_strength_error_shoot/_pass``,
    ``multi_kick_penalty``, ``approach_after_kick_penalty``,
    ``target_progress_pass_window``, ``ball_at_target_terminal``). The
    legacy ``goal_scored_reward`` / ``pass_landing_reward`` /
    ``kick_strength_error`` terms are dropped to avoid double-counting.
  * Term: identical to legacy single-agent kick.

Registered as ``Booster-Soccer-Kick-Skill-v0`` via the sibling ``__init__``.
"""
from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp as mdp
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_commands import (
    SoccerKickCommandCfg,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_perception import (
    VirtualPerceptionCfg,
    soccer_vision_repair_cfg,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.robots.k1.soccer_kick_amp.tracking_env_cfg import (
    ActionsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg as _LegacyObservationsCfg,
    RewardsCfg as _LegacyRewardsCfg,
    SoccerKickEnvCfg,
    SoccerSceneCfg,
    TerminationsCfg,
)


# =========================================================================
# Commands — single-agent kicker with role flag + history buffer
# =========================================================================


@configclass
class CommandsCfg:
    """Stage 1 command: shoot/pass kicker, role="kicker", history_len=10."""

    soccer_kick = SoccerKickCommandCfg(
        debug_vis=False,
        # Head camera perception (same DR envelope as the V3.3 baseline).
        perception=VirtualPerceptionCfg(),
        kick_ball_speed_thresh=1.0,
        kick_success_speed_thresh=1.5,
        # V4 — single-agent kick skill.
        receiver_name=None,
        role="kicker",
        ball_history_len=10,
        shoot_prob=0.5,
        # V4.1: widen the pass landing window so passes are physically
        # achievable. With peak_kick_speed ~6 m/s and a 6 m/s upper speed
        # window the ball can land while still moving fast; 1.5 m radius
        # accommodates the policy's first-pass aim variance.
        pass_landing_radius=1.5,
        pass_landing_speed_window=(0.5, 6.0),
        # V4.4: split strength ranges per mode.
        # Shoot starts at (8, 14) — already above the V4.1 baseline peak
        # (~8.5 m/s lifetime) but not so wide that the Gaussian power
        # gradient becomes flat. After the policy stabilises at the new
        # mean we can move to (10, 15).
        # Pass is clamped to (3, 6) so commanded strength stays within the
        # ``pass_landing_speed_window = (0.5, 6.0)`` upper bound — a pass
        # commanded faster than the landing window can never land.
        target_strength_range=(3.0, 6.0),
        shoot_target_strength_range=(8.0, 14.0),
        # V4.1: enable multi-attempt scoring in shoot mode — the kick
        # latches clear once the ball has stopped AND the foot is away, so
        # the policy can re-engage for a second shot if the first missed.
        enable_multi_attempt_shoot=True,
        multi_attempt_ball_speed_thresh=0.5,
        multi_attempt_foot_clear_dist=0.5,
    )


@configclass
class CommandsCfgV5(CommandsCfg):
    """Kick V5 command distribution: harder shoot, retained pass discipline."""

    soccer_kick = SoccerKickCommandCfg(
        debug_vis=False,
        # Hold last detection on miss so a short perception dropout does not
        # collapse the actor input to an all-zero OOD ball observation.
        perception=VirtualPerceptionCfg(
            hold_last_on_miss=True,
            blind_prob=0.05,
            max_detection_range=8.0,
            detection_prob_in_fov_range=(0.45, 0.98),
        ),
        kick_ball_speed_thresh=1.0,
        kick_success_speed_thresh=1.5,
        # V5: make shoot success strength-aware so cheap 1.5 m/s taps no
        # longer collect the same success bonus as real shots.
        shoot_success_target_fraction=0.60,
        receiver_name=None,
        role="kicker",
        ball_history_len=10,
        # Slight shoot bias for V5 because goal scoring is the bottleneck,
        # while pass still receives enough samples to keep the skill alive.
        shoot_prob=0.65,
        pass_landing_radius=1.5,
        pass_landing_speed_window=(0.5, 6.0),
        # Pass remains controlled; shoot is allowed to push into the 15 m/s
        # regime requested for realistic downstream trap/defend training.
        target_strength_range=(3.0, 6.0),
        shoot_target_strength_range=(8.0, 15.0),
        enable_multi_attempt_shoot=True,
        multi_attempt_ball_speed_thresh=0.5,
        multi_attempt_foot_clear_dist=0.5,
    )


@configclass
class CommandsCfgV51(CommandsCfgV5):
    """Kick V5.1 command distribution: contact-first, then hard-shot curriculum."""

    soccer_kick = SoccerKickCommandCfg(
        debug_vis=False,
        perception=VirtualPerceptionCfg(
            hold_last_on_miss=True,
            blind_prob=0.05,
            max_detection_range=8.0,
            detection_prob_in_fov_range=(0.45, 0.98),
        ),
        # Keep the same real-contact definition; do not inflate contact metrics
        # by counting taps below 1 m/s.
        kick_ball_speed_thresh=1.0,
        kick_success_speed_thresh=1.5,
        shoot_success_target_fraction=0.60,
        receiver_name=None,
        role="kicker",
        ball_history_len=10,
        # More shoot samples because shoot contact/goal is the bottleneck.
        shoot_prob=0.70,
        # Start closer and slightly narrower; curriculum expands distance.
        ball_spawn_distance_range=(0.45, 1.25),
        ball_spawn_angle_range=(-0.85, 0.85),
        pass_landing_radius=1.5,
        pass_landing_speed_window=(0.5, 6.0),
        target_strength_range=(3.0, 6.0),
        # V5.1 starts with achievable hard-ish shots. A curriculum raises this
        # to (12, 15) once contact has stabilized.
        shoot_target_strength_range=(8.0, 11.0),
        enable_multi_attempt_shoot=True,
        multi_attempt_ball_speed_thresh=0.5,
        multi_attempt_foot_clear_dist=0.5,
    )


@configclass
class CommandsCfgV52(CommandsCfgV51):
    """Kick V5.2 command distribution: clean approach before hard power."""

    soccer_kick = SoccerKickCommandCfg(
        debug_vis=False,
        perception=VirtualPerceptionCfg(
            hold_last_on_miss=True,
            blind_prob=0.05,
            max_detection_range=8.0,
            detection_prob_in_fov_range=(0.45, 0.98),
        ),
        kick_ball_speed_thresh=1.0,
        kick_success_speed_thresh=1.5,
        shoot_success_target_fraction=0.60,
        receiver_name=None,
        role="kicker",
        ball_history_len=10,
        shoot_prob=0.65,
        # Avoid near-foot reset states that reward accidental bumps during the
        # walk-up. Curriculum later widens this to composition-like distances.
        ball_spawn_distance_range=(0.8, 1.8),
        ball_spawn_angle_range=(-0.75, 0.75),
        pass_landing_radius=1.5,
        pass_landing_speed_window=(0.5, 6.0),
        target_strength_range=(3.0, 6.0),
        # V5.2 is a clean-strike recovery phase, not the final 15 m/s phase.
        shoot_target_strength_range=(7.0, 10.0),
        # Single-shot first-contact discipline: a bad first touch should be
        # penalized, not cleaned up by chasing a second attempt.
        enable_multi_attempt_shoot=False,
        multi_attempt_ball_speed_thresh=0.5,
        multi_attempt_foot_clear_dist=0.5,
    )


@configclass
class CommandsCfgV53(CommandsCfgV52):
    """Kick V5.3 command distribution: recover approach with easy perception."""

    soccer_kick = SoccerKickCommandCfg(
        debug_vis=False,
        perception=soccer_vision_repair_cfg(),
        kick_ball_speed_thresh=1.0,
        kick_success_speed_thresh=1.5,
        shoot_success_target_fraction=0.60,
        receiver_name=None,
        role="kicker",
        ball_history_len=50,
        shoot_prob=0.70,
        ball_spawn_distance_range=(0.55, 1.45),
        ball_spawn_angle_range=(-0.65, 0.65),
        pass_landing_radius=1.5,
        pass_landing_speed_window=(0.5, 6.0),
        target_strength_range=(3.0, 6.0),
        shoot_target_strength_range=(7.0, 10.0),
        enable_multi_attempt_shoot=False,
        multi_attempt_ball_speed_thresh=0.5,
        multi_attempt_foot_clear_dist=0.5,
    )


# =========================================================================
# Observations — actor adds role + ball history (Step A)
# =========================================================================


@configclass
class ObservationsCfg(_LegacyObservationsCfg):
    """Extends the legacy soccer-kick observation cfg with V4 additions."""

    @configclass
    class PolicyCfg(_LegacyObservationsCfg.PolicyCfg):
        # V4: per-agent role one-hot. Constant per env in Stage 1 (= kicker).
        role_one_hot = ObsTerm(
            func=mdp.soccer_role_obs.role_one_hot,
            params={"command_name": "soccer_kick"},
        )
        # V4 Step A: temporal information — 10-frame perceived ball history.
        ball_history = ObsTerm(
            func=mdp.soccer_role_obs.ball_history_flat,
            params={"command_name": "soccer_kick"},
            clip=(-30.0, 30.0),
        )

    @configclass
    class PrivilegedCfg(_LegacyObservationsCfg.PrivilegedCfg):
        # Mirror role/history into critic for symmetric value-function input.
        role_one_hot = ObsTerm(
            func=mdp.soccer_role_obs.role_one_hot,
            params={"command_name": "soccer_kick"},
        )
        ball_history = ObsTerm(
            func=mdp.soccer_role_obs.ball_history_flat,
            params={"command_name": "soccer_kick"},
            clip=(-30.0, 30.0),
        )

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


# =========================================================================
# Rewards — extend with V4 mode-conditional shaping
# =========================================================================


@configclass
class RewardsCfg(_LegacyRewardsCfg):
    """V4.1 Stage 1 rewards = rebalanced shaping + mode-conditional bonuses.

    Key V4.1 changes versus V4.0:
      * ``target_progress`` (mode-agnostic, was weight 8) is disabled and
        replaced by :func:`target_progress_shoot` (is_shoot-gated, weight 2).
        This stops pass-mode envs from getting a continuous shaping reward
        that competes with the single-shot ``ball_at_target_terminal``.
      * Added :func:`kick_aim_at_goal_shoot` — continuous shoot-mode reward
        on ``cos(ball_vel, ball→goal_center)``, so the policy is rewarded
        for aiming at the *center* of the goal not just any point on the
        goal line.
      * ``kick_angle_error`` weight -1 → -5 (stronger angle penalty).
      * ``kick_strength_error_shoot`` weight -0.5 → -1.5 (more strength
        precision).
      * ``goal_scored_quick`` base 30+50*remaining → 50+150*remaining
        (4× larger terminal bonus).
      * ``time_to_goal_penalty`` -0.02 → -0.05 (more urgency).
      * Pass landing window widened to ``(0.5, 6.0)`` m/s + radius 1.5 m
        (see ``CommandsCfg``), making first-shot passes physically
        reachable given peak_kick_speed ~6 m/s.
      * Legacy ``goal_scored`` / ``pass_landing`` / ``kick_strength_error``
        kept at weight 0 (replaced by V4 variants).
    """

    # ---- V4.1 mode-gated continuous shaping ----
    target_progress_shoot = RewTerm(
        func=mdp.soccer_rewards.target_progress_shoot,
        weight=2.0,
        params={"command_name": "soccer_kick"},
    )
    # V4.2: boost aim precision — V4.1 plateaued at goal_done 14 % because
    # the policy aimed ~67° off the goal center on average (cos_angle ~0.39).
    # Tripling the weight pushes the policy toward better aim convergence.
    kick_aim_at_goal_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_aim_at_goal_shoot,
        weight=10.0,
        params={"command_name": "soccer_kick", "min_ball_speed": 1.0},
    )
    # V4.2: new pass-mode aim precision reward — same shape as the shoot
    # version but oriented at the *pass target* (live receiver xy).
    kick_aim_at_pass_target_pass = RewTerm(
        func=mdp.soccer_rewards.kick_aim_at_pass_target_pass,
        weight=8.0,
        params={"command_name": "soccer_kick", "min_ball_speed": 1.0},
    )
    # V4.2: new strength-tracking rewards. V4.1 ``kick_strength_error_*``
    # only fired on the rising edge of ``kick_contact_new`` (1 step per
    # contact), giving per-episode contribution ~-0.005 — far too small to
    # shape behaviour. peak_kick_speed plateaued at ~5 m/s regardless of
    # commanded strength (range 3-8 m/s). The new ``kick_power_track_*``
    # rewards fire every step in the 30/15-step post-kick window with a
    # Gaussian centered on the commanded strength, so the policy gets a
    # dense gradient pulling the peak speed toward the command.
    # V4.4: widen shoot sigma 1.5 → 2.5 since the commanded strength range
    # is now (8, 14) — a Gaussian with σ=1.5 would give vanishing gradient
    # at peak~8 when cmd=14 (err=6, exp(-16) ≈ 1e-7). σ=2.5 keeps the
    # gradient non-negligible across the new wider range.
    kick_power_track_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_power_track_shoot,
        weight=5.0,
        params={"command_name": "soccer_kick", "sigma": 2.5, "window_steps": 30},
    )
    kick_power_track_pass = RewTerm(
        func=mdp.soccer_rewards.kick_power_track_pass,
        weight=6.0,
        params={"command_name": "soccer_kick", "sigma": 1.0, "window_steps": 15},
    )

    # ---- V4 mode-conditional rewards (re-tuned weights) ----
    goal_scored_quick = RewTerm(
        func=mdp.soccer_rewards.goal_scored_quick,
        weight=1.0,  # magnitude lives inside the function (base + time_bonus * remaining)
        params={
            "command_name": "soccer_kick",
            "base_bonus": 50.0,
            "time_bonus": 150.0,
        },
    )
    time_to_goal_penalty = RewTerm(
        func=mdp.soccer_rewards.time_to_goal_penalty,
        weight=-0.05,
        params={"command_name": "soccer_kick"},
    )
    kick_success_first_bonus = RewTerm(
        func=mdp.soccer_rewards.kick_success_first_bonus,
        weight=1.0,
        params={
            "command_name": "soccer_kick",
            "first_bonus": 10.0,
            "repeat_bonus": 3.0,
        },
    )
    # V4.2: boost edge penalty too — the rising-edge contribution stays
    # small per episode (~-0.04 with the new weight) but now reinforces
    # the dense ``kick_power_track_*`` signal at impact time.
    kick_strength_error_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_strength_error_shoot,
        weight=-5.0,
        params={"command_name": "soccer_kick"},
    )
    kick_strength_error_pass = RewTerm(
        func=mdp.soccer_rewards.kick_strength_error_pass,
        weight=-5.0,
        params={"command_name": "soccer_kick"},
    )
    # V4.3: new mode-conditional angle precision penalties. Replace the
    # legacy ``kick_angle_error`` (which was disabled below). Speed-
    # independent (``angle²`` only) so the policy isn't punished for
    # kicking harder; live ball→target direction so it stays aligned with
    # the corresponding ``kick_aim_at_*`` positive reward (no off-axis
    # double-counting from a stale spawn-time direction).
    kick_angle_error_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_angle_error_shoot,
        weight=-3.0,
        params={
            "command_name": "soccer_kick",
            "min_ball_speed": 1.0,
            "window_steps": 15,
        },
    )
    kick_angle_error_pass = RewTerm(
        func=mdp.soccer_rewards.kick_angle_error_pass,
        weight=-3.0,
        params={
            "command_name": "soccer_kick",
            "min_ball_speed": 1.0,
            "window_steps": 15,
        },
    )
    multi_kick_penalty = RewTerm(
        func=mdp.soccer_rewards.multi_kick_penalty,
        weight=-10.0,
        params={"command_name": "soccer_kick"},
    )
    approach_after_kick_penalty = RewTerm(
        func=mdp.soccer_rewards.approach_after_kick_penalty,
        weight=-3.0,
        params={"command_name": "soccer_kick"},
    )
    target_progress_pass_window = RewTerm(
        func=mdp.soccer_rewards.target_progress_pass_window,
        weight=4.0,
        params={"command_name": "soccer_kick", "window_steps": 20},
    )
    ball_at_target_terminal = RewTerm(
        func=mdp.soccer_rewards.ball_at_target_terminal,
        weight=20.0,
        params={"command_name": "soccer_kick", "radius": 1.5},
    )

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        # Disable legacy terms that the V4 variants replace, to avoid double-
        # counting. Zeroing the weight is sufficient — the reward manager
        # still computes the term but it contributes 0 to the total.
        self.goal_scored.weight = 0.0
        self.pass_landing.weight = 0.0
        self.kick_strength_error.weight = 0.0
        # V4.1: disable mode-agnostic target_progress; the shoot-only
        # variant ``target_progress_shoot`` takes over for shoot mode and
        # ``target_progress_pass_window`` (windowed) handles pass mode.
        self.target_progress.weight = 0.0
        # V4.3: disable legacy speed-dependent ``kick_angle_error``. The
        # ``angle² × ball_xy_speed`` formula caused the V4.2 plateau (policy
        # kicked weaker to reduce angle penalty). Replaced with
        # mode-conditional speed-independent variants
        # ``kick_angle_error_shoot`` / ``kick_angle_error_pass`` above.
        self.kick_angle_error.weight = 0.0


@configclass
class RewardsCfgV5(RewardsCfg):
    """Kick V5 rewards: hard shoot power, recovery, and lost-ball robustness."""

    target_progress_shoot = RewTerm(
        func=mdp.soccer_rewards.target_progress_shoot,
        weight=1.5,
        params={"command_name": "soccer_kick"},
    )
    kick_aim_at_goal_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_aim_at_goal_shoot,
        weight=6.0,
        params={"command_name": "soccer_kick", "min_ball_speed": 1.0},
    )
    kick_power_track_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_power_track_shoot,
        weight=4.0,
        params={"command_name": "soccer_kick", "sigma": 4.0, "window_steps": 45},
    )
    kick_power_progress_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_power_progress_shoot,
        weight=8.0,
        params={"command_name": "soccer_kick", "floor_speed": 3.0, "window_steps": 50},
    )
    shoot_goal_line_speed = RewTerm(
        func=mdp.soccer_rewards.shoot_goal_line_speed,
        weight=5.0,
        params={"command_name": "soccer_kick", "floor_speed": 3.0, "window_steps": 50},
    )
    shoot_underpowered_penalty = RewTerm(
        func=mdp.soccer_rewards.shoot_underpowered_penalty,
        weight=-1.5,
        params={
            "command_name": "soccer_kick",
            "target_fraction": 0.85,
            "window_steps": 45,
        },
    )
    shoot_power_hard_bounded = RewTerm(
        func=mdp.soccer_rewards.shoot_power_hard_bounded,
        weight=2.0,
        params={
            "command_name": "soccer_kick",
            "min_margin": 0.0,
            "saturation_margin": 3.5,
            "window_steps": 45,
            "min_goal_alignment": 0.25,
            "min_height": 0.42,
            "max_tilt": 0.65,
        },
    )
    goal_scored_quick = RewTerm(
        func=mdp.soccer_rewards.goal_scored_quick,
        weight=1.0,
        params={
            "command_name": "soccer_kick",
            "base_bonus": 80.0,
            "time_bonus": 250.0,
        },
    )
    kick_success_first_bonus = RewTerm(
        func=mdp.soccer_rewards.kick_success_first_bonus,
        weight=1.0,
        params={
            "command_name": "soccer_kick",
            "first_bonus": 15.0,
            "repeat_bonus": 4.0,
        },
    )
    # High shoot commands make the old edge abs-error penalty counterproductive
    # early in training. Keep a tiny nudge; let dense positive power terms lead.
    kick_strength_error_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_strength_error_shoot,
        weight=-0.3,
        params={"command_name": "soccer_kick"},
    )
    kick_angle_error_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_angle_error_shoot,
        weight=-2.0,
        params={
            "command_name": "soccer_kick",
            "min_ball_speed": 1.0,
            "window_steps": 15,
        },
    )

    post_kick_recovery_stability = RewTerm(
        func=mdp.soccer_rewards.post_kick_recovery_stability,
        weight=1.5,
        params={
            "command_name": "soccer_kick",
            "mode": "any",
            "start_step": 5,
            "window_steps": 90,
            "min_ball_speed": 0.5,
        },
    )
    post_kick_no_fall_bounded = RewTerm(
        func=mdp.soccer_rewards.post_kick_no_fall_bounded,
        weight=0.8,
        params={
            "command_name": "soccer_kick",
            "mode": "any",
            "start_step": 0,
            "window_steps": 90,
            "min_height": 0.42,
            "max_tilt": 0.65,
        },
    )
    post_kick_fall_risk_penalty = RewTerm(
        func=mdp.soccer_rewards.post_kick_fall_risk_penalty,
        weight=-3.0,
        params={
            "command_name": "soccer_kick",
            "start_step": 0,
            "end_step": 90,
            "min_height": 0.42,
        },
    )

    lost_ball_search_bounded = RewTerm(
        func=mdp.soccer_rewards.lost_ball_search_bounded,
        weight=2.0,
        params={
            "command_name": "soccer_kick",
            "mode": "any",
            "min_lost_dt": 0.10,
            "max_ball_speed": 2.0,
        },
    )
    lost_ball_freeze_penalty_bounded = RewTerm(
        func=mdp.soccer_rewards.lost_ball_freeze_penalty_bounded,
        weight=-2.0,
        params={
            "command_name": "soccer_kick",
            "mode": "any",
            "min_lost_dt": 0.15,
            "max_ball_speed": 2.0,
        },
    )
    lost_ball_reacquire_bonus = RewTerm(
        func=mdp.soccer_rewards.lost_ball_reacquire_bonus,
        weight=2.0,
        params={
            "command_name": "soccer_kick",
            "mode": "any",
            "min_lost_dt": 0.15,
            "max_ball_speed": 2.5,
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # Re-enable immediate pass landing feedback; the terminal-only reward
        # was too delayed for one-shot pass precision.
        self.pass_landing.weight = 20.0
        # Prefer bounded V5 lost-ball terms over legacy unbounded spin rewards.
        self.search_yaw_velocity.weight = 0.0
        self.last_seen_dt_penalty.weight = 0.0
        self.head_yaw_search.weight = 0.0


@configclass
class RewardsCfgV51(RewardsCfgV5):
    """Kick V5.1 rewards: recover contact/approach first, retain V5 safety."""

    ball_approach = RewTerm(
        func=mdp.soccer_rewards.ball_approach_progress,
        weight=6.0,
        params={"command_name": "soccer_kick", "max_delta": 0.05},
    )
    foot_ball_proximity = RewTerm(
        func=mdp.soccer_rewards.foot_ball_proximity,
        weight=8.0,
        params={"command_name": "soccer_kick", "sigma": 0.45},
    )
    kick_contact = RewTerm(
        func=mdp.soccer_rewards.kick_contact_bonus,
        weight=24.0,
        params={"command_name": "soccer_kick"},
    )
    visible_pre_kick_time = RewTerm(
        func=mdp.soccer_rewards.visible_pre_kick_time,
        weight=-0.15,
        params={"command_name": "soccer_kick", "require_visible": True},
    )
    close_pre_kick_penalty = RewTerm(
        func=mdp.soccer_rewards.foot_ball_proximity,
        weight=-6.0,
        params={"command_name": "soccer_kick", "sigma": 0.45},
    )
    target_progress_shoot = RewTerm(
        func=mdp.soccer_rewards.target_progress_shoot,
        weight=1.0,
        params={"command_name": "soccer_kick"},
    )
    kick_power_progress_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_power_progress_shoot,
        weight=6.0,
        params={"command_name": "soccer_kick", "floor_speed": 3.0, "window_steps": 50},
    )
    shoot_goal_line_speed = RewTerm(
        func=mdp.soccer_rewards.shoot_goal_line_speed,
        weight=4.0,
        params={"command_name": "soccer_kick", "floor_speed": 3.0, "window_steps": 50},
    )
    shoot_underpowered_penalty = RewTerm(
        func=mdp.soccer_rewards.shoot_underpowered_penalty,
        weight=-1.0,
        params={
            "command_name": "soccer_kick",
            "target_fraction": 0.80,
            "window_steps": 45,
        },
    )


@configclass
class RewardsCfgV52(RewardsCfgV51):
    """Kick V5.2 rewards: clean first touch, target-facing approach, reacquire."""

    ball_approach = RewTerm(
        func=mdp.soccer_rewards.ball_approach_progress,
        weight=8.0,
        params={"command_name": "soccer_kick", "max_delta": 0.05},
    )
    foot_ball_proximity = RewTerm(
        func=mdp.soccer_rewards.foot_ball_proximity,
        weight=4.0,
        params={"command_name": "soccer_kick", "sigma": 0.35},
    )
    close_pre_kick_penalty = RewTerm(
        func=mdp.soccer_rewards.foot_ball_proximity,
        weight=-4.0,
        params={"command_name": "soccer_kick", "sigma": 0.35},
    )
    # Raw contact remains visible in logs but is no longer the main objective.
    kick_contact = RewTerm(
        func=mdp.soccer_rewards.kick_contact_bonus,
        weight=3.0,
        params={"command_name": "soccer_kick"},
    )
    clean_kick_contact = RewTerm(
        func=mdp.soccer_rewards.clean_kick_contact_bonus,
        weight=35.0,
        params={
            "command_name": "soccer_kick",
            "target_fraction": 0.45,
            "min_projected_speed": 2.0,
            "min_alignment": 0.55,
            "min_body_target_cos": 0.35,
        },
    )
    unclean_kick_contact_penalty = RewTerm(
        func=mdp.soccer_rewards.unclean_kick_contact_penalty,
        weight=-25.0,
        params={
            "command_name": "soccer_kick",
            "target_fraction": 0.45,
            "min_projected_speed": 2.0,
            "min_alignment": 0.55,
            "min_body_target_cos": 0.35,
        },
    )
    pre_kick_swing_miss_penalty = RewTerm(
        func=mdp.soccer_rewards.pre_kick_swing_miss_penalty,
        weight=-1.5,
        params={
            "command_name": "soccer_kick",
            "near_radius": 0.45,
            "foot_speed_thresh": 1.2,
            "max_ball_speed": 0.8,
        },
    )
    visible_pre_kick_time = RewTerm(
        func=mdp.soccer_rewards.visible_pre_kick_time,
        weight=-0.05,
        # Penalize delayed first touch even when perception is lost. Otherwise
        # the policy can avoid time pressure by looking away from the ball.
        params={"command_name": "soccer_kick", "require_visible": False},
    )
    pre_kick_yaw_align = RewTerm(
        func=mdp.soccer_rewards.pre_kick_body_yaw_alignment,
        weight=-1.5,
        params={"command_name": "soccer_kick"},
    )
    head_yaw_align = RewTerm(
        func=mdp.soccer_rewards.head_yaw_alignment_to_ball,
        weight=-0.8,
        params={"command_name": "soccer_kick"},
    )
    head_pitch_align = RewTerm(
        func=mdp.soccer_rewards.head_pitch_alignment_to_ball,
        weight=-0.5,
        params={"command_name": "soccer_kick"},
    )
    support_foot = RewTerm(
        func=mdp.soccer_rewards.support_foot_proximity,
        weight=1.0,
        params={"command_name": "soccer_kick"},
    )
    kick_power_progress_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_power_progress_shoot,
        weight=4.0,
        params={"command_name": "soccer_kick", "floor_speed": 3.0, "window_steps": 45},
    )
    shoot_goal_line_speed = RewTerm(
        func=mdp.soccer_rewards.shoot_goal_line_speed,
        weight=3.0,
        params={"command_name": "soccer_kick", "floor_speed": 3.0, "window_steps": 45},
    )
    shoot_underpowered_penalty = RewTerm(
        func=mdp.soccer_rewards.shoot_underpowered_penalty,
        weight=-0.5,
        params={
            "command_name": "soccer_kick",
            "target_fraction": 0.70,
            "window_steps": 40,
        },
    )
    lost_ball_search_bounded = RewTerm(
        func=mdp.soccer_rewards.lost_ball_search_bounded,
        weight=1.0,
        params={
            "command_name": "soccer_kick",
            "mode": "any",
            "min_lost_dt": 0.12,
            "max_ball_speed": 2.0,
            "allow_post_kick": True,
        },
    )
    lost_ball_freeze_penalty_bounded = RewTerm(
        func=mdp.soccer_rewards.lost_ball_freeze_penalty_bounded,
        weight=-4.0,
        params={
            "command_name": "soccer_kick",
            "mode": "any",
            "min_lost_dt": 0.12,
            "max_ball_speed": 2.0,
            "allow_post_kick": True,
        },
    )
    lost_ball_reacquire_bonus = RewTerm(
        func=mdp.soccer_rewards.lost_ball_reacquire_bonus,
        weight=18.0,
        params={
            "command_name": "soccer_kick",
            "mode": "any",
            "min_lost_dt": 0.12,
            "max_ball_speed": 3.0,
            "allow_post_kick": True,
        },
    )
    lost_ball_time_penalty = RewTerm(
        func=mdp.soccer_rewards.last_seen_dt_penalty,
        weight=-1.5,
        params={"command_name": "soccer_kick", "max_dt": 2.0},
    )


@configclass
class RewardsCfgV53(RewardsCfgV52):
    """Kick V5.3 rewards: repair foot placement and memory-guided search."""

    ball_approach = RewTerm(
        func=mdp.soccer_rewards.ball_approach_progress,
        weight=30.0,
        params={"command_name": "soccer_kick", "max_delta": 0.05, "recent_visible_dt": 0.25},
    )
    goal_progress_shoot = RewTerm(
        func=mdp.soccer_rewards.goal_progress_shoot,
        weight=55.0,
        params={
            "command_name": "soccer_kick",
            "max_delta": 0.10,
            "max_backslide": 0.04,
            "min_ball_speed": 0.35,
        },
    )
    target_progress_shoot = RewTerm(
        func=mdp.soccer_rewards.target_progress_shoot,
        weight=0.0,
        params={"command_name": "soccer_kick"},
    )
    target_progress_pass_window = RewTerm(
        func=mdp.soccer_rewards.target_progress_pass_window,
        weight=3.0,
        params={"command_name": "soccer_kick", "window_steps": 24},
    )
    goal_scored_quick = RewTerm(
        func=mdp.soccer_rewards.goal_scored_quick,
        weight=1.0,
        params={
            "command_name": "soccer_kick",
            "base_bonus": 60.0,
            "time_bonus": 180.0,
        },
    )
    kick_aim_at_goal_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_aim_at_goal_shoot,
        weight=0.0,
        params={"command_name": "soccer_kick", "min_ball_speed": 1.0},
    )
    kick_aim_at_pass_target_pass = RewTerm(
        func=mdp.soccer_rewards.kick_aim_at_pass_target_pass,
        weight=2.0,
        params={"command_name": "soccer_kick", "min_ball_speed": 1.0},
    )
    kick_angle_error_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_angle_error_shoot,
        weight=-1.0,
        params={
            "command_name": "soccer_kick",
            "min_ball_speed": 1.0,
            "window_steps": 15,
        },
    )
    kick_angle_error_pass = RewTerm(
        func=mdp.soccer_rewards.kick_angle_error_pass,
        weight=-1.0,
        params={
            "command_name": "soccer_kick",
            "min_ball_speed": 1.0,
            "window_steps": 15,
        },
    )
    kick_power_track_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_power_track_shoot,
        weight=1.5,
        params={"command_name": "soccer_kick", "sigma": 3.0, "window_steps": 45},
    )
    kick_power_track_pass = RewTerm(
        func=mdp.soccer_rewards.kick_power_track_pass,
        weight=2.0,
        params={"command_name": "soccer_kick", "sigma": 1.2, "window_steps": 20},
    )
    kick_power_progress_shoot = RewTerm(
        func=mdp.soccer_rewards.kick_power_progress_shoot,
        weight=2.0,
        params={"command_name": "soccer_kick", "floor_speed": 3.0, "window_steps": 45},
    )
    shoot_goal_line_speed = RewTerm(
        func=mdp.soccer_rewards.shoot_goal_line_speed,
        weight=0.0,
        params={"command_name": "soccer_kick", "floor_speed": 3.0, "window_steps": 45},
    )
    shoot_underpowered_penalty = RewTerm(
        func=mdp.soccer_rewards.shoot_underpowered_penalty,
        weight=-0.5,
        params={
            "command_name": "soccer_kick",
            "target_fraction": 0.70,
            "window_steps": 40,
        },
    )
    # V5.2 accidentally canceled the foot-to-ball shaping by using the same
    # proximity function with +4 and -4 weights. Keep a strong approach signal
    # and only penalize pathological near-overlap.
    foot_ball_proximity = RewTerm(
        func=mdp.soccer_rewards.foot_ball_proximity,
        weight=4.0,
        params={"command_name": "soccer_kick", "sigma": 0.45, "recent_visible_dt": 0.25},
    )
    close_pre_kick_penalty = RewTerm(
        func=mdp.soccer_rewards.foot_ball_proximity,
        weight=-1.0,
        params={"command_name": "soccer_kick", "sigma": 0.18},
    )
    kick_contact = RewTerm(
        func=mdp.soccer_rewards.kick_contact_bonus,
        weight=2.0,
        params={"command_name": "soccer_kick"},
    )
    clean_kick_contact = RewTerm(
        func=mdp.soccer_rewards.clean_kick_contact_bonus,
        weight=0.0,
        params={
            "command_name": "soccer_kick",
            "target_fraction": 0.45,
            "min_projected_speed": 2.0,
            "min_alignment": 0.55,
            "min_body_target_cos": 0.35,
            "recent_visible_dt": 0.25,
        },
    )
    unclean_kick_contact_penalty = RewTerm(
        func=mdp.soccer_rewards.unclean_kick_contact_penalty,
        weight=0.0,
        params={
            "command_name": "soccer_kick",
            "target_fraction": 0.45,
            "min_projected_speed": 2.0,
            "min_alignment": 0.55,
            "min_body_target_cos": 0.35,
            "recent_visible_dt": 0.25,
        },
    )
    arch_contact_lateral = RewTerm(
        func=mdp.soccer_rewards.arch_contact_lateral_bonus,
        weight=20.0,
        params={
            "command_name": "soccer_kick",
            "min_lateral_speed": 0.35,
            "saturation_speed": 2.5,
            "recent_visible_dt": 0.25,
        },
    )
    toe_poke_contact_penalty = RewTerm(
        func=mdp.soccer_rewards.toe_poke_contact_penalty,
        weight=-12.0,
        params={
            "command_name": "soccer_kick",
            "min_forward_speed": 0.30,
            "saturation_speed": 1.8,
            "recent_visible_dt": 0.25,
        },
    )
    visible_pre_kick_time = RewTerm(
        func=mdp.soccer_rewards.visible_pre_kick_time,
        weight=-0.02,
        params={"command_name": "soccer_kick", "require_visible": False},
    )
    lost_ball_last_seen_search = RewTerm(
        func=mdp.soccer_rewards.lost_ball_last_seen_search,
        weight=5.0,
        params={
            "command_name": "soccer_kick",
            "mode": "any",
            "min_lost_dt": 0.08,
            "max_last_seen_dt": 1.20,
            "max_ball_speed": 2.0,
            "allow_post_kick": False,
        },
    )
    # Generic sweep is a fallback after memory is stale; keep it weaker so the
    # policy does not learn to look away and farm reacquisition events.
    lost_ball_search_bounded = RewTerm(
        func=mdp.soccer_rewards.lost_ball_search_bounded,
        weight=1.0,
        params={
            "command_name": "soccer_kick",
            "mode": "any",
            "min_lost_dt": 0.35,
            "max_ball_speed": 2.0,
            "allow_post_kick": False,
        },
    )
    lost_ball_freeze_penalty_bounded = RewTerm(
        func=mdp.soccer_rewards.lost_ball_freeze_penalty_bounded,
        weight=-4.0,
        params={
            "command_name": "soccer_kick",
            "mode": "any",
            "min_lost_dt": 0.12,
            "max_ball_speed": 2.0,
            "allow_post_kick": False,
        },
    )
    lost_ball_reacquire_bonus = RewTerm(
        func=mdp.soccer_rewards.lost_ball_reacquire_bonus,
        weight=4.0,
        params={
            "command_name": "soccer_kick",
            "mode": "any",
            "min_lost_dt": 0.12,
            "max_ball_speed": 3.0,
            "allow_post_kick": False,
        },
    )
    lost_ball_time_penalty = RewTerm(
        func=mdp.soccer_rewards.pre_kick_last_seen_dt_penalty,
        weight=-2.0,
        params={"command_name": "soccer_kick", "max_dt": 2.0},
    )

    def __post_init__(self):
        super().__post_init__()
        # V5.3 follows the LVDRS reward shape more closely: goal progress and
        # contact style are primary, while old overlapping precision terms stay
        # weak or disabled.
        self.support_foot.weight = 0.0
        self.kick_strength_error_pass.weight = -1.0
        self.pre_kick_yaw_align.weight = -1.0
        self.head_yaw_align.weight = -0.8
        self.head_pitch_align.weight = -0.7
        self.shoot_power_hard_bounded.weight = 1.0


@configclass
class CurriculumCfgV51(CurriculumCfg):
    """V5.1 contact-first curricula."""

    ball_distance = CurrTerm(
        func=mdp.soccer_curriculums.ball_distance_curriculum,
        params={
            "command_name": "soccer_kick",
            "lo_start": 0.45,
            "hi_start": 1.25,
            "lo_final": 0.8,
            "hi_final": 2.5,
            # common_step_counter advances once per vectorized env step. With
            # the PPO rollout length of 24, this reaches final near iter 2500.
            "num_steps_to_final": 2500 * 24,
        },
    )
    shoot_strength = CurrTerm(
        func=mdp.soccer_curriculums.shoot_strength_curriculum,
        params={
            "command_name": "soccer_kick",
            "lo_start": 8.0,
            "hi_start": 11.0,
            "lo_final": 12.0,
            "hi_final": 15.0,
            # Reach 12-15 m/s near iter 3500 instead of staying at 9.5 m/s.
            "num_steps_to_final": 3500 * 24,
        },
    )


@configclass
class CurriculumCfgV52(CurriculumCfg):
    """V5.2 clean approach curricula."""

    ball_distance = CurrTerm(
        func=mdp.soccer_curriculums.ball_distance_curriculum,
        params={
            "command_name": "soccer_kick",
            "lo_start": 0.8,
            "hi_start": 1.8,
            "lo_final": 1.0,
            "hi_final": 2.8,
            "num_steps_to_final": 2500 * 24,
        },
    )
    shoot_strength = CurrTerm(
        func=mdp.soccer_curriculums.shoot_strength_curriculum,
        params={
            "command_name": "soccer_kick",
            "lo_start": 7.0,
            "hi_start": 10.0,
            "lo_final": 10.0,
            "hi_final": 13.0,
            "num_steps_to_final": 3000 * 24,
        },
    )


@configclass
class CurriculumCfgV53(CurriculumCfg):
    """V5.3 LVDRS-style approach-first curricula."""

    ball_distance = CurrTerm(
        func=mdp.soccer_curriculums.ball_distance_curriculum,
        params={
            "command_name": "soccer_kick",
            "lo_start": 0.55,
            "hi_start": 1.45,
            "lo_final": 0.9,
            "hi_final": 2.6,
            "num_steps_to_final": 2500 * 24,
        },
    )
    shoot_strength = CurrTerm(
        func=mdp.soccer_curriculums.shoot_strength_curriculum,
        params={
            "command_name": "soccer_kick",
            "lo_start": 7.0,
            "hi_start": 10.0,
            "lo_final": 11.0,
            "hi_final": 14.0,
            "num_steps_to_final": 3500 * 24,
        },
    )


# =========================================================================
# Env config
# =========================================================================


@configclass
class SoccerKickSkillEnvCfg(SoccerKickEnvCfg):
    """V4 Stage 1 env cfg — adds role/history obs + mode-conditional rewards."""

    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()


@configclass
class SoccerKickSkillV5EnvCfg(SoccerKickSkillEnvCfg):
    """Kick V5 env cfg — hard-shoot curriculum with recovery/search fixes."""

    commands: CommandsCfgV5 = CommandsCfgV5()
    rewards: RewardsCfgV5 = RewardsCfgV5()


@configclass
class SoccerKickSkillV51EnvCfg(SoccerKickSkillEnvCfg):
    """Kick V5.1 env cfg — contact-first phase before hard-power push."""

    commands: CommandsCfgV51 = CommandsCfgV51()
    rewards: RewardsCfgV51 = RewardsCfgV51()
    curriculum: CurriculumCfgV51 = CurriculumCfgV51()


@configclass
class SoccerKickSkillV52EnvCfg(SoccerKickSkillEnvCfg):
    """Kick V5.2 env cfg — clean approach and reacquisition repair phase."""

    commands: CommandsCfgV52 = CommandsCfgV52()
    rewards: RewardsCfgV52 = RewardsCfgV52()
    curriculum: CurriculumCfgV52 = CurriculumCfgV52()


@configclass
class SoccerKickSkillV53EnvCfg(SoccerKickSkillEnvCfg):
    """Kick V5.3 env cfg — LVDRS-style history latent, rewards, and perception."""

    commands: CommandsCfgV53 = CommandsCfgV53()
    rewards: RewardsCfgV53 = RewardsCfgV53()
    curriculum: CurriculumCfgV53 = CurriculumCfgV53()

"""V3.2 soccer kick+trap env config — kicker + receiver, both controlled.

Builds on the V3.1 ``soccer_kick_trap_amp`` env-cfg (two K1 articulations
under sibling prim paths, receiver xy reset via ``reset_receiver_to_default``,
pass target tracks the receiver's world xy) and adds:

* a second action term ``joint_pos_receiver`` driving the receiver — combined
  with the existing kicker ``joint_pos`` this gives a 44-dim joint action;
* receiver-side observations in both PolicyCfg & PrivilegedCfg (proprio + ball
  pose in receiver yaw frame + direction to kicker);
* receiver trap rewards (foot-ball Gaussian, face-ball penalty, trap-success
  one-shot bonus, survival, terminated penalty);
* receiver fall-height / fall-tilt terminations.

The AMP discriminator group stays kicker-only (single-robot motion priors).
"""
from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

import booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp as mdp
from booster_rl_tasks.tasks.manager_based.beyond_mimic.robots.k1.soccer_kick_trap_amp.tracking_env_cfg import (
    ActionsCfg as V31ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg as V31EventCfg,
    ObservationsCfg as V31ObservationsCfg,
    RewardsCfg as V31RewardsCfg,
    SoccerKickTrapEnvCfg,
    SoccerKickTrapSceneCfg,
    TerminationsCfg as V31TerminationsCfg,
)


# =========================================================================
# Actions — V3.1 kicker term + NEW receiver term (44-dim total)
# =========================================================================


@configclass
class ActionsCfgV2(V31ActionsCfg):
    """Adds a receiver joint-position action term to the kicker action term."""

    joint_pos_receiver = mdp.JointPositionActionCfg(
        asset_name="receiver", joint_names=[".*"], use_default_offset=False
    )


# =========================================================================
# Observations — V3.1 stack + receiver-side proprio / task-conditioning
# =========================================================================


@configclass
class ObservationsCfgV2(V31ObservationsCfg):
    """Adds receiver observations to PolicyCfg and PrivilegedCfg.

    The new actor terms give the policy enough state to act as the trapper
    via the receiver half of the 44-dim joint action vector. AMP obs are
    untouched (single-robot discriminator, kicker only).
    """

    @configclass
    class PolicyCfg(V31ObservationsCfg.PolicyCfg):
        # ----- receiver proprio (noisy, mirroring kicker terms) ------------
        receiver_base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("receiver")},
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        receiver_projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("receiver")},
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        receiver_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("receiver")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        receiver_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("receiver")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        receiver_actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "joint_pos_receiver"},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # ----- receiver task-conditioning (GT ball / kicker pose) ----------
        receiver_ball_pos_b = ObsTerm(
            func=mdp.soccer_observations.receiver_ball_pos_b,
            params={"command_name": "soccer_kick"},
            clip=(-30.0, 30.0),
            scale=1.0,
        )
        receiver_ball_mask = ObsTerm(
            func=mdp.soccer_observations.receiver_ball_mask,
            params={"command_name": "soccer_kick"},
        )
        receiver_kicker_dir_b = ObsTerm(
            func=mdp.soccer_observations.receiver_kicker_dir_b,
            params={"command_name": "soccer_kick"},
        )

    @configclass
    class PrivilegedCfg(V31ObservationsCfg.PrivilegedCfg):
        # mirror policy receiver terms but without noise + add receiver xy to
        # the kicker's body-yaw frame for symmetry with the kicker's privileged
        # ``receiver_pos_b`` (already attached in the V3.1 base env cfg).
        receiver_base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("receiver")},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        receiver_projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("receiver")},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        receiver_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("receiver")},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        receiver_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("receiver")},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        receiver_actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "joint_pos_receiver"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        receiver_ball_pos_b = ObsTerm(
            func=mdp.soccer_observations.receiver_ball_pos_b,
            params={"command_name": "soccer_kick"},
            clip=(-30.0, 30.0),
            scale=1.0,
        )
        receiver_kicker_dir_b = ObsTerm(
            func=mdp.soccer_observations.receiver_kicker_dir_b,
            params={"command_name": "soccer_kick"},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


# =========================================================================
# Rewards — V3.1 kicker stack + receiver-side trap rewards
# =========================================================================


CRITIC_GROUPS_MAP = {
    # ---- goal-related (drive the kick toward the target) ----
    "target_progress": "goal",
    "ball_approach": "goal",
    "foot_ball_proximity": "goal",
    "kick_contact": "goal",
    "kick_success": "goal",
    "goal_scored": "goal",
    "pass_landing": "goal",
    "kick_angle_error": "goal",
    "kick_strength_error": "goal",
    # ---- V3.2 receiver "goal" ----
    "trap_success": "goal",
    "receiver_ball_at_feet": "goal",
    # ---- V3.3 search shaping (aux: posture-style, not directly task-goal) ----
    "search_yaw_velocity": "aux",
    "last_seen_dt_penalty": "aux",
    "head_yaw_search": "aux",
    # everything else (posture, regularizers, receiver_alive/face/terminated)
    # falls back to "aux" inside the multi-critic runner.
}


@configclass
class RewardsCfgV2(V31RewardsCfg):
    """V3.2 reward stack: kicker rewards (inherited) + receiver rewards."""

    # ---- V3.2 receiver goal-related ----
    receiver_ball_at_feet = RewTerm(
        func=mdp.soccer_trap_rewards.receiver_ball_at_feet,
        weight=6.0,
        params={"command_name": "soccer_kick", "sigma": 0.30},
    )
    trap_success = RewTerm(
        func=mdp.soccer_trap_rewards.trap_success,
        weight=30.0,
        params={"command_name": "soccer_kick"},
    )
    # ---- V3.2 receiver auxiliary ----
    receiver_face_ball = RewTerm(
        func=mdp.soccer_trap_rewards.receiver_face_ball,
        weight=-0.4,
        params={"command_name": "soccer_kick"},
    )
    receiver_alive = RewTerm(
        func=mdp.soccer_trap_rewards.receiver_alive, weight=0.5
    )
    receiver_terminated = RewTerm(
        func=mdp.soccer_trap_rewards.receiver_terminated, weight=-200.0
    )


# Attach as a *class attribute*, after @configclass has run (see V1.2 cfg for
# the mechanism — RewardManager won't accept non-RewardTermCfg in __dict__).
RewardsCfgV2.CRITIC_GROUPS = dict(CRITIC_GROUPS_MAP)


def _strip_critic_groups_from_instance_v2(self) -> None:
    self.__dict__.pop("CRITIC_GROUPS", None)


_orig_post_init_v2 = RewardsCfgV2.__post_init__


def _patched_post_init_v2(self):
    _orig_post_init_v2(self)
    _strip_critic_groups_from_instance_v2(self)


RewardsCfgV2.__post_init__ = _patched_post_init_v2


# =========================================================================
# Terminations — V3.1 kicker stack + receiver fall terms
# =========================================================================


@configclass
class TerminationsCfgV2(V31TerminationsCfg):
    """Adds receiver fall-height / fall-tilt terminations."""

    receiver_fall_height = DoneTerm(
        func=mdp.soccer_terminations.receiver_fall_height,
        params={"min_height": 0.30},
    )
    receiver_fall_tilt = DoneTerm(
        func=mdp.soccer_terminations.receiver_fall_tilt,
        params={"max_tilt": 1.3},
    )


# =========================================================================
# Env config
# =========================================================================


@configclass
class SoccerKickTrapV2EnvCfg(SoccerKickTrapEnvCfg):
    """V3.2 base env cfg — kicker + receiver, both controlled by one policy."""

    # Scene / commands / curriculum / events are inherited unchanged from V3.1.
    observations: ObservationsCfgV2 = ObservationsCfgV2()
    actions: ActionsCfgV2 = ActionsCfgV2()
    rewards: RewardsCfgV2 = RewardsCfgV2()
    terminations: TerminationsCfgV2 = TerminationsCfgV2()

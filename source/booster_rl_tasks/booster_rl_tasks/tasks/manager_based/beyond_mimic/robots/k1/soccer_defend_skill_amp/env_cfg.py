"""V4 Stage 3 — Defend Skill, multi-critic AMP-PPO env cfg.

Adds the ``CRITIC_GROUPS`` mapping that partitions reward terms into the
``goal`` (sparse / task-completion) and ``aux`` (style / posture / regularizer)
columns consumed by :class:`MultiCriticAmpOnPolicyRunner`. The mapping must
cover every reward term defined in :mod:`tracking_env_cfg.RewardsCfg`; unknown
terms default to ``aux`` inside the runner.

Also binds the K1 robot articulation + action scales (mirroring the Stage 1
flat env cfg).
"""
from __future__ import annotations

from isaaclab.utils import configclass

from booster_rl_tasks.assets.robots.booster import (
    BOOSTER_K1_CFG as ROBOT_CFG,
    K1_ACTION_SCALE,
)

from .tracking_env_cfg import RewardsCfg, SoccerDefendEnvCfg


# Mapping: reward term name -> critic group. Any term not in this map falls
# back to "aux" inside the multi-critic runner.
CRITIC_GROUPS_MAP = {
    # ---- goal-group (defend rewards & terminal) ----
    "block_success": "goal",
    "steal_success": "goal",
    "defensive_line_hold": "goal",
    "ball_intercept_proximity": "goal",
    "defender_approach_ball": "goal",
    "ball_pushed_away": "goal",
    "own_goal_proximity": "goal",
    # ---- aux-group (alive / terminated / posture / regs) ----
    "alive": "aux",
    "terminated": "aux",
    "pelvis_orientation": "aux",
    "feet_proximity": "aux",
    "action_rate_l2": "aux",
    "dof_torques_l2": "aux",
    "dof_acc_l2": "aux",
    "dof_pos_limits": "aux",
    "undesired_contacts": "aux",
}


@configclass
class RewardsCfgMultiCritic(RewardsCfg):
    """RewardsCfg with a ``CRITIC_GROUPS`` class attribute (see soccer_kick_skill_amp)."""

    pass


# Attach as a class attribute *after* @configclass runs so dataclass-style
# field detection doesn't pick it up as a missing RewardTermCfg field.
RewardsCfgMultiCritic.CRITIC_GROUPS = dict(CRITIC_GROUPS_MAP)


def _strip_critic_groups_from_instance(self) -> None:
    self.__dict__.pop("CRITIC_GROUPS", None)


_orig_post_init = RewardsCfgMultiCritic.__post_init__


def _patched_post_init(self):
    _orig_post_init(self)
    _strip_critic_groups_from_instance(self)


RewardsCfgMultiCritic.__post_init__ = _patched_post_init


@configclass
class SoccerDefendSkillMultiCriticEnvCfg(SoccerDefendEnvCfg):
    """V4 Stage 3 env — multi-critic rewards split."""

    rewards: RewardsCfgMultiCritic = RewardsCfgMultiCritic()


@configclass
class FlatSoccerDefendSkillEnvCfg(SoccerDefendSkillMultiCriticEnvCfg):
    """Flat-plane V4 Stage 3 default — binds K1 robot + action scales."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = dict(K1_ACTION_SCALE)

        # Same arm-scale tightening as the Stage 1 flat env cfg — keeps the
        # policy from learning a T-pose to clear the FOV.
        for jn in [
            "ALeft_Shoulder_Pitch",
            "ARight_Shoulder_Pitch",
            "Left_Elbow_Pitch",
            "Right_Elbow_Pitch",
        ]:
            if jn in self.actions.joint_pos.scale:
                self.actions.joint_pos.scale[jn] = 0.5
        for jn in ["Left_Shoulder_Roll", "Right_Shoulder_Roll"]:
            if jn in self.actions.joint_pos.scale:
                self.actions.joint_pos.scale[jn] = 0.05
        for jn in ["Left_Elbow_Yaw", "Right_Elbow_Yaw"]:
            if jn in self.actions.joint_pos.scale:
                self.actions.joint_pos.scale[jn] = 0.3
        for jn in ["AAHead_yaw", "Head_pitch"]:
            if jn in self.actions.joint_pos.scale:
                self.actions.joint_pos.scale[jn] = 0.19

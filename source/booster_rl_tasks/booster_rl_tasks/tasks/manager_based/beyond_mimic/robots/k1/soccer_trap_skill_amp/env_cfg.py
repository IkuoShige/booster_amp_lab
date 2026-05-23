"""V4 Stage 2 — Trap Skill, multi-critic AMP-PPO env cfg.

Adds the ``CRITIC_GROUPS`` mapping that partitions reward terms into the
``goal`` (sparse / task-completion) and ``aux`` (style / posture / regularizer)
columns consumed by :class:`MultiCriticAmpOnPolicyRunner`. Any term not in
the map falls back to ``aux`` inside the runner.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from booster_rl_tasks.assets.robots.booster import (
    BOOSTER_K1_CFG as ROBOT_CFG,
    K1_ACTION_SCALE,
)

from .tracking_env_cfg import RewardsCfg, SoccerTrapSkillEnvCfg


# Reward term name -> critic group. Unknown terms default to ``aux``.
CRITIC_GROUPS_MAP = {
    # ---- trap "goal" rewards ----
    "receiver_ball_at_feet": "goal",
    "trap_anticipation": "goal",
    "trap_success": "goal",
    # ---- everything else (intercept_alignment, idle_when_blind, search shaping,
    # posture, regs, alive/terminated) falls back to "aux".
}


@configclass
class RewardsCfgMultiCritic(RewardsCfg):
    """RewardsCfg with a ``CRITIC_GROUPS`` class attribute."""

    pass


# Attach as a *class attribute* after @configclass runs (mirrors Stage 1).
RewardsCfgMultiCritic.CRITIC_GROUPS = dict(CRITIC_GROUPS_MAP)


def _strip_critic_groups_from_instance(self) -> None:
    self.__dict__.pop("CRITIC_GROUPS", None)


_orig_post_init = getattr(RewardsCfgMultiCritic, "__post_init__", None)


def _patched_post_init(self):
    if _orig_post_init is not None:
        _orig_post_init(self)
    _strip_critic_groups_from_instance(self)


RewardsCfgMultiCritic.__post_init__ = _patched_post_init


@configclass
class SoccerTrapSkillMultiCriticEnvCfg(SoccerTrapSkillEnvCfg):
    """V4 Stage 2 env — multi-critic rewards split."""

    rewards: RewardsCfgMultiCritic = RewardsCfgMultiCritic()


@configclass
class FlatSoccerTrapSkillEnvCfg(SoccerTrapSkillMultiCriticEnvCfg):
    """Flat-plane V4 Stage 2 default — binds K1 robot + action scales."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = dict(K1_ACTION_SCALE)

        # Same arm-scale tightening as Stage 1 — keeps the policy from
        # learning a T-pose to clear the FOV.
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

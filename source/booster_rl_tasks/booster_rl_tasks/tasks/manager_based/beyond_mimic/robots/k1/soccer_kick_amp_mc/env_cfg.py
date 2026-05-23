"""Concrete env-cfg subclasses for the multi-critic soccer kick AMP task.

Mirrors ``soccer_kick_amp.env_cfg`` but installs a ``CRITIC_GROUPS`` class
attribute on the rewards cfg so the multi-critic runner can partition reward
terms into ``goal`` vs ``aux`` columns.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from booster_rl_tasks.assets.robots.booster import (
    BOOSTER_K1_CFG as ROBOT_CFG,
    K1_ACTION_SCALE,
)

from booster_rl_tasks.tasks.manager_based.beyond_mimic.robots.k1.soccer_kick_amp.tracking_env_cfg import (
    RewardsCfg,
    SoccerKickEnvCfg,
)


# Mapping from RewardsCfg term name -> critic group name.
# Any term not in this dict defaults to "aux" inside the runner.
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
    # ---- V3.3 search shaping (aux: posture-style, not directly task-goal) ----
    "search_yaw_velocity": "aux",
    "last_seen_dt_penalty": "aux",
    "head_yaw_search": "aux",
    # ---- everything else is auxiliary (posture / regularizers) ----
}


@configclass
class RewardsCfgMultiCritic(RewardsCfg):
    """RewardsCfg with a CRITIC_GROUPS metadata attribute.

    ``CRITIC_GROUPS`` is attached as a *class attribute* below, after the
    ``@configclass`` decorator runs (so dataclass does not treat it as a
    field). IsaacLab's :class:`isaaclab.managers.RewardManager` iterates
    ``cfg.__dict__`` and requires every entry to be a :class:`RewardTermCfg`;
    we therefore wrap ``__post_init__`` after configclass has applied its own
    wrapping so that our cleanup runs *last* and the instance ``__dict__``
    stays free of ``CRITIC_GROUPS``. Attribute lookups still resolve via the
    class. The runner reads it via ``env.unwrapped.cfg.rewards.CRITIC_GROUPS``.
    """

    pass


# Attach as a *class attribute*, after the @configclass decorator runs.
RewardsCfgMultiCritic.CRITIC_GROUPS = dict(CRITIC_GROUPS_MAP)


def _strip_critic_groups_from_instance(self) -> None:
    """Remove CRITIC_GROUPS from the instance __dict__ (post-everything)."""
    # configclass's _custom_post_init deep-copies every class attr into the
    # instance via setattr(); we need to undo that here so RewardManager's
    # __dict__ scan does not encounter the non-RewardTermCfg entry.
    self.__dict__.pop("CRITIC_GROUPS", None)


_orig_post_init = RewardsCfgMultiCritic.__post_init__


def _patched_post_init(self):
    _orig_post_init(self)
    _strip_critic_groups_from_instance(self)


RewardsCfgMultiCritic.__post_init__ = _patched_post_init


@configclass
class SoccerKickMultiCriticEnvCfg(SoccerKickEnvCfg):
    """Soccer-kick env cfg that uses the multi-critic rewards cfg."""

    rewards: RewardsCfgMultiCritic = RewardsCfgMultiCritic()


@configclass
class FlatSoccerKickMultiCriticEnvCfg(SoccerKickMultiCriticEnvCfg):
    """Flat-plane variant — V1 multi-critic default."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = dict(K1_ACTION_SCALE)

        # Same arm-scale tightening as the single-critic baseline.
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

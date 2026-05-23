"""V4 Stage 1 — Kick Skill, multi-critic AMP-PPO env cfg.

Adds the ``CRITIC_GROUPS`` mapping that partitions reward terms into the
``goal`` (sparse / task-completion) and ``aux`` (style / posture / regularizer)
columns consumed by :class:`MultiCriticAmpOnPolicyRunner`. The mapping must
cover every reward term added in :mod:`tracking_env_cfg.RewardsCfg`; unknown
terms default to ``aux`` inside the runner.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from booster_rl_tasks.assets.robots.booster import (
    BOOSTER_K1_CFG as ROBOT_CFG,
    K1_ACTION_SCALE,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_perception import (
    soccer_vision_repair_cfg,
)

from .tracking_env_cfg import (
    RewardsCfg,
    RewardsCfgV5,
    RewardsCfgV51,
    RewardsCfgV52,
    RewardsCfgV53,
    SoccerKickSkillEnvCfg,
    SoccerKickSkillV5EnvCfg,
    SoccerKickSkillV51EnvCfg,
    SoccerKickSkillV52EnvCfg,
    SoccerKickSkillV53EnvCfg,
)


# Mapping: reward term name -> critic group. Any term not in this map falls
# back to "aux" inside the multi-critic runner.
CRITIC_GROUPS_MAP = {
    # ---- legacy goal-related shaping (inherited from soccer_kick_amp) ----
    "target_progress": "goal",
    "ball_approach": "goal",
    "foot_ball_proximity": "goal",
    "kick_contact": "goal",
    "clean_kick_contact": "goal",
    "unclean_kick_contact_penalty": "goal",
    "pre_kick_swing_miss_penalty": "goal",
    "visible_pre_kick_time": "goal",
    "close_pre_kick_penalty": "goal",
    "kick_success": "goal",
    "goal_scored": "goal",  # legacy term, weight is zero in V4 but stays in cfg
    "pass_landing": "goal",  # ditto
    "kick_angle_error": "goal",
    "kick_strength_error": "goal",  # ditto
    # ---- V3.3 search shaping ----
    "search_yaw_velocity": "aux",
    "last_seen_dt_penalty": "aux",
    "head_yaw_search": "aux",
    # ---- V4 mode-conditional bonuses (all goal-group) ----
    "goal_scored_quick": "goal",
    "time_to_goal_penalty": "goal",
    "kick_success_first_bonus": "goal",
    "kick_strength_error_shoot": "goal",
    "kick_strength_error_pass": "goal",
    "multi_kick_penalty": "goal",
    "approach_after_kick_penalty": "goal",
    "target_progress_pass_window": "goal",
    "ball_at_target_terminal": "goal",
    # ---- V4.1 mode-gated continuous shaping ----
    "target_progress_shoot": "goal",
    "kick_aim_at_goal_shoot": "goal",
    # ---- V4.2 precision + power tracking ----
    "kick_aim_at_pass_target_pass": "goal",
    "kick_power_track_shoot": "goal",
    "kick_power_track_pass": "goal",
    # ---- V4.3 speed-independent angle penalty (mode-conditional) ----
    "kick_angle_error_shoot": "goal",
    "kick_angle_error_pass": "goal",
    # ---- V5 hard-shoot / recovery / lost-ball robustness ----
    "kick_power_progress_shoot": "goal",
    "shoot_goal_line_speed": "goal",
    "shoot_underpowered_penalty": "goal",
    "shoot_power_hard_bounded": "goal",
    "post_kick_recovery_stability": "aux",
    "post_kick_no_fall_bounded": "aux",
    "post_kick_fall_risk_penalty": "aux",
    "lost_ball_search_bounded": "goal",
    "lost_ball_freeze_penalty_bounded": "goal",
    "lost_ball_reacquire_bonus": "goal",
    "lost_ball_time_penalty": "goal",
    "lost_ball_last_seen_search": "goal",
    "goal_progress_shoot": "goal",
    "arch_contact_lateral": "goal",
    "toe_poke_contact_penalty": "goal",
    # ---- everything else (posture, regs, alive/terminated) falls to "aux" ----
}


# V5.3 follows the LVDRS multi-critic split more closely: the task critic should
# estimate terminal task outcomes and the two potential-based progress rewards.
# Contact style, precision, power shaping, perception/search, and regularizers
# stay in aux so their advantages do not dominate the goal-progress head.
CRITIC_GROUPS_MAP_V53 = {name: "aux" for name in CRITIC_GROUPS_MAP}
CRITIC_GROUPS_MAP_V53.update(
    {
        "ball_approach": "goal",
        "goal_progress_shoot": "goal",
        "goal_scored": "goal",
        "goal_scored_quick": "goal",
        "time_to_goal_penalty": "goal",
        "kick_success": "goal",
        "kick_success_first_bonus": "goal",
        # Pass is a terminal task outcome for the kick skill, but dense pass
        # aim/power/progress shaping remains aux to avoid over-optimizing pass
        # at the expense of shoot strength.
        "pass_landing": "goal",
        "ball_at_target_terminal": "goal",
    }
)


@configclass
class RewardsCfgMultiCritic(RewardsCfg):
    """RewardsCfg with a ``CRITIC_GROUPS`` class attribute (see soccer_kick_amp_mc)."""

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
class SoccerKickSkillMultiCriticEnvCfg(SoccerKickSkillEnvCfg):
    """V4 Stage 1 env — multi-critic rewards split."""

    rewards: RewardsCfgMultiCritic = RewardsCfgMultiCritic()


@configclass
class FlatSoccerKickSkillEnvCfg(SoccerKickSkillMultiCriticEnvCfg):
    """Flat-plane V4 Stage 1 default — binds K1 robot + action scales."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = dict(K1_ACTION_SCALE)

        # Same arm-scale tightening as soccer_kick_amp_mc — keeps the policy
        # from learning a T-pose to clear the FOV.
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


@configclass
class RewardsCfgV5MultiCritic(RewardsCfgV5):
    """V5 RewardsCfg with the same multi-critic partition map."""

    pass


RewardsCfgV5MultiCritic.CRITIC_GROUPS = dict(CRITIC_GROUPS_MAP)


_orig_v5_post_init = RewardsCfgV5MultiCritic.__post_init__


def _patched_v5_post_init(self):
    _orig_v5_post_init(self)
    _strip_critic_groups_from_instance(self)


RewardsCfgV5MultiCritic.__post_init__ = _patched_v5_post_init


@configclass
class SoccerKickSkillV5MultiCriticEnvCfg(SoccerKickSkillV5EnvCfg):
    """Kick V5 env — multi-critic rewards split."""

    rewards: RewardsCfgV5MultiCritic = RewardsCfgV5MultiCritic()


@configclass
class FlatSoccerKickSkillV5EnvCfg(SoccerKickSkillV5MultiCriticEnvCfg):
    """Flat-plane Kick V5 default — binds K1 robot + action scales."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = dict(K1_ACTION_SCALE)

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


@configclass
class RewardsCfgV51MultiCritic(RewardsCfgV51):
    """V5.1 RewardsCfg with the same multi-critic partition map."""

    pass


RewardsCfgV51MultiCritic.CRITIC_GROUPS = dict(CRITIC_GROUPS_MAP)


_orig_v51_post_init = RewardsCfgV51MultiCritic.__post_init__


def _patched_v51_post_init(self):
    _orig_v51_post_init(self)
    _strip_critic_groups_from_instance(self)


RewardsCfgV51MultiCritic.__post_init__ = _patched_v51_post_init


@configclass
class SoccerKickSkillV51MultiCriticEnvCfg(SoccerKickSkillV51EnvCfg):
    """Kick V5.1 env — multi-critic rewards split."""

    rewards: RewardsCfgV51MultiCritic = RewardsCfgV51MultiCritic()


@configclass
class FlatSoccerKickSkillV51EnvCfg(SoccerKickSkillV51MultiCriticEnvCfg):
    """Flat-plane Kick V5.1 default — binds K1 robot + action scales."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = dict(K1_ACTION_SCALE)

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


@configclass
class RewardsCfgV52MultiCritic(RewardsCfgV52):
    """V5.2 RewardsCfg with the same multi-critic partition map."""

    pass


RewardsCfgV52MultiCritic.CRITIC_GROUPS = dict(CRITIC_GROUPS_MAP)


_orig_v52_post_init = RewardsCfgV52MultiCritic.__post_init__


def _patched_v52_post_init(self):
    _orig_v52_post_init(self)
    _strip_critic_groups_from_instance(self)


RewardsCfgV52MultiCritic.__post_init__ = _patched_v52_post_init


@configclass
class SoccerKickSkillV52MultiCriticEnvCfg(SoccerKickSkillV52EnvCfg):
    """Kick V5.2 env — multi-critic rewards split."""

    rewards: RewardsCfgV52MultiCritic = RewardsCfgV52MultiCritic()


@configclass
class FlatSoccerKickSkillV52EnvCfg(SoccerKickSkillV52MultiCriticEnvCfg):
    """Flat-plane Kick V5.2 default — binds K1 robot + action scales."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = dict(K1_ACTION_SCALE)

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


@configclass
class RewardsCfgV53MultiCritic(RewardsCfgV53):
    """V5.3 RewardsCfg with the same multi-critic partition map."""

    pass


RewardsCfgV53MultiCritic.CRITIC_GROUPS = dict(CRITIC_GROUPS_MAP_V53)


_orig_v53_post_init = RewardsCfgV53MultiCritic.__post_init__


def _patched_v53_post_init(self):
    _orig_v53_post_init(self)
    _strip_critic_groups_from_instance(self)


RewardsCfgV53MultiCritic.__post_init__ = _patched_v53_post_init


@configclass
class SoccerKickSkillV53MultiCriticEnvCfg(SoccerKickSkillV53EnvCfg):
    """Kick V5.3 env — multi-critic rewards split."""

    rewards: RewardsCfgV53MultiCritic = RewardsCfgV53MultiCritic()


@configclass
class FlatSoccerKickSkillV53EnvCfg(SoccerKickSkillV53MultiCriticEnvCfg):
    """Flat-plane Kick V5.3 default — binds K1 robot + action scales."""

    def __post_init__(self):
        super().__post_init__()

        # Be explicit here: some Isaac Lab config inheritance paths preserve the
        # parent command object even when the reward/curriculum subclass is used.
        # V5.3 needs 50 frames for the encoder slice and the easy-perception
        # repair preset.
        cmd = self.commands.soccer_kick
        cmd.perception = soccer_vision_repair_cfg()
        cmd.ball_history_len = 50
        cmd.shoot_prob = 0.70
        cmd.ball_spawn_distance_range = (0.55, 1.45)
        cmd.ball_spawn_angle_range = (-0.65, 0.65)
        cmd.target_strength_range = (3.0, 6.0)
        cmd.shoot_target_strength_range = (7.0, 10.0)
        cmd.enable_multi_attempt_shoot = False
        cmd.pass_landing_radius = 1.5
        cmd.pass_landing_speed_window = (0.5, 6.0)
        cmd.kick_ball_speed_thresh = 1.0
        cmd.kick_success_speed_thresh = 1.5
        cmd.shoot_success_target_fraction = 0.60

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = dict(K1_ACTION_SCALE)

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

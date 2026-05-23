"""Concrete env-cfg subclasses for the soccer kick AMP task."""
from __future__ import annotations

from isaaclab.utils import configclass

from booster_rl_tasks.assets.robots.booster import (
    BOOSTER_K1_CFG as ROBOT_CFG,
    K1_ACTION_SCALE,
)

from .tracking_env_cfg import SoccerKickEnvCfg


@configclass
class FlatSoccerKickEnvCfg(SoccerKickEnvCfg):
    """Flat-plane variant — V1 default. Field is a flat 14×9 m painted plane."""

    def __post_init__(self):
        super().__post_init__()

        # Wire up the K1 robot to the scene + apply joint scale.
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = dict(K1_ACTION_SCALE)

        # LVDRS / mjlab V1.10 lesson: tighten arm scales to suppress T-pose flailing.
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

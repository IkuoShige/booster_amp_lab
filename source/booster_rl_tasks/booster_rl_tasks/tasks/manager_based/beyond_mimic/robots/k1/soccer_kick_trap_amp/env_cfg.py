"""Concrete env-cfg subclass for the V3 soccer kick+trap scaffold."""
from __future__ import annotations

from isaaclab.utils import configclass

from booster_rl_tasks.assets.robots.booster import (
    BOOSTER_K1_CFG as ROBOT_CFG,
    K1_ACTION_SCALE,
)

from .tracking_env_cfg import SoccerKickTrapEnvCfg


@configclass
class FlatSoccerKickTrapEnvCfg(SoccerKickTrapEnvCfg):
    """Flat-plane variant — V3 scaffold default.

    Wires up two K1 articulations under sibling prim paths:
      * ``{ENV_REGEX_NS}/Robot``    — the kicker (action target)
      * ``{ENV_REGEX_NS}/Receiver`` — the passive receiver (no action term)
    """

    def __post_init__(self):
        super().__post_init__()

        # Kicker — full controllable K1.
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Receiver — same K1, sibling prim path, default spawn offset (the
        # reset event overwrites this each episode). We just need a non-
        # overlapping startup pose so PhysX doesn't trip over the two roots.
        self.scene.receiver = ROBOT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Receiver",
            init_state=ROBOT_CFG.init_state.replace(pos=(3.0, 0.0, 0.57)),
        )

        # Action scale — same arm-deviation suppression as V1.
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

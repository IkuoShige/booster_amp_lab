"""Concrete env-cfg for the V4 Stage 4 composition validation task.

Binds two K1 articulations (kicker = ``robot``, receiver = ``receiver``)
under sibling prim paths and applies the standard arm-scale tightening
used by every other K1 soccer task.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from booster_rl_tasks.assets.robots.booster import (
    BOOSTER_K1_CFG as ROBOT_CFG,
    K1_ACTION_SCALE,
)

from .tracking_env_cfg import SoccerCompositionEnvCfg


def _apply_arm_scale_tightening(scale: dict) -> None:
    """In-place — match the arm/head scale overrides used by every K1 task."""
    for jn in [
        "ALeft_Shoulder_Pitch",
        "ARight_Shoulder_Pitch",
        "Left_Elbow_Pitch",
        "Right_Elbow_Pitch",
    ]:
        if jn in scale:
            scale[jn] = 0.5
    for jn in ["Left_Shoulder_Roll", "Right_Shoulder_Roll"]:
        if jn in scale:
            scale[jn] = 0.05
    for jn in ["Left_Elbow_Yaw", "Right_Elbow_Yaw"]:
        if jn in scale:
            scale[jn] = 0.3
    for jn in ["AAHead_yaw", "Head_pitch"]:
        if jn in scale:
            scale[jn] = 0.19


@configclass
class FlatSoccerCompositionEnvCfg(SoccerCompositionEnvCfg):
    """Flat-plane V4 Stage 4 default — binds both K1 robots + action scales."""

    def __post_init__(self):
        super().__post_init__()

        # Kicker — full controllable K1 under ``/Robot``.
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Receiver — same K1 under ``/Receiver``. The reset event overwrites
        # its pose each episode; the startup pose just needs to not overlap
        # the kicker so PhysX initialization succeeds.
        self.scene.receiver = ROBOT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Receiver",
            init_state=ROBOT_CFG.init_state.replace(pos=(3.0, 0.0, 0.57)),
        )

        # Apply the standard K1 action-scale tightening to both action terms.
        self.actions.joint_pos.scale = dict(K1_ACTION_SCALE)
        _apply_arm_scale_tightening(self.actions.joint_pos.scale)
        self.actions.joint_pos_receiver.scale = dict(K1_ACTION_SCALE)
        _apply_arm_scale_tightening(self.actions.joint_pos_receiver.scale)

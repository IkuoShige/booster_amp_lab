"""Concrete env-cfg subclass for the V3.2 soccer kick+trap task.

Wires up two controllable K1 articulations + sets joint scales for BOTH the
kicker (``joint_pos``) and the receiver (``joint_pos_receiver``).
"""
from __future__ import annotations

from isaaclab.utils import configclass

from booster_rl_tasks.assets.robots.booster import (
    BOOSTER_K1_CFG as ROBOT_CFG,
    K1_ACTION_SCALE,
)

from .tracking_env_cfg import SoccerKickTrapV2EnvCfg


def _apply_k1_action_scale(action_term) -> None:
    """Set the V1/V3.1 arm-deviation suppression scales on a JointPositionActionCfg.

    Mirrors the manual override the V1 / V3.1 ``__post_init__`` blocks apply
    after ``dict(K1_ACTION_SCALE)``. Centralised so the kicker and receiver
    get identical action-scale treatment.
    """
    action_term.scale = dict(K1_ACTION_SCALE)
    for jn in [
        "ALeft_Shoulder_Pitch",
        "ARight_Shoulder_Pitch",
        "Left_Elbow_Pitch",
        "Right_Elbow_Pitch",
    ]:
        if jn in action_term.scale:
            action_term.scale[jn] = 0.5
    for jn in ["Left_Shoulder_Roll", "Right_Shoulder_Roll"]:
        if jn in action_term.scale:
            action_term.scale[jn] = 0.05
    for jn in ["Left_Elbow_Yaw", "Right_Elbow_Yaw"]:
        if jn in action_term.scale:
            action_term.scale[jn] = 0.3
    for jn in ["AAHead_yaw", "Head_pitch"]:
        if jn in action_term.scale:
            action_term.scale[jn] = 0.19


@configclass
class FlatSoccerKickTrapV2EnvCfg(SoccerKickTrapV2EnvCfg):
    """Flat-plane variant — V3.2 default."""

    def __post_init__(self):
        super().__post_init__()

        # Both robots wired in. The kicker keeps the canonical prim path; the
        # receiver gets a sibling prim path so PhysX articulations don't
        # collide on root resolution.
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.receiver = ROBOT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Receiver",
            init_state=ROBOT_CFG.init_state.replace(pos=(3.0, 0.0, 0.57)),
        )

        # Apply arm-scale tightening to BOTH robot action terms.
        _apply_k1_action_scale(self.actions.joint_pos)
        _apply_k1_action_scale(self.actions.joint_pos_receiver)

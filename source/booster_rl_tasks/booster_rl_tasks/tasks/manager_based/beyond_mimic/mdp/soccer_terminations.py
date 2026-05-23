"""Termination terms for the soccer kick task (V1)."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_commands import (
    SoccerKickCommand,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fall_height(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.30,
) -> torch.Tensor:
    """Terminate when the robot trunk drops below ``min_height``."""
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return z < min_height


def fall_tilt(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_tilt: float = 1.3,
) -> torch.Tensor:
    """Terminate when the trunk tilt vs world-z exceeds ``max_tilt`` (rad)."""
    asset: Articulation = env.scene[asset_cfg.name]
    proj_grav = asset.data.projected_gravity_b  # (N, 3) — z-axis pointing down in body
    # angle between body-z and world-z: acos(|grav_z|).
    cos_t = (-proj_grav[:, 2]).clamp(-1.0, 1.0)
    angle = torch.acos(cos_t)
    return angle > max_tilt


def ball_out_of_field(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """Terminate when the ball leaves the field bounds (latched on command term)."""
    cmd: SoccerKickCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
    return cmd.ball_out_of_field


def goal_scored(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """Terminate when the ball has entered the goal."""
    cmd: SoccerKickCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
    return cmd.goal_awarded


# =========================================================================
# V3.2 — receiver fall terminations
# =========================================================================


def receiver_fall_height(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("receiver"),
    min_height: float = 0.30,
) -> torch.Tensor:
    """Terminate when the receiver trunk drops below ``min_height``.

    Shape: (num_envs,) bool.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return z < min_height


def receiver_fall_tilt(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("receiver"),
    max_tilt: float = 1.3,
) -> torch.Tensor:
    """Terminate when the receiver trunk tilt vs world-z exceeds ``max_tilt`` (rad).

    Shape: (num_envs,) bool.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    proj_grav = asset.data.projected_gravity_b
    cos_t = (-proj_grav[:, 2]).clamp(-1.0, 1.0)
    angle = torch.acos(cos_t)
    return angle > max_tilt


# =========================================================================
# V4 Stage 3 — defend skill terminations
# =========================================================================


def own_goal_scored(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_defend"
) -> torch.Tensor:
    """Terminate when the ball crosses the own-goal line.

    Reads the latch maintained by :class:`SoccerDefendCommand`. Shape:
    (num_envs,) bool.
    """
    from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_defend_commands import (
        SoccerDefendCommand,
    )

    cmd: SoccerDefendCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
    return cmd.own_goal_scored

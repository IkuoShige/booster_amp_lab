"""Observation functions for the V4 Stage 2 trap-skill task.

Reads from :class:`SoccerTrapCommand`. Mirrors the kicker-side obs (perceived
ball xy / mask / last_seen_dt for actor; GT ball pos/vel for critic) but with
no shoot/pass mode flag and a single robot.

The role one-hot and ball-history flatten obs from
:mod:`soccer_role_obs` already work against any command term that exposes
``role`` / ``ball_history`` / ``num_envs`` / ``device``, so we don't need to
duplicate them here.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_trap_commands import (
    SoccerTrapCommand,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: "ManagerBasedRLEnv", name: str) -> SoccerTrapCommand:
    return env.command_manager.get_term(name)  # type: ignore[return-value]


# --- ACTOR observations ----------------------------------------------------
def trap_ball_pos_b(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_trap"
) -> torch.Tensor:
    """Perceived ball xy in robot's body-yaw frame.

    Returns GT when virtual perception is disabled; otherwise the noisy /
    delayed perceived xy from the head-camera detection pipeline.

    Shape: (num_envs, 2).
    """
    cmd = _cmd(env, command_name)
    if cmd.perception is None:
        return cmd.ball_pos_b[:, :2]
    return cmd.ball_pos_b_perceived[:, :2]


def trap_ball_mask(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_trap"
) -> torch.Tensor:
    """Binary "ball detected" flag. Shape: (num_envs, 1)."""
    cmd = _cmd(env, command_name)
    if cmd.perception is None:
        return torch.ones(cmd.num_envs, 1, device=cmd.device)
    return cmd.ball_mask_perceived.unsqueeze(-1)


def trap_last_seen_dt(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_trap"
) -> torch.Tensor:
    """Time since the most recent ball detection (s). Shape: (num_envs, 1)."""
    cmd = _cmd(env, command_name)
    if cmd.perception is None:
        return torch.zeros(cmd.num_envs, 1, device=cmd.device)
    return cmd.last_seen_dt.unsqueeze(-1)


# --- CRITIC (privileged) observations -------------------------------------
def trap_ball_pos_b_gt(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_trap"
) -> torch.Tensor:
    """Ground-truth ball xyz in body-yaw frame. Shape: (num_envs, 3)."""
    return _cmd(env, command_name).ball_pos_b


def trap_ball_vel_b_gt(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_trap"
) -> torch.Tensor:
    """Ground-truth ball linear velocity in body-yaw frame. Shape: (num_envs, 3)."""
    return _cmd(env, command_name).ball_vel_b


def trap_predicted_contact_distance(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_trap"
) -> torch.Tensor:
    """Predicted closest foot↔ball xy distance in the next 0.5 s window.

    Clamped to a finite range for numerical stability before clipping in the
    obs cfg. Shape: (num_envs, 1).
    """
    cmd = _cmd(env, command_name)
    val = cmd.predicted_contact_distance.clamp_max(10.0)
    return val.unsqueeze(-1)

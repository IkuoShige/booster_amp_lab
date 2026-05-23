"""Observation functions for the V4 Stage 3 Soccer Defend Skill task.

Three groups:
  * actor (policy)  — perceived ball xy + mask + last_seen_dt + own-goal dir
  * critic (priv.)  — GT ball xy + GT ball vel + own-goal dir + lane flag +
                      block / steal latches
  * AMP discr.     — reused from the legacy soccer obs cfg (joint+EE pos).
                      Defined in the task env cfg, not here.

Functions are called with ``(env, command_name=...)`` and read the
:class:`SoccerDefendCommand` instance from the command manager.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_defend_commands import (
    SoccerDefendCommand,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: "ManagerBasedRLEnv", name: str) -> SoccerDefendCommand:
    return env.command_manager.get_term(name)  # type: ignore[return-value]


# =========================================================================
# Actor (policy) — perception-based
# =========================================================================


def defend_ball_pos_b(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_defend"
) -> torch.Tensor:
    """Perceived ball xy in defender body-yaw frame.

    Falls back to GT when perception is disabled. Shape: (num_envs, 2).
    """
    cmd = _cmd(env, command_name)
    if cmd.perception is None:
        return cmd.ball_pos_b[:, :2]
    return cmd.ball_pos_b_perceived[:, :2]


def defend_ball_mask(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_defend"
) -> torch.Tensor:
    """Binary "ball detected" flag. Shape: (num_envs, 1)."""
    cmd = _cmd(env, command_name)
    if cmd.perception is None:
        return torch.ones(cmd.num_envs, 1, device=cmd.device)
    return cmd.ball_mask_perceived.unsqueeze(-1)


def defend_last_seen_dt(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_defend"
) -> torch.Tensor:
    """Time since the most recent ball detection (s). Shape: (num_envs, 1)."""
    cmd = _cmd(env, command_name)
    if cmd.perception is None:
        return torch.zeros(cmd.num_envs, 1, device=cmd.device)
    return cmd.last_seen_dt.unsqueeze(-1)


def defend_own_goal_dir_b(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_defend"
) -> torch.Tensor:
    """Unit direction to the own-goal center in body-yaw frame (cos, sin).

    Exposed to both policy and critic so the defender always knows which
    way it should keep its lane. Shape: (num_envs, 2).
    """
    return _cmd(env, command_name).own_goal_dir_b


# =========================================================================
# Critic (privileged) — GT state
# =========================================================================


def defend_ball_pos_b_gt(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_defend"
) -> torch.Tensor:
    """GT ball xyz in defender body-yaw frame. Shape: (num_envs, 3)."""
    return _cmd(env, command_name).ball_pos_b


def defend_ball_vel_b_gt(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_defend"
) -> torch.Tensor:
    """GT ball linear velocity in body-yaw frame. Shape: (num_envs, 3)."""
    return _cmd(env, command_name).ball_vel_b


def defend_is_in_lane(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_defend"
) -> torch.Tensor:
    """1 if defender currently lies in the ball↔own-goal lane, else 0.

    Shape: (num_envs, 1).
    """
    cmd = _cmd(env, command_name)
    return cmd.is_in_lane.float().unsqueeze(-1)


def defend_block_flags(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_defend"
) -> torch.Tensor:
    """Two latch flags: ``[block_success_awarded, steal_success_awarded]``.

    Shape: (num_envs, 2). Each entry is 1 once the corresponding event has
    fired in this episode, 0 otherwise.
    """
    cmd = _cmd(env, command_name)
    return torch.stack(
        [
            cmd.block_success_awarded.float(),
            cmd.steal_success_awarded.float(),
        ],
        dim=-1,
    )

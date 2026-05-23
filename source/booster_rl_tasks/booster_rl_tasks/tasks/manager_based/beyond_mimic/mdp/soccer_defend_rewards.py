"""Reward terms for the V4 Stage 3 Soccer Defend Skill task.

Reward groups (multi-critic split, see :mod:`soccer_amp_skill_library_design` §4.3):

  * "goal" group: block_success, steal_success, defensive_line_hold,
                  ball_pushed_away, own_goal_proximity_penalty
  * "aux" group:  alive, terminated, posture / regularizer terms (inherited
                  via the env config — not defined here).

All terms are pure-tensor (no Python loops over envs). Rising-edge detection
uses a side-channel latch stored on the command term, mirroring
``soccer_rewards._emit_once``.
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
# Goal-group rewards
# =========================================================================


def block_success(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_defend"
) -> torch.Tensor:
    """One-shot reward when the ball touches a non-foot body of the defender.

    Caller supplies the magnitude (design §4.3: +30). The returned tensor is
    1 on the rising edge of ``cmd.block_success_awarded`` and 0 otherwise.
    """
    cmd = _cmd(env, command_name)
    fresh = _emit_once(cmd, "_block_success_emitted", cmd.block_success_awarded)
    return fresh.float()


def steal_success(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_defend"
) -> torch.Tensor:
    """One-shot reward when the defender steals the ball (foot near + decel).

    Caller supplies +25. Rising edge of ``cmd.steal_success_awarded``.
    """
    cmd = _cmd(env, command_name)
    fresh = _emit_once(cmd, "_steal_success_emitted", cmd.steal_success_awarded)
    return fresh.float()


def defensive_line_hold(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_defend"
) -> torch.Tensor:
    """Per-step +1 while the defender is in the ball↔own-goal lane.

    Active only while the episode is alive (i.e. ``own_goal_scored`` has not
    yet fired). Caller supplies +5/step from the §4.3 spec.
    """
    cmd = _cmd(env, command_name)
    alive = ~cmd.own_goal_scored
    return (cmd.is_in_lane & alive).float()


def ball_pushed_away(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_defend"
) -> torch.Tensor:
    """One-shot reward on the rising edge of ``ball_pushed_away_flag``.

    The flag is post-contact (block or steal awarded) AND ball velocity
    projected onto the (own_goal − ball) direction is negative — i.e. ball
    is moving away from the own-goal. Caller supplies +10.
    """
    cmd = _cmd(env, command_name)
    fresh = _emit_once(cmd, "_ball_pushed_away_emitted", cmd.ball_pushed_away_flag)
    return fresh.float()


def own_goal_proximity_penalty(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_defend"
) -> torch.Tensor:
    """One-shot penalty fired when the ball crosses the own-goal line.

    Caller supplies -15 (sign lives in the weight). The returned tensor is
    1 on the rising edge of ``cmd.own_goal_scored`` and 0 otherwise.
    """
    cmd = _cmd(env, command_name)
    fresh = _emit_once(cmd, "_own_goal_emitted", cmd.own_goal_scored)
    return fresh.float()


# =========================================================================
# V4.1 — Continuous interception shaping
# =========================================================================


def ball_intercept_proximity(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_defend",
    sigma: float = 0.8,
) -> torch.Tensor:
    """Continuous reward for sitting on the ball→own-goal segment.

    Replaces the dense ``defensive_line_hold`` binary signal with a smooth
    Gaussian on the defender's perpendicular distance to the line segment.
    Designed to reward incremental progress toward the optimal interception
    spot rather than the all-or-nothing "in lane" predicate, which the
    policy learned to game by parking on the line.

    Returns ``exp(-d² / σ²)`` where ``d`` is the perpendicular distance
    to the ball→own-goal segment (capped by the segment endpoints).

    Shape: ``(num_envs,)``.
    """
    cmd = _cmd(env, command_name)
    d = cmd.intercept_distance
    return torch.exp(-(d * d) / (sigma * sigma))


def defender_approach_ball(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_defend",
    sigma: float = 1.5,
) -> torch.Tensor:
    """Continuous reward for closing the defender→ball distance.

    Encourages the defender to actively close on the incoming ball
    (necessary for steal / block contact), complementing the position
    signal from :func:`ball_intercept_proximity`. Active only when the
    ball is moving toward the own-goal (incoming) — once the ball has
    been pushed away we don't want the defender to chase it.

    Shape: ``(num_envs,)``.
    """
    cmd = _cmd(env, command_name)
    ball_xy = cmd.ball_pos_w[:, :2]
    defender_xy = cmd.robot.data.root_pos_w[:, :2]
    # Incoming? Project ball velocity onto direction-from-ball-to-own-goal.
    env_origins = env.scene.env_origins
    og_x = env_origins[:, 0] + float(cmd.cfg.own_goal_line_x)
    og_y = env_origins[:, 1] + 0.0
    rel_to_og_x = og_x - ball_xy[:, 0]
    rel_to_og_y = og_y - ball_xy[:, 1]
    rel_to_og_norm = torch.sqrt(
        rel_to_og_x * rel_to_og_x + rel_to_og_y * rel_to_og_y
    ).clamp_min(1e-3)
    incoming = (
        cmd.ball_vel_w[:, 0] * rel_to_og_x + cmd.ball_vel_w[:, 1] * rel_to_og_y
    ) / rel_to_og_norm
    incoming_mask = (incoming > 0.5).float()
    # Distance from defender to ball.
    dx = ball_xy[:, 0] - defender_xy[:, 0]
    dy = ball_xy[:, 1] - defender_xy[:, 1]
    dist = torch.sqrt(dx * dx + dy * dy)
    return incoming_mask * torch.exp(-(dist * dist) / (sigma * sigma))


# =========================================================================
# Helpers
# =========================================================================


def _emit_once(cmd: SoccerDefendCommand, attr_name: str, latch: torch.Tensor) -> torch.Tensor:
    """Detect rising edge of a latched per-env bool tensor.

    Mirrors :func:`soccer_rewards._emit_once`. Stores prev state on the
    command term under ``attr_name`` (created lazily). Returns a bool
    tensor that is True only on the step the latch flips on.
    """
    prev = getattr(cmd, attr_name, None)
    if prev is None or prev.shape != latch.shape:
        prev = torch.zeros_like(latch, dtype=torch.bool)
        setattr(cmd, attr_name, prev)
    fresh = latch & ~prev
    prev.copy_(latch)
    return fresh

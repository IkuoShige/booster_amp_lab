"""Reward terms for the V3.2 soccer kick-trap task — receiver-side rewards.

The V3.2 task ``Booster-Soccer-KickTrap-AMP-V2-v0`` trains BOTH the kicker and
the receiver with a single shared 44-dim-action policy. These rewards drive
the receiver half of the policy. They mirror the kicker rewards in
``soccer_rewards.py`` where helpful:

* :func:`receiver_ball_at_feet`   — Gaussian on min(receiver_foot - ball_xy)
* :func:`receiver_face_ball`      — penalty on yaw error to the ball
* :func:`trap_success`            — rising edge of the per-env trap latch
* :func:`receiver_alive`          — basic per-step survival bonus
* :func:`receiver_terminated`     — non-timeout termination penalty
* :func:`receiver_face_kicker`    — (optional) face the kicker, useful for
                                    catching incoming passes (not exposed by
                                    default — kept here for symmetry).

All terms are vectorized; no Python-level loops over envs.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import yaw_quat

from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_commands import (
    SoccerKickCommand,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: "ManagerBasedRLEnv", name: str) -> SoccerKickCommand:
    return env.command_manager.get_term(name)  # type: ignore[return-value]


# =========================================================================
# Receiver "goal" rewards
# =========================================================================


def receiver_ball_at_feet(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    sigma: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("receiver"),
) -> torch.Tensor:
    """Gaussian reward on minimum receiver-foot to ball xy distance.

    Encourages the receiver to physically place a foot near the ball — the
    pre-condition for actually trapping it. Active in every step (no pre/post
    gating) so the gradient is always available.

    Shape: (num_envs,).
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    # Cache foot ids on the command term to avoid per-call lookups (the
    # latch installs them too, but we cannot rely on that path being taken
    # in case ``cfg.receiver_name`` was left None somehow).
    foot_ids = cmd.receiver_foot_ids
    if foot_ids is None:
        foot_ids_local, _ = asset.find_bodies(
            list(cmd.cfg.foot_body_names), preserve_order=True
        )
        foot_ids = torch.as_tensor(foot_ids_local, dtype=torch.long, device=cmd.device)
    foot_pos = asset.data.body_pos_w[:, foot_ids, :]  # (N, 2, 3)
    ball = cmd.ball_pos_w.unsqueeze(1)
    foot_ball_xy = torch.linalg.norm(foot_pos[..., :2] - ball[..., :2], dim=-1)
    min_d = foot_ball_xy.amin(dim=-1)
    return torch.exp(-min_d * min_d / (sigma * sigma))


def trap_success(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """One-shot rising edge of the trap latch (pass mode only).

    The latch is gated in :meth:`SoccerKickCommand._update_command` on
    ``~is_shoot``; we AND with ``~is_shoot`` again here as a safety net.

    Shape: (num_envs,).
    """
    cmd = _cmd(env, command_name)
    fresh = _emit_once(cmd, "_trap_success_emitted", cmd.trap_success_awarded)
    return (fresh & (~cmd.is_shoot)).float()


# =========================================================================
# Receiver "aux" rewards (posture / survival)
# =========================================================================


def receiver_face_ball(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_kick",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("receiver"),
) -> torch.Tensor:
    """Penalize yaw mismatch between receiver body and direction-to-ball.

    Returns a non-negative scalar per env — callers should attach a *negative*
    weight (see ``tracking_env_cfg``).

    Shape: (num_envs,).
    """
    cmd = _cmd(env, command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    recv_pos = asset.data.root_pos_w
    ball_xy = cmd.ball_pos_w[:, :2]
    rel = ball_xy - recv_pos[:, :2]
    desired_yaw = torch.atan2(rel[:, 1], rel[:, 0])
    # receiver yaw extracted from the root quat (use yaw_quat for parity).
    yq = yaw_quat(asset.data.root_quat_w)
    body_yaw = torch.atan2(
        2.0 * (yq[:, 0] * yq[:, 3] + yq[:, 1] * yq[:, 2]),
        1.0 - 2.0 * (yq[:, 2] * yq[:, 2] + yq[:, 3] * yq[:, 3]),
    )
    err = desired_yaw - body_yaw
    err = torch.atan2(torch.sin(err), torch.cos(err))  # wrap to [-pi, pi]
    return err * err


def receiver_alive(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("receiver"),
) -> torch.Tensor:
    """Constant per-step survival bonus for the receiver.

    Shape: (num_envs,).
    """
    return torch.ones(env.num_envs, device=env.device)


def receiver_terminated(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("receiver"),
) -> torch.Tensor:
    """Penalty on receiver-specific non-timeout termination.

    Approximation: any non-timeout termination this step gets the penalty,
    because the termination manager doesn't break out per-asset. In practice
    this means the kicker's falls / ball-out / goal-scored also penalize the
    receiver. That's acceptable for V3.2 (single shared policy — both must
    survive together). When per-agent termination splits are introduced this
    function should be specialized.

    Shape: (num_envs,).
    """
    if not hasattr(env, "termination_manager"):
        return torch.zeros(env.num_envs, device=env.device)
    dones = env.termination_manager.dones.float()
    timeouts = env.termination_manager.time_outs.float()
    return (dones * (1.0 - timeouts)).clamp_max(1.0)


# =========================================================================
# Helpers
# =========================================================================


def _emit_once(cmd: SoccerKickCommand, attr_name: str, latch: torch.Tensor) -> torch.Tensor:
    """Detect rising edge of a latched per-env bool tensor.

    Stores prev state on the command term under ``attr_name`` (created lazily).
    """
    prev = getattr(cmd, attr_name, None)
    if prev is None or prev.shape != latch.shape:
        prev = torch.zeros_like(latch, dtype=torch.bool)
        setattr(cmd, attr_name, prev)
    fresh = latch & ~prev
    prev.copy_(latch)
    return fresh


# =========================================================================
# V4 Stage 2 — Trap-skill rewards (operate on SoccerTrapCommand)
# =========================================================================
#
# These mirror the V3.2 ``trap_success`` / ``receiver_ball_at_feet`` /
# ``receiver_face_ball`` but read from the single-robot
# :class:`SoccerTrapCommand` (no ``is_shoot`` mask, no separate receiver
# articulation — the robot itself is the receiver).


from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_trap_commands import (  # noqa: E402
    SoccerTrapCommand,
)


def _trap_cmd(env: "ManagerBasedRLEnv", name: str) -> SoccerTrapCommand:
    return env.command_manager.get_term(name)  # type: ignore[return-value]


def trap_success_skill(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_trap"
) -> torch.Tensor:
    """One-shot rising-edge bonus when the trap latch flips on.

    Reads :class:`SoccerTrapCommand` (Stage 2). Returns 1.0 on the step the
    trap latch transitions False -> True for an env, 0 otherwise.

    Shape: (num_envs,).
    """
    cmd = _trap_cmd(env, command_name)
    fresh = _emit_once(cmd, "_trap_skill_emitted", cmd.trap_success_awarded)
    return fresh.float()


def receiver_ball_at_feet_skill(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_trap",
    sigma: float = 0.30,
) -> torch.Tensor:
    """Gaussian on minimum receiver-foot to ball xy distance.

    Encourages the receiver to physically place a foot near the ball — the
    pre-condition for actually trapping it. Active every step (no pre/post
    gating) so the gradient is always available.

    Shape: (num_envs,).
    """
    cmd = _trap_cmd(env, command_name)
    asset: Articulation = cmd.robot
    foot_ids = cmd.foot_ids
    foot_pos = asset.data.body_pos_w[:, foot_ids, :]  # (N, 2, 3)
    ball = cmd.ball_pos_w.unsqueeze(1)
    foot_ball_xy = torch.linalg.norm(foot_pos[..., :2] - ball[..., :2], dim=-1)
    min_d = foot_ball_xy.amin(dim=-1)
    return torch.exp(-min_d * min_d / (sigma * sigma))


def intercept_alignment_skill(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_trap"
) -> torch.Tensor:
    """Penalize yaw mismatch between robot body forward and direction-to-ball.

    Returns a non-negative scalar per env — callers should attach a *negative*
    weight (see ``tracking_env_cfg``).

    Shape: (num_envs,).
    """
    cmd = _trap_cmd(env, command_name)
    asset: Articulation = cmd.robot
    pos = asset.data.root_pos_w
    ball_xy = cmd.ball_pos_w[:, :2]
    rel = ball_xy - pos[:, :2]
    desired_yaw = torch.atan2(rel[:, 1], rel[:, 0])
    yq = yaw_quat(asset.data.root_quat_w)
    body_yaw = torch.atan2(
        2.0 * (yq[:, 0] * yq[:, 3] + yq[:, 1] * yq[:, 2]),
        1.0 - 2.0 * (yq[:, 2] * yq[:, 2] + yq[:, 3] * yq[:, 3]),
    )
    err = desired_yaw - body_yaw
    err = torch.atan2(torch.sin(err), torch.cos(err))
    return err * err


def trap_anticipation(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_trap",
    contact_radius: float = 0.5,
    min_approach_speed: float = 0.5,
) -> torch.Tensor:
    """Reward when the ball is approaching and predicted-contact distance is small.

    Returns 1.0 when:
      * ``predicted_contact_distance < contact_radius`` (a foot will be close
        to the projected ball trajectory in the next ~0.5 s), AND
      * the ball is moving *toward* the receiver at > ``min_approach_speed`` m/s.

    The "approaching" check projects the ball's velocity onto the
    receiver→ball unit direction: a positive projection means the ball is
    moving away, so we flip the sign and require it to exceed
    ``min_approach_speed``.

    Shape: (num_envs,).
    """
    cmd = _trap_cmd(env, command_name)
    rel = cmd.robot.data.root_pos_w[:, :2] - cmd.ball_pos_w[:, :2]  # ball -> receiver
    norm = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp_min(1e-6)
    dir_to_recv = rel / norm
    ball_vxy = cmd.ball_vel_w[:, :2]
    approach_speed = (ball_vxy * dir_to_recv).sum(-1)  # +ve when moving toward recv
    approaching = approach_speed > float(min_approach_speed)
    close = cmd.predicted_contact_distance < float(contact_radius)
    return (approaching & close).float()


def receiver_idle_when_blind(
    env: "ManagerBasedRLEnv",
    command_name: str = "soccer_trap",
    posture_tolerance: float = 0.3,
) -> torch.Tensor:
    """Reward stance posture when the ball is not perceived.

    Returns 1.0 when:
      * ``ball_mask == 0`` (camera doesn't see the ball), AND
      * the robot is in stance posture
        (||joint_pos - default_joint_pos||_1 < ``posture_tolerance`` * num_joints / 22).

    The tolerance scaling normalises the threshold so it is approximately
    valid regardless of joint count. Shape: (num_envs,).
    """
    cmd = _trap_cmd(env, command_name)
    asset: Articulation = cmd.robot
    err = (asset.data.joint_pos - asset.data.default_joint_pos).abs().sum(-1)
    n_joints = asset.data.joint_pos.shape[-1]
    threshold = float(posture_tolerance) * float(n_joints) / 22.0
    in_stance = err < threshold
    blind = cmd.ball_mask_perceived < 0.5
    return (blind & in_stance).float()


def trap_alive(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Constant per-step survival bonus for the trap skill."""
    return torch.ones(env.num_envs, device=env.device)


def trap_terminated(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalty on non-timeout termination (rising edge from termination manager).

    Returns 1.0 on the step a non-timeout termination fires; 0 otherwise.
    Shape: (num_envs,).
    """
    if not hasattr(env, "termination_manager"):
        return torch.zeros(env.num_envs, device=env.device)
    dones = env.termination_manager.dones.float()
    timeouts = env.termination_manager.time_outs.float()
    return (dones * (1.0 - timeouts)).clamp_max(1.0)

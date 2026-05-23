"""V4 — role-conditional observation helpers.

The V4 skill library uses a single shared actor network conditioned on a
4-dim ``role`` one-hot (kicker / receiver / defender / idle). Each Stage 1-3
task fixes ``cfg.role`` in its env config; at deploy time the operator
overrides the per-env role flag to switch behaviours on a single checkpoint.

Also exposes :func:`ball_history_flat`, the Step A temporal-information
observation: flattened past ``history_len`` slots of perceived ball state.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_commands import (
    SoccerKickCommand,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_ROLE_NAMES = ("kicker", "receiver", "defender", "idle")
_ROLE_INDEX = {name: idx for idx, name in enumerate(_ROLE_NAMES)}


def _cmd(env: "ManagerBasedRLEnv", name: str) -> SoccerKickCommand:
    return env.command_manager.get_term(name)  # type: ignore[return-value]


def role_one_hot(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """Per-env role one-hot vector ``(N, 4)`` in order ``(kicker, receiver, defender, idle)``.

    The role is constant per skill task (Stage 1 → kicker, Stage 2 → receiver,
    Stage 3 → defender). At deploy time downstream code can override the tensor
    on a per-env basis. Returns a fresh tensor every call so callers may mutate
    safely.
    """
    cmd = _cmd(env, command_name)
    role_name = cmd.role
    idx = _ROLE_INDEX.get(role_name, _ROLE_INDEX["idle"])
    out = torch.zeros(cmd.num_envs, 4, device=cmd.device)
    out[:, idx] = 1.0
    return out


def ball_history_flat(
    env: "ManagerBasedRLEnv", command_name: str = "soccer_kick"
) -> torch.Tensor:
    """Flattened ball observation history.

    Returns the past ``history_len`` perceived-ball slots flattened to a single
    vector per env. With the default ``history_len=10`` and 5 dims per slot,
    the output is ``(N, 50)``. The most recent slot occupies the *last* 5
    entries — ordering matches how the V4 encoder (Step B) will consume it.

    Layout per slot (5 scalars):
      ``[ball_pos_b_x, ball_pos_b_y, ball_mask, last_seen_dt, ball_speed_b]``

    Shape: ``(num_envs, history_len * 5)``.
    """
    cmd = _cmd(env, command_name)
    # Buffer is (history_len, N, dim); transpose to (N, history_len, dim) then flatten.
    hist = cmd.ball_history.permute(1, 0, 2).contiguous()
    return hist.flatten(start_dim=1)


def role_id(role_name: str) -> int:
    """Public helper — return the integer index of a role name in the one-hot."""
    return int(_ROLE_INDEX[role_name])

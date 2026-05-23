"""Passive trap command for the composition validation task.

Subclasses :class:`SoccerTrapCommand` but turns ``_resample_command`` into a
no-op: in the composition scene the receiver pose is reset by the V3.1
``reset_receiver_to_default`` event and the ball is driven by the kicker via
:class:`SoccerKickCommand`. We still want the receiver-side bookkeeping
(perceived ball state in the receiver's body-yaw frame, ball history,
trap-success latch, predicted contact distance) so the receiver policy can be
evaluated on its native observation layout.

This module is intentionally local to the composition task — it doesn't
modify the shared MDP modules.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass

from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_trap_commands import (
    SoccerTrapCommand,
    SoccerTrapCommandCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PassiveTrapCommand(SoccerTrapCommand):
    """Trap command that does NOT respawn the robot/ball on reset.

    All other behaviour (per-step bookkeeping in :meth:`_update_command`,
    properties, metrics, debug viz) is inherited unchanged so observation
    and reward functions registered for ``SoccerTrapCommand`` keep working.
    """

    cfg: "PassiveTrapCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):  # type: ignore[override]
        if isinstance(env_ids, slice):
            env_ids_t = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        n = int(env_ids_t.numel())
        if n == 0:
            return

        # Reset latches and per-episode counters; do NOT touch robot / ball
        # poses (those are driven by the kicker command + reset event).
        self._trap_success_awarded[env_ids_t] = False
        self._predicted_contact_distance[env_ids_t] = float("inf")
        self._steps_since_launch[env_ids_t] = 0
        self._time_to_trap[env_ids_t] = 0.0

        if self._perception is not None:
            self._perception.reset(env_ids_t)
            self._ball_pos_b_perceived[env_ids_t] = 0.0
            self._ball_mask_perceived[env_ids_t] = 0.0
            self._last_seen_dt_buf[env_ids_t] = 0.0

        self._ball_history_buf[:, env_ids_t, :] = 0.0


@configclass
class PassiveTrapCommandCfg(SoccerTrapCommandCfg):
    """Configuration for :class:`PassiveTrapCommand`."""

    class_type: type = PassiveTrapCommand
    asset_name: str = "receiver"
    ball_name: str = "ball"
    role: str = "receiver"

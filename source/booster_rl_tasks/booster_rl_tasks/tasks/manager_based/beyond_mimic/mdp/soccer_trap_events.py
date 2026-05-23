"""Event-manager helpers for the V3 soccer kick-trap task.

V3 introduces a *passive receiver* K1 sharing the scene with the kicker.
:meth:`reset_receiver_to_default` resets the receiver back to its default
joint pose and zero velocity each episode, and reseats its base at a random
xy offset in front of the kicker (so the scene doesn't keep the same fixed
spawn forever). The receiver has no action term in V3, so its PD actuators
will hold the default pose against gravity once the joint state is written.

The kicker / ball spawn pipeline is unchanged — that's still owned by
``SoccerKickCommand._resample_command``.
"""
from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_receiver_to_default(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_name: str = "receiver",
    offset_x_range: tuple[float, float] = (2.5, 4.0),
    offset_y_range: tuple[float, float] = (-1.0, 1.0),
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    spawn_z: float = 0.57,
) -> None:
    """Reset the passive receiver K1 to default joint pose at a random xy / yaw.

    This is intended to run with ``mode="reset"``. The receiver xy is drawn in
    each env's local frame, then offset by ``env_origins`` so it sits in front
    of the kicker. With the K1 default joint stiffness/damping holding the
    pose, the receiver stays roughly upright for the duration of the episode.
    """
    receiver: Articulation = env.scene[asset_name]
    device = receiver.device

    if env_ids is None:
        env_ids_t = torch.arange(env.scene.num_envs, device=device)
    else:
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=device)
    n = int(env_ids_t.numel())
    if n == 0:
        return

    env_origins = env.scene.env_origins[env_ids_t]

    # Sample local xy offset + yaw.
    def _uniform(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(n, device=device) * (hi - lo) + lo

    off_x = _uniform(*offset_x_range)
    off_y = _uniform(*offset_y_range)
    off_yaw = _uniform(*yaw_range)

    root_pose = torch.zeros(n, 7, device=device)
    root_pose[:, 0] = env_origins[:, 0] + off_x
    root_pose[:, 1] = env_origins[:, 1] + off_y
    root_pose[:, 2] = env_origins[:, 2] + spawn_z
    root_pose[:, 3:7] = quat_from_euler_xyz(
        torch.zeros_like(off_yaw),
        torch.zeros_like(off_yaw),
        off_yaw,
    )
    receiver.write_root_pose_to_sim(root_pose, env_ids=env_ids_t)
    receiver.write_root_velocity_to_sim(
        torch.zeros(n, 6, device=device), env_ids=env_ids_t
    )

    default_jp = receiver.data.default_joint_pos[env_ids_t]
    default_jv = receiver.data.default_joint_vel[env_ids_t]
    receiver.write_joint_state_to_sim(default_jp, default_jv, env_ids=env_ids_t)

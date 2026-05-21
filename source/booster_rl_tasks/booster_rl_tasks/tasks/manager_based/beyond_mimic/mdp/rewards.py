from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Union

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor,RayCaster
from isaaclab.utils.math import quat_error_magnitude

from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.commands import MotionCommand
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
import booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def feet_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    feet_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].clone()
    mean_feet_quat = torch.mean(feet_quat,dim=1)
    feet_projected_gravity_b = math_utils.quat_apply_inverse(mean_feet_quat, asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(feet_projected_gravity_b[:, :2]), dim=1)

def liedown_desired_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    pen = torch.sum(torch.square(asset.data.projected_gravity_b[:, 1:]), dim=1)
    pen *= asset.data.projected_gravity_b[:, 0] > 0
    pen *= asset.data.root_link_pos_w[:, 2] < 0.25

    return pen

def donot_falling(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    pen = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    pen *= asset.data.root_link_pos_w[:, 2] > 0.25
    return pen

def contact_force(env: ManagerBasedRLEnv,sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    forces = torch.norm(contact_sensor.data.net_forces_w ,dim=-1)
    mean_contact_force = torch.mean(forces,dim=-1)
    # Penalize feet hitting vertical surfaces
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    mean_contact_force *= contact_num > 2
    return mean_contact_force

def in_the_sky(env: ManagerBasedRLEnv,sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Penalize feet hitting vertical surfaces
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    no_contack = (contact_num == 0)
    rew = no_contack.float() * 5
    return rew

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.2
    return reward

def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def desired_hand_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, height: float = 0.3, donot_touch : list = [".*Hip.*", ".*Shank.*"], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize if none of the desired contacts are present."""
    asset: RigidObject = env.scene[asset_cfg.name]

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    hand_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(hand_contact, dim=1)
    hand_is_contact = (contact_num == 1)
    hand_is_contact_and_height_ok = torch.logical_and (hand_is_contact , asset.data.root_link_pos_w[:, 2] > height)
    donot_touch_ids, _ = asset.find_bodies(name_keys=donot_touch, preserve_order=True)

    net_contact_forces = contact_sensor.data.net_forces_w_history
    donot_touch_bool = torch.sum(torch.norm(torch.norm(net_contact_forces[:, :, donot_touch_ids], dim=-1), dim=-1),dim=-1) > 1.25
    reward = torch.logical_xor(hand_is_contact_and_height_ok, donot_touch_bool)
    reward = reward.float() * 5
    return reward

def tracking_base_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """reward asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        adjusted_target_height = target_height
    error = torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return torch.exp(-error / std**2)

def get_stand_rew(
    env: ManagerBasedRLEnv,
    target_height: float = 0.57,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = 5 * torch.ones_like(mdp.joint_deviation_l1(env, asset_cfg))
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    reward *= torch.norm(forces_z, dim=1) > 20
    reward *= torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() < 0.3
    reward *= (asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] > target_height).squeeze(-1)
    return reward

def stand_still2(
    env: ManagerBasedRLEnv,
    limit_angle: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    reward *= torch.norm(forces_z, dim=1) > 20

    reward *= torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() < limit_angle
    return reward

def tracking_head_height(
    env: ManagerBasedRLEnv,
    target_head_height: float,
    threshold :float | None,
    std: float, 
    command_name: str,
    command_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Head_2"]),
) -> torch.Tensor:
    """reward asset height from its target using L2 squared kernel.

    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if threshold is not None:
        adjust_height = torch.clip( asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - threshold , min = 0) 
    else:
        adjust_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    error = torch.squeeze((torch.abs(adjust_height - target_head_height)), dim=1)
    reward = torch.exp(-error / std**2)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > command_threshold
    
    return reward


def yaw_only_track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    lin_threshold: float = 0.15,
    yaw_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Extra yaw tracking reward for near-in-place turn commands."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    yaw_only = (torch.norm(command[:, :2], dim=1) < lin_threshold) & (torch.abs(command[:, 2]) > yaw_threshold)
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2) * yaw_only


def low_speed_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.18,
    min_command: float = 0.10,
    max_command: float = 0.45,
    max_lateral: float = 0.08,
    max_yaw: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sharper XY tracking reward for low-speed commands, excluding true zero command."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_abs = torch.abs(command)
    low_speed = (
        (command_abs[:, 0] >= min_command)
        & (command_abs[:, 0] <= max_command)
        & (command_abs[:, 1] <= max_lateral)
        & (command_abs[:, 2] <= max_yaw)
    )
    lin_vel_error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2) * low_speed


def _push_active_mask(env: ManagerBasedRLEnv, command_name: str, device: torch.device | str) -> torch.Tensor:
    try:
        command_term = env.command_manager.get_term(command_name)
        push_active = command_term.metrics.get("failure_push_active", None)
    except Exception:
        push_active = None
    if isinstance(push_active, torch.Tensor):
        return push_active.to(device=device) > 0.0
    return torch.zeros(env.num_envs, dtype=torch.bool, device=device)


def _push_state(
    env: ManagerBasedRLEnv,
    command_name: str,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    elapsed = torch.zeros(env.num_envs, device=device)
    delta_xy = torch.zeros(env.num_envs, 2, device=device)
    try:
        command_term = env.command_manager.get_term(command_name)
    except Exception:
        return active, elapsed, delta_xy

    metrics = getattr(command_term, "metrics", {})
    push_active = metrics.get("failure_push_active")
    push_elapsed = metrics.get("failure_push_elapsed_s")
    if isinstance(push_active, torch.Tensor):
        active = push_active.to(device=device) > 0.0
    if isinstance(push_elapsed, torch.Tensor):
        elapsed = push_elapsed.to(device=device).float()

    push_delta_xy = getattr(command_term, "_push_delta_xy", None)
    if isinstance(push_delta_xy, torch.Tensor):
        delta_xy = push_delta_xy.to(device=device).float()
    else:
        delta_x = metrics.get("failure_push_delta_x")
        delta_y = metrics.get("failure_push_delta_y")
        if isinstance(delta_x, torch.Tensor) and isinstance(delta_y, torch.Tensor):
            delta_xy = torch.stack((delta_x.to(device=device), delta_y.to(device=device)), dim=-1).float()
    return active, elapsed, delta_xy


def _force_push_active_mask(env: ManagerBasedRLEnv, command_name: str, device: torch.device | str) -> torch.Tensor:
    try:
        command_term = env.command_manager.get_term(command_name)
        force_push_active = command_term.metrics.get("force_push_active", None)
    except Exception:
        force_push_active = None
    if isinstance(force_push_active, torch.Tensor):
        return force_push_active.to(device=device) > 0.0
    return torch.zeros(env.num_envs, dtype=torch.bool, device=device)


def _foot_positions_b(
    env: ManagerBasedRLEnv,
    foot_asset_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_link_pos_w[:, foot_asset_cfg.body_ids, :3]
    root_pos_w = asset.data.root_pos_w[:, :3].unsqueeze(1)
    rel_w = foot_pos_w - root_pos_w
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, rel_w.shape[1], -1)
    rel_b = math_utils.quat_apply_inverse(root_quat_w.reshape(-1, 4), rel_w.reshape(-1, 3))
    return rel_b.reshape_as(rel_w)


def single_axis_lin_vel_track_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    axis: str,
    std: float = 0.20,
    min_command: float = 0.05,
    max_cross_command: float = 0.05,
    max_yaw_command: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking for pure x-only or y-only command samples."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name).to(asset.data.root_lin_vel_b.device)
    if axis == "x":
        axis_index, cross_index = 0, 1
    elif axis == "y":
        axis_index, cross_index = 1, 0
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}.")
    mask = (
        (torch.abs(command[:, axis_index]) >= min_command)
        & (torch.abs(command[:, cross_index]) <= max_cross_command)
        & (torch.abs(command[:, 2]) <= max_yaw_command)
    )
    error = torch.square(command[:, axis_index] - asset.data.root_lin_vel_b[:, axis_index])
    return torch.exp(-error / std**2) * mask


def single_axis_lin_crosstalk_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    axis: str,
    min_command: float = 0.05,
    max_cross_command: float = 0.05,
    max_yaw_command: float = 0.05,
    yaw_scale: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize unwanted cross-axis motion for pure x-only or y-only command samples."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name).to(asset.data.root_lin_vel_b.device)
    if axis == "x":
        axis_index, cross_index = 0, 1
    elif axis == "y":
        axis_index, cross_index = 1, 0
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}.")
    mask = (
        (torch.abs(command[:, axis_index]) >= min_command)
        & (torch.abs(command[:, cross_index]) <= max_cross_command)
        & (torch.abs(command[:, 2]) <= max_yaw_command)
    )
    crosstalk = torch.square(asset.data.root_lin_vel_b[:, cross_index]) + torch.square(
        yaw_scale * asset.data.root_ang_vel_b[:, 2]
    )
    return crosstalk * mask


def yaw_only_track_ang_vel_z_rel_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    lin_threshold: float = 0.05,
    yaw_threshold: float = 0.10,
    min_command: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw-only tracking reward using relative error so low yaw commands still matter."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name).to(asset.data.root_ang_vel_b.device)
    yaw_only = (torch.norm(command[:, :2], dim=1) <= lin_threshold) & (torch.abs(command[:, 2]) >= yaw_threshold)
    scale = torch.clamp(torch.abs(command[:, 2]), min=min_command)
    rel_error = (command[:, 2] - asset.data.root_ang_vel_b[:, 2]) / scale
    return torch.exp(-torch.square(rel_error) / std**2) * yaw_only


def yaw_only_translation_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    lin_threshold: float = 0.05,
    yaw_threshold: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize translational drift during pure yaw commands."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name).to(asset.data.root_lin_vel_b.device)
    yaw_only = (torch.norm(command[:, :2], dim=1) <= lin_threshold) & (torch.abs(command[:, 2]) >= yaw_threshold)
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1) * yaw_only


def zero_command_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    yaw_scale: float = 0.30,
    skip_push_active: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base motion at zero command, while allowing push recovery movement."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name).to(asset.data.root_lin_vel_b.device)
    zero_command = torch.norm(command, dim=1) <= command_threshold
    if skip_push_active:
        zero_command &= ~_push_active_mask(env, command_name, asset.data.root_lin_vel_b.device)
    velocity = torch.stack(
        [
            asset.data.root_lin_vel_b[:, 0],
            asset.data.root_lin_vel_b[:, 1],
            yaw_scale * asset.data.root_ang_vel_b[:, 2],
        ],
        dim=-1,
    )
    return torch.sum(torch.square(velocity), dim=1) * zero_command


def zero_command_joint_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    skip_push_active: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize unnecessary joint motion at zero command, except during push recovery."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name).to(asset.data.joint_vel.device)
    zero_command = torch.norm(command, dim=1) <= command_threshold
    if skip_push_active:
        zero_command &= ~_push_active_mask(env, command_name, asset.data.joint_vel.device)
    return torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1) * zero_command


def axis_yaw_track_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    axis: str,
    lin_std: float = 0.28,
    yaw_std: float = 0.45,
    min_lin_command: float = 0.10,
    min_yaw_command: float = 0.20,
    max_cross_command: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Secondary reward for vx+yaw or vy+yaw command samples."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name).to(asset.data.root_lin_vel_b.device)
    if axis == "x":
        axis_index, cross_index = 0, 1
    elif axis == "y":
        axis_index, cross_index = 1, 0
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}.")
    mask = (
        (torch.abs(command[:, axis_index]) >= min_lin_command)
        & (torch.abs(command[:, 2]) >= min_yaw_command)
        & (torch.abs(command[:, cross_index]) <= max_cross_command)
    )
    lin_error = torch.square(command[:, axis_index] - asset.data.root_lin_vel_b[:, axis_index])
    yaw_error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-(lin_error / lin_std**2 + yaw_error / yaw_std**2)) * mask


def axis_yaw_crosstalk_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    axis: str,
    min_lin_command: float = 0.10,
    min_yaw_command: float = 0.20,
    max_cross_command: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize unused translational axis during vx+yaw or vy+yaw command samples."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name).to(asset.data.root_lin_vel_b.device)
    if axis == "x":
        axis_index, cross_index = 0, 1
    elif axis == "y":
        axis_index, cross_index = 1, 0
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}.")
    mask = (
        (torch.abs(command[:, axis_index]) >= min_lin_command)
        & (torch.abs(command[:, 2]) >= min_yaw_command)
        & (torch.abs(command[:, cross_index]) <= max_cross_command)
    )
    return torch.square(asset.data.root_lin_vel_b[:, cross_index]) * mask


def push_recovery_velocity_track_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.65,
    yaw_scale: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward returning to the commanded velocity during externally pushed windows."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name).to(asset.data.root_lin_vel_b.device)
    actual = torch.stack(
        [
            asset.data.root_lin_vel_b[:, 0],
            asset.data.root_lin_vel_b[:, 1],
            yaw_scale * asset.data.root_ang_vel_b[:, 2],
        ],
        dim=-1,
    )
    target = torch.stack([command[:, 0], command[:, 1], yaw_scale * command[:, 2]], dim=-1)
    error = torch.sum(torch.square(actual - target), dim=1)
    return torch.exp(-error / std**2) * _push_active_mask(env, command_name, asset.data.root_lin_vel_b.device)


def push_recovery_delayed_velocity_track_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_elapsed_s: float = 0.35,
    max_elapsed_s: float = 0.0,
    std: float = 0.65,
    yaw_scale: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward return-to-command only after the early post-kick survival window."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name).to(asset.data.root_lin_vel_b.device)
    actual = torch.stack(
        [
            asset.data.root_lin_vel_b[:, 0],
            asset.data.root_lin_vel_b[:, 1],
            yaw_scale * asset.data.root_ang_vel_b[:, 2],
        ],
        dim=-1,
    )
    target = torch.stack([command[:, 0], command[:, 1], yaw_scale * command[:, 2]], dim=-1)
    error = torch.sum(torch.square(actual - target), dim=1)

    try:
        command_term = env.command_manager.get_term(command_name)
        active = command_term.metrics.get("failure_push_active")
        elapsed = command_term.metrics.get("failure_push_elapsed_s")
    except Exception:
        active = None
        elapsed = None
    if active is None or elapsed is None:
        return torch.zeros(env.num_envs, device=asset.data.root_lin_vel_b.device)

    active = active.to(asset.data.root_lin_vel_b.device).float().clamp(0.0, 1.0)
    elapsed = elapsed.to(asset.data.root_lin_vel_b.device).float()
    delayed = active * (elapsed >= float(min_elapsed_s)).float()
    if max_elapsed_s > 0.0:
        delayed = delayed * (elapsed <= float(max_elapsed_s)).float()
    return torch.exp(-error / std**2) * delayed


def recovery_push_velocity_cancel_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_elapsed_s: float = 0.0,
    max_elapsed_s: float = 0.60,
    min_push_norm: float = 0.05,
    std: float = 0.35,
    yaw_std: float = 1.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward quickly cancelling velocity that continues in the push direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    active, elapsed, push_delta_xy = _push_state(env, command_name, asset.data.root_lin_vel_b.device)
    push_norm = torch.linalg.norm(push_delta_xy, dim=1).clamp_min(1.0e-6)
    push_dir = push_delta_xy / push_norm.unsqueeze(-1)
    valid = active & (push_norm >= float(min_push_norm)) & (elapsed >= float(min_elapsed_s))
    if max_elapsed_s > 0.0:
        valid = valid & (elapsed <= float(max_elapsed_s))

    command = env.command_manager.get_command(command_name).to(asset.data.root_lin_vel_b.device)
    velocity_error_xy = asset.data.root_lin_vel_b[:, :2] - command[:, :2]
    outward_velocity = torch.sum(velocity_error_xy * push_dir, dim=1).clamp_min(0.0)
    yaw_error = torch.abs(asset.data.root_ang_vel_b[:, 2] - command[:, 2])
    reward = torch.exp(
        -(torch.square(outward_velocity) / float(std) ** 2 + torch.square(yaw_error) / float(yaw_std) ** 2)
    )
    return reward * valid.float()


def yaw_only_feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    lin_threshold: float = 0.15,
    yaw_threshold: float = 0.15,
) -> torch.Tensor:
    """Small stepping incentive for near-in-place yaw commands."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    command = env.command_manager.get_command(command_name)
    yaw_only = (torch.norm(command[:, :2], dim=1) < lin_threshold) & (torch.abs(command[:, 2]) > yaw_threshold)
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward * yaw_only


def body_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch angular velocity of selected body links."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.square(ang_vel_xy), dim=(1, 2))

def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    return torch.sum(torch.abs(angle), dim=1)

def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)


def recovery_upright_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense upright reward used while the recovery gate is active."""
    asset: Articulation = env.scene[asset_cfg.name]
    tilt_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-tilt_error / std**2)


def recovery_base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float = 0.57,
    std: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the root staying close to a recoverable standing height."""
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-height_error / std**2)


def recovery_ang_vel_xy_damp_exp(
    env: ManagerBasedRLEnv,
    std: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward damping roll/pitch angular velocity during recovery."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    return torch.exp(-ang_vel_error / std**2)


def recovery_no_fall_alive(
    env: ManagerBasedRLEnv,
    min_height: float = 0.35,
    max_tilt: float = 0.75,
    undesired_sensor_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Binary reward for staying in a recoverable non-fallen state."""
    asset: Articulation = env.scene[asset_cfg.name]
    alive = (asset.data.root_pos_w[:, 2] > min_height) & (
        torch.norm(asset.data.projected_gravity_b[:, :2], dim=1) < max_tilt
    )
    if undesired_sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[undesired_sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w_history
        bad_contact = torch.max(torch.norm(forces[:, :, undesired_sensor_cfg.body_ids], dim=-1), dim=1)[0]
        alive &= ~torch.any(bad_contact > contact_threshold, dim=1)
    return alive.float()


def recovery_no_base_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward recovery windows that avoid trunk/body contact entirely."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    bad_contact = torch.max(torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    return (~torch.any(bad_contact > contact_threshold, dim=1)).float()


def recovery_base_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    grace_s: float = 0.05,
) -> torch.Tensor:
    """Penalize trunk/body contact during the post-push recovery window after a short grace period."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    bad_contact = torch.max(torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    bad_contact = torch.any(bad_contact > contact_threshold, dim=1).float()
    active, elapsed, _ = _push_state(env, command_name, bad_contact.device)
    after_grace = active.float() * (elapsed >= float(grace_s)).float()
    return bad_contact * after_grace


def recovery_foot_placement_exp(
    env: ManagerBasedRLEnv,
    foot_asset_cfg: SceneEntityCfg,
    target_height: float = 0.57,
    std: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward feet being close enough to the approximate capture point."""
    from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.robustness import capture_point_error

    error = capture_point_error(env, foot_asset_cfg=foot_asset_cfg, target_height=target_height, asset_cfg=asset_cfg)
    return torch.exp(-torch.square(error) / std**2)


def recovery_push_capture_point_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    foot_asset_cfg: SceneEntityCfg,
    target_height: float = 0.57,
    std: float = 0.35,
    min_elapsed_s: float = 0.0,
    max_elapsed_s: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Push-window capture-point reward so the nearest support foot catches the moving base."""
    from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.robustness import capture_point_error

    asset: Articulation = env.scene[asset_cfg.name]
    active, elapsed, _ = _push_state(env, command_name, asset.data.root_lin_vel_w.device)
    mask = active.float() * (elapsed >= float(min_elapsed_s)).float()
    if max_elapsed_s > 0.0:
        mask = mask * (elapsed <= float(max_elapsed_s)).float()
    error = capture_point_error(env, foot_asset_cfg=foot_asset_cfg, target_height=target_height, asset_cfg=asset_cfg)
    return torch.exp(-torch.square(error) / std**2) * mask


def recovery_directional_step_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    foot_asset_cfg: SceneEntityCfg,
    min_projection: float = 0.10,
    target_projection: float = 0.26,
    projection_std: float = 0.20,
    lateral_std: float = 0.35,
    min_push_norm: float = 0.05,
    min_elapsed_s: float = 0.0,
    max_elapsed_s: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward placing at least one foot in the direction of the latest external impulse."""
    asset: Articulation = env.scene[asset_cfg.name]
    active, elapsed, push_delta_xy = _push_state(env, command_name, asset.data.root_lin_vel_b.device)
    push_norm = torch.linalg.norm(push_delta_xy, dim=1).clamp_min(1.0e-6)
    push_dir = push_delta_xy / push_norm.unsqueeze(-1)
    valid = active & (push_norm >= float(min_push_norm)) & (elapsed >= float(min_elapsed_s))
    if max_elapsed_s > 0.0:
        valid = valid & (elapsed <= float(max_elapsed_s))

    foot_xy_b = _foot_positions_b(env, foot_asset_cfg=foot_asset_cfg, asset_cfg=asset_cfg)[:, :, :2]
    projection = torch.sum(foot_xy_b * push_dir.unsqueeze(1), dim=-1)
    lateral = torch.abs(foot_xy_b[:, :, 0] * push_dir[:, 1:2] - foot_xy_b[:, :, 1] * push_dir[:, 0:1])
    best_projection, best_index = torch.max(projection, dim=1)
    best_lateral = torch.gather(lateral, 1, best_index.unsqueeze(1)).squeeze(1)

    projection_error = torch.clamp(float(target_projection) - best_projection, min=0.0)
    min_projection_gate = (best_projection >= float(min_projection)).float()
    projection_reward = torch.exp(-torch.square(projection_error) / projection_std**2)
    lateral_reward = torch.exp(-torch.square(best_lateral) / lateral_std**2)
    return projection_reward * lateral_reward * min_projection_gate * valid.float()


def recovery_directional_support_contact_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    foot_asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_projection: float = 0.08,
    target_projection: float = 0.30,
    projection_std: float = 0.18,
    lateral_std: float = 0.32,
    contact_threshold: float = 1.0,
    min_push_norm: float = 0.05,
    min_elapsed_s: float = 0.04,
    max_elapsed_s: float = 1.60,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a contacted support foot placed in the latest push direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    active, elapsed, push_delta_xy = _push_state(env, command_name, asset.data.root_lin_vel_b.device)
    push_norm = torch.linalg.norm(push_delta_xy, dim=1).clamp_min(1.0e-6)
    push_dir = push_delta_xy / push_norm.unsqueeze(-1)
    valid = active & (push_norm >= float(min_push_norm)) & (elapsed >= float(min_elapsed_s))
    if max_elapsed_s > 0.0:
        valid = valid & (elapsed <= float(max_elapsed_s))

    foot_xy_b = _foot_positions_b(env, foot_asset_cfg=foot_asset_cfg, asset_cfg=asset_cfg)[:, :, :2]
    projection = torch.sum(foot_xy_b * push_dir.unsqueeze(1), dim=-1)
    lateral = torch.abs(foot_xy_b[:, :, 0] * push_dir[:, 1:2] - foot_xy_b[:, :, 1] * push_dir[:, 0:1])

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    in_contact = torch.max(torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > contact_threshold
    in_contact = in_contact.to(device=projection.device)
    if in_contact.shape != projection.shape:
        raise RuntimeError(
            "recovery_directional_support_contact_exp requires matching foot asset and contact sensor body ids, "
            f"got contact shape {tuple(in_contact.shape)} and foot shape {tuple(projection.shape)}."
        )

    projection_error = torch.clamp(float(target_projection) - projection, min=0.0)
    projection_reward = torch.exp(-torch.square(projection_error) / projection_std**2)
    lateral_reward = torch.exp(-torch.square(lateral) / lateral_std**2)
    projection_gate = (projection >= float(min_projection)).float()
    support_score = projection_reward * lateral_reward * projection_gate * in_contact.float()
    best_support = torch.max(support_score, dim=1)[0]
    return best_support * valid.float()


def recovery_directional_first_contact_step_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    foot_asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_projection: float = 0.08,
    target_projection: float = 0.22,
    projection_std: float = 0.16,
    lateral_std: float = 0.28,
    contact_threshold: float = 1.0,
    min_push_norm: float = 0.02,
    min_elapsed_s: float = 0.10,
    max_elapsed_s: float = 2.40,
    min_air_time_s: float = 0.06,
    max_command_norm: float = 0.10,
    command_yaw_scale: float = 0.30,
    require_force_active: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a new contacted support step in the push direction.

    Unlike support-position rewards, this only pays on a fresh foot contact after
    air time, which discourages solving slow pushes with ankle/toe bracing alone.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    active, elapsed, push_delta_xy = _push_state(env, command_name, asset.data.root_lin_vel_b.device)
    push_norm = torch.linalg.norm(push_delta_xy, dim=1).clamp_min(1.0e-6)
    push_dir = push_delta_xy / push_norm.unsqueeze(-1)
    valid = active & (push_norm >= float(min_push_norm)) & (elapsed >= float(min_elapsed_s))
    if max_elapsed_s > 0.0:
        valid = valid & (elapsed <= float(max_elapsed_s))
    if require_force_active:
        valid = valid & _force_push_active_mask(env, command_name, asset.data.root_lin_vel_b.device)
    if max_command_norm > 0.0:
        command = env.command_manager.get_command(command_name).to(asset.data.root_lin_vel_b.device)
        command_norm = torch.norm(
            torch.stack(
                (
                    command[:, 0],
                    command[:, 1],
                    max(float(command_yaw_scale), 0.0) * command[:, 2],
                ),
                dim=-1,
            ),
            dim=-1,
        )
        valid = valid & (command_norm <= float(max_command_norm))

    foot_xy_b = _foot_positions_b(env, foot_asset_cfg=foot_asset_cfg, asset_cfg=asset_cfg)[:, :, :2]
    projection = torch.sum(foot_xy_b * push_dir.unsqueeze(1), dim=-1)
    lateral = torch.abs(foot_xy_b[:, :, 0] * push_dir[:, 1:2] - foot_xy_b[:, :, 1] * push_dir[:, 0:1])

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].to(device=projection.device)
    first_contact = first_contact > 0
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids].to(device=projection.device)
    if first_contact.shape != projection.shape:
        raise RuntimeError(
            "recovery_directional_first_contact_step_exp requires matching foot asset and contact sensor body ids, "
            f"got contact shape {tuple(first_contact.shape)} and foot shape {tuple(projection.shape)}."
        )

    projection_error = torch.clamp(float(target_projection) - projection, min=0.0)
    projection_reward = torch.exp(-torch.square(projection_error) / projection_std**2)
    lateral_reward = torch.exp(-torch.square(lateral) / lateral_std**2)
    projection_gate = projection >= float(min_projection)
    air_gate = last_air_time >= float(min_air_time_s)
    step_score = projection_reward * lateral_reward * projection_gate.float() * first_contact.float() * air_gate.float()
    best_step = torch.max(step_score, dim=1)[0]
    return best_step * valid.float()


def recovery_body_frame_stance_width(
    env: ManagerBasedRLEnv,
    foot_asset_cfg: SceneEntityCfg,
    min_width: float = 0.12,
    target_width: float = 0.26,
    max_width: float = 0.55,
    std: float = 0.16,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a non-crossed stance measured in the robot body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = list(foot_asset_cfg.body_ids)
    left_ids = [body_id for body_id in body_ids if "left" in asset.body_names[body_id].lower()]
    right_ids = [body_id for body_id in body_ids if "right" in asset.body_names[body_id].lower()]
    feet_b = _foot_positions_b(env, foot_asset_cfg=foot_asset_cfg, asset_cfg=asset_cfg)
    if left_ids and right_ids:
        local_index = {body_id: index for index, body_id in enumerate(body_ids)}
        left_pos = feet_b[:, [local_index[body_id] for body_id in left_ids], :2].mean(dim=1)
        right_pos = feet_b[:, [local_index[body_id] for body_id in right_ids], :2].mean(dim=1)
    else:
        left_pos = feet_b[:, 0, :2]
        right_pos = feet_b[:, min(1, feet_b.shape[1] - 1), :2]
    signed_width = left_pos[:, 1] - right_pos[:, 1]
    width = torch.abs(signed_width)
    in_range = ((signed_width > 0.0) & (width >= float(min_width)) & (width <= float(max_width))).float()
    width_error = torch.square(width - float(target_width))
    return torch.exp(-width_error / std**2) * in_range


def recovery_return_to_command_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.6,
    yaw_scale: float = 0.30,
    stable_score_threshold: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward command tracking again once the body is nearly stable."""
    from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.robustness import disturbance_score

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name).to(asset.data.root_lin_vel_b.device)
    actual = torch.stack(
        [
            asset.data.root_lin_vel_b[:, 0],
            asset.data.root_lin_vel_b[:, 1],
            yaw_scale * asset.data.root_ang_vel_b[:, 2],
        ],
        dim=-1,
    )
    target = torch.stack([command[:, 0], command[:, 1], yaw_scale * command[:, 2]], dim=-1)
    error = torch.sum(torch.square(actual - target), dim=1)
    stable = disturbance_score(
        env,
        command_name=command_name,
        yaw_scale=yaw_scale,
        contact_weight=0.0,
        slip_weight=0.0,
        asset_cfg=asset_cfg,
    ) < stable_score_threshold
    return torch.exp(-error / std**2) * stable


def recovery_feet_support_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    min_contacts: int = 1,
) -> torch.Tensor:
    """Reward maintaining at least one supporting foot contact during recovery."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    in_contact = torch.max(torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > contact_threshold
    return (torch.sum(in_contact.float(), dim=1) >= min_contacts).float()


def recovery_lateral_step_allowance(
    env: ManagerBasedRLEnv,
    foot_asset_cfg: SceneEntityCfg,
    min_width: float = 0.12,
    max_width: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a non-crossed but not excessively wide stance during recovery."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = list(foot_asset_cfg.body_ids)
    left_ids = [body_id for body_id in body_ids if "left" in asset.body_names[body_id].lower()]
    right_ids = [body_id for body_id in body_ids if "right" in asset.body_names[body_id].lower()]
    if left_ids and right_ids:
        left_pos = asset.data.body_link_pos_w[:, left_ids, :2].mean(dim=1)
        right_pos = asset.data.body_link_pos_w[:, right_ids, :2].mean(dim=1)
    else:
        foot_pos = asset.data.body_link_pos_w[:, body_ids[:2], :2]
        left_pos = foot_pos[:, 0]
        right_pos = foot_pos[:, 1]
    width = torch.norm(left_pos - right_pos, dim=1)
    lower_margin = torch.clamp(width - min_width, min=0.0)
    upper_margin = torch.clamp(max_width - width, min=0.0)
    return torch.clamp(torch.minimum(lower_margin, upper_margin) / max(min_width, 1.0e-6), 0.0, 1.0)


def _sudden_stop_mask(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    moving_speed_threshold: float = 0.10,
    yaw_scale: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name).to(asset.data.root_lin_vel_b.device)
    zero_command = torch.norm(command, dim=1) < command_threshold
    actual_speed = torch.norm(
        torch.stack(
            [asset.data.root_lin_vel_b[:, 0], asset.data.root_lin_vel_b[:, 1], yaw_scale * asset.data.root_ang_vel_b[:, 2]],
            dim=-1,
        ),
        dim=1,
    )
    moving = actual_speed > moving_speed_threshold
    try:
        command_term = env.command_manager.get_term(command_name)
        timer = getattr(command_term, "sudden_stop_timer", None)
    except Exception:
        timer = None
    if timer is not None:
        return ((timer.to(asset.data.root_lin_vel_b.device) > 0.0) | (zero_command & moving)).float()
    return (zero_command & moving).float()


def recovery_sudden_stop_upright_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.30,
    command_threshold: float = 0.05,
    moving_speed_threshold: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward staying upright while recovering from an abrupt all-zero command."""
    asset: Articulation = env.scene[asset_cfg.name]
    tilt_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    mask = _sudden_stop_mask(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        moving_speed_threshold=moving_speed_threshold,
        asset_cfg=asset_cfg,
    )
    return torch.exp(-tilt_error / std**2) * mask


def recovery_sudden_stop_velocity_damp_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.45,
    yaw_scale: float = 0.30,
    command_threshold: float = 0.05,
    moving_speed_threshold: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward quickly damping residual base velocity after command snaps to zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_error = torch.sum(
        torch.square(
            torch.stack(
                [
                    asset.data.root_lin_vel_b[:, 0],
                    asset.data.root_lin_vel_b[:, 1],
                    yaw_scale * asset.data.root_ang_vel_b[:, 2],
                ],
                dim=-1,
            )
        ),
        dim=1,
    )
    mask = _sudden_stop_mask(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        moving_speed_threshold=moving_speed_threshold,
        yaw_scale=yaw_scale,
        asset_cfg=asset_cfg,
    )
    return torch.exp(-velocity_error / std**2) * mask


def recovery_sudden_stop_default_pose_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 1.6,
    command_threshold: float = 0.05,
    moving_speed_threshold: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward settling toward default joints after an abrupt stop, without forcing it during braking."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = torch.mean(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    mask = _sudden_stop_mask(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        moving_speed_threshold=moving_speed_threshold,
        asset_cfg=asset_cfg,
    )
    return torch.exp(-joint_error / std) * mask


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal foot velocity while the foot is in contact.

    Uses a contact threshold on net_forces_w to detect ground contact, then
    accumulates the horizontal (xy) linear velocity magnitude of the foot
    bodies that are currently in contact. Encourages the policy to reach
    commanded base speeds through real stepping rather than by sliding feet
    against the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # shape: (num_envs, history, num_bodies, 3)
    forces_history = contact_sensor.data.net_forces_w_history
    # max over history to be robust to single-step noise
    in_contact = (
        torch.max(torch.norm(forces_history[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
    )  # (num_envs, num_bodies)
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel_xy = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]  # (num_envs, num_bodies, 2)
    speed_xy = torch.norm(body_vel_xy, dim=-1)  # (num_envs, num_bodies)
    return torch.sum(speed_xy * in_contact, dim=1)

def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _get_adaptive_sigma(env, key: str | float, error: Union[float, torch.Tensor]):
    if isinstance(key, float):
        return key
    sigma_update_rate = 0.9
    if not hasattr(env, 'reward_sigmas_ema'):
        env.reward_sigmas_ema = {}
        env.reward_sigmas = {}

    env.reward_sigmas_ema[key] = (
        sigma_update_rate * env.reward_sigmas_ema.get(key, torch.tensor([100.], device=env.device)) + (1 - sigma_update_rate) * error
    )
    env.reward_sigmas[key] = torch.minimum(env.reward_sigmas_ema[key], env.reward_sigmas.get(key, torch.tensor([100.], device=env.device))).clip(min=1e-8)
    return torch.sqrt(env.reward_sigmas[key])


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float | str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
        env: ManagerBasedRLEnv, command_name: str, std: float | str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def feet_stance_time(
        env: ManagerBasedRLEnv, asset_name: str, feet_names: list[str], vel_threshold: float, desired_time: float
) -> torch.Tensor:
    if not hasattr(env, '_buf_feet_stance_time'):
        env._buf_feet_stance_time = torch.zeros(env.num_envs, 2, device=env.device)

    robot = env.scene.articulations[asset_name]
    feet_indexes = [robot.body_names.index(name) for name in feet_names]

    stance = robot.data.body_link_lin_vel_w[:, feet_indexes].norm(dim=-1) < vel_threshold

    first_slide = (env._buf_feet_stance_time > 0.) * (~stance)
    rew_stanceTime = torch.sum((env._buf_feet_stance_time - desired_time).clip(max=0.) * first_slide, dim=1)

    env._buf_feet_stance_time += env.step_dt
    env._buf_feet_stance_time *= stance
    return rew_stanceTime

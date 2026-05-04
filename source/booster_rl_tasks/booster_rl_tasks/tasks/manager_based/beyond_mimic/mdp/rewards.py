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

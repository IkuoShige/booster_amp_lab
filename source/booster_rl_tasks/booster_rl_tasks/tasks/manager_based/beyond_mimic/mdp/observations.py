from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.commands import MotionCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate
from isaaclab.sensors import Camera, ContactSensor, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def get_lefthand_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """AMP type observations"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = getattr(env, "device", None)
    if device is None:
        try:
            device = env.scene["robot"].device
        except Exception:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(device)
    elbow_body_ids, _ = asset.find_bodies(name_keys=["left_hand_link", "right_hand_link"], preserve_order=True)
    left_arm_local_vec = torch.tensor([0.0, 0.2, 0.0], device=device).repeat((env.num_envs, 1))
    left_hand_pos = (asset.data.body_state_w[:, elbow_body_ids[0], :3] - asset.data.root_state_w[:, 0:3] + quat_apply(asset.data.body_state_w[:, elbow_body_ids[0], 3:7], left_arm_local_vec))
    left_hand_pos = quat_apply(quat_conjugate(asset.data.root_state_w[:, 3:7]), left_hand_pos)

    return left_hand_pos

def get_righthand_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """AMP type observations"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elbow_body_ids, _ = asset.find_bodies(name_keys=["left_hand_link", "right_hand_link"], preserve_order=True)
    right_arm_local_vec = torch.tensor([0.0, -0.2, 0.0], device=device).repeat((env.num_envs, 1))
    right_hand_pos = (asset.data.body_state_w[:, elbow_body_ids[1], :3] - asset.data.root_state_w[:, 0:3] + quat_apply(asset.data.body_state_w[:, elbow_body_ids[1], 3:7], right_arm_local_vec))
    right_hand_pos = quat_apply(quat_conjugate(asset.data.root_state_w[:, 3:7]), right_hand_pos)

    return right_hand_pos

def get_leftfoot_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """AMP type observations"""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_body_ids, _ = asset.find_bodies(name_keys=["left_foot_link", "right_foot_link"], preserve_order=True)
    left_foot_pos = (asset.data.body_state_w[:, feet_body_ids[0], :3] - asset.data.root_state_w[:, 0:3])
    left_foot_pos = quat_apply(quat_conjugate(asset.data.root_state_w[:, 3:7]), left_foot_pos)

    return left_foot_pos

def get_rightfoot_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """AMP type observations"""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_body_ids, _ = asset.find_bodies(name_keys=["left_foot_link", "right_foot_link"], preserve_order=True)
    right_foot_pos = (asset.data.body_state_w[:, feet_body_ids[1], :3] - asset.data.root_state_w[:, 0:3])
    right_foot_pos = quat_apply(quat_conjugate(asset.data.root_state_w[:, 3:7]), right_foot_pos)

    return right_foot_pos

def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

def robot_joint_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint torque of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.applied_torque.to(device)


def robot_joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint acc of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.joint_acc.to(device)


def robot_feet_contact_force(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg):
    """contact force of the robot feet"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    contact_force_tensor = contact_sensor.data.net_forces_w_history.to(device)
    return contact_force_tensor.view(contact_force_tensor.shape[0], -1)


def robot_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_mass.to(device)


def robot_inertia(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """inertia of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inertia_tensor = asset.data.default_inertia.to(device)
    return inertia_tensor.view(inertia_tensor.shape[0], -1)


def robot_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint positions of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_pos.to(device)


def robot_joint_stiffness(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint stiffness of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_stiffness.to(device)


def robot_joint_damping(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint damping of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_damping.to(device)
def robot_base_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot base"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pose_w[:,3:].to(device)

def robot_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)


def robot_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """velocity of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_vel_w.to(device)


def robot_material_properties(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """material properties of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    material_tensor = asset.root_physx_view.get_material_properties().to(device)
    return material_tensor.view(material_tensor.shape[0], -1)


def robot_center_of_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """center of mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    com_tensor = asset.root_physx_view.get_coms().clone().to(device)
    return com_tensor.view(com_tensor.shape[0], -1)


def robot_contact_force(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The contact forces of the body."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    body_contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]

    return body_contact_force.reshape(body_contact_force.shape[0], -1)


def track_adapter_recovery_privileged_state(
    env: ManagerBasedEnv,
    command_name: str = "base_velocity",
    force_scale: float = 300.0,
    duration_scale: float = 3.0,
    foot_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=[".*_foot_.*"]),
    trunk_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["Trunk"]),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Privileged recovery-only features for Track Adapter teacher training.

    The deployable actor observation is intentionally unchanged.  These
    features are only added to the critic/privileged group when explicitly
    enabled, so the recovery teacher can see physical push state and contacts.
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_envs = env.num_envs
    force_scale = max(float(force_scale), 1.0e-6)
    duration_scale = max(float(duration_scale), 1.0e-6)

    def _resolve_body_ids(contact_sensor: ContactSensor, cfg: SceneEntityCfg):
        body_ids = getattr(cfg, "body_ids", None)
        if body_ids is None:
            body_names = getattr(cfg, "body_names", None)
            if body_names is not None and hasattr(contact_sensor, "find_bodies"):
                try:
                    body_ids, _ = contact_sensor.find_bodies(body_names, preserve_order=True)
                except Exception:
                    body_ids = None
        if body_ids is None:
            return slice(None)
        return body_ids

    def _body_id_list(body_ids, body_count: int) -> list[int]:
        if body_ids is None:
            return list(range(body_count))
        if isinstance(body_ids, slice):
            return list(range(body_count))[body_ids]
        if isinstance(body_ids, int):
            return [int(body_ids)]
        if isinstance(body_ids, torch.Tensor):
            return [int(index) for index in body_ids.detach().cpu().reshape(-1).tolist()]
        return [int(index) for index in body_ids]

    force_xy = torch.zeros(num_envs, 2, device=device)
    force_norm = torch.zeros(num_envs, 1, device=device)
    force_remaining = torch.zeros(num_envs, 1, device=device)
    force_elapsed = torch.zeros(num_envs, 1, device=device)
    force_duration = torch.zeros(num_envs, 1, device=device)

    states = getattr(env, "_booster_external_wrench_pulse_states", None)
    if isinstance(states, dict):
        for state in states.values():
            forces = state.get("forces") if isinstance(state, dict) else None
            if not isinstance(forces, torch.Tensor) or forces.numel() == 0:
                continue
            forces = forces.to(device)
            resultant = forces.sum(dim=1)
            norms = torch.norm(resultant[:, :2], dim=-1, keepdim=True)
            replace = norms.squeeze(-1) > force_norm.squeeze(-1)
            if torch.any(replace):
                force_xy[replace] = resultant[replace, :2] / force_scale
                force_norm[replace] = norms[replace] / force_scale
                remaining = state.get("remaining_s")
                duration = state.get("duration_s")
                if isinstance(remaining, torch.Tensor):
                    force_remaining[replace] = remaining.to(device)[replace].unsqueeze(-1) / duration_scale
                if isinstance(duration, torch.Tensor):
                    duration_value = duration.to(device)[replace].unsqueeze(-1)
                    force_duration[replace] = duration_value / duration_scale
                    elapsed_value = torch.clamp(duration_value - force_remaining[replace] * duration_scale, min=0.0)
                    force_elapsed[replace] = elapsed_value / duration_scale

    command_features = torch.zeros(num_envs, 9, device=device)
    command_manager = getattr(env, "command_manager", None)
    if command_manager is not None and hasattr(command_manager, "get_term"):
        try:
            command_term = command_manager.get_term(command_name)
            metrics = getattr(command_term, "metrics", {})
            for index, name in enumerate(
                (
                    "force_push_active",
                    "force_push_started",
                    "force_push_curriculum_alpha",
                    "force_push_applied_force_n",
                    "force_push_force_max_n",
                    "failure_push_elapsed_s",
                    "failure_push_delta_x",
                    "failure_push_delta_y",
                    "failure_push_delta_norm",
                )
            ):
                value = metrics.get(name) if isinstance(metrics, dict) else None
                if isinstance(value, torch.Tensor):
                    value = value.to(device=device, dtype=command_features.dtype).reshape(-1)
                    if value.shape[0] == num_envs:
                        command_features[:, index] = value
            command_features[:, 3:5] = command_features[:, 3:5] / force_scale
            command_features[:, 5] = command_features[:, 5] / duration_scale
        except Exception:
            pass

    foot_contact = torch.zeros(num_envs, 2, device=device)
    try:
        contact_sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w_history.to(device)
        body_ids = _resolve_body_ids(contact_sensor, foot_sensor_cfg)
        body_indices = _body_id_list(body_ids, forces.shape[2])
        contact_norm = torch.max(torch.norm(forces[:, :, body_ids], dim=-1), dim=1)[0]
        contacts = contact_norm > float(contact_threshold)
        body_names = list(getattr(contact_sensor, "body_names", []))
        selected_names = [body_names[index] for index in body_indices if index < len(body_names)]
        left_ids = [index for index, name in enumerate(selected_names) if "left" in name.lower()]
        right_ids = [index for index, name in enumerate(selected_names) if "right" in name.lower()]
        if left_ids:
            foot_contact[:, 0] = torch.any(contacts[:, left_ids], dim=1).float()
        elif contacts.shape[1] > 0:
            foot_contact[:, 0] = contacts[:, 0].float()
        if right_ids:
            foot_contact[:, 1] = torch.any(contacts[:, right_ids], dim=1).float()
        elif contacts.shape[1] > 1:
            foot_contact[:, 1] = contacts[:, 1].float()
    except Exception:
        pass

    trunk_contact = torch.zeros(num_envs, 1, device=device)
    try:
        contact_sensor: ContactSensor = env.scene.sensors[trunk_sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w_history.to(device)
        body_ids = _resolve_body_ids(contact_sensor, trunk_sensor_cfg)
        contact_norm = torch.max(torch.norm(forces[:, :, body_ids], dim=-1), dim=1)[0]
        trunk_contact[:, 0] = torch.any(contact_norm > float(contact_threshold), dim=1).float()
    except Exception:
        pass

    return torch.cat(
        (
            force_xy,
            force_norm,
            force_remaining,
            force_elapsed,
            force_duration,
            command_features,
            foot_contact,
            trunk_contact,
        ),
        dim=-1,
    )

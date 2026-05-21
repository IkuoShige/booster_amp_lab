from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def finite_clamp(x: torch.Tensor, min_value: float = 0.0, max_value: float = 1.0e6) -> torch.Tensor:
    """Replace non-finite values and clamp to a bounded range."""
    return torch.nan_to_num(x, nan=max_value, posinf=max_value, neginf=min_value).clamp(min_value, max_value)


def base_tilt_norm(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return finite_clamp(torch.norm(asset.data.projected_gravity_b[:, :2], dim=1), max_value=10.0)


def base_ang_vel_xy_norm(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return finite_clamp(torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1), max_value=50.0)


def velocity_tracking_error_norm(
    env,
    command_name: str = "base_velocity",
    yaw_scale: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
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
    return finite_clamp(torch.norm(actual - target, dim=1), max_value=50.0)


def base_height_error(
    env,
    target_height: float = 0.57,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return finite_clamp(torch.abs(asset.data.root_pos_w[:, 2] - target_height), max_value=10.0)


def body_contact_indicator(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    in_contact = torch.max(torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.any(in_contact, dim=1).float()


def illegal_contact_mask(
    env,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    contact_force = torch.max(torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    return torch.any(contact_force > threshold, dim=1)


def _record_command_failure(
    env,
    failure_mask: torch.Tensor,
    command_name: str | None,
    base_contact_mask: torch.Tensor | None = None,
):
    if not command_name or not torch.any(failure_mask):
        return
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None or not hasattr(command_manager, "get_term"):
        return
    try:
        command_term = command_manager.get_term(command_name)
    except Exception:
        return
    recorder = getattr(command_term, "record_failure_mask", None)
    if callable(recorder):
        recorder(failure_mask, base_contact_mask=base_contact_mask if base_contact_mask is not None else failure_mask)


def illegal_contact_recording(
    env,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    command_name: str | None = "base_velocity",
) -> torch.Tensor:
    """Illegal-contact termination that records failures into Track Adapter mining bins."""
    contact = illegal_contact_mask(env, threshold, sensor_cfg)
    _record_command_failure(env, contact, command_name, base_contact_mask=contact)
    return contact


def illegal_contact_after_duration(
    env,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    duration_s: float = 0.0,
    command_name: str | None = "base_velocity",
) -> torch.Tensor:
    """Terminate only after undesired contact persists for a short duration.

    Contact is still recorded immediately for failure mining, but PPO receives a
    brief recovery window instead of ending on the first trunk touch.
    """
    contact = illegal_contact_mask(env, threshold, sensor_cfg)
    _record_command_failure(env, contact, command_name, base_contact_mask=contact)
    if duration_s <= 0.0:
        return contact

    timers = getattr(env, "_booster_illegal_contact_timers", None)
    if timers is None:
        timers = {}
        setattr(env, "_booster_illegal_contact_timers", timers)
    key = (sensor_cfg.name, str(sensor_cfg.body_ids))
    timer = timers.get(key)
    if timer is None or timer.shape[0] != env.scene.num_envs:
        timer = torch.zeros(env.scene.num_envs, device=contact.device)
        timers[key] = timer
    step_dt = float(getattr(env, "step_dt", 0.02))
    timer[contact] += step_dt
    timer[~contact] = 0.0
    return timer >= float(duration_s)


def foot_slip_norm(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    in_contact = torch.max(torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > contact_threshold
    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel_xy = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
    slip = torch.sum(torch.norm(foot_vel_xy, dim=-1) * in_contact, dim=1)
    return finite_clamp(slip, max_value=50.0)


def capture_point_error(
    env,
    foot_asset_cfg: SceneEntityCfg,
    target_height: float = 0.57,
    gravity: float = 9.81,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Approximate capture-point distance to the closest foot in the horizontal plane."""
    asset: Articulation = env.scene[asset_cfg.name]
    omega = torch.sqrt(
        torch.tensor(gravity / max(target_height, 1.0e-3), device=asset.device, dtype=torch.float32)
    )
    com_xy = asset.data.root_pos_w[:, :2]
    vel_xy = asset.data.root_lin_vel_w[:, :2]
    cp_xy = com_xy + vel_xy / omega
    foot_xy = asset.data.body_link_pos_w[:, foot_asset_cfg.body_ids, :2]
    distances = torch.norm(cp_xy.unsqueeze(1) - foot_xy, dim=-1)
    return finite_clamp(torch.min(distances, dim=1)[0], max_value=50.0)


def disturbance_score(
    env,
    command_name: str = "base_velocity",
    yaw_scale: float = 0.30,
    target_height: float = 0.57,
    tilt_weight: float = 1.8,
    ang_vel_weight: float = 0.25,
    velocity_weight: float = 0.75,
    height_weight: float = 3.0,
    capture_weight: float = 0.0,
    contact_weight: float = 1.5,
    slip_weight: float = 0.25,
    undesired_contact_sensor_cfg: SceneEntityCfg | None = None,
    foot_sensor_cfg: SceneEntityCfg | None = None,
    foot_asset_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_score: float = 8.0,
) -> torch.Tensor:
    """Weighted disturbance score used by recovery-gated AMP."""
    score = (
        tilt_weight * base_tilt_norm(env, asset_cfg)
        + ang_vel_weight * base_ang_vel_xy_norm(env, asset_cfg)
        + velocity_weight * velocity_tracking_error_norm(env, command_name, yaw_scale, asset_cfg)
        + height_weight * base_height_error(env, target_height, asset_cfg)
    )
    if capture_weight > 0.0 and foot_asset_cfg is not None:
        score = score + capture_weight * capture_point_error(env, foot_asset_cfg, target_height, asset_cfg=asset_cfg)
    if contact_weight > 0.0 and undesired_contact_sensor_cfg is not None:
        score = score + contact_weight * body_contact_indicator(env, undesired_contact_sensor_cfg)
    if slip_weight > 0.0 and foot_sensor_cfg is not None and foot_asset_cfg is not None:
        score = score + slip_weight * foot_slip_norm(env, foot_sensor_cfg, foot_asset_cfg)
    return finite_clamp(score, max_value=max_score)


def style_gate_from_score(
    score: torch.Tensor,
    beta: float = 1.0,
    min_gate: float = 0.0,
    max_gate: float = 1.0,
) -> torch.Tensor:
    gate = torch.exp(-float(beta) * finite_clamp(score, max_value=50.0))
    return finite_clamp(gate, min_value=min_gate, max_value=max_gate)


def command_gate_from_style_gate(style_gate: torch.Tensor, min_command_weight: float = 0.35) -> torch.Tensor:
    min_command_weight = float(min_command_weight)
    return min_command_weight + (1.0 - min_command_weight) * finite_clamp(style_gate, max_value=1.0)

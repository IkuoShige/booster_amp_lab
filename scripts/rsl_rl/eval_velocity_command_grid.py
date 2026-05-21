"""Evaluate fixed velocity-command tracking and simple gait symmetry metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate fixed velocity command tracking for RSL-RL policies.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=350)
parser.add_argument("--settle_steps", type=int, default=100)
parser.add_argument(
    "--commands",
    type=str,
    default="0.0,0.0,0.0;0.2,0.0,0.0;0.4,0.0,0.0;0.8,0.0,0.0",
    help="Semicolon-separated vx,vy,wz commands.",
)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--output_csv", type=str, default=None)
parser.add_argument("--disable_push", action="store_true", default=False)
parser.add_argument("--push_xy", type=float, default=None)
parser.add_argument("--push_yaw", type=float, default=None)
parser.add_argument("--push_interval_min", type=float, default=None)
parser.add_argument("--push_interval_max", type=float, default=None)
parser.add_argument(
    "--non_strict_checkpoint_load",
    action="store_true",
    default=False,
    help="Allow loading older Track Adapter checkpoints with missing auxiliary state keys.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_conjugate
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner, TrackAdapterRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_rl_tasks.tasks  # noqa: F401


def _split_obs(obs_td):
    groups = dict(obs_td.items()) if hasattr(obs_td, "items") else dict(obs_td)
    policy_obs = groups.pop("policy")
    return policy_obs, groups


class _UnwrappedProxy:
    def __init__(self, unwrapped):
        self.env = unwrapped


class _LegacyRslRlEnv:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def env(self):
        return _UnwrappedProxy(self._env.unwrapped)

    def get_observations(self):
        obs, others = _split_obs(self._env.get_observations())
        return obs, {"observations": others}

    def reset(self):
        obs_td, extras = self._env.reset()
        obs, others = _split_obs(obs_td)
        extras = dict(extras) if extras is not None else {}
        extras["observations"] = others
        return obs, extras

    def step(self, actions):
        obs_td, rew, dones, infos = self._env.step(actions)
        obs, others = _split_obs(obs_td)
        infos = dict(infos) if infos is not None else {}
        infos["observations"] = others
        return obs, rew, dones, infos

    def close(self):
        return self._env.close()


def _parse_commands(value: str):
    commands = []
    for raw in value.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        parts = [float(part.strip()) for part in raw.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Command must have vx,vy,wz: {raw}")
        commands.append(tuple(parts))
    if not commands:
        raise ValueError("At least one command is required.")
    return commands


def _resolve_base_contact(env_cfg, base_env):
    threshold = 2.0
    terminations = getattr(env_cfg, "terminations", None)
    base_contact = getattr(terminations, "base_contact", None)
    params = getattr(base_contact, "params", {}) if base_contact is not None else {}
    threshold = float(params.get("threshold", threshold))
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=["Trunk"])
    sensor_cfg.resolve(base_env.scene)
    return sensor_cfg, threshold


def _contact_mask(base_env, sensor_cfg: SceneEntityCfg, threshold: float):
    contact_sensor = base_env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    force = torch.max(torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    return torch.any(force > threshold, dim=1)


def _foot_contact_rates(base_env, foot_sensor_cfg: SceneEntityCfg, threshold: float = 1.0):
    contact_sensor = base_env.scene.sensors[foot_sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    force = torch.max(torch.norm(forces[:, :, foot_sensor_cfg.body_ids], dim=-1), dim=1)[0]
    contact = force > threshold
    if contact.ndim == 1:
        contact = contact.unsqueeze(-1)
    left = contact[:, 0].float().mean().item() if contact.shape[1] > 0 else None
    right = contact[:, 1].float().mean().item() if contact.shape[1] > 1 else None
    return left, right


def _foot_relative_positions(robot, foot_body_ids):
    root_pos = robot.data.root_state_w[:, :3]
    root_quat = robot.data.root_state_w[:, 3:7]
    foot_pos = robot.data.body_state_w[:, foot_body_ids, :3] - root_pos.unsqueeze(1)
    flat = foot_pos.reshape(-1, 3)
    quat = root_quat.unsqueeze(1).expand(-1, len(foot_body_ids), -1).reshape(-1, 4)
    rel = quat_apply(quat_conjugate(quat), flat).reshape(root_pos.shape[0], len(foot_body_ids), 3)
    return rel


def _residual_norm(runner):
    residual = getattr(runner.alg.policy, "latest_scaled_residual_action_mean", None)
    if residual is None:
        return None
    return torch.norm(residual.detach(), dim=-1)


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _local_horizontal_displacement(
    current_pos: torch.Tensor, reference_pos: torch.Tensor, reference_yaw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = current_pos[:, :2] - reference_pos[:, :2]
    cos_yaw = torch.cos(reference_yaw)
    sin_yaw = torch.sin(reference_yaw)
    local_x = cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
    local_y = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]
    return local_x, local_y


def _mean_or_none(values):
    if not values:
        return None
    return float(sum(values) / len(values))


def _configure_push_event(env_cfg):
    events = getattr(env_cfg, "events", None)
    if events is None or not hasattr(events, "push_robot"):
        return
    push_robot = getattr(events, "push_robot")
    if args_cli.disable_push:
        events.push_robot = None
        return
    if push_robot is None:
        return
    if args_cli.push_interval_min is not None or args_cli.push_interval_max is not None:
        current = getattr(push_robot, "interval_range_s", (10.0, 15.0))
        min_s = args_cli.push_interval_min if args_cli.push_interval_min is not None else current[0]
        max_s = args_cli.push_interval_max if args_cli.push_interval_max is not None else current[1]
        push_robot.interval_range_s = (float(min_s), float(max_s))
    if args_cli.push_xy is not None or args_cli.push_yaw is not None:
        params = dict(getattr(push_robot, "params", {}))
        velocity_range = dict(params.get("velocity_range", {}))
        if args_cli.push_xy is not None:
            push_xy = float(args_cli.push_xy)
            velocity_range["x"] = (-push_xy, push_xy)
            velocity_range["y"] = (-push_xy, push_xy)
        if args_cli.push_yaw is not None:
            push_yaw = float(args_cli.push_yaw)
            velocity_range["yaw"] = (-push_yaw, push_yaw)
        params["velocity_range"] = velocity_range
        push_robot.params = params


def _push_event_enabled(env_cfg) -> bool:
    events = getattr(env_cfg, "events", None)
    return bool(events is not None and getattr(events, "push_robot", None) is not None)


def _force_push_event_enabled(env_cfg) -> bool:
    events = getattr(env_cfg, "events", None)
    return bool(events is not None and getattr(events, "external_wrench_push", None) is not None)


def _summary_for_command(samples, command):
    vx, vy, wz = command
    summary = {
        "cmd_vx": vx,
        "cmd_vy": vy,
        "cmd_wz": wz,
        "num_samples": len(samples),
    }
    keys = sorted({key for row in samples for key in row.keys() if key not in {"phase", "step"}})
    for key in keys:
        values = [row[key] for row in samples if row.get(key) is not None]
        summary[f"{key}_mean"] = _mean_or_none(values)
        if key.startswith(("left_hand_", "right_hand_")) and values:
            summary[f"{key}_range"] = float(max(values) - min(values))
        if key in {"abs_local_y_displacement", "abs_yaw_drift"} and values:
            summary[f"{key}_max"] = float(max(values))
        if key in {"local_x_displacement", "local_y_displacement", "yaw_drift"} and values:
            summary[f"{key}_final"] = float(values[-1])
    if summary.get("actual_vx_mean") is not None:
        summary["vx_error_mean"] = summary["actual_vx_mean"] - vx
    if summary.get("actual_vy_mean") is not None:
        summary["vy_error_mean"] = summary["actual_vy_mean"] - vy
    if summary.get("actual_wz_mean") is not None:
        summary["wz_error_mean"] = summary["actual_wz_mean"] - wz
    left_x = summary.get("left_foot_x_mean")
    right_x = summary.get("right_foot_x_mean")
    if left_x is not None and right_x is not None:
        summary["foot_x_mean_left_minus_right"] = left_x - right_x
    left_z = summary.get("left_foot_z_mean")
    right_z = summary.get("right_foot_z_mean")
    if left_z is not None and right_z is not None:
        summary["foot_z_mean_left_minus_right"] = left_z - right_z
    left_hand_x = summary.get("left_hand_x_mean")
    right_hand_x = summary.get("right_hand_x_mean")
    if left_hand_x is not None and right_hand_x is not None:
        summary["hand_x_mean_left_minus_right"] = left_hand_x - right_hand_x
    left_hand_y = summary.get("left_hand_y_mean")
    right_hand_y = summary.get("right_hand_y_mean")
    if left_hand_y is not None and right_hand_y is not None:
        summary["hand_y_mean_left_minus_right"] = left_hand_y - right_hand_y
    final_local_x = summary.get("local_x_displacement_final")
    final_local_y = summary.get("local_y_displacement_final")
    if final_local_x is not None and final_local_y is not None and abs(final_local_x) > 1.0e-6:
        summary["lateral_drift_per_forward_meter"] = abs(final_local_y) / max(abs(final_local_x), 1.0e-6)
    return summary


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.non_strict_checkpoint_load and hasattr(agent_cfg, "strict_base_auxiliary_state"):
        agent_cfg.strict_base_auxiliary_state = False
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    _configure_push_event(env_cfg)
    push_event_enabled = _push_event_enabled(env_cfg)
    force_push_event_enabled = _force_push_event_enabled(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env = _LegacyRslRlEnv(env)

    runner_class_name = getattr(agent_cfg, "runner_class_name", "OnPolicyRunner")
    if runner_class_name == "TrackAdapterRunner":
        runner = TrackAdapterRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif runner_class_name == "AmpOnPolicyRunner":
        runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    cmd_term = base_env.command_manager.get_term("base_velocity")
    robot = base_env.scene["robot"]
    base_sensor_cfg, base_contact_threshold = _resolve_base_contact(env_cfg, base_env)
    foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"])
    foot_sensor_cfg.resolve(base_env.scene)
    foot_body_ids, _ = robot.find_bodies(["left_foot_link", "right_foot_link"], preserve_order=True)
    foot_body_ids = [int(index) for index in foot_body_ids]
    hand_body_ids, _ = robot.find_bodies(["left_hand_link", "right_hand_link"], preserve_order=True)
    hand_body_ids = [int(index) for index in hand_body_ids]

    summaries = []
    rows = []

    def set_command(command):
        cmd_term.vel_command_b[:, 0] = command[0]
        cmd_term.vel_command_b[:, 1] = command[1]
        cmd_term.vel_command_b[:, 2] = command[2]

    with torch.inference_mode():
        for command_index, command in enumerate(_parse_commands(args_cli.commands)):
            obs, _ = env.reset()
            dones = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
            set_command(command)
            command_rows = []
            ever_done = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
            ever_base_contact = torch.zeros_like(ever_done)
            reference_root_pos = None
            reference_root_yaw = None
            for step in range(args_cli.steps):
                set_command(command)
                if runner_class_name == "TrackAdapterRunner":
                    actions = policy(obs, dones=dones)
                else:
                    actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                set_command(command)

                done_bool = dones.to(dtype=torch.bool, device=base_env.device)
                base_contact = _contact_mask(base_env, base_sensor_cfg, base_contact_threshold)
                ever_done |= done_bool
                ever_base_contact |= base_contact

                if step < args_cli.settle_steps:
                    continue

                root_state = robot.data.root_state_w
                root_pos = root_state[:, :3]
                root_yaw = _yaw_from_quat_wxyz(root_state[:, 3:7])
                if reference_root_pos is None or reference_root_yaw is None:
                    reference_root_pos = root_pos.detach().clone()
                    reference_root_yaw = root_yaw.detach().clone()
                local_x, local_y = _local_horizontal_displacement(root_pos, reference_root_pos, reference_root_yaw)
                yaw_drift = _wrap_to_pi(root_yaw - reference_root_yaw)
                root_lin = robot.data.root_lin_vel_b
                root_ang = robot.data.root_ang_vel_b
                residual = _residual_norm(runner)
                rel_foot = _foot_relative_positions(robot, foot_body_ids)
                rel_hand = _foot_relative_positions(robot, hand_body_ids)
                left_contact, right_contact = _foot_contact_rates(base_env, foot_sensor_cfg)
                push_active = None
                push_delta_norm = None
                if hasattr(cmd_term, "metrics") and "failure_push_active" in cmd_term.metrics:
                    push_active = cmd_term.metrics["failure_push_active"].float().mean().item()
                    if "failure_push_delta_norm" in cmd_term.metrics:
                        push_delta_norm = cmd_term.metrics["failure_push_delta_norm"].float().mean().item()
                row = {
                    "command_index": command_index,
                    "cmd_vx": command[0],
                    "cmd_vy": command[1],
                    "cmd_wz": command[2],
                    "step": step,
                    "actual_vx": root_lin[:, 0].mean().item(),
                    "actual_vy": root_lin[:, 1].mean().item(),
                    "actual_wz": root_ang[:, 2].mean().item(),
                    "abs_vx_error": torch.abs(root_lin[:, 0] - command[0]).mean().item(),
                    "abs_vy_error": torch.abs(root_lin[:, 1] - command[1]).mean().item(),
                    "abs_wz_error": torch.abs(root_ang[:, 2] - command[2]).mean().item(),
                    "local_x_displacement": local_x.mean().item(),
                    "local_y_displacement": local_y.mean().item(),
                    "abs_local_y_displacement": torch.abs(local_y).mean().item(),
                    "yaw_drift": yaw_drift.mean().item(),
                    "abs_yaw_drift": torch.abs(yaw_drift).mean().item(),
                    "fall_rate": done_bool.float().mean().item(),
                    "base_contact_rate": base_contact.float().mean().item(),
                    "ever_fall_rate": ever_done.float().mean().item(),
                    "ever_base_contact_rate": ever_base_contact.float().mean().item(),
                    "residual_norm": residual.mean().item() if residual is not None else None,
                    "left_foot_x": rel_foot[:, 0, 0].mean().item(),
                    "right_foot_x": rel_foot[:, 1, 0].mean().item(),
                    "left_foot_y": rel_foot[:, 0, 1].mean().item(),
                    "right_foot_y": rel_foot[:, 1, 1].mean().item(),
                    "left_foot_z": rel_foot[:, 0, 2].mean().item(),
                    "right_foot_z": rel_foot[:, 1, 2].mean().item(),
                    "left_hand_x": rel_hand[:, 0, 0].mean().item(),
                    "right_hand_x": rel_hand[:, 1, 0].mean().item(),
                    "left_hand_y": rel_hand[:, 0, 1].mean().item(),
                    "right_hand_y": rel_hand[:, 1, 1].mean().item(),
                    "left_hand_z": rel_hand[:, 0, 2].mean().item(),
                    "right_hand_z": rel_hand[:, 1, 2].mean().item(),
                    "left_foot_contact": left_contact,
                    "right_foot_contact": right_contact,
                    "push_active": push_active,
                    "push_delta_norm": push_delta_norm,
                    "push_enabled": float(push_event_enabled),
                    "force_push_enabled": float(force_push_event_enabled),
                }
                command_rows.append(row)
                rows.append(row)
            summaries.append(_summary_for_command(command_rows, command))

    result = {
        "checkpoint": args_cli.checkpoint,
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "settle_steps": args_cli.settle_steps,
        "push_enabled": bool(push_event_enabled),
        "force_push_enabled": bool(force_push_event_enabled),
        "summaries": summaries,
    }
    print(json.dumps(result, indent=2))
    if args_cli.output_json:
        with open(args_cli.output_json, "w", encoding="utf-8") as file:
            json.dump({"summary": result, "rows": rows}, file, indent=2)
    if args_cli.output_csv and rows:
        with open(args_cli.output_csv, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

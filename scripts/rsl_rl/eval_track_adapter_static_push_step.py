"""Evaluate zero-command static push recovery by measuring capture/support stepping."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--force_n", type=float, default=120.0)
parser.add_argument("--force_body_x", type=float, default=1.0)
parser.add_argument("--force_body_y", type=float, default=0.0)
parser.add_argument("--force_body_z", type=float, default=0.0)
parser.add_argument("--cmd_x", type=float, default=0.0)
parser.add_argument("--cmd_y", type=float, default=0.0)
parser.add_argument("--cmd_yaw", type=float, default=0.0)
parser.add_argument("--body_name", type=str, default="Trunk")
parser.add_argument("--push_start_step", type=int, default=80)
parser.add_argument("--push_duration_steps", type=int, default=5)
parser.add_argument("--post_push_steps", type=int, default=170)
parser.add_argument("--recovery_window_s", type=float, default=2.5)
parser.add_argument("--equivalent_push_vel", type=float, default=1.0)
parser.add_argument(
    "--push_metric_mode",
    choices=("event", "none"),
    default="event",
    help=(
        "'event' injects failure_push_* metrics so Track Adapter recovery gates open; "
        "'none' leaves those metrics inactive, closer to deployment when no push event is available."
    ),
)
parser.add_argument("--step_success_threshold", type=float, default=0.08)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--output_csv", type=str, default=None)
parser.add_argument(
    "--non_strict_checkpoint_load",
    action="store_true",
    default=False,
    help="Allow older non-layerwise Track Adapter checkpoints to load into the current module definition.",
)
parser.add_argument(
    "--force_residual_gate",
    type=float,
    default=None,
    help="Override TrackAdapterRunner residual action gate, e.g. 1.0 to match constant-gate ONNX deployment.",
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


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _local_horizontal_displacement(
    current_pos: torch.Tensor, reference_pos: torch.Tensor, reference_yaw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = current_pos[:, :2] - reference_pos[:, :2]
    cos_yaw = torch.cos(reference_yaw)
    sin_yaw = torch.sin(reference_yaw)
    local_x = cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
    local_y = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]
    return local_x, local_y


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


def _body_contact_mask(base_env, body_names, threshold: float = 1.0):
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=body_names)
    sensor_cfg.resolve(base_env.scene)
    contact_sensor = base_env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    force = torch.max(torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    contact = force > threshold
    if contact.ndim == 1:
        contact = contact.unsqueeze(-1)
    return contact


def _relative_body_positions(robot, body_ids: list[int]):
    root_pos = robot.data.root_state_w[:, :3]
    root_quat = robot.data.root_state_w[:, 3:7]
    body_pos = robot.data.body_state_w[:, body_ids, :3] - root_pos.unsqueeze(1)
    flat = body_pos.reshape(-1, 3)
    quat = root_quat.unsqueeze(1).expand(-1, len(body_ids), -1).reshape(-1, 4)
    rel = quat_apply(quat_conjugate(quat), flat).reshape(root_pos.shape[0], len(body_ids), 3)
    return rel


def _residual_norm(runner):
    residual = getattr(runner.alg.policy, "latest_scaled_residual_action_mean", None)
    if residual is None:
        return None
    return torch.norm(residual.detach(), dim=-1)


def _runner_scalar(runner, name: str):
    value = getattr(runner, "_runner_scalars", {}).get(name)
    return None if value is None else float(value)


def _mean(value: torch.Tensor) -> float:
    return float(value.float().mean().item())


def _configure_eval_env(env_cfg):
    events = getattr(env_cfg, "events", None)
    if events is not None:
        if hasattr(events, "push_robot"):
            events.push_robot = None
        if hasattr(events, "external_wrench_push"):
            events.external_wrench_push = None


def _set_command(base_env, command: torch.Tensor):
    cmd_term = base_env.command_manager.get_term("base_velocity")
    cmd_term.vel_command_b[:, :3] = command.to(device=cmd_term.vel_command_b.device, dtype=cmd_term.vel_command_b.dtype)
    return cmd_term


def _set_push_metrics(cmd_term, active: bool, elapsed_s: float, push_delta_xy: torch.Tensor):
    if not hasattr(cmd_term, "metrics"):
        return
    metrics = cmd_term.metrics
    device = push_delta_xy.device
    num_envs = push_delta_xy.shape[0]
    active_value = 1.0 if active else 0.0
    metrics["failure_push_active"] = torch.full((num_envs,), active_value, device=device)
    metrics["failure_push_elapsed_s"] = torch.full((num_envs,), float(elapsed_s), device=device)
    metrics["failure_push_delta_x"] = push_delta_xy[:, 0] * active_value
    metrics["failure_push_delta_y"] = push_delta_xy[:, 1] * active_value
    metrics["failure_push_delta_norm"] = torch.linalg.norm(push_delta_xy[:, :2], dim=1) * active_value


def _clear_push_metrics(cmd_term, num_envs: int, device: torch.device):
    if not hasattr(cmd_term, "metrics"):
        return
    metrics = cmd_term.metrics
    zeros = torch.zeros(num_envs, device=device)
    metrics["failure_push_active"] = zeros
    metrics["failure_push_elapsed_s"] = zeros
    metrics["failure_push_delta_x"] = zeros
    metrics["failure_push_delta_y"] = zeros
    metrics["failure_push_delta_norm"] = zeros


def _apply_body_force(robot, env_ids, body_ids, force_vector_b: torch.Tensor):
    num_envs = env_ids.numel()
    num_bodies = len(body_ids) if isinstance(body_ids, list) else 1
    forces = torch.zeros((num_envs, num_bodies, 3), device=robot.device)
    torques = torch.zeros_like(forces)
    forces[:, :, :] = force_vector_b.reshape(1, 1, 3)
    robot.set_external_force_and_torque(
        forces=forces,
        torques=torques,
        env_ids=env_ids,
        body_ids=body_ids,
        is_global=False,
    )


def _clear_body_force(robot, env_ids, body_ids):
    num_envs = env_ids.numel()
    num_bodies = len(body_ids) if isinstance(body_ids, list) else 1
    forces = torch.zeros((num_envs, num_bodies, 3), device=robot.device)
    torques = torch.zeros_like(forces)
    robot.set_external_force_and_torque(
        forces=forces,
        torques=torques,
        env_ids=env_ids,
        body_ids=body_ids,
        is_global=False,
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.non_strict_checkpoint_load and hasattr(agent_cfg, "strict_base_auxiliary_state"):
        agent_cfg.strict_base_auxiliary_state = False
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    _configure_eval_env(env_cfg)

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
    if runner_class_name == "TrackAdapterRunner" and args_cli.force_residual_gate is not None:
        forced_gate_value = max(0.0, min(1.0, float(args_cli.force_residual_gate)))

        def _forced_residual_action_gate(style_gate=None, recovery_gate=None):
            if style_gate is None:
                base = torch.ones(runner.env.num_envs, device=runner.device)
            else:
                base = torch.ones_like(style_gate, device=runner.device)
            gate = base * forced_gate_value
            runner._runner_scalars.update(
                {
                    "Gate/residual_action_mean": gate.mean().item(),
                    "Gate/residual_action_min": gate.min().item(),
                    "Gate/residual_action_max": gate.max().item(),
                    "Gate/residual_action_forced": forced_gate_value,
                }
            )
            return gate.unsqueeze(-1)

        runner._compute_residual_action_gate = _forced_residual_action_gate
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    env_ids = torch.arange(base_env.num_envs, device=base_env.device)
    eval_command = torch.tensor(
        [args_cli.cmd_x, args_cli.cmd_y, args_cli.cmd_yaw],
        dtype=torch.float32,
        device=base_env.device,
    )
    cmd_term = _set_command(base_env, eval_command)

    force_cfg = SceneEntityCfg("robot", body_names=args_cli.body_name)
    force_cfg.resolve(base_env.scene)
    body_ids = force_cfg.body_ids
    if isinstance(body_ids, int):
        body_ids = [body_ids]

    foot_body_ids, _ = robot.find_bodies(["left_foot_link", "right_foot_link"], preserve_order=True)
    foot_body_ids = [int(index) for index in foot_body_ids]
    base_sensor_cfg, base_contact_threshold = _resolve_base_contact(env_cfg, base_env)

    direction = torch.tensor(
        [args_cli.force_body_x, args_cli.force_body_y, args_cli.force_body_z],
        dtype=torch.float32,
        device=base_env.device,
    )
    direction_norm = torch.linalg.norm(direction)
    if float(direction_norm.item()) <= 1.0e-6:
        raise ValueError("Push direction must be non-zero.")
    direction = direction / direction_norm
    force_vector_b = direction * float(args_cli.force_n)
    direction_xy = direction[:2]
    direction_xy_norm = torch.linalg.norm(direction_xy).clamp_min(1.0e-6)
    direction_xy = direction_xy / direction_xy_norm
    push_delta_xy = direction_xy.reshape(1, 2).repeat(base_env.num_envs, 1) * float(args_cli.equivalent_push_vel)

    total_steps = int(args_cli.push_start_step + args_cli.push_duration_steps + args_cli.post_push_steps)
    recovery_window_steps = max(int(math.ceil(float(args_cli.recovery_window_s) / float(base_env.step_dt))), 1)
    push_start = int(args_cli.push_start_step)
    push_end = push_start + int(args_cli.push_duration_steps)
    recovery_end = push_start + recovery_window_steps

    obs, _ = env.reset()
    if hasattr(policy, "reset_history"):
        policy.reset_history()
    _set_command(base_env, eval_command)
    dones = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)

    pre_root_pos = None
    pre_root_yaw = None
    pre_foot_rel = None
    max_step_proj = torch.full((base_env.num_envs,), -1.0e9, device=base_env.device)
    max_contacted_step_proj = torch.full_like(max_step_proj, -1.0e9)
    max_speed_xy = torch.zeros(base_env.num_envs, device=base_env.device)
    last_speed_samples: list[torch.Tensor] = []
    last_cmd_error_samples: list[torch.Tensor] = []
    ever_done = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
    ever_base_contact = torch.zeros_like(ever_done)
    rows = []

    with torch.inference_mode():
        for step in range(total_steps):
            _set_command(base_env, eval_command)

            active_window = push_start <= step < recovery_end
            if step == push_start and args_cli.push_metric_mode == "event" and hasattr(cmd_term, "record_external_push"):
                sampled_mask = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
                velocity_range = {
                    "x": (-abs(float(args_cli.equivalent_push_vel)), abs(float(args_cli.equivalent_push_vel))),
                    "y": (-abs(float(args_cli.equivalent_push_vel)), abs(float(args_cli.equivalent_push_vel))),
                    "yaw": (0.0, 0.0),
                }
                push_delta = torch.zeros((base_env.num_envs, 6), device=base_env.device)
                push_delta[:, :2] = push_delta_xy
                cmd_term.record_external_push(env_ids, push_delta, sampled_mask, velocity_range)

            elapsed_s = max(0.0, (step - push_start) * float(base_env.step_dt))
            if args_cli.push_metric_mode == "event":
                _set_push_metrics(cmd_term, active_window, elapsed_s, push_delta_xy)
            else:
                _clear_push_metrics(cmd_term, base_env.num_envs, base_env.device)

            if step == push_start:
                root_state = robot.data.root_state_w
                pre_root_pos = root_state[:, :3].detach().clone()
                pre_root_yaw = _yaw_from_quat_wxyz(root_state[:, 3:7]).detach().clone()
                pre_foot_rel = _relative_body_positions(robot, foot_body_ids).detach().clone()

            if push_start <= step < push_end:
                _apply_body_force(robot, env_ids, body_ids, force_vector_b)
            else:
                _clear_body_force(robot, env_ids, body_ids)

            if runner_class_name == "TrackAdapterRunner":
                actions = policy(obs, dones=dones)
            else:
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            _set_command(base_env, eval_command)

            done_bool = dones.to(dtype=torch.bool, device=base_env.device)
            base_contact = _contact_mask(base_env, base_sensor_cfg, base_contact_threshold)
            ever_done |= done_bool
            ever_base_contact |= base_contact

            if step < push_start or pre_root_pos is None or pre_root_yaw is None or pre_foot_rel is None:
                continue

            root_state = robot.data.root_state_w
            root_pos = root_state[:, :3]
            local_x, local_y = _local_horizontal_displacement(root_pos, pre_root_pos, pre_root_yaw)
            root_lin = robot.data.root_lin_vel_b
            root_ang = robot.data.root_ang_vel_b
            speed_xy = torch.linalg.norm(root_lin[:, :2], dim=1)
            max_speed_xy = torch.maximum(max_speed_xy, speed_xy)
            if step >= total_steps - 20:
                last_speed_samples.append(speed_xy.detach().clone())
                command_error = torch.stack(
                    [
                        root_lin[:, 0] - eval_command[0],
                        root_lin[:, 1] - eval_command[1],
                        root_ang[:, 2] - eval_command[2],
                    ],
                    dim=-1,
                )
                last_cmd_error_samples.append(torch.abs(command_error).detach().clone())

            rel_foot = _relative_body_positions(robot, foot_body_ids)
            foot_delta = rel_foot[:, :, :2] - pre_foot_rel[:, :, :2]
            projection = torch.sum(foot_delta * direction_xy.reshape(1, 1, 2), dim=-1)
            max_step_proj = torch.maximum(max_step_proj, projection.max(dim=1).values)

            foot_contact = _body_contact_mask(base_env, ["left_foot_link", "right_foot_link"], threshold=1.0)
            contacted_projection = projection.masked_fill(~foot_contact, -1.0e9)
            max_contacted_step_proj = torch.maximum(max_contacted_step_proj, contacted_projection.max(dim=1).values)

            residual = _residual_norm(runner)
            rows.append(
                {
                    "step": step,
                    "push_active": float(active_window),
                    "force_active": float(push_start <= step < push_end),
                    "fall_rate": _mean(done_bool),
                    "base_contact_rate": _mean(base_contact),
                    "ever_fall_rate": _mean(ever_done),
                    "ever_base_contact_rate": _mean(ever_base_contact),
                    "max_step_projection_mean": _mean(max_step_proj.clamp_min(-1.0)),
                    "max_contacted_step_projection_mean": _mean(max_contacted_step_proj.clamp_min(-1.0)),
                    "current_best_step_projection_mean": _mean(projection.max(dim=1).values),
                    "current_best_contacted_step_projection_mean": _mean(contacted_projection.max(dim=1).values.clamp_min(-1.0)),
                    "local_x_displacement_mean": _mean(local_x),
                    "local_y_displacement_mean": _mean(local_y),
                    "speed_xy_mean": _mean(speed_xy),
                    "abs_vx_error_mean": _mean(torch.abs(root_lin[:, 0] - eval_command[0])),
                    "abs_vy_error_mean": _mean(torch.abs(root_lin[:, 1] - eval_command[1])),
                    "abs_wz_error_mean": _mean(torch.abs(root_ang[:, 2] - eval_command[2])),
                    "residual_norm_mean": residual.mean().item() if residual is not None else None,
                    "gate_residual_action_mean": _runner_scalar(runner, "Gate/residual_action_mean"),
                    "gate_push_active_mean": _runner_scalar(runner, "Gate/push_active_mean"),
                }
            )

    _clear_body_force(robot, env_ids, body_ids)
    final_speed_xy = (
        torch.stack(last_speed_samples, dim=0).mean(dim=0)
        if last_speed_samples
        else torch.zeros(base_env.num_envs, device=base_env.device)
    )
    final_cmd_error = (
        torch.stack(last_cmd_error_samples, dim=0).mean(dim=0)
        if last_cmd_error_samples
        else torch.zeros(base_env.num_envs, 3, device=base_env.device)
    )
    clean = ~(ever_done | ever_base_contact)
    step_success = max_step_proj > float(args_cli.step_success_threshold)
    support_step_success = max_contacted_step_proj > float(args_cli.step_success_threshold)
    clean_support_step_success = clean & support_step_success

    summary = {
        "checkpoint": args_cli.checkpoint,
        "task": args_cli.task,
        "num_envs": base_env.num_envs,
        "force_n": float(args_cli.force_n),
        "force_body_direction": [float(direction[0].item()), float(direction[1].item()), float(direction[2].item())],
        "command": [float(eval_command[0].item()), float(eval_command[1].item()), float(eval_command[2].item())],
        "body_name": args_cli.body_name,
        "push_start_step": push_start,
        "push_duration_steps": int(args_cli.push_duration_steps),
        "push_duration_s": float(args_cli.push_duration_steps) * float(base_env.step_dt),
        "post_push_steps": int(args_cli.post_push_steps),
        "recovery_window_s": float(args_cli.recovery_window_s),
        "push_metric_mode": args_cli.push_metric_mode,
        "force_residual_gate": args_cli.force_residual_gate,
        "step_success_threshold": float(args_cli.step_success_threshold),
        "ever_fall_rate": _mean(ever_done),
        "ever_base_contact_rate": _mean(ever_base_contact),
        "clean_rate": _mean(clean),
        "step_success_rate": _mean(step_success),
        "support_step_success_rate": _mean(support_step_success),
        "clean_support_step_success_rate": _mean(clean_support_step_success),
        "max_step_projection_mean": _mean(max_step_proj),
        "max_step_projection_p50": float(torch.quantile(max_step_proj, 0.50).item()),
        "max_step_projection_p90": float(torch.quantile(max_step_proj, 0.90).item()),
        "max_contacted_step_projection_mean": _mean(max_contacted_step_proj.clamp_min(-1.0)),
        "max_contacted_step_projection_p50": float(torch.quantile(max_contacted_step_proj.clamp_min(-1.0), 0.50).item()),
        "max_contacted_step_projection_p90": float(torch.quantile(max_contacted_step_proj.clamp_min(-1.0), 0.90).item()),
        "max_speed_xy_mean": _mean(max_speed_xy),
        "final_speed_xy_mean": _mean(final_speed_xy),
        "final_abs_vx_error_mean": _mean(final_cmd_error[:, 0]),
        "final_abs_vy_error_mean": _mean(final_cmd_error[:, 1]),
        "final_abs_wz_error_mean": _mean(final_cmd_error[:, 2]),
    }
    result = {"summary": summary, "rows": rows}
    print(json.dumps(result, indent=2))

    if args_cli.output_json:
        with open(args_cli.output_json, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
    if args_cli.output_csv and rows:
        with open(args_cli.output_csv, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

# Evaluate Track Adapter recovery from an abrupt all-zero command.

import argparse
import csv
import json
import sys
from dataclasses import dataclass

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate Track Adapter stop/recovery metrics.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--move_steps", type=int, default=150)
parser.add_argument("--stop_steps", type=int, default=150)
parser.add_argument("--cmd_x", type=float, default=0.8)
parser.add_argument("--cmd_y", type=float, default=0.0)
parser.add_argument("--cmd_yaw", type=float, default=0.0)
parser.add_argument(
    "--commands",
    type=str,
    default=None,
    help="Optional semicolon-separated vx,vy,wz commands. Overrides --cmd_x/--cmd_y/--cmd_yaw.",
)
parser.add_argument(
    "--scenario_set",
    choices=["single", "high_speed_jerk"],
    default="single",
    help="Use --cmd/--commands or the preset vx=2->0, vx=1->0, vy=1->0, vx/vy->0 suite.",
)
parser.add_argument(
    "--push_modes",
    choices=["none", "push", "both"],
    default=None,
    help="Evaluation push coverage. Default: none for single, both for high_speed_jerk.",
)
parser.add_argument("--eval_push_speed", type=float, default=0.5)
parser.add_argument("--eval_push_yaw", type=float, default=0.0)
parser.add_argument("--eval_push_step", type=int, default=0)
parser.add_argument("--settle_speed", type=float, default=0.10)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--output_csv", type=str, default=None)
parser.add_argument("--disable_push", action="store_true", default=False)
parser.add_argument(
    "--keep_env_push",
    action="store_true",
    default=False,
    help="Keep configured interval push events; default disables them for clean fixed-command A/B eval.",
)
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

from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner, TrackAdapterRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_rl_tasks.tasks  # noqa: F401


@dataclass(frozen=True)
class _StopScenario:
    name: str
    command: tuple[float, float, float]
    eval_push: bool


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


def _base_contact_mask(base_env, sensor_cfg: SceneEntityCfg, threshold: float):
    contact_sensor = base_env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    contact_force = torch.max(torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    return torch.any(contact_force > threshold, dim=1)


def _speed(robot, yaw_scale=0.30):
    return torch.norm(
        torch.stack(
            [
                robot.data.root_lin_vel_b[:, 0],
                robot.data.root_lin_vel_b[:, 1],
                yaw_scale * robot.data.root_ang_vel_b[:, 2],
            ],
            dim=-1,
        ),
        dim=1,
    )


def _residual_norm(runner):
    residual = getattr(runner.alg.policy, "latest_scaled_residual_action_mean", None)
    if residual is None:
        return None
    return torch.norm(residual.detach(), dim=-1)


def _amp_score(runner, amp_obs, next_amp_obs):
    try:
        reward = runner.alg.discriminator.predict_amp_reward(
            amp_obs,
            next_amp_obs,
            torch.zeros(amp_obs.shape[0], device=amp_obs.device),
            normalizer=runner.alg.amp_normalizer,
        )[0]
        return reward.detach()
    except Exception:
        return None


def _termination_contact_threshold(env_cfg, default: float = 1.0) -> float:
    terminations = getattr(env_cfg, "terminations", None)
    base_contact = getattr(terminations, "base_contact", None)
    params = getattr(base_contact, "params", {}) if base_contact is not None else {}
    return float(params.get("threshold", default))


def _parse_commands(value: str | None):
    if args_cli.scenario_set == "high_speed_jerk" and value is None:
        return [
            (2.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ]
    if value is None:
        return [(args_cli.cmd_x, args_cli.cmd_y, args_cli.cmd_yaw)]
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


def _push_mode_flags():
    if args_cli.push_modes is None:
        return [False, True] if args_cli.scenario_set == "high_speed_jerk" else [False]
    if args_cli.push_modes == "none":
        return [False]
    if args_cli.push_modes == "push":
        return [True]
    return [False, True]


def _scenario_name(command: tuple[float, float, float], eval_push: bool):
    vx, vy, yaw = command
    suffix = "push" if eval_push else "no_push"
    return f"vx{vx:g}_vy{vy:g}_yaw{yaw:g}_to_0_{suffix}".replace("-", "neg")


def _build_scenarios():
    return [
        _StopScenario(name=_scenario_name(command, eval_push), command=command, eval_push=eval_push)
        for command in _parse_commands(args_cli.commands)
        for eval_push in _push_mode_flags()
    ]


def _configure_push_event(env_cfg):
    events = getattr(env_cfg, "events", None)
    if events is None or not hasattr(events, "push_robot"):
        return
    push_robot = getattr(events, "push_robot")
    if args_cli.disable_push or not args_cli.keep_env_push:
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


def _masked_mean(values, mask):
    if values is None or not torch.any(mask):
        return None
    return values[mask].float().mean().item()


def _masked_quantile(values, mask, q: float):
    if values is None or not torch.any(mask):
        return None
    return torch.quantile(values[mask].float(), q).item()


def _apply_eval_push(robot, cmd_term, command: tuple[float, float, float]):
    speed = max(float(args_cli.eval_push_speed), 0.0)
    yaw = float(args_cli.eval_push_yaw)
    if speed <= 0.0 and abs(yaw) <= 0.0:
        return

    root_vel = robot.data.root_vel_w.clone()
    device = root_vel.device
    env_ids = torch.arange(root_vel.shape[0], dtype=torch.long, device=device)
    command_xy = torch.tensor(command[:2], dtype=root_vel.dtype, device=device)
    command_norm = torch.linalg.norm(command_xy)
    if command_norm > 1.0e-6:
        push_xy_b = -speed * command_xy / command_norm
    else:
        push_xy_b = torch.tensor([-speed, 0.0], dtype=root_vel.dtype, device=device)

    push_lin_b = torch.zeros((root_vel.shape[0], 3), dtype=root_vel.dtype, device=device)
    push_lin_b[:, 0] = push_xy_b[0]
    push_lin_b[:, 1] = push_xy_b[1]
    push_lin_w = quat_apply(robot.data.root_state_w[:, 3:7], push_lin_b)

    push_delta_w = torch.zeros_like(root_vel)
    push_delta_w[:, :3] = push_lin_w
    push_delta_w[:, 5] = yaw
    robot.write_root_velocity_to_sim(root_vel + push_delta_w, env_ids=env_ids)

    if hasattr(cmd_term, "record_external_push"):
        velocity_range = {
            "x": (-abs(float(push_xy_b[0])), abs(float(push_xy_b[0]))),
            "y": (-abs(float(push_xy_b[1])), abs(float(push_xy_b[1]))),
            "yaw": (-abs(yaw), abs(yaw)),
        }
        sampled_mask = torch.zeros(env_ids.numel(), dtype=torch.bool, device=device)
        cmd_term.record_external_push(env_ids, push_delta_w, sampled_mask, velocity_range)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.non_strict_checkpoint_load and hasattr(agent_cfg, "strict_base_auxiliary_state"):
        agent_cfg.strict_base_auxiliary_state = False
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    _configure_push_event(env_cfg)
    env_push_event_enabled = _push_event_enabled(env_cfg)

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
    base_contact_sensor_cfg = SceneEntityCfg("contact_forces", body_names=["Trunk"])
    base_contact_sensor_cfg.resolve(base_env.scene)
    base_contact_threshold = _termination_contact_threshold(env_cfg, default=2.0)

    rows = []
    summaries = []

    def run_scenario(scenario: _StopScenario):
        obs, extras = env.reset()
        dones = torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device)
        settled_step = torch.full((base_env.num_envs,), -1, dtype=torch.long, device=base_env.device)
        ever_done = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
        ever_base_contact = torch.zeros_like(ever_done)
        pre_stop_failed = torch.zeros_like(ever_done)

        def refresh_observations():
            nonlocal obs, extras
            obs, extras = env.get_observations()

        def set_command(x, y, yaw):
            cmd_term.vel_command_b[:, 0] = x
            cmd_term.vel_command_b[:, 1] = y
            cmd_term.vel_command_b[:, 2] = yaw
            refresh_observations()

        def step_once(phase, step_idx, eval_push_applied=False):
            nonlocal obs, dones, extras
            amp_obs = extras["observations"].get("amp_observations")
            if runner_class_name == "TrackAdapterRunner":
                actions = policy(obs, dones=dones)
            else:
                actions = policy(obs)
            next_obs, _, dones, infos = env.step(actions)
            _, next_extras = env.get_observations()
            next_amp_obs = next_extras["observations"].get("amp_observations")
            done_bool = dones.to(dtype=torch.bool, device=base_env.device)
            speed = _speed(robot)
            residual = _residual_norm(runner)
            amp = _amp_score(runner, amp_obs, next_amp_obs) if amp_obs is not None and next_amp_obs is not None else None
            contact = _base_contact_mask(base_env, base_contact_sensor_cfg, base_contact_threshold)
            rows.append(
                {
                    "scenario": scenario.name,
                    "cmd_x": scenario.command[0],
                    "cmd_y": scenario.command[1],
                    "cmd_yaw": scenario.command[2],
                    "eval_push": scenario.eval_push,
                    "eval_push_applied": eval_push_applied,
                    "env_push_event_enabled": env_push_event_enabled,
                    "phase": phase,
                    "step": step_idx,
                    "speed_mean": speed.mean().item(),
                    "speed_p95": torch.quantile(speed, 0.95).item(),
                    "done_rate": done_bool.float().mean().item(),
                    "base_contact_rate": contact.float().mean().item(),
                    "residual_norm_mean": residual.mean().item() if residual is not None else None,
                    "amp_score_mean": amp.mean().item() if amp is not None else None,
                }
            )
            obs = next_obs
            extras = next_extras
            return speed, contact, done_bool, residual, amp

        cmd_x, cmd_y, cmd_yaw = scenario.command
        set_command(cmd_x, cmd_y, cmd_yaw)
        for step in range(args_cli.warmup_steps):
            _, contact, done_bool, _, _ = step_once("warmup", step)
            ever_done |= done_bool
            ever_base_contact |= contact
            pre_stop_failed |= done_bool | contact
            set_command(cmd_x, cmd_y, cmd_yaw)
        for step in range(args_cli.move_steps):
            _, contact, done_bool, _, _ = step_once("move", step)
            ever_done |= done_bool
            ever_base_contact |= contact
            pre_stop_failed |= done_bool | contact
            set_command(cmd_x, cmd_y, cmd_yaw)

        set_command(0.0, 0.0, 0.0)
        final_speed = None
        final_residual = None
        final_amp = None
        for step in range(args_cli.stop_steps):
            eval_push_applied = bool(scenario.eval_push and step == max(int(args_cli.eval_push_step), 0))
            if eval_push_applied:
                _apply_eval_push(robot, cmd_term, scenario.command)
                refresh_observations()
            speed, contact, done_bool, residual, amp = step_once("stop", step, eval_push_applied)
            ever_done |= done_bool
            ever_base_contact |= contact
            failed = pre_stop_failed | ever_done | ever_base_contact
            newly_settled = (settled_step < 0) & (speed < args_cli.settle_speed) & ~failed
            settled_step[newly_settled] = step
            final_speed = speed.detach()
            final_residual = residual.detach() if residual is not None else None
            final_amp = amp.detach() if amp is not None else None
            set_command(0.0, 0.0, 0.0)

        stop_rows = [row for row in rows if row["scenario"] == scenario.name and row["phase"] == "stop"]
        valid_envs = ~(pre_stop_failed | ever_done | ever_base_contact)
        settled = (settled_step >= 0) & valid_envs
        return {
            "scenario": scenario.name,
            "num_envs": int(base_env.num_envs),
            "cmd": [cmd_x, cmd_y, cmd_yaw],
            "eval_push": scenario.eval_push,
            "eval_push_speed": args_cli.eval_push_speed if scenario.eval_push else 0.0,
            "eval_push_yaw": args_cli.eval_push_yaw if scenario.eval_push else 0.0,
            "eval_push_step": args_cli.eval_push_step if scenario.eval_push else None,
            "env_push_event_enabled": env_push_event_enabled,
            "fall_rate": ever_done.float().mean().item(),
            "base_contact_rate": ever_base_contact.float().mean().item(),
            "pre_stop_failure_rate": pre_stop_failed.float().mean().item(),
            "valid_stop_env_rate": valid_envs.float().mean().item(),
            "settle_success_rate": settled.float().mean().item(),
            "settle_success_rate_valid_envs": (
                settled[valid_envs].float().mean().item() if torch.any(valid_envs) else None
            ),
            "mean_recovery_steps": settled_step[settled].float().mean().item() if torch.any(settled) else None,
            "final_stop_speed_mean": _masked_mean(final_speed, valid_envs),
            "final_stop_speed_p95": _masked_quantile(final_speed, valid_envs, 0.95),
            "final_residual_norm_mean": _masked_mean(final_residual, valid_envs),
            "final_amp_score_mean": _masked_mean(final_amp, valid_envs),
            "final_stop_speed_mean_all": stop_rows[-1]["speed_mean"] if stop_rows else None,
            "final_stop_speed_p95_all": stop_rows[-1]["speed_p95"] if stop_rows else None,
            "final_residual_norm_mean_all": stop_rows[-1]["residual_norm_mean"] if stop_rows else None,
            "final_amp_score_mean_all": stop_rows[-1]["amp_score_mean"] if stop_rows else None,
        }

    with torch.inference_mode():
        for scenario in _build_scenarios():
            summaries.append(run_scenario(scenario))

    result = summaries[0] if len(summaries) == 1 else {"summaries": summaries}
    print(json.dumps(result, indent=2))

    if args_cli.output_json:
        with open(args_cli.output_json, "w", encoding="utf-8") as file:
            key = "summary" if len(summaries) == 1 else "summaries"
            value = summaries[0] if len(summaries) == 1 else summaries
            json.dump({key: value, "rows": rows}, file, indent=2)
    if args_cli.output_csv and rows:
        with open(args_cli.output_csv, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

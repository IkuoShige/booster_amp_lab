# Evaluate velocity tracking per command band for a trained AMP policy.
#
# Rolls out a fixed forward velocity command for a few seconds per target,
# records actual base lin_vel_x / ang_vel_z, and prints a table of command
# vs. achieved speed + error.

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate velocity tracking of RSL-RL AMP policy.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--measure_steps", type=int, default=300)
parser.add_argument(
    "--axis",
    choices=["x", "yaw"],
    default="x",
    help="Which command axis to probe: lin_vel_x or ang_vel_z.",
)
parser.add_argument(
    "--lin_vel_x_during_yaw",
    type=float,
    default=0.5,
    help="Fixed forward velocity command while sweeping yaw (m/s).",
)
parser.add_argument(
    "--commands",
    type=float,
    nargs="+",
    default=[0.2, 0.5, 0.8, 1.0, 1.3, 1.6, 1.9],
    help="Commands to probe. Units depend on --axis (m/s for x, rad/s for yaw).",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--viser",
    action="store_true",
    default=False,
    help="Stream eval to a viser web viewer for visual inspection while bands run.",
)
parser.add_argument("--viser_host", type=str, default="0.0.0.0")
parser.add_argument("--viser_port", type=int, default=8080)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# viser reads tensors directly so we still run headless
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time

import torch
import gymnasium as gym

from rsl_rl.runners import AmpOnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_rl_tasks.tasks  # noqa: F401


def _split_obs(obs_td):
    groups = dict(obs_td.items()) if hasattr(obs_td, "items") else dict(obs_td)
    policy_obs = groups.pop("policy")
    return policy_obs, groups


class _LegacyRslRlEnv:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def env(self):
        # dummy two-layer chain so `self.env.env.env` reaches the unwrapped base env
        class _P:
            def __init__(s, u):
                s.env = u

        return _P(self._env.unwrapped)

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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env = _LegacyRslRlEnv(env)

    runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]

    bridge = None
    if args_cli.viser:
        from viser_bridge import BoosterViserBridge

        bridge = BoosterViserBridge(
            env.unwrapped, host=args_cli.viser_host, port=args_cli.viser_port,
        )

    sim_dt = float(env.unwrapped.step_dt)

    obs, _ = env.get_observations()
    if args_cli.axis == "x":
        label = "cmd_vx"
        unit = "m/s"
    else:
        label = "cmd_yaw"
        unit = "rad/s"
    print(f"\n{label:>8} | {'mean':>9} | {'std':>7} | {'err':>7} | {'ratio':>6}    [{unit}]")
    print("-" * 60)

    def set_cmd(value):
        if args_cli.axis == "x":
            cmd_term.vel_command_b[:, 0] = value
            cmd_term.vel_command_b[:, 1] = 0.0
            cmd_term.vel_command_b[:, 2] = 0.0
        else:
            cmd_term.vel_command_b[:, 0] = args_cli.lin_vel_x_during_yaw
            cmd_term.vel_command_b[:, 1] = 0.0
            cmd_term.vel_command_b[:, 2] = value

    def measured(_robot):
        if args_cli.axis == "x":
            return _robot.data.root_lin_vel_b[:, 0].detach().clone()
        return _robot.data.root_ang_vel_b[:, 2].detach().clone()

    def _step_with_bridge(action):
        nonlocal obs
        if bridge is not None:
            bridge.apply_joystick(cmd_term)
        out_obs, _, _, _ = env.step(action)
        obs = out_obs
        if bridge is not None:
            bridge.update()

    results = []
    for c in args_cli.commands:
        with torch.inference_mode():
            set_cmd(c)
            for _ in range(args_cli.warmup_steps):
                t0 = time.time()
                actions = policy(obs)
                _step_with_bridge(actions)
                set_cmd(c)
                if bridge is not None:
                    sleep = sim_dt - (time.time() - t0)
                    if sleep > 0:
                        time.sleep(sleep)

            samples = []
            for _ in range(args_cli.measure_steps):
                t0 = time.time()
                actions = policy(obs)
                _step_with_bridge(actions)
                set_cmd(c)
                samples.append(measured(robot))
                if bridge is not None:
                    sleep = sim_dt - (time.time() - t0)
                    if sleep > 0:
                        time.sleep(sleep)

            arr = torch.stack(samples, dim=0)
            mean_v = arr.mean().item()
            std_v = arr.std().item()
            err_v = (arr - c).abs().mean().item()
            ratio = mean_v / c if abs(c) > 1e-6 else float("nan")
            results.append((c, mean_v, std_v, err_v, ratio))
            print(f"{c:8.2f} | {mean_v:9.3f} | {std_v:7.3f} | {err_v:7.3f} | {ratio:6.2f}")

    if bridge is not None:
        bridge.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

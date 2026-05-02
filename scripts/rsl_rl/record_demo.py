# Record a demo video where commands are scripted on a timeline so that the
# resulting clip clearly shows the policy responding to a variety of inputs:
#   - lin_vel_x sweep at zero yaw
#   - yaw sweep at moderate forward speed
#   - combined lin_vel_x + yaw inputs
# Camera follows the robot via env_cfg.viewer.
#
# Usage:
#   python scripts/rsl_rl/record_demo.py \
#     --task=Booster-Run-AMP-v0 \
#     --checkpoint=path/to/model.pt \
#     --headless

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Record a scripted-command demo of an AMP policy.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seconds_per_command", type=float, default=2.5)
parser.add_argument("--video_name", type=str, default="demo.mp4")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

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


# Each entry is (lin_vel_x [m/s], lin_vel_y [m/s], ang_vel_z [rad/s])
COMMAND_SCHEDULE = [
    # forward speed sweep at zero yaw
    (0.5, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.3, 0.0, 0.0),
    (1.6, 0.0, 0.0),
    (1.9, 0.0, 0.0),
    # yaw sweep at moderate forward speed
    (0.8, 0.0, +0.3),
    (0.8, 0.0, -0.3),
    # combined x + yaw
    (1.3, 0.0, +0.2),
    (1.3, 0.0, -0.2),
    (1.6, 0.0, +0.15),
]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed

    # follow the robot so it stays in frame
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (2.5, -2.5, 0.8)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.3)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    log_dir = os.path.dirname(args_cli.checkpoint)
    video_dir = os.path.join(log_dir, "videos", "demo")
    os.makedirs(video_dir, exist_ok=True)

    # decimation gives env step time; we drive each command for `seconds_per_command`
    decimation = getattr(env.unwrapped.cfg, "decimation", 1)
    sim_dt = env.unwrapped.cfg.sim.dt
    env_dt = sim_dt * decimation
    steps_per_command = max(1, int(round(args_cli.seconds_per_command / env_dt)))
    total_steps = steps_per_command * len(COMMAND_SCHEDULE)
    print(
        f"[demo] env_dt={env_dt:.4f}s -> {steps_per_command} steps per command, "
        f"{total_steps} total ({total_steps * env_dt:.1f}s)."
    )

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        name_prefix=os.path.splitext(args_cli.video_name)[0],
        step_trigger=lambda step: step == 0,
        video_length=total_steps,
        disable_logger=True,
    )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env = _LegacyRslRlEnv(env)

    runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")

    obs, _ = env.get_observations()
    with torch.inference_mode():
        for vx, vy, wz in COMMAND_SCHEDULE:
            cmd_term.vel_command_b[:, 0] = vx
            cmd_term.vel_command_b[:, 1] = vy
            cmd_term.vel_command_b[:, 2] = wz
            for _ in range(steps_per_command):
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                # re-pin commands every env step so the resampler does not overwrite them
                cmd_term.vel_command_b[:, 0] = vx
                cmd_term.vel_command_b[:, 1] = vy
                cmd_term.vel_command_b[:, 2] = wz

    env.close()
    print(f"[demo] saved video to {video_dir}/{os.path.splitext(args_cli.video_name)[0]}-*.mp4")


if __name__ == "__main__":
    main()
    simulation_app.close()

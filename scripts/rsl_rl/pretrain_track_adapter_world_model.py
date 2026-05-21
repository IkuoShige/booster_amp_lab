# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 3 Track Adapter world-model pretraining entrypoint."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Pretrain Track Adapter history world model with frozen base rollouts.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--pretrain_iterations", type=int, default=16, help="Number of pretraining iterations.")
parser.add_argument("--pretrain_steps_per_env", type=int, default=None, help="Rollout steps per environment.")
parser.add_argument("--pretrain_epochs", type=int, default=None, help="World-model update epochs per rollout.")
parser.add_argument("--pretrain_mini_batches", type=int, default=None, help="World-model mini-batches per epoch.")
parser.add_argument("--pretrain_learning_rate", type=float, default=None, help="World-model pretraining LR.")
parser.add_argument("--pretrain_max_grad_norm", type=float, default=None, help="World-model gradient clip norm.")
parser.add_argument("--pretrain_horizon", type=int, default=None, help="Future target horizon.")
parser.add_argument(
    "--pretrain_target_source",
    type=str,
    default=None,
    choices={
        "observations",
        "next_observations",
        "obs",
        "next_obs",
        "world_model_state",
        "wm_state",
        "structured",
        "psi_s",
    },
    help="Future target source.",
)
parser.add_argument("--pretrain_target_dim", type=int, default=None, help="Future target feature dimension.")
parser.add_argument(
    "--pretrain_target_groups",
    type=str,
    default=None,
    help="Comma-separated world-model target groups, for observation-layout targets.",
)
parser.add_argument("--pretrain_save_interval", type=int, default=None, help="Checkpoint save interval.")
parser.add_argument(
    "--init_at_random_ep_len",
    action="store_true",
    default=False,
    help="Start environments at randomized episode lengths.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners.track_adapter_world_model_pretrain_runner import TrackAdapterWorldModelPretrainRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_rl_tasks.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


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


def _put_if_not_none(target: dict, name: str, value):
    if value is not None:
        target[name] = value


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    runner_class_name = getattr(agent_cfg, "runner_class_name", "OnPolicyRunner")
    if runner_class_name != "TrackAdapterRunner":
        raise ValueError(
            "Track Adapter world-model pretraining requires a TrackAdapterRunner config, "
            f"got {runner_class_name}."
        )

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = agent_cfg.run_name or "track_adapter_wm_pretrain"
    log_dir = os.path.join(log_root_path, f"{log_dir}_{run_name}")
    print(f"[INFO] Logging Track Adapter world-model pretrain in directory: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env = _LegacyRslRlEnv(env)

    pretrain_cfg = {}
    _put_if_not_none(pretrain_cfg, "num_steps_per_env", args_cli.pretrain_steps_per_env)
    _put_if_not_none(pretrain_cfg, "num_learning_epochs", args_cli.pretrain_epochs)
    _put_if_not_none(pretrain_cfg, "num_mini_batches", args_cli.pretrain_mini_batches)
    _put_if_not_none(pretrain_cfg, "learning_rate", args_cli.pretrain_learning_rate)
    _put_if_not_none(pretrain_cfg, "max_grad_norm", args_cli.pretrain_max_grad_norm)
    _put_if_not_none(pretrain_cfg, "horizon", args_cli.pretrain_horizon)
    _put_if_not_none(pretrain_cfg, "target_source", args_cli.pretrain_target_source)
    _put_if_not_none(pretrain_cfg, "target_dim", args_cli.pretrain_target_dim)
    _put_if_not_none(pretrain_cfg, "target_groups", args_cli.pretrain_target_groups)
    _put_if_not_none(pretrain_cfg, "save_interval", args_cli.pretrain_save_interval)

    train_cfg = agent_cfg.to_dict()
    train_cfg["world_model_pretrain"] = pretrain_cfg

    runner = TrackAdapterWorldModelPretrainRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    if args_cli.checkpoint:
        print(f"[INFO]: Loading Track Adapter policy checkpoint from: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint, load_optimizer=False)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(
        num_learning_iterations=args_cli.pretrain_iterations,
        init_at_random_ep_len=args_cli.init_at_random_ep_len,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

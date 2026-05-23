# Copyright (c) 2025-2026, The Booster Soccer Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Multi-critic AMP on-policy runner.

This runner mirrors :class:`AmpOnPolicyRunner` but:
  * Builds a :class:`MultiCriticAMPPPO` algorithm + :class:`MultiCriticActorCritic` policy.
  * Partitions the env's per-term rewards into per-critic-group rewards each
    step (defaulting any unknown term to ``aux``).
  * Folds the AMP style reward into the ``aux`` group.
"""
from __future__ import annotations

import os
import statistics
import time
from collections import deque
from typing import Sequence

import torch

import rsl_rl
from rsl_rl.algorithms import MultiCriticAMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    Discriminator,
    EmpiricalNormalization,
    MultiCriticActorCritic,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.utils import AMPLoader, Normalizer, store_code_state
from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner


class MultiCriticAmpOnPolicyRunner(AmpOnPolicyRunner):
    """AMP on-policy runner with per-term reward partitioning into G critic groups."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # We intentionally do *not* call ``super().__init__`` because the parent
        # constructor builds an ``AMPPPO`` + ``ActorCritic`` and a single-critic
        # storage. We replicate its setup with the multi-critic substitutions.
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        self._configure_multi_gpu()

        if self.alg_cfg["class_name"] not in ("PPO", "AMPPPO", "MultiCriticAMPPPO"):
            raise ValueError(f"Unsupported algorithm class for MultiCriticAmpOnPolicyRunner: {self.alg_cfg['class_name']}")
        self.training_type = "rl"

        # Resolve obs.
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            self.privileged_obs_type = "critic"
        else:
            self.privileged_obs_type = None
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # Resolve critic-group configuration from the algorithm cfg. These
        # mirror the keys consumed by :class:`MultiCriticAMPPPO`.
        self.num_critic_groups = int(self.alg_cfg.get("num_critic_groups", 2))
        self.critic_group_names = tuple(
            str(name) for name in self.alg_cfg.get("critic_group_names", ("goal", "aux"))
        )
        self.critic_group_weights = tuple(
            float(w) for w in self.alg_cfg.get("critic_group_weights", (2.0, 1.0))
        )
        if len(self.critic_group_names) != self.num_critic_groups:
            raise ValueError("critic_group_names length must equal num_critic_groups.")
        if len(self.critic_group_weights) != self.num_critic_groups:
            raise ValueError("critic_group_weights length must equal num_critic_groups.")
        # Group used for AMP style reward; defaults to the last group ("aux").
        amp_group_name = str(self.alg_cfg.get("amp_critic_group", self.critic_group_names[-1]))
        if amp_group_name not in self.critic_group_names:
            raise ValueError(
                f"amp_critic_group '{amp_group_name}' not in critic_group_names {self.critic_group_names}."
            )
        self._amp_group_idx = self.critic_group_names.index(amp_group_name)

        # Build policy. Force MultiCriticActorCritic by passing the multi-critic
        # parameters to whichever class is configured (MultiCriticActorCritic or
        # something inheriting from it).
        policy_class_name = self.policy_cfg.pop("class_name")
        try:
            policy_class = eval(policy_class_name)
        except NameError as exc:
            raise ValueError(f"Unknown policy class '{policy_class_name}' for MultiCriticAmpOnPolicyRunner.") from exc
        policy_cfg = dict(self.policy_cfg)
        policy_cfg.setdefault("num_critic_groups", self.num_critic_groups)
        policy_cfg.setdefault("critic_group_names", self.critic_group_names)
        policy = policy_class(num_obs, num_privileged_obs, self.env.num_actions, **policy_cfg).to(self.device)

        # AMP data + discriminator + normalizer (same as AmpOnPolicyRunner).
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            num_rnd_state = rnd_state.shape[1]
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        amp_data = AMPLoader(
            device,
            time_between_frames=self.env.env.env.step_dt,
            preload_transitions=True,
            num_preload_transitions=train_cfg["amp_num_preload_transitions"],
            motion_files=train_cfg["amp_motion_files"],
        )
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = Discriminator(
            amp_data.observation_dim * 2,
            train_cfg["amp_reward_coef"],
            train_cfg["amp_discr_hidden_dims"],
            device,
            train_cfg["amp_task_reward_lerp"],
        ).to(self.device)
        min_std = torch.zeros(len(train_cfg["min_normalized_std"]), device=self.device, requires_grad=False)

        # Build the algorithm. We pop our extra keys to avoid the parent class
        # complaining about unknown kwargs.
        alg_class_name = self.alg_cfg.pop("class_name")
        alg_class = eval(alg_class_name)
        alg_kwargs = dict(self.alg_cfg)
        # Pop multi-critic-only fields so we can pass them explicitly.
        alg_kwargs.pop("num_critic_groups", None)
        alg_kwargs.pop("critic_group_names", None)
        alg_kwargs.pop("critic_group_weights", None)
        alg_kwargs.pop("amp_critic_group", None)
        self.alg: MultiCriticAMPPPO = alg_class(
            policy,
            discriminator,
            amp_data,
            amp_normalizer,
            device=self.device,
            min_std=min_std,
            num_critic_groups=self.num_critic_groups,
            critic_group_names=self.critic_group_names,
            critic_group_weights=self.critic_group_weights,
            **alg_kwargs,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(
                shape=[num_privileged_obs], until=1.0e8
            ).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

        # Init storage + reward-partitioning lookups.
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        # Build per-term -> group index map from env reward config.
        self._build_reward_group_map()

        # Logging.
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    # ------------------------------------------------------------- helpers
    def _build_reward_group_map(self) -> None:
        """Resolve the per-term -> critic-group index map once at init."""
        env_unwrapped = self.env.unwrapped
        reward_manager = env_unwrapped.reward_manager
        rewards_cfg = env_unwrapped.cfg.rewards
        critic_groups = getattr(rewards_cfg, "CRITIC_GROUPS", None) or {}
        if not isinstance(critic_groups, dict):
            critic_groups = dict(critic_groups)

        # Default group: last in the configured list ("aux").
        default_group_idx = self.critic_group_names.index(self.critic_group_names[-1])
        name_to_idx = {name: idx for idx, name in enumerate(self.critic_group_names)}

        term_names: Sequence[str] = reward_manager.active_terms
        num_terms = len(term_names)
        idx_tensor = torch.full((num_terms,), default_group_idx, dtype=torch.long, device=self.device)
        unknown_groups: list[str] = []
        for term_idx, term_name in enumerate(term_names):
            group_name = critic_groups.get(term_name)
            if group_name is None:
                # Term not listed -> default ("aux").
                continue
            if group_name not in name_to_idx:
                unknown_groups.append(group_name)
                continue
            idx_tensor[term_idx] = name_to_idx[group_name]

        if unknown_groups:
            print(
                "[MultiCriticAmpOnPolicyRunner] WARNING: ignoring unknown critic group names "
                f"{sorted(set(unknown_groups))}. Valid groups: {self.critic_group_names}"
            )

        self._term_group_idx = idx_tensor  # (num_terms,)
        self._num_terms = num_terms
        self._step_dt = float(env_unwrapped.step_dt)
        # Pretty print for visibility.
        groups_summary = {
            name: [term for term, idx in zip(term_names, idx_tensor.tolist()) if idx == g_idx]
            for g_idx, name in enumerate(self.critic_group_names)
        }
        print(f"[MultiCriticAmpOnPolicyRunner] Reward group partition: {groups_summary}")

    def _compute_amp_style_reward(
        self,
        amp_obs: torch.Tensor,
        next_amp_obs: torch.Tensor,
        amp_reward_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute the discriminator-based AMP style reward as a 1-D tensor.

        Mirrors ``Discriminator.predict_amp_reward`` but skips the task-reward
        lerp; the resulting style reward is folded into the aux critic group.
        """
        disc = self.alg.discriminator
        with torch.no_grad():
            disc.eval()
            normalizer = self.alg.amp_normalizer
            if normalizer is not None:
                s = normalizer.normalize_torch(amp_obs, disc.device)
                sn = normalizer.normalize_torch(next_amp_obs, disc.device)
            else:
                s = amp_obs
                sn = next_amp_obs
            d = disc.amp_linear(disc.trunk(torch.cat([s, sn], dim=-1)))
            reward = disc.amp_reward_coef * torch.clamp(1 - 0.25 * torch.square(d - 1), min=0)
            if amp_reward_scale is not None:
                reward = reward * amp_reward_scale.unsqueeze(-1)
            disc.train()
        return reward.squeeze(-1)

    def _per_group_rewards(self) -> torch.Tensor:
        """Read the env's per-term step rewards and aggregate them per group.

        Returns:
            Tensor of shape ``(num_envs, G)``.
        """
        reward_manager = self.env.unwrapped.reward_manager
        # _step_reward stores value/dt; multiply by dt to get the per-step contribution.
        step_rewards = reward_manager._step_reward  # (N, num_terms), on env device
        step_rewards = step_rewards.to(self.device) * self._step_dt
        # Scatter-add along the term axis to a (N, G) tensor.
        num_envs = step_rewards.shape[0]
        out = torch.zeros(num_envs, self.num_critic_groups, device=self.device)
        # _term_group_idx: (num_terms,) — broadcast to (N, num_terms).
        term_group = self._term_group_idx.unsqueeze(0).expand(num_envs, -1)
        out.scatter_add_(1, term_group, step_rewards)
        return out

    # --------------------------------------------------------------- learn
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        amp_obs = extras["observations"].get("amp_observations")
        obs, privileged_obs, amp_obs = obs.to(self.device), privileged_obs.to(self.device), amp_obs.to(self.device)
        self.train_mode()

        ep_infos: list = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs, amp_obs)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    _, extras = self.env.get_observations()
                    next_amp_obs = extras["observations"].get("amp_observations")

                    obs, rewards, dones, next_amp_obs = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                        next_amp_obs.to(self.device),
                    )
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    # ------ split the env's per-term rewards into per-group rewards ------
                    per_group_rewards = self._per_group_rewards()  # (N, G)

                    # ------ AMP style reward; replaces the env's task reward in the parent
                    # runner. Here we read the discriminator's style reward and add it to
                    # the aux group (without lerping against the task reward, since the
                    # task reward is already represented in per_group_rewards in raw form).
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    reset_env_ids = self.env.env.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                    terminal_amp_states = extras["observations"].get("amp_observations")[reset_env_ids]
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    amp_reward_scale = None
                    zero_command_amp_scale = self.cfg.get("amp_zero_command_scale", 1.0)
                    if zero_command_amp_scale != 1.0:
                        command_name = self.cfg.get("amp_command_name", "base_velocity")
                        command_threshold = self.cfg.get("amp_zero_command_threshold", 1.0e-6)
                        command = self.env.env.env.command_manager.get_command(command_name).to(self.device)
                        zero_command = torch.norm(command, dim=1) < command_threshold
                        amp_reward_scale = torch.ones_like(rewards)
                        amp_reward_scale[zero_command] = zero_command_amp_scale

                    # Compute the AMP style reward directly (no task lerp) so we
                    # can fold it cleanly into the aux group.
                    style_reward = self._compute_amp_style_reward(
                        amp_obs, next_amp_obs_with_term, amp_reward_scale
                    )
                    per_group_rewards[:, self._amp_group_idx] = (
                        per_group_rewards[:, self._amp_group_idx] + style_reward
                    )
                    amp_obs = torch.clone(next_amp_obs)

                    # The transition needs values for bootstrapping; the act() call
                    # populated self.alg.transition.values with shape (N, G).
                    self.alg.process_env_step(per_group_rewards, dones, infos, next_amp_obs_with_term)

                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # Book-keeping (use weighted total reward for the per-episode mean).
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        weighted = (
                            per_group_rewards * self.alg.storage.critic_group_weights.to(self.device)
                        ).sum(dim=-1)
                        if self.alg.rnd:
                            cur_ereward_sum += weighted
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += weighted + intrinsic_rewards
                        else:
                            cur_reward_sum += weighted
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # Bootstrap values for the final state (per-group).
                self.alg.compute_returns(privileged_obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

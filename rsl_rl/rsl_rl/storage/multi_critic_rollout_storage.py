# Copyright (c) 2025-2026, The Booster Soccer Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Rollout storage with per-critic-group rewards/values/returns/advantages.

This storage variant carries an extra trailing dimension ``G`` for the rewards,
values, returns and advantages tensors so we can run a separate GAE per critic
group (e.g. ``goal`` vs. ``aux``) and combine the resulting advantages with
configurable weights ``w_g`` to drive a single policy.
"""
from __future__ import annotations

from typing import Sequence

import torch

from rsl_rl.storage.rollout_storage import RolloutStorage
from rsl_rl.utils import split_and_pad_trajectories


class MultiCriticRolloutStorage(RolloutStorage):
    """RolloutStorage with G value heads and per-group GAE."""

    class Transition(RolloutStorage.Transition):
        """Same as the parent transition; ``rewards``/``values`` are ``(N, G)``."""

        pass

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device: str = "cpu",
        num_critic_groups: int = 2,
        critic_group_weights: Sequence[float] = (2.0, 1.0),
    ) -> None:
        if num_critic_groups <= 0:
            raise ValueError(f"num_critic_groups must be positive, got {num_critic_groups}.")
        if len(critic_group_weights) != num_critic_groups:
            raise ValueError(
                "critic_group_weights length does not match num_critic_groups: "
                f"{len(critic_group_weights)} != {num_critic_groups}."
            )

        # Stash before super().__init__() — parent ctor allocates rewards/values
        # via standard (T, N, 1) shapes that we then re-allocate to (T, N, G).
        self.num_critic_groups = int(num_critic_groups)
        self.critic_group_weights = torch.tensor(
            list(critic_group_weights), dtype=torch.float, device=device
        )

        super().__init__(
            training_type=training_type,
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=obs_shape,
            privileged_obs_shape=privileged_obs_shape,
            actions_shape=actions_shape,
            rnd_state_shape=rnd_state_shape,
            device=device,
        )

        # Reallocate the per-group buffers.
        G = self.num_critic_groups
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, G, device=self.device)
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, G, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, G, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, G, device=self.device)
            # Combined advantage (T, N, 1) used for the surrogate loss.
            self.combined_advantages = torch.zeros(
                num_transitions_per_env, num_envs, 1, device=self.device
            )

    # ------------------------------------------------------------------ insert
    def add_transitions(self, transition: "MultiCriticRolloutStorage.Transition") -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core observations / actions / dones.
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # Per-group rewards & values, expected shape (N, G).
        rewards = transition.rewards
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if rewards.shape[-1] != self.num_critic_groups:
            raise ValueError(
                "Reward tensor must have last dim equal to num_critic_groups "
                f"({self.num_critic_groups}); got shape {tuple(rewards.shape)}."
            )
        self.rewards[self.step].copy_(rewards.view(-1, self.num_critic_groups))

        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        if self.training_type == "rl":
            values = transition.values
            if values.dim() == 1:
                values = values.unsqueeze(-1)
            if values.shape[-1] != self.num_critic_groups:
                raise ValueError(
                    "Value tensor must have last dim equal to num_critic_groups "
                    f"({self.num_critic_groups}); got shape {tuple(values.shape)}."
                )
            self.values[self.step].copy_(values.view(-1, self.num_critic_groups))
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        if self.rnd_state_shape is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)

        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    # ---------------------------------------------------------------- returns
    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        normalize_advantage: bool = True,
    ) -> None:
        """Per-group GAE; combined advantage stacked separately."""
        # Ensure last_values is (N, G).
        if last_values.dim() == 1:
            last_values = last_values.unsqueeze(-1)
        if last_values.shape[-1] != self.num_critic_groups:
            raise ValueError(
                "last_values must have last dim equal to num_critic_groups "
                f"({self.num_critic_groups}); got shape {tuple(last_values.shape)}."
            )

        advantage = torch.zeros_like(self.values[0])  # (N, G)
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # (N, 1) -> broadcast across G.
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Per-group advantages.
        self.advantages = self.returns - self.values
        # Normalize each group's advantage independently when requested.
        if normalize_advantage:
            mean = self.advantages.mean(dim=(0, 1), keepdim=True)
            std = self.advantages.std(dim=(0, 1), keepdim=True)
            self.advantages = (self.advantages - mean) / (std + 1e-8)

        # Combined advantage: weighted sum across G -> (T, N, 1)
        # Weights live on the same device as advantages.
        weights = self.critic_group_weights.to(self.advantages.device).view(1, 1, -1)
        self.combined_advantages = (self.advantages * weights).sum(dim=-1, keepdim=True)

    # ----------------------------------------------------- mini batch generator
    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Flatten time/env.
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        # Shape (T*N, G):
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        # Per-group advantages (T*N, G); combined advantage (T*N, 1):
        advantages = self.advantages.flatten(0, 1)
        combined_advantages = self.combined_advantages.flatten(0, 1)

        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                actions_batch = actions[batch_idx]

                target_values_batch = values[batch_idx]            # (B, G)
                returns_batch = returns[batch_idx]                  # (B, G)
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]            # (B, G)
                combined_adv_batch = combined_advantages[batch_idx]  # (B, 1)
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None

                yield (
                    obs_batch,
                    privileged_observations_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    combined_adv_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (None, None),
                    None,
                    rnd_state_batch,
                )

    # ------------------------------------------ recurrent mini batch generator
    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        # Recurrent path is not used by the soccer-kick task; raise to avoid silent issues.
        raise NotImplementedError(
            "MultiCriticRolloutStorage.recurrent_mini_batch_generator is not implemented. "
            "Use a feed-forward policy (MultiCriticActorCritic) with this storage."
        )

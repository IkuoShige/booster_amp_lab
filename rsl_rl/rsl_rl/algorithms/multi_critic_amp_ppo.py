# Copyright (c) 2025-2026, The Booster Soccer Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Multi-critic AMP-PPO algorithm (LVDRS §3.2).

This extends :class:`AMPPPO` by using :class:`MultiCriticRolloutStorage`. Each
critic head produces its own value/return/advantage trace; the surrogate loss
uses a weighted sum of the per-group advantages, while the value loss is the
sum of per-group MSEs against the per-group returns.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.storage.multi_critic_rollout_storage import MultiCriticRolloutStorage


class MultiCriticAMPPPO(AMPPPO):
    """AMP-PPO variant with ``G`` value heads + weighted-advantage surrogate."""

    def __init__(
        self,
        *args,
        num_critic_groups: int = 2,
        critic_group_names: Sequence[str] = ("goal", "aux"),
        critic_group_weights: Sequence[float] = (2.0, 1.0),
        **kwargs,
    ) -> None:
        if num_critic_groups <= 0:
            raise ValueError(f"num_critic_groups must be positive, got {num_critic_groups}.")
        if len(critic_group_names) != num_critic_groups or len(critic_group_weights) != num_critic_groups:
            raise ValueError(
                "critic_group_names / critic_group_weights length must match num_critic_groups."
            )

        super().__init__(*args, **kwargs)

        self.num_critic_groups = int(num_critic_groups)
        self.critic_group_names = tuple(str(n) for n in critic_group_names)
        self.critic_group_weights = tuple(float(w) for w in critic_group_weights)

    # --------------------------------------------------------- init storage
    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ) -> None:
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        self.storage = MultiCriticRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
            num_critic_groups=self.num_critic_groups,
            critic_group_weights=self.critic_group_weights,
        )
        # Replace the inherited transition object so its ``rewards``/``values``
        # fields naturally hold the (N, G) tensors we will provide.
        self.transition = MultiCriticRolloutStorage.Transition()

    # ----------------------------------------------------- env step bookkeeping
    def process_env_step(self, rewards, dones, infos, amp_obs) -> None:
        """Per-group rewards/values bootstrap.

        ``rewards`` is expected to be ``(N, G)`` already partitioned by the
        runner. The discriminator-shaped AMP reward should already be folded in
        by the caller (typically into the ``aux`` slice).
        """
        # Bootstrap on time outs across all groups simultaneously.
        rewards = rewards.clone()
        if "time_outs" in infos:
            time_outs = infos["time_outs"].to(self.device).float().unsqueeze(1)  # (N, 1)
            # transition.values is (N, G); broadcast across G.
            rewards = rewards + self.gamma * (self.transition.values * time_outs)

        self.transition.rewards = rewards
        self.transition.dones = dones

        # RND intrinsic reward (always 1-D); add uniformly to all groups so it
        # remains a soft regularizer regardless of the group weights.
        if self.rnd:
            rnd_state = infos["observations"]["rnd_state"]
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            self.transition.rewards = self.transition.rewards + self.intrinsic_rewards.unsqueeze(-1)
            self.transition.rnd_state = rnd_state.clone()

        # AMP replay + rollout storage.
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.policy.reset(dones)

    # ------------------------------------------------------ update / loss path
    def update(self):  # noqa: C901
        # Per-group value losses; we also track a scalar overall sum.
        mean_value_loss = 0.0
        mean_value_loss_per_group = [0.0 for _ in range(self.num_critic_groups)]
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0

        if self.rnd:
            mean_rnd_loss = 0.0
        else:
            mean_rnd_loss = None
        if self.symmetry:
            mean_symmetry_loss = 0.0
        else:
            mean_symmetry_loss = None

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,    # (B, G)
                advantages_batch,       # (B, G) (unused for surrogate; kept for diagnostics)
                combined_adv_batch,     # (B, 1)
                returns_batch,          # (B, G)
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                _hid_states_batch,
                _masks_batch,
                rnd_state_batch,
            ) = sample

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    combined_adv_batch = (combined_adv_batch - combined_adv_batch.mean()) / (
                        combined_adv_batch.std() + 1e-8
                    )

            # No data augmentation here (matching the AMPPPO non-symmetry path).
            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch)  # (B, G)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # KL adaptive LR.
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss using combined advantage.
            combined_adv = combined_adv_batch.squeeze(-1)  # (B,)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -combined_adv * ratio
            surrogate_clipped = -combined_adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Per-group value loss.
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)               # (B, G)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)     # (B, G)
                per_group_value_loss = torch.max(value_losses, value_losses_clipped).mean(dim=0)  # (G,)
            else:
                per_group_value_loss = (returns_batch - value_batch).pow(2).mean(dim=0)  # (G,)
            value_loss = per_group_value_loss.sum()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # RND loss (kept from parent for completeness; usually disabled for this task).
            if self.rnd:
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Discriminator loss.
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            expert_loss = nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
            policy_loss = nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(*sample_amp_expert, lambda_=10)
            loss += self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss

            # Backprop.
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore[union-attr]
                rnd_loss.backward()  # type: ignore[possibly-undefined]
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())

            # Accumulate stats.
            mean_value_loss += value_loss.item()
            for g in range(self.num_critic_groups):
                mean_value_loss_per_group[g] += per_group_value_loss[g].item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()  # type: ignore[possibly-undefined]

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        for g in range(self.num_critic_groups):
            mean_value_loss_per_group[g] /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp": mean_amp_loss,
            "amp_grad_pen": mean_grad_pen_loss,
            "amp_policy_pred": mean_policy_pred,
            "amp_expert_pred": mean_expert_pred,
        }
        for g, name in enumerate(self.critic_group_names):
            loss_dict[f"value_function_{name}"] = mean_value_loss_per_group[g]
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        return loss_dict

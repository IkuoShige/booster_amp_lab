# Copyright (c) 2025-2026, The Booster Soccer Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""V4 Stage 5 — Multi-critic AMP-PPO with privileged-state decoder aux-loss.

This is :class:`MultiCriticAMPPPO` augmented with an auxiliary decoder MSE
loss. The encoder/decoder live on the policy (see
:class:`rsl_rl.modules.EncoderActorCritic`); this algorithm reads the cached
latent after each ``policy.act`` call during update and minimizes

    L_decode = MSE(decoder(encoder(history_slice)), critic_obs[target_slice])

scaled by ``decoder_loss_coef`` and added to the total PPO loss. No changes
to storage are required — both the actor obs and the critic obs are already
provided by the existing :class:`MultiCriticRolloutStorage` mini-batch.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from rsl_rl.algorithms.multi_critic_amp_ppo import MultiCriticAMPPPO


class EncoderMultiCriticAMPPPO(MultiCriticAMPPPO):
    """Stage 5 multi-critic AMP-PPO with decoder MSE auxiliary loss."""

    def __init__(
        self,
        *args,
        decoder_target_slice: Sequence[int] = (0, 9),
        decoder_loss_coef: float = 1.0,
        **kwargs,
    ) -> None:
        if len(decoder_target_slice) != 2:
            raise ValueError(
                f"decoder_target_slice must be a 2-tuple (start, end); got {decoder_target_slice!r}."
            )
        super().__init__(*args, **kwargs)
        target_start = int(decoder_target_slice[0])
        target_end = int(decoder_target_slice[1])
        if not (0 <= target_start < target_end):
            raise ValueError(
                f"decoder_target_slice [{target_start}, {target_end}) is invalid."
            )
        self.decoder_target_slice = (target_start, target_end)
        self.decoder_loss_coef = float(decoder_loss_coef)
        # MSE module — instantiate once to avoid per-batch object churn.
        self._mse = nn.MSELoss()

    # ----------------------------------------------------------- update path
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
        mean_decoder_loss = 0.0

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

        target_start, target_end = self.decoder_target_slice

        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                combined_adv_batch,
                returns_batch,
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

            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # Cache the actor-side latent (encoder output for this mini-batch).
            actor_latent = self.policy.latest_latent
            value_batch = self.policy.evaluate(critic_obs_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

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
            combined_adv = combined_adv_batch.squeeze(-1)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -combined_adv * ratio
            surrogate_clipped = -combined_adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Per-group value loss.
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                per_group_value_loss = torch.max(value_losses, value_losses_clipped).mean(dim=0)
            else:
                per_group_value_loss = (returns_batch - value_batch).pow(2).mean(dim=0)
            value_loss = per_group_value_loss.sum()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # ---- Stage 5 decoder aux loss --------------------------------
            # Target is a privileged slice of the critic obs (e.g. GT ball
            # position/velocity expressed in base frame). The decoder is run on
            # the cached actor-side latent so gradients flow encoder -> decoder
            # only (no double encoder pass through the critic obs).
            target_end_eff = min(target_end, critic_obs_batch.shape[-1])
            decoder_target = critic_obs_batch[..., target_start:target_end_eff].detach()
            decoder_output = self.policy.decoder(actor_latent)
            if decoder_output.shape[-1] != decoder_target.shape[-1]:
                raise RuntimeError(
                    f"Decoder output dim ({decoder_output.shape[-1]}) does not match decoder_target "
                    f"slice width ({decoder_target.shape[-1]}). Check decoder_target_dim vs "
                    f"decoder_target_slice."
                )
            decoder_loss = self._mse(decoder_output, decoder_target)
            loss = loss + self.decoder_loss_coef * decoder_loss

            # RND loss (kept from parent for completeness; usually disabled).
            if self.rnd:
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = self._mse(predicted_embedding, target_embedding)

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
            mean_decoder_loss += decoder_loss.item()
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
        mean_decoder_loss /= num_updates
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp": mean_amp_loss,
            "amp_grad_pen": mean_grad_pen_loss,
            "amp_policy_pred": mean_policy_pred,
            "amp_expert_pred": mean_expert_pred,
            "decoder": mean_decoder_loss,
        }
        for g, name in enumerate(self.critic_group_names):
            loss_dict[f"value_function_{name}"] = mean_value_loss_per_group[g]
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        return loss_dict

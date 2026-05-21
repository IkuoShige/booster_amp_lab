# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from __future__ import annotations

import torch

from rsl_rl.modules.track_adapter_actor_critic import _target_feature_indices
from rsl_rl.utils import split_and_pad_trajectories


class TrackAdapterRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.next_observations = None
            self.world_model_state = None
            self.world_model_reference_state = None
            self.privileged_observations = None
            self.history = None
            self.residual_action_gate = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.rnd_state = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        history_shape,
        world_model_state_shape=None,
        rnd_state_shape=None,
        device="cpu",
    ):
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.rnd_state_shape = rnd_state_shape
        self.actions_shape = actions_shape
        self.history_shape = self._normalize_shape(history_shape)
        self.world_model_state_shape = self._normalize_shape(world_model_state_shape) if world_model_state_shape else None

        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.next_observation_valid = torch.zeros(
            num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=self.device
        )
        if self.world_model_state_shape is not None:
            self.world_model_states = torch.zeros(
                num_transitions_per_env, num_envs, *self.world_model_state_shape, device=self.device
            )
            self.world_model_reference_states = torch.zeros(
                num_transitions_per_env, num_envs, *self.world_model_state_shape, device=self.device
            )
            self.world_model_state_valid = torch.zeros(
                num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=self.device
            )
            self.world_model_reference_state_valid = torch.zeros(
                num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=self.device
            )
        else:
            self.world_model_states = None
            self.world_model_reference_states = None
            self.world_model_state_valid = None
            self.world_model_reference_state_valid = None
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        self.history = torch.zeros(num_transitions_per_env, num_envs, *self.history_shape, device=self.device)
        self.residual_action_gates = torch.ones(num_transitions_per_env, num_envs, 1, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        if rnd_state_shape is not None:
            self.rnd_state = torch.zeros(num_transitions_per_env, num_envs, *rnd_state_shape, device=self.device)

        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None
        self.step = 0

    @staticmethod
    def _normalize_shape(shape):
        if isinstance(shape, int):
            return (shape,)
        return tuple(shape)

    def _select_world_model_source(self, target_source):
        if target_source in ("world_model_state", "wm_state", "structured", "psi_s"):
            if self.world_model_states is None:
                raise ValueError("World-model state targets were requested but storage has no world_model_state buffer.")
            return self.world_model_states
        if target_source in ("next_actor", "next_obs", "next_observation", "next_observations"):
            return self.next_observations
        if target_source in ("critic", "privileged", "privileged_observations"):
            if self.privileged_observations is not None:
                return self.privileged_observations
            return self.observations
        if target_source in ("actor", "obs", "observation", "observations", None):
            return self.observations
        raise ValueError(f"Unknown world model target source: {target_source}")

    @staticmethod
    def _unpack_world_model_target_config(target_source, target_dim=None, target_slices=None, target_groups=None):
        if isinstance(target_source, dict):
            config = target_source
            target_source = config.get(
                "source",
                config.get("target_source", "observations"),
            )
            target_dim = config.get("target_dim", config.get("dim", target_dim))
            target_slices = config.get("target_slices", config.get("slices", target_slices))
            target_groups = config.get("target_groups", config.get("groups", target_groups))
        elif isinstance(target_source, (tuple, list)):
            target_groups = target_source if target_groups is None else target_groups
            target_source = "observations"
        elif isinstance(target_source, str) and "::" in target_source:
            source_name, group_spec = target_source.split("::", 1)
            target_source = source_name
            target_groups = group_spec if target_groups is None else target_groups
        return target_source, target_dim, target_slices, target_groups

    def _select_world_model_features(self, features, target_slices=None, target_groups=None):
        indices = _target_feature_indices(
            features.shape[-1],
            self._normalize_shape(self.actions_shape)[-1],
            target_slices=target_slices,
            target_groups=target_groups,
        )
        if not indices:
            return features
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=features.device)
        return features.index_select(-1, index_tensor)

    @staticmethod
    def _fit_feature_dim(features, target_dim):
        if target_dim is None:
            return features
        target_dim = int(target_dim)
        if features.shape[-1] == target_dim:
            return features
        if features.shape[-1] > target_dim:
            return features[..., :target_dim]
        pad_width = target_dim - features.shape[-1]
        padding = features.new_zeros(*features.shape[:-1], pad_width)
        return torch.cat((features, padding), dim=-1)

    def world_model_targets(
        self,
        horizon=1,
        target_source="observations",
        target_dim=None,
        target_slices=None,
        target_groups=None,
    ):
        target_source, target_dim, target_slices, target_groups = self._unpack_world_model_target_config(
            target_source,
            target_dim=target_dim,
            target_slices=target_slices,
            target_groups=target_groups,
        )
        horizon = max(1, int(horizon))
        observation_sources = ("actor", "obs", "observation", "observations", None)
        structured_source = target_source in ("world_model_state", "wm_state", "structured", "psi_s")
        explicit_next_source = target_source in observation_sources and bool(torch.all(self.next_observation_valid))
        source = self.next_observations if explicit_next_source else self._select_world_model_source(target_source)
        source = self._select_world_model_features(
            source,
            target_slices=target_slices,
            target_groups=target_groups,
        )
        source = self._fit_feature_dim(source, target_dim)
        source_is_next = structured_source or explicit_next_source or target_source in (
            "next_actor",
            "next_obs",
            "next_observation",
            "next_observations",
        )
        targets = source.new_zeros(
            self.num_transitions_per_env,
            self.num_envs,
            horizon,
            source.shape[-1],
        )
        valid_mask = source.new_zeros(self.num_transitions_per_env, self.num_envs, horizon)
        dones = self.dones.squeeze(-1).bool()

        for target_index in range(horizon):
            source_start = target_index if source_is_next else target_index + 1
            if source_start >= self.num_transitions_per_env:
                continue
            upper = self.num_transitions_per_env - source_start
            targets[:upper, :, target_index].copy_(source[source_start:])
            valid = torch.ones(
                upper,
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
            for done_offset in range(target_index + 1):
                valid = valid & ~dones[done_offset : upper + done_offset]
            if structured_source and self.world_model_state_valid is not None:
                valid = valid & self.world_model_state_valid[source_start : source_start + upper].squeeze(-1).bool()
            valid_mask[:upper, :, target_index].copy_(valid.to(dtype=valid_mask.dtype))
        return targets, valid_mask

    def world_model_references(
        self,
        target_source="observations",
        target_dim=None,
        target_slices=None,
        target_groups=None,
    ):
        target_source, target_dim, target_slices, target_groups = self._unpack_world_model_target_config(
            target_source,
            target_dim=target_dim,
            target_slices=target_slices,
            target_groups=target_groups,
        )
        if target_source in ("world_model_state", "wm_state", "structured", "psi_s"):
            if self.world_model_reference_states is None:
                raise ValueError("World-model state references were requested but storage has no reference buffer.")
            source = self.world_model_reference_states
        else:
            source = self.observations
        source = self._select_world_model_features(
            source,
            target_slices=target_slices,
            target_groups=target_groups,
        )
        return self._fit_feature_dim(source, target_dim)

    def world_model_action_sequences(self, horizon=1):
        horizon = max(1, int(horizon))
        action_sequences = self.actions.new_zeros(
            self.num_transitions_per_env,
            self.num_envs,
            horizon,
            *self._normalize_shape(self.actions_shape),
        )
        for action_index in range(horizon):
            if action_index >= self.num_transitions_per_env:
                continue
            upper = self.num_transitions_per_env - action_index
            action_sequences[:upper, :, action_index].copy_(self.actions[action_index:])
        return action_sequences

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        self.observations[self.step].copy_(transition.observations)
        if transition.next_observations is None:
            self.next_observations[self.step].copy_(transition.observations)
            self.next_observation_valid[self.step].zero_()
        else:
            self.next_observations[self.step].copy_(transition.next_observations)
            self.next_observation_valid[self.step].fill_(True)
        if self.world_model_states is not None:
            if transition.world_model_state is None:
                self.world_model_states[self.step].zero_()
                self.world_model_state_valid[self.step].zero_()
            else:
                self.world_model_states[self.step].copy_(transition.world_model_state)
                self.world_model_state_valid[self.step].fill_(True)
            if transition.world_model_reference_state is None:
                self.world_model_reference_states[self.step].zero_()
                self.world_model_reference_state_valid[self.step].zero_()
            else:
                self.world_model_reference_states[self.step].copy_(transition.world_model_reference_state)
                self.world_model_reference_state_valid[self.step].fill_(True)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.history[self.step].copy_(transition.history)
        if transition.residual_action_gate is None:
            self.residual_action_gates[self.step].fill_(1.0)
        else:
            gate = transition.residual_action_gate
            if gate.dim() == 1:
                gate = gate.unsqueeze(-1)
            if gate.shape[-1] != 1:
                raise ValueError(
                    f"TrackAdapterRolloutStorage stores scalar residual gates, got shape {tuple(gate.shape)}."
                )
            self.residual_action_gates[self.step].copy_(gate)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        if self.rnd_state_shape is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)

        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def generator(self):
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            if self.privileged_observations is not None:
                privileged_observations = self.privileged_observations[i]
            else:
                privileged_observations = self.observations[i]
            yield (
                self.observations[i],
                privileged_observations,
                self.history[i],
                self.actions[i],
                self.privileged_actions[i],
                self.dones[i],
            )

    def mini_batch_generator(
        self,
        num_mini_batches,
        num_epochs=8,
        include_world_model=False,
        world_model_horizon=1,
        world_model_target_source="observations",
        world_model_target_dim=None,
        world_model_target_slices=None,
        world_model_target_groups=None,
    ):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations
        history = self.history.flatten(0, 1)
        residual_action_gates = self.residual_action_gates.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)

        if include_world_model:
            wm_references = self.world_model_references(
                target_source=world_model_target_source,
                target_dim=world_model_target_dim,
                target_slices=world_model_target_slices,
                target_groups=world_model_target_groups,
            )
            wm_targets, wm_valid_mask = self.world_model_targets(
                horizon=world_model_horizon,
                target_source=world_model_target_source,
                target_dim=world_model_target_dim,
                target_slices=world_model_target_slices,
                target_groups=world_model_target_groups,
            )
            wm_action_sequences = self.world_model_action_sequences(horizon=world_model_horizon)
            wm_references = wm_references.flatten(0, 1)
            wm_targets = wm_targets.flatten(0, 1)
            wm_valid_mask = wm_valid_mask.flatten(0, 1)
            wm_action_sequences = wm_action_sequences.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None

                sample = (
                    observations[batch_idx],
                    privileged_observations[batch_idx],
                    actions[batch_idx],
                    history[batch_idx],
                    residual_action_gates[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),
                    None,
                    rnd_state_batch,
                )
                if include_world_model:
                    sample = sample + (
                        wm_references[batch_idx],
                        wm_targets[batch_idx],
                        wm_valid_mask[batch_idx],
                        wm_action_sequences[batch_idx],
                    )
                yield sample

    def recurrent_mini_batch_generator(
        self,
        num_mini_batches,
        num_epochs=8,
        include_world_model=False,
        world_model_horizon=1,
        world_model_target_source="observations",
        world_model_target_dim=None,
        world_model_target_slices=None,
        world_model_target_groups=None,
    ):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        if self.saved_hidden_states_a is None or self.saved_hidden_states_c is None:
            raise RuntimeError("Recurrent mini-batches require saved hidden states.")

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        padded_history_trajectories, _ = split_and_pad_trajectories(self.history, self.dones)
        padded_residual_gate_trajectories, _ = split_and_pad_trajectories(self.residual_action_gates, self.dones)
        if self.privileged_observations is not None:
            padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_privileged_obs_trajectories = padded_obs_trajectories

        if self.rnd_state_shape is not None:
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        else:
            padded_rnd_state_trajectories = None

        if include_world_model:
            wm_references = self.world_model_references(
                target_source=world_model_target_source,
                target_dim=world_model_target_dim,
                target_slices=world_model_target_slices,
                target_groups=world_model_target_groups,
            )
            wm_targets, wm_valid_mask = self.world_model_targets(
                horizon=world_model_horizon,
                target_source=world_model_target_source,
                target_dim=world_model_target_dim,
                target_slices=world_model_target_slices,
                target_groups=world_model_target_groups,
            )
            wm_action_sequences = self.world_model_action_sequences(horizon=world_model_horizon)

        mini_batch_size = self.num_envs // num_mini_batches
        for _ in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]
                history_batch = padded_history_trajectories[:, first_traj:last_traj]
                residual_action_gate_batch = padded_residual_gate_trajectories[:, first_traj:last_traj]

                if padded_rnd_state_trajectories is not None:
                    rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
                else:
                    rnd_state_batch = None

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]
                if include_world_model:
                    wm_references_batch = wm_references[:, start:stop]
                    wm_targets_batch = wm_targets[:, start:stop]
                    wm_valid_mask_batch = wm_valid_mask[:, start:stop]
                    wm_action_sequences_batch = wm_action_sequences[:, start:stop]

                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                sample = (
                    obs_batch,
                    privileged_obs_batch,
                    actions_batch,
                    history_batch,
                    residual_action_gate_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (hid_a_batch, hid_c_batch),
                    masks_batch,
                    rnd_state_batch,
                )
                if include_world_model:
                    sample = sample + (
                        wm_references_batch,
                        wm_targets_batch,
                        wm_valid_mask_batch,
                        wm_action_sequences_batch,
                    )
                yield sample

                first_traj = last_traj

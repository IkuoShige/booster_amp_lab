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

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import ReplayBuffer, TrackAdapterRolloutStorage, TrackAdapterWorldModelReplayBuffer
from rsl_rl.utils import string_to_callable


class PrivilegedTeacherDistillReplayBuffer:
    """Persistent DAgger/BC replay for privileged-teacher recovery targets."""

    def __init__(
        self,
        obs_shape,
        critic_obs_shape,
        history_shape,
        gate_shape,
        action_shape,
        buffer_size,
        sample_device,
        storage_device="cpu",
        storage_dtype=torch.float16,
    ):
        self.obs_shape = tuple(obs_shape)
        self.critic_obs_shape = tuple(critic_obs_shape)
        self.history_shape = tuple(history_shape)
        self.gate_shape = tuple(gate_shape)
        self.action_shape = tuple(action_shape)
        self.buffer_size = int(buffer_size)
        if self.buffer_size <= 0:
            raise ValueError(f"PrivilegedTeacherDistillReplayBuffer buffer_size must be positive, got {buffer_size}.")
        self.sample_device = torch.device(sample_device)
        self.storage_device = torch.device(storage_device)
        self.storage_dtype = storage_dtype
        self.observations = torch.zeros(self.buffer_size, *self.obs_shape, device=self.storage_device, dtype=storage_dtype)
        self.critic_observations = torch.zeros(
            self.buffer_size, *self.critic_obs_shape, device=self.storage_device, dtype=storage_dtype
        )
        self.history = torch.zeros(self.buffer_size, *self.history_shape, device=self.storage_device, dtype=storage_dtype)
        self.residual_gates = torch.zeros(self.buffer_size, *self.gate_shape, device=self.storage_device, dtype=storage_dtype)
        self.teacher_actions = torch.zeros(
            self.buffer_size, *self.action_shape, device=self.storage_device, dtype=storage_dtype
        )
        self.step = 0
        self.num_samples = 0
        self.total_inserted = 0

    @staticmethod
    def _flatten(tensor, shape):
        return tensor.detach().reshape(-1, *tuple(shape))

    @staticmethod
    def _finite_rows(tensor):
        return torch.isfinite(tensor.reshape(tensor.shape[0], -1)).all(dim=1)

    def insert(self, observations, critic_observations, history, residual_gates, teacher_actions, mask=None):
        observations = self._flatten(observations, self.obs_shape)
        critic_observations = self._flatten(critic_observations, self.critic_obs_shape)
        history = self._flatten(history, self.history_shape)
        residual_gates = self._flatten(residual_gates, self.gate_shape)
        teacher_actions = self._flatten(teacher_actions, self.action_shape)
        count = observations.shape[0]
        if not (
            critic_observations.shape[0]
            == history.shape[0]
            == residual_gates.shape[0]
            == teacher_actions.shape[0]
            == count
        ):
            raise ValueError("Privileged teacher distill replay samples have inconsistent leading dimensions.")
        valid_rows = (
            self._finite_rows(observations)
            & self._finite_rows(critic_observations)
            & self._finite_rows(history)
            & self._finite_rows(residual_gates)
            & self._finite_rows(teacher_actions)
        )
        if mask is not None:
            valid_rows = valid_rows & mask.detach().reshape(-1).to(device=valid_rows.device, dtype=torch.bool)
        if not bool(valid_rows.any()):
            return 0
        observations = observations[valid_rows].to(device=self.storage_device, dtype=self.storage_dtype)
        critic_observations = critic_observations[valid_rows].to(device=self.storage_device, dtype=self.storage_dtype)
        history = history[valid_rows].to(device=self.storage_device, dtype=self.storage_dtype)
        residual_gates = residual_gates[valid_rows].to(device=self.storage_device, dtype=self.storage_dtype)
        teacher_actions = teacher_actions[valid_rows].to(device=self.storage_device, dtype=self.storage_dtype)
        num_samples = observations.shape[0]
        if num_samples >= self.buffer_size:
            observations = observations[-self.buffer_size :]
            critic_observations = critic_observations[-self.buffer_size :]
            history = history[-self.buffer_size :]
            residual_gates = residual_gates[-self.buffer_size :]
            teacher_actions = teacher_actions[-self.buffer_size :]
            num_samples = self.buffer_size
        end_idx = self.step + num_samples
        if end_idx <= self.buffer_size:
            target = slice(self.step, end_idx)
            self.observations[target].copy_(observations)
            self.critic_observations[target].copy_(critic_observations)
            self.history[target].copy_(history)
            self.residual_gates[target].copy_(residual_gates)
            self.teacher_actions[target].copy_(teacher_actions)
        else:
            first = self.buffer_size - self.step
            second = end_idx - self.buffer_size
            self.insert(observations[:first], critic_observations[:first], history[:first], residual_gates[:first], teacher_actions[:first])
            self.insert(observations[first:first + second], critic_observations[first:first + second], history[first:first + second], residual_gates[first:first + second], teacher_actions[first:first + second])
            return num_samples
        self.step = (self.step + num_samples) % self.buffer_size
        self.num_samples = min(self.buffer_size, self.num_samples + num_samples)
        self.total_inserted += num_samples
        return num_samples

    def sample(self, batch_size):
        if self.num_samples <= 0:
            raise RuntimeError("Cannot sample an empty privileged teacher distill replay buffer.")
        batch_size = max(1, min(int(batch_size), self.num_samples))
        indices = torch.randint(self.num_samples, (batch_size,), device=self.storage_device)
        return (
            self.observations[indices].to(device=self.sample_device, dtype=torch.float32),
            self.critic_observations[indices].to(device=self.sample_device, dtype=torch.float32),
            self.history[indices].to(device=self.sample_device, dtype=torch.float32),
            self.residual_gates[indices].to(device=self.sample_device, dtype=torch.float32),
            self.teacher_actions[indices].to(device=self.sample_device, dtype=torch.float32),
        )

    def save(self, path, max_samples=None):
        num_samples = self.num_samples if max_samples is None else min(self.num_samples, int(max_samples))
        num_samples = max(0, min(num_samples, self.num_samples))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "obs_shape": self.obs_shape,
                "critic_obs_shape": self.critic_obs_shape,
                "history_shape": self.history_shape,
                "gate_shape": self.gate_shape,
                "action_shape": self.action_shape,
                "num_samples": num_samples,
                "observations": self.observations[:num_samples].cpu(),
                "critic_observations": self.critic_observations[:num_samples].cpu(),
                "history": self.history[:num_samples].cpu(),
                "residual_gates": self.residual_gates[:num_samples].cpu(),
                "teacher_actions": self.teacher_actions[:num_samples].cpu(),
            },
            path,
        )

    def load(self, path):
        loaded = torch.load(path, map_location=self.storage_device, weights_only=False)
        expected_shapes = {
            "obs_shape": self.obs_shape,
            "critic_obs_shape": self.critic_obs_shape,
            "history_shape": self.history_shape,
            "gate_shape": self.gate_shape,
            "action_shape": self.action_shape,
        }
        mismatched = [
            f"{name}: checkpoint{tuple(loaded.get(name, ())) }->current{shape}"
            for name, shape in expected_shapes.items()
            if tuple(loaded.get(name, ())) != tuple(shape)
        ]
        if mismatched:
            raise RuntimeError("Privileged teacher distill replay shape mismatch: " + ", ".join(mismatched))
        num_samples = int(loaded.get("num_samples", 0))
        num_samples = max(0, min(num_samples, self.buffer_size))
        if num_samples == 0:
            return 0
        for name, target in (
            ("observations", self.observations),
            ("critic_observations", self.critic_observations),
            ("history", self.history),
            ("residual_gates", self.residual_gates),
            ("teacher_actions", self.teacher_actions),
        ):
            source = loaded[name][-num_samples:].to(device=self.storage_device, dtype=self.storage_dtype)
            target[:num_samples].copy_(source)
        self.num_samples = num_samples
        self.step = num_samples % self.buffer_size
        self.total_inserted = max(self.total_inserted, num_samples)
        return num_samples


class TrackAdapterAMPPPO:
    """AMP-PPO variant that threads Track Adapter history through actor and critic calls."""

    def __init__(
        self,
        policy,
        discriminator,
        amp_data,
        amp_normalizer,
        amp_replay_buffer_size=100000,
        min_std=None,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        freeze_discriminator=False,
        residual_penalty_coef=0.0,
        residual_penalty_cfg: dict | None = None,
        recovery_teacher_cfg: dict | None = None,
        privileged_teacher_cfg: dict | None = None,
        privileged_teacher_distill_cfg: dict | None = None,
        world_model_loss_coef=0.0,
        world_model_cfg: dict | None = None,
        wm_loss_coef=None,
        wm_horizon=None,
        wm_target_source=None,
        wm_target_dim=None,
        wm_target_slices=None,
        wm_target_groups=None,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "TrackAdapterAMPPPO.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        self.device = device
        self.phase_trace_enabled = os.getenv("BOOSTER_TRACK_ADAPTER_PHASE_TRACE", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        if rnd_cfg is not None:
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        if symmetry_cfg is not None and (
            symmetry_cfg.get("use_data_augmentation", False) or symmetry_cfg.get("use_mirror_loss", False)
        ):
            raise ValueError(
                "TrackAdapterAMPPPO does not support symmetry augmentation yet because history stores "
                "observation plus action frames. Set symmetry_cfg=None for Track Adapter."
            )

        if symmetry_cfg is not None:
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        self.amploss_coef = 1.0
        self.min_std = min_std
        self.freeze_discriminator = freeze_discriminator
        self.residual_penalty_cfg = residual_penalty_cfg or {}
        self.residual_penalty_coef = residual_penalty_coef
        self.ungated_residual_policy_loss_weight = float(
            self.residual_penalty_cfg.get(
                "ungated_policy_loss_weight",
                self.residual_penalty_cfg.get("ungated_stable_weight", 0.0),
            )
        )
        self.ungated_residual_policy_gate_threshold = float(
            self.residual_penalty_cfg.get("ungated_policy_gate_threshold", 0.10)
        )
        self.recovery_residual_activation_weight = float(
            self.residual_penalty_cfg.get("recovery_activation_weight", 0.0)
        )
        self.recovery_residual_activation_gate_threshold = float(
            self.residual_penalty_cfg.get("recovery_activation_gate_threshold", 0.65)
        )
        self.recovery_residual_activation_target_norm = float(
            self.residual_penalty_cfg.get("recovery_activation_target_norm", 0.02)
        )
        self.recovery_residual_activation_max_norm = float(
            self.residual_penalty_cfg.get("recovery_activation_max_norm", 0.0)
        )
        self.recovery_residual_activation_overgrowth_weight = float(
            self.residual_penalty_cfg.get("recovery_activation_overgrowth_weight", 0.0)
        )
        self.recovery_ppo_sample_weight = float(self.residual_penalty_cfg.get("recovery_ppo_sample_weight", 0.0))
        self.recovery_ppo_sample_gate_threshold = float(
            self.residual_penalty_cfg.get("recovery_ppo_sample_gate_threshold", 0.45)
        )
        self.recovery_ppo_sample_power = float(self.residual_penalty_cfg.get("recovery_ppo_sample_power", 1.0))
        self.recovery_teacher_cfg = recovery_teacher_cfg or {}
        self.recovery_teacher_enabled = bool(self.recovery_teacher_cfg.get("enabled", False))
        self.recovery_teacher_coef = float(
            self.recovery_teacher_cfg.get("coef", self.recovery_teacher_cfg.get("loss_coef", 0.0))
        )
        self.recovery_teacher_gate_threshold = float(self.recovery_teacher_cfg.get("gate_threshold", 0.55))
        self.recovery_teacher_score_threshold = float(self.recovery_teacher_cfg.get("score_threshold", 0.18))
        self.recovery_teacher_velocity_gain = float(self.recovery_teacher_cfg.get("velocity_gain", 0.55))
        self.recovery_teacher_tilt_gain = float(self.recovery_teacher_cfg.get("tilt_gain", 0.35))
        self.recovery_teacher_ang_vel_gain = float(self.recovery_teacher_cfg.get("ang_vel_gain", 0.10))
        self.recovery_teacher_force_gain = float(self.recovery_teacher_cfg.get("force_gain", 0.0))
        self.recovery_teacher_push_delta_gain = float(self.recovery_teacher_cfg.get("push_delta_gain", 0.0))
        self.recovery_teacher_max_delta = float(self.recovery_teacher_cfg.get("max_delta", 0.12))
        self.recovery_teacher_non_leg_weight = float(self.recovery_teacher_cfg.get("non_leg_weight", 0.02))
        self.recovery_teacher_joint_indices = dict(self.recovery_teacher_cfg.get("joint_indices", {}))
        self.recovery_teacher_apply_to_privileged = bool(
            self.recovery_teacher_cfg.get("apply_to_privileged", False)
        )
        self.recovery_teacher_separate_actor_grad_clip = bool(
            self.recovery_teacher_cfg.get("separate_actor_grad_clip", False)
        )
        self.recovery_teacher_actor_max_grad_norm = float(
            self.recovery_teacher_cfg.get("actor_max_grad_norm", max_grad_norm)
        )
        self.privileged_teacher_cfg = privileged_teacher_cfg or {}
        self.train_privileged_teacher = bool(
            self.privileged_teacher_cfg.get(
                "train",
                self.privileged_teacher_cfg.get("enabled", False),
            )
        )
        self.privileged_teacher_distill_cfg = privileged_teacher_distill_cfg or {}
        self.privileged_teacher_distill_enabled = bool(
            self.privileged_teacher_distill_cfg.get("enabled", False)
        )
        self.privileged_teacher_distill_coef = float(
            self.privileged_teacher_distill_cfg.get("coef", self.privileged_teacher_distill_cfg.get("loss_coef", 0.0))
        )
        self.privileged_teacher_distill_gate_threshold = float(
            self.privileged_teacher_distill_cfg.get("gate_threshold", 0.55)
        )
        self.privileged_teacher_distill_score_threshold = float(
            self.privileged_teacher_distill_cfg.get("score_threshold", 0.12)
        )
        self.privileged_teacher_distill_replay_enabled = bool(
            self.privileged_teacher_distill_cfg.get("replay_enabled", False)
        )
        self.privileged_teacher_distill_replay_buffer_size = int(
            self.privileged_teacher_distill_cfg.get("replay_buffer_size", 262144)
        )
        self.privileged_teacher_distill_replay_batch_size = int(
            self.privileged_teacher_distill_cfg.get("replay_batch_size", 4096)
        )
        self.privileged_teacher_distill_replay_updates_per_iteration = int(
            self.privileged_teacher_distill_cfg.get("replay_updates_per_iteration", 0)
        )
        self.privileged_teacher_distill_replay_warm_start_updates = int(
            self.privileged_teacher_distill_cfg.get("replay_warm_start_updates", 0)
        )
        self.privileged_teacher_distill_replay_min_samples = int(
            self.privileged_teacher_distill_cfg.get("replay_min_samples", 4096)
        )
        self.privileged_teacher_distill_capture_teacher_replay = bool(
            self.privileged_teacher_distill_cfg.get("capture_teacher_replay", False)
        )
        self.privileged_teacher_distill_replay_path = str(
            self.privileged_teacher_distill_cfg.get("replay_path", "") or ""
        )
        self.privileged_teacher_distill_replay_load_path = str(
            self.privileged_teacher_distill_cfg.get("replay_load_path", "") or ""
        )
        self.privileged_teacher_distill_replay_save_interval = int(
            self.privileged_teacher_distill_cfg.get("replay_save_interval", 0)
        )
        self.privileged_teacher_distill_replay_save_on_checkpoint = bool(
            self.privileged_teacher_distill_cfg.get("replay_save_on_checkpoint", True)
        )
        self.privileged_teacher_distill_replay_save_max_samples = int(
            self.privileged_teacher_distill_cfg.get(
                "replay_save_max_samples", self.privileged_teacher_distill_replay_buffer_size
            )
        )
        self._privileged_teacher_distill_replay_loaded_from_path = False
        self._privileged_teacher_distill_warm_start_done = False
        self._privileged_teacher_distill_update_calls = 0
        self.world_model_cfg = world_model_cfg or {}
        if wm_loss_coef is not None:
            world_model_loss_coef = wm_loss_coef
        configured_wm_coef = self.world_model_cfg.get(
            "loss_coef",
            self.world_model_cfg.get("coef", world_model_loss_coef),
        )
        if self.world_model_cfg.get("enabled", False) and configured_wm_coef == 0:
            configured_wm_coef = 1.0
        self.world_model_loss_coef = float(configured_wm_coef)
        self.world_model_horizon = int(
            wm_horizon
            if wm_horizon is not None
            else self.world_model_cfg.get("horizon", getattr(policy, "world_model_horizon", 1))
        )
        self.world_model_target_source = (
            wm_target_source
            if wm_target_source is not None
            else self.world_model_cfg.get("target_source", "observations")
        )
        target_dim = (
            wm_target_dim
            if wm_target_dim is not None
            else self.world_model_cfg.get("target_dim", getattr(policy, "world_model_target_dim", None))
        )
        self.world_model_target_dim = int(target_dim) if target_dim is not None else None
        self.world_model_target_slices = (
            wm_target_slices
            if wm_target_slices is not None
            else self.world_model_cfg.get("target_slices", self.world_model_cfg.get("slices", None))
        )
        self.world_model_target_groups = (
            wm_target_groups
            if wm_target_groups is not None
            else self.world_model_cfg.get("target_groups", self.world_model_cfg.get("groups", None))
        )
        self.world_model_alternating_updates = bool(
            self.world_model_cfg.get("alternating_updates", self.world_model_cfg.get("use_replay_buffer", True))
        )
        self.world_model_joint_loss = bool(self.world_model_cfg.get("joint_loss", False))
        self.world_model_updates_per_iteration = int(self.world_model_cfg.get("updates_per_iteration", 1))
        self.world_model_mini_batch_size = int(self.world_model_cfg.get("mini_batch_size", 1024))
        self.world_model_replay_buffer_size = int(self.world_model_cfg.get("replay_buffer_size", 0))
        self.world_model_replay_device = self.world_model_cfg.get("replay_device", "cpu")
        self.world_model_replay_dtype = TrackAdapterWorldModelReplayBuffer._dtype_from_name(
            self.world_model_cfg.get("replay_dtype", "float16")
        )
        self.world_model_learning_rate = float(self.world_model_cfg.get("learning_rate", learning_rate))
        self.world_model_max_grad_norm = float(self.world_model_cfg.get("max_grad_norm", max_grad_norm))
        self.world_model_optimizer_loss_scale = float(self.world_model_cfg.get("optimizer_loss_scale", 1.0))
        self.world_model_ppo_updates_history_encoder = bool(
            self.world_model_cfg.get(
                "ppo_updates_history_encoder",
                not (self.world_model_cfg.get("enabled", False) and self.world_model_alternating_updates),
            )
        )
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        if self.freeze_discriminator:
            self.discriminator.requires_grad_(False)
        self.amp_transition = TrackAdapterRolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(discriminator.input_dim // 2, amp_replay_buffer_size, device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        self.policy = policy
        self.policy.to(self.device)
        self.use_world_model_loss = self.world_model_loss_coef > 0 and callable(
            getattr(self.policy, "compute_world_model_loss", None)
        )
        if hasattr(self.policy, "detach_history_latent_for_policy"):
            self.policy.detach_history_latent_for_policy = not self.world_model_ppo_updates_history_encoder
        if self.train_privileged_teacher:
            teacher_parameters = getattr(self.policy, "privileged_teacher_ppo_parameters", None)
            if not callable(teacher_parameters):
                raise ValueError("TrackAdapterAMPPPO privileged teacher mode requires policy.privileged_teacher_ppo_parameters().")
            self.policy_trainable_parameters = self._unique_trainable_parameters(teacher_parameters())
        else:
            ppo_parameters = getattr(self.policy, "ppo_parameters", None)
            if callable(ppo_parameters):
                self.policy_trainable_parameters = self._unique_trainable_parameters(
                    ppo_parameters(include_history_encoder=self.world_model_ppo_updates_history_encoder)
                )
            else:
                self.policy_trainable_parameters = [
                    param
                    for name, param in self.policy.named_parameters()
                    if param.requires_grad and not name.startswith("base_actor_critic.")
                    and not name.startswith("world_model_predictor.")
                    and not name.startswith("privileged_teacher_residual_actor.")
                    and (self.world_model_ppo_updates_history_encoder or not name.startswith("history_encoder."))
                ]
        if not self.policy_trainable_parameters:
            raise ValueError("TrackAdapterAMPPPO requires at least one trainable policy parameter.")
        self.privileged_teacher_actor_parameters = []
        if self.train_privileged_teacher:
            privileged_teacher_actor = getattr(self.policy, "privileged_teacher_residual_actor", None)
            if privileged_teacher_actor is not None:
                self.privileged_teacher_actor_parameters = self._unique_trainable_parameters(
                    param for param in privileged_teacher_actor.parameters() if param.requires_grad
                )
        base_parameters = list(getattr(self.policy, "base_actor_critic", nn.Module()).parameters())
        optimized_ids = {id(trainable) for trainable in self.policy_trainable_parameters}
        if any(id(param) in optimized_ids for param in base_parameters):
            raise ValueError("TrackAdapterAMPPPO optimizer cannot include frozen base actor parameters.")
        base_parameter_ids = {id(base) for base in base_parameters}
        self.world_model_trainable_parameters = []
        if self.use_world_model_loss and self.world_model_alternating_updates:
            world_model_parameters = getattr(self.policy, "world_model_parameters", None)
            if not callable(world_model_parameters):
                raise ValueError("TrackAdapterAMPPPO requires policy.world_model_parameters() for alternating WM updates.")
            self.world_model_trainable_parameters = self._unique_trainable_parameters(world_model_parameters())
            if any(id(param) in base_parameter_ids for param in self.world_model_trainable_parameters):
                raise ValueError("TrackAdapterAMPPPO world-model optimizer cannot include frozen base actor parameters.")
            if not self.world_model_trainable_parameters:
                raise ValueError("TrackAdapterAMPPPO world-model optimizer has no trainable parameters.")

        params = [{"params": self.policy_trainable_parameters, "name": "policy"}]
        self.discriminator_trainable_parameters = []
        if not self.freeze_discriminator:
            self.discriminator_trainable_parameters = [
                param for param in self.discriminator.parameters() if param.requires_grad
            ]
            if self.discriminator_trainable_parameters:
                trunk_parameters = [param for param in self.discriminator.trunk.parameters() if param.requires_grad]
                head_parameters = [
                    param for param in self.discriminator.amp_linear.parameters() if param.requires_grad
                ]
                if trunk_parameters:
                    params.append(
                        {
                            "params": trunk_parameters,
                            "weight_decay": 10e-4,
                            "name": "amp_trunk",
                        }
                    )
                if head_parameters:
                    params.append(
                        {
                            "params": head_parameters,
                            "weight_decay": 10e-2,
                            "name": "amp_head",
                        }
                    )
        self.train_discriminator = not self.freeze_discriminator and bool(self.discriminator_trainable_parameters)
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.world_model_optimizer = (
            optim.Adam(self.world_model_trainable_parameters, lr=self.world_model_learning_rate)
            if self.world_model_trainable_parameters
            else None
        )
        self.world_model_replay_buffer = None
        self.privileged_teacher_distill_replay_buffer = None

        self.storage: TrackAdapterRolloutStorage = None  # type: ignore
        self.transition = TrackAdapterRolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    @staticmethod
    def _unique_trainable_parameters(parameters):
        unique = []
        seen = set()
        for parameter in parameters:
            if parameter is None or not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id in seen:
                continue
            seen.add(parameter_id)
            unique.append(parameter)
        return unique

    def _clip_policy_gradients(self):
        if not (
            self.recovery_teacher_separate_actor_grad_clip
            and self.train_privileged_teacher
            and self.privileged_teacher_actor_parameters
        ):
            nn.utils.clip_grad_norm_(self.policy_trainable_parameters, self.max_grad_norm)
            return

        actor_ids = {id(parameter) for parameter in self.privileged_teacher_actor_parameters}
        other_parameters = [
            parameter for parameter in self.policy_trainable_parameters if id(parameter) not in actor_ids
        ]
        nn.utils.clip_grad_norm_(
            self.privileged_teacher_actor_parameters,
            self.recovery_teacher_actor_max_grad_norm,
        )
        if other_parameters:
            nn.utils.clip_grad_norm_(other_parameters, self.max_grad_norm)

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
        history_shape,
        world_model_state_shape=None,
    ):
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        self.storage = TrackAdapterRolloutStorage(
            training_type=training_type,
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=actor_obs_shape,
            privileged_obs_shape=critic_obs_shape,
            actions_shape=actions_shape,
            history_shape=history_shape,
            world_model_state_shape=world_model_state_shape,
            rnd_state_shape=rnd_state_shape,
            device=self.device,
        )

    def test_mode(self):
        self.policy.test()

    def train_mode(self):
        self.policy.train()

    def act(self, obs, critic_obs, amp_obs, history, residual_action_gate=None):
        if getattr(self.policy, "is_recurrent", False):
            self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.history = history
        self.transition.residual_action_gate = (
            residual_action_gate.detach() if isinstance(residual_action_gate, torch.Tensor) else None
        )
        if self.train_privileged_teacher:
            self.transition.actions = self.policy.privileged_teacher_act(
                obs,
                critic_obs,
                history=history,
                residual_gate=residual_action_gate,
            ).detach()
        else:
            self.transition.actions = self.policy.act(obs, history, residual_gate=residual_action_gate).detach()
        self.transition.values = self.policy.evaluate(critic_obs, history).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(
        self,
        rewards,
        dones,
        infos,
        amp_obs,
        next_observations=None,
        world_model_state=None,
        world_model_reference_state=None,
    ):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if next_observations is not None:
            self.transition.next_observations = next_observations.detach()
        if world_model_state is not None:
            self.transition.world_model_state = world_model_state.detach()
        if world_model_reference_state is not None:
            self.transition.world_model_reference_state = world_model_reference_state.detach()

        if self.rnd:
            rnd_state = infos["observations"]["rnd_state"]
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            self.transition.rewards += self.intrinsic_rewards
            self.transition.rnd_state = rnd_state.clone()

        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self.amp_storage.insert(self.amp_transition.observations, amp_obs)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs, history):
        last_values = self.policy.evaluate(last_critic_obs, history).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def _world_model_rollout_kwargs(self):
        return {
            "world_model_horizon": self.world_model_horizon,
            "world_model_target_source": self.world_model_target_source,
            "world_model_target_dim": self.world_model_target_dim,
            "world_model_target_slices": self.world_model_target_slices,
            "world_model_target_groups": self.world_model_target_groups,
        }

    def _ensure_world_model_replay_buffer(self):
        if self.world_model_replay_buffer is not None:
            return self.world_model_replay_buffer
        if self.storage is None:
            raise RuntimeError("TrackAdapterAMPPPO storage must be initialized before creating WM replay.")
        target_dim = self.world_model_target_dim
        if target_dim is None:
            target_dim = self.storage.obs_shape[-1]
        rollout_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env
        buffer_size = self.world_model_replay_buffer_size
        if buffer_size <= 0:
            buffer_size = min(max(rollout_batch_size, 32768), 65536)
        self.world_model_replay_buffer = TrackAdapterWorldModelReplayBuffer(
            history_shape=self.storage.history_shape,
            reference_shape=(target_dim,),
            target_shape=(self.world_model_horizon, target_dim),
            valid_mask_shape=(self.world_model_horizon,),
            action_sequence_shape=(self.world_model_horizon, *self.storage.actions_shape),
            buffer_size=buffer_size,
            sample_device=self.device,
            storage_device=self.world_model_replay_device,
            storage_dtype=self.world_model_replay_dtype,
        )
        return self.world_model_replay_buffer

    def _insert_world_model_rollout_in_replay(self):
        if not (self.use_world_model_loss and self.world_model_alternating_updates):
            return 0
        replay = self._ensure_world_model_replay_buffer()
        rollout_kwargs = self._world_model_rollout_kwargs()
        references = self.storage.world_model_references(
            target_source=self.world_model_target_source,
            target_dim=self.world_model_target_dim,
            target_slices=self.world_model_target_slices,
            target_groups=self.world_model_target_groups,
        )
        targets, valid_mask = self.storage.world_model_targets(
            horizon=self.world_model_horizon,
            target_source=self.world_model_target_source,
            target_dim=self.world_model_target_dim,
            target_slices=self.world_model_target_slices,
            target_groups=self.world_model_target_groups,
        )
        action_sequences = self.storage.world_model_action_sequences(
            horizon=rollout_kwargs["world_model_horizon"]
        )
        return replay.insert(self.storage.history, references, targets, valid_mask, action_sequences)

    @staticmethod
    def _add_world_model_metrics(metric_totals, metrics):
        for name, value in metrics.items():
            if name == "loss":
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    continue
                metric_value = value.detach().mean().item()
            else:
                metric_value = float(value)
            metric_totals[name] = metric_totals.get(name, 0.0) + metric_value

    def _update_world_model_from_replay(self):
        if not (
            self.use_world_model_loss
            and self.world_model_alternating_updates
            and self.world_model_optimizer is not None
            and self.world_model_updates_per_iteration > 0
        ):
            return 0, 0.0, {}
        replay = self._ensure_world_model_replay_buffer()
        if replay.num_samples <= 0:
            return 0, 0.0, {}

        loss_total = 0.0
        metric_totals = {}
        update_count = 0
        generator = replay.feed_forward_generator(
            self.world_model_updates_per_iteration,
            self.world_model_mini_batch_size,
        )
        for history, references, targets, valid_mask, action_sequences in generator:
            world_model_loss, world_model_metrics = self.policy.compute_world_model_loss(
                history=history,
                actions=action_sequences,
                targets=targets,
                reference=references,
                valid_mask=valid_mask,
            )
            optimized_loss = self.world_model_optimizer_loss_scale * world_model_loss
            self.world_model_optimizer.zero_grad()
            optimized_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters(self.world_model_trainable_parameters)
            nn.utils.clip_grad_norm_(self.world_model_trainable_parameters, self.world_model_max_grad_norm)
            self.world_model_optimizer.step()
            loss_total += world_model_loss.item()
            self._add_world_model_metrics(metric_totals, world_model_metrics)
            update_count += 1
        return update_count, loss_total, metric_totals

    def _ensure_privileged_teacher_distill_replay_buffer(self):
        if self.privileged_teacher_distill_replay_buffer is not None:
            return self.privileged_teacher_distill_replay_buffer
        if self.storage is None:
            raise RuntimeError("TrackAdapterAMPPPO storage must be initialized before creating distill replay.")
        self.privileged_teacher_distill_replay_buffer = PrivilegedTeacherDistillReplayBuffer(
            obs_shape=self.storage.obs_shape,
            critic_obs_shape=self.storage.privileged_obs_shape,
            history_shape=self.storage.history_shape,
            gate_shape=(1,),
            action_shape=self.storage.actions_shape,
            buffer_size=self.privileged_teacher_distill_replay_buffer_size,
            sample_device=self.device,
        )
        load_path = self.privileged_teacher_distill_replay_load_path
        if load_path and not self._privileged_teacher_distill_replay_loaded_from_path:
            if os.path.exists(load_path):
                loaded_count = self.privileged_teacher_distill_replay_buffer.load(load_path)
                print(f"[TrackAdapterAMPPPO] Loaded privileged teacher replay: {load_path} ({loaded_count} samples)")
            else:
                print(f"[TrackAdapterAMPPPO] Privileged teacher replay load path not found: {load_path}")
            self._privileged_teacher_distill_replay_loaded_from_path = True
        return self.privileged_teacher_distill_replay_buffer

    def _privileged_teacher_distill_mask(self, obs, critic_obs, residual_gate, action_dim=None, dones=None):
        gate = residual_gate.to(device=obs.device, dtype=obs.dtype)
        if gate.dim() == 1:
            gate = gate.unsqueeze(-1)
        gate_scalar = gate.mean(dim=-1)
        if action_dim is None:
            action_dim = getattr(self.policy, "num_actions", 0)
        score = self._recovery_score_from_batches(obs, critic_obs, int(action_dim)).to(device=obs.device, dtype=obs.dtype)
        mask = (
            (gate_scalar >= self.privileged_teacher_distill_gate_threshold)
            & (score >= self.privileged_teacher_distill_score_threshold)
        )
        if dones is not None:
            mask = mask & ~dones.reshape(-1).to(device=mask.device, dtype=torch.bool)
        return mask

    def _insert_privileged_teacher_capture_rollout_in_replay(self):
        if (
            not self.train_privileged_teacher
            or not self.privileged_teacher_distill_capture_teacher_replay
            or not self.privileged_teacher_distill_replay_enabled
        ):
            return 0
        replay = self._ensure_privileged_teacher_distill_replay_buffer()
        observations = self.storage.observations.flatten(0, 1)
        if self.storage.privileged_observations is not None:
            critic_observations = self.storage.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
        history = self.storage.history.flatten(0, 1)
        residual_gates = self.storage.residual_action_gates.flatten(0, 1)
        teacher_actions = self.storage.mu.flatten(0, 1)
        dones = self.storage.dones.flatten(0, 1).bool()
        total_inserted = 0
        chunk_size = max(1024, min(self.privileged_teacher_distill_replay_batch_size, observations.shape[0]))
        for start in range(0, observations.shape[0], chunk_size):
            end = min(start + chunk_size, observations.shape[0])
            with torch.no_grad():
                mask = self._privileged_teacher_distill_mask(
                    observations[start:end],
                    critic_observations[start:end],
                    residual_gates[start:end],
                    action_dim=teacher_actions.shape[-1],
                    dones=dones[start:end],
                )
            total_inserted += replay.insert(
                observations[start:end],
                critic_observations[start:end],
                history[start:end],
                residual_gates[start:end],
                teacher_actions[start:end],
                mask=mask,
            )
        return total_inserted

    def _insert_privileged_teacher_distill_rollout_in_replay(self):
        if (
            self.train_privileged_teacher
            or not self.privileged_teacher_distill_enabled
            or not self.privileged_teacher_distill_replay_enabled
            or self.privileged_teacher_distill_coef <= 0.0
        ):
            return 0
        target_source = getattr(self.policy, "privileged_teacher_action_target", None)
        if not callable(target_source):
            return 0
        replay = self._ensure_privileged_teacher_distill_replay_buffer()
        observations = self.storage.observations.flatten(0, 1)
        if self.storage.privileged_observations is not None:
            critic_observations = self.storage.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
        history = self.storage.history.flatten(0, 1)
        residual_gates = self.storage.residual_action_gates.flatten(0, 1)
        dones = self.storage.dones.flatten(0, 1).bool()
        total_inserted = 0
        chunk_size = max(1024, min(self.privileged_teacher_distill_replay_batch_size, observations.shape[0]))
        for start in range(0, observations.shape[0], chunk_size):
            end = min(start + chunk_size, observations.shape[0])
            obs_chunk = observations[start:end]
            critic_chunk = critic_observations[start:end]
            gate_chunk = residual_gates[start:end]
            with torch.no_grad():
                target_actions = target_source(
                    obs_chunk,
                    critic_chunk,
                    history=history[start:end],
                    residual_gate=gate_chunk,
                )
                mask = self._privileged_teacher_distill_mask(
                    obs_chunk,
                    critic_chunk,
                    gate_chunk,
                    action_dim=target_actions.shape[-1],
                    dones=dones[start:end],
                )
            total_inserted += replay.insert(
                obs_chunk,
                critic_chunk,
                history[start:end],
                gate_chunk,
                target_actions,
                mask=mask,
            )
        return total_inserted

    def _update_privileged_teacher_distill_from_replay(self, num_updates):
        if (
            self.train_privileged_teacher
            or not self.privileged_teacher_distill_enabled
            or not self.privileged_teacher_distill_replay_enabled
            or self.privileged_teacher_distill_coef <= 0.0
            or num_updates <= 0
        ):
            return 0, 0.0
        replay = self._ensure_privileged_teacher_distill_replay_buffer()
        if replay.num_samples < max(1, self.privileged_teacher_distill_replay_min_samples):
            return 0, 0.0
        update_count = 0
        loss_total = 0.0
        for _ in range(int(num_updates)):
            obs_batch, _critic_batch, history_batch, gate_batch, teacher_actions = replay.sample(
                self.privileged_teacher_distill_replay_batch_size
            )
            self.policy.act(obs_batch, history_batch, residual_gate=gate_batch)
            student_actions = self.policy.action_mean
            raw_loss = F.mse_loss(student_actions, teacher_actions)
            loss = self.privileged_teacher_distill_coef * raw_loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters(self.policy_trainable_parameters)
            self._clip_policy_gradients()
            self.optimizer.step()
            loss_total += raw_loss.item()
            update_count += 1
        return update_count, loss_total

    def warm_start_privileged_teacher_distill_replay(self):
        if self._privileged_teacher_distill_warm_start_done:
            return 0, 0.0
        self._privileged_teacher_distill_warm_start_done = True
        return self._update_privileged_teacher_distill_from_replay(
            self.privileged_teacher_distill_replay_warm_start_updates
        )

    def save_privileged_teacher_distill_replay(self, force=False):
        path = self.privileged_teacher_distill_replay_path
        replay = self.privileged_teacher_distill_replay_buffer
        if not path or replay is None or replay.num_samples <= 0:
            return None
        if force and not self.privileged_teacher_distill_replay_save_on_checkpoint:
            return None
        if not force and self.privileged_teacher_distill_replay_save_interval <= 0:
            return None
        if not force and self.privileged_teacher_distill_replay_save_interval > 0:
            if self._privileged_teacher_distill_update_calls % self.privileged_teacher_distill_replay_save_interval != 0:
                return None
        replay.save(path, max_samples=self.privileged_teacher_distill_replay_save_max_samples)
        return path

    def _trace_phase(self, phase: str):
        if not self.phase_trace_enabled:
            return
        print(f"[TrackAdapterPPOTrace] phase={phase} t={time.time():.3f}", flush=True)

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        mean_residual_penalty = None
        mean_ungated_residual_policy_penalty = None
        mean_recovery_residual_activation = None
        mean_recovery_ppo_sample_weight_max = None
        mean_recovery_teacher = None
        mean_recovery_teacher_metrics = {}
        mean_privileged_teacher_distill = None
        mean_privileged_teacher_distill_metrics = {}
        mean_adapter_diagnostics = {}
        world_model_loss_total = 0.0
        world_model_update_count = 0
        world_model_metrics_total = {}
        self._trace_phase("update_start")
        self._trace_phase("before_world_model_replay_insert")
        world_model_replay_inserted = self._insert_world_model_rollout_in_replay()
        self._trace_phase("after_world_model_replay_insert")
        self._trace_phase("before_world_model_replay_update")
        wm_replay_updates, wm_replay_loss, wm_replay_metrics = self._update_world_model_from_replay()
        self._trace_phase("after_world_model_replay_update")
        if wm_replay_updates > 0:
            world_model_loss_total += wm_replay_loss
            world_model_update_count += wm_replay_updates
            for name, value in wm_replay_metrics.items():
                world_model_metrics_total[name] = world_model_metrics_total.get(name, 0.0) + value
        self._trace_phase("before_privileged_teacher_capture_insert")
        privileged_teacher_capture_inserted = self._insert_privileged_teacher_capture_rollout_in_replay()
        self._trace_phase("after_privileged_teacher_capture_insert")
        self._trace_phase("before_privileged_teacher_distill_insert")
        privileged_distill_replay_inserted = self._insert_privileged_teacher_distill_rollout_in_replay()
        self._trace_phase("after_privileged_teacher_distill_insert")
        privileged_distill_warm_updates = 0
        privileged_distill_warm_loss = 0.0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        generator_kwargs = {}
        if self.use_world_model_loss and self.world_model_joint_loss:
            generator_kwargs = {
                "include_world_model": True,
                "world_model_horizon": self.world_model_horizon,
                "world_model_target_source": self.world_model_target_source,
                "world_model_target_dim": self.world_model_target_dim,
                "world_model_target_slices": self.world_model_target_slices,
                "world_model_target_groups": self.world_model_target_groups,
            }

        if getattr(self.policy, "is_recurrent", False):
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs, **generator_kwargs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs, **generator_kwargs
            )

        if self.train_discriminator:
            amp_policy_generator = self.amp_storage.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
            )
            amp_expert_generator = self.amp_data.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
            )
            batch_iterator = zip(generator, amp_policy_generator, amp_expert_generator)
        else:
            batch_iterator = ((sample, None, None) for sample in generator)

        self._trace_phase("before_ppo_minibatches")
        for sample, sample_amp_policy, sample_amp_expert in batch_iterator:
            base_sample = sample[:14]
            extras = sample[14:]
            next_obs_batch = None
            wm_reference_batch = None
            wm_targets_batch = None
            wm_valid_mask_batch = None
            wm_action_sequences_batch = None
            if len(extras) == 1:
                next_obs_batch = extras[0]
            elif len(extras) == 2:
                wm_targets_batch, wm_valid_mask_batch = extras
            elif len(extras) >= 3:
                if len(extras) >= 4:
                    wm_reference_batch, wm_targets_batch, wm_valid_mask_batch, wm_action_sequences_batch = extras[:4]
                else:
                    next_obs_batch, wm_targets_batch, wm_valid_mask_batch = extras[:3]

            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                history_batch,
                residual_action_gate_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                rnd_state_batch,
            ) = base_sample

            num_aug = 1
            original_batch_size = obs_batch.shape[0]
            if wm_reference_batch is None:
                wm_reference_batch = obs_batch
            wm_actions_batch = actions_batch
            wm_history_batch = history_batch
            if wm_action_sequences_batch is not None:
                wm_actions_batch = wm_action_sequences_batch

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                history_batch = history_batch.repeat(num_aug, *([1] * (history_batch.dim() - 1)))
                residual_action_gate_batch = residual_action_gate_batch.repeat(
                    num_aug, *([1] * (residual_action_gate_batch.dim() - 1))
                )
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            if self.train_privileged_teacher:
                self.policy.privileged_teacher_act(
                    obs_batch,
                    critic_obs_batch,
                    history=history_batch,
                    residual_gate=residual_action_gate_batch,
                    masks=masks_batch,
                    hidden_states=hid_states_batch[0],
                )
            else:
                self.policy.act(
                    obs_batch,
                    history_batch,
                    residual_gate=residual_action_gate_batch,
                    masks=masks_batch,
                    hidden_states=hid_states_batch[0],
                )
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                critic_obs_batch, history_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

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

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_terms = torch.max(surrogate, surrogate_clipped)
            ppo_sample_weights = self._get_recovery_ppo_sample_weights(
                residual_action_gate_batch,
                reference=surrogate_terms,
            )
            if ppo_sample_weights is not None:
                surrogate_loss = (surrogate_terms * ppo_sample_weights).mean()
                mean_recovery_ppo_sample_weight_max = (
                    (mean_recovery_ppo_sample_weight_max or 0) + ppo_sample_weights.max().item()
                )
            else:
                surrogate_loss = surrogate_terms.mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            residual_penalty = self._get_residual_penalty()
            if residual_penalty is not None:
                loss += self.residual_penalty_coef * residual_penalty
                mean_residual_penalty = (mean_residual_penalty or 0) + residual_penalty.item()
            ungated_residual_policy_penalty = self._get_ungated_residual_policy_penalty(residual_action_gate_batch)
            if ungated_residual_policy_penalty is not None:
                loss += ungated_residual_policy_penalty
                mean_ungated_residual_policy_penalty = (
                    (mean_ungated_residual_policy_penalty or 0) + ungated_residual_policy_penalty.item()
                )
            recovery_residual_activation = self._get_recovery_residual_activation_loss(residual_action_gate_batch)
            if recovery_residual_activation is not None:
                loss += recovery_residual_activation
                mean_recovery_residual_activation = (
                    (mean_recovery_residual_activation or 0) + recovery_residual_activation.item()
                )
            privileged_teacher_distill_loss, privileged_teacher_distill_metrics = (
                self._get_privileged_teacher_distill_loss(
                    obs_batch,
                    critic_obs_batch,
                    history_batch,
                    residual_action_gate_batch,
                    self.policy.action_mean,
                )
            )
            if privileged_teacher_distill_loss is not None:
                loss += privileged_teacher_distill_loss
                mean_privileged_teacher_distill = (
                    (mean_privileged_teacher_distill or 0) + privileged_teacher_distill_loss.item()
                )
                for name, value in privileged_teacher_distill_metrics.items():
                    mean_privileged_teacher_distill_metrics[name] = (
                        mean_privileged_teacher_distill_metrics.get(name, 0.0) + value
                    )
            if not self.train_privileged_teacher or self.recovery_teacher_apply_to_privileged:
                recovery_teacher_loss, recovery_teacher_metrics = self._get_recovery_teacher_loss(
                    obs_batch,
                    critic_obs_batch,
                    residual_action_gate_batch,
                )
                if recovery_teacher_loss is not None:
                    loss += recovery_teacher_loss
                    mean_recovery_teacher = (mean_recovery_teacher or 0) + recovery_teacher_loss.item()
                    for name, value in recovery_teacher_metrics.items():
                        mean_recovery_teacher_metrics[name] = mean_recovery_teacher_metrics.get(name, 0.0) + value

            if self.use_world_model_loss and self.world_model_joint_loss and wm_targets_batch is not None:
                world_model_loss, world_model_metrics = self.policy.compute_world_model_loss(
                    history=wm_history_batch,
                    actions=wm_actions_batch,
                    targets=wm_targets_batch,
                    reference=wm_reference_batch,
                    valid_mask=wm_valid_mask_batch,
                )
                loss += self.world_model_loss_coef * world_model_loss
            elif self.use_world_model_loss and self.world_model_joint_loss and next_obs_batch is not None:
                fallback_targets = next_obs_batch.unsqueeze(-2)
                fallback_valid_mask = torch.ones(
                    *fallback_targets.shape[:-1],
                    device=fallback_targets.device,
                    dtype=fallback_targets.dtype,
                )
                world_model_loss, world_model_metrics = self.policy.compute_world_model_loss(
                    history=wm_history_batch,
                    actions=wm_actions_batch,
                    targets=fallback_targets,
                    reference=wm_reference_batch,
                    valid_mask=fallback_valid_mask,
                )
                loss += self.world_model_loss_coef * world_model_loss
            else:
                world_model_loss = None
                world_model_metrics = {}

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                    history_batch = history_batch.repeat(num_aug, *([1] * (history_batch.dim() - 1)))
                    residual_action_gate_batch = residual_action_gate_batch.repeat(
                        num_aug, *([1] * (residual_action_gate_batch.dim() - 1))
                    )

                mean_actions_batch = self.policy.act_inference(
                    obs_batch.detach().clone(),
                    history_batch,
                    residual_gate=residual_action_gate_batch,
                )
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )
                symmetry_loss = torch.nn.MSELoss()(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            if self.train_discriminator:
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
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                grad_pen_loss = self.discriminator.compute_grad_pen(*sample_amp_expert, lambda_=10)
                loss += self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss

            self.optimizer.zero_grad()
            if self.rnd_optimizer:
                self.rnd_optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            self._clip_policy_gradients()
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            if self.train_discriminator and self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if self.train_discriminator:
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            if self.use_world_model_loss and self.world_model_joint_loss:
                if world_model_loss is not None:
                    world_model_loss_total += world_model_loss.item()
                    world_model_update_count += 1
                self._add_world_model_metrics(world_model_metrics_total, world_model_metrics)
            for name, value in self._collect_adapter_diagnostics().items():
                mean_adapter_diagnostics[name] = mean_adapter_diagnostics.get(name, 0.0) + value

        self._trace_phase("after_ppo_minibatches")
        num_updates = self.num_learning_epochs * self.num_mini_batches
        if not self._privileged_teacher_distill_warm_start_done:
            self._trace_phase("before_privileged_teacher_warm_start")
            privileged_distill_warm_updates, privileged_distill_warm_loss = (
                self.warm_start_privileged_teacher_distill_replay()
            )
            self._trace_phase("after_privileged_teacher_warm_start")
        self._trace_phase("before_privileged_teacher_replay_update")
        privileged_distill_replay_updates, privileged_distill_replay_loss = (
            self._update_privileged_teacher_distill_from_replay(
                self.privileged_teacher_distill_replay_updates_per_iteration
            )
        )
        self._trace_phase("after_privileged_teacher_replay_update")
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        if mean_residual_penalty is not None:
            mean_residual_penalty /= num_updates
        if mean_ungated_residual_policy_penalty is not None:
            mean_ungated_residual_policy_penalty /= num_updates
        if mean_recovery_residual_activation is not None:
            mean_recovery_residual_activation /= num_updates
        if mean_recovery_ppo_sample_weight_max is not None:
            mean_recovery_ppo_sample_weight_max /= num_updates
        if mean_recovery_teacher is not None:
            mean_recovery_teacher /= num_updates
            mean_recovery_teacher_metrics = {
                f"recovery_teacher_{name}": value / num_updates
                for name, value in mean_recovery_teacher_metrics.items()
            }
        if mean_privileged_teacher_distill is not None:
            mean_privileged_teacher_distill /= num_updates
            mean_privileged_teacher_distill_metrics = {
                f"privileged_teacher_distill_{name}": value / num_updates
                for name, value in mean_privileged_teacher_distill_metrics.items()
            }
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if world_model_update_count > 0:
            mean_world_model_loss = world_model_loss_total / world_model_update_count
            mean_world_model_metrics = {
                f"wm_{name}": value / world_model_update_count for name, value in world_model_metrics_total.items()
            }
        else:
            mean_world_model_loss = None
            mean_world_model_metrics = {}
        mean_adapter_diagnostics = {
            f"adapter_{name}": value / num_updates for name, value in mean_adapter_diagnostics.items()
        }

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
        if mean_residual_penalty is not None:
            loss_dict["residual_penalty"] = mean_residual_penalty
        if mean_ungated_residual_policy_penalty is not None:
            loss_dict["ungated_residual_policy_penalty"] = mean_ungated_residual_policy_penalty
        if mean_recovery_residual_activation is not None:
            loss_dict["recovery_residual_activation"] = mean_recovery_residual_activation
        if mean_recovery_ppo_sample_weight_max is not None:
            loss_dict["recovery_ppo_sample_weight_max"] = mean_recovery_ppo_sample_weight_max
        if mean_recovery_teacher is not None:
            loss_dict["recovery_teacher"] = mean_recovery_teacher
            loss_dict.update(mean_recovery_teacher_metrics)
        if mean_privileged_teacher_distill is not None:
            loss_dict["privileged_teacher_distill"] = mean_privileged_teacher_distill
            loss_dict.update(mean_privileged_teacher_distill_metrics)
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if mean_world_model_loss is not None:
            loss_dict["wm"] = mean_world_model_loss
            loss_dict.update(mean_world_model_metrics)
        if self.use_world_model_loss and self.world_model_alternating_updates:
            replay_size = 0 if self.world_model_replay_buffer is None else self.world_model_replay_buffer.num_samples
            loss_dict["wm_replay_inserted"] = float(world_model_replay_inserted)
            loss_dict["wm_replay_size"] = float(replay_size)
            loss_dict["wm_updates"] = float(world_model_update_count)
        if (
            self.privileged_teacher_distill_replay_enabled
            and self.privileged_teacher_distill_replay_buffer is not None
        ):
            loss_dict["privileged_teacher_capture_replay_inserted"] = float(privileged_teacher_capture_inserted)
            loss_dict["privileged_teacher_distill_replay_inserted"] = float(privileged_distill_replay_inserted)
            loss_dict["privileged_teacher_distill_replay_size"] = float(
                self.privileged_teacher_distill_replay_buffer.num_samples
            )
            loss_dict["privileged_teacher_distill_warm_updates"] = float(privileged_distill_warm_updates)
            loss_dict["privileged_teacher_distill_replay_updates"] = float(privileged_distill_replay_updates)
            if privileged_distill_warm_updates > 0:
                loss_dict["privileged_teacher_distill_warm_bc"] = (
                    privileged_distill_warm_loss / privileged_distill_warm_updates
                )
            if privileged_distill_replay_updates > 0:
                loss_dict["privileged_teacher_distill_replay_bc"] = (
                    privileged_distill_replay_loss / privileged_distill_replay_updates
                )
        loss_dict.update(mean_adapter_diagnostics)
        self._privileged_teacher_distill_update_calls += 1
        self._trace_phase("before_privileged_teacher_replay_save")
        saved_replay_path = self.save_privileged_teacher_distill_replay(force=False)
        self._trace_phase("after_privileged_teacher_replay_save")
        if saved_replay_path:
            loss_dict["privileged_teacher_distill_replay_saved"] = 1.0

        self._trace_phase("update_end")
        return loss_dict

    def _get_residual_penalty(self):
        if self.residual_penalty_coef == 0:
            return None
        penalty_source = getattr(self.policy, "get_residual_penalty", None)
        if callable(penalty_source):
            penalty = penalty_source()
        else:
            penalty = getattr(self.policy, "residual_penalty", None)
        if penalty is None:
            return None
        if not isinstance(penalty, torch.Tensor):
            penalty = torch.as_tensor(penalty, device=self.device, dtype=torch.float32)
        return penalty.mean()

    def _get_ungated_residual_policy_penalty(self, residual_action_gate_batch):
        weight = self.ungated_residual_policy_loss_weight
        if weight <= 0.0:
            return None
        ungated = getattr(self.policy, "latest_ungated_scaled_residual_action_mean_for_loss", None)
        if not isinstance(ungated, torch.Tensor):
            return None
        gate = residual_action_gate_batch.to(device=ungated.device, dtype=ungated.dtype)
        if gate.dim() == 1:
            gate = gate.unsqueeze(-1)
        gate_scalar = gate.mean(dim=-1)
        ungated_l2 = torch.sum(torch.square(ungated), dim=-1)
        if gate_scalar.shape != ungated_l2.shape:
            gate_scalar = gate_scalar.reshape(ungated_l2.shape)
        stable_mask = (gate_scalar <= self.ungated_residual_policy_gate_threshold).to(dtype=ungated_l2.dtype)
        stable_count = stable_mask.sum()
        if stable_count <= 0:
            return None
        penalty = (ungated_l2 * stable_mask).sum() / stable_count.clamp_min(1.0)
        return weight * penalty

    def _get_recovery_residual_activation_loss(self, residual_action_gate_batch):
        weight = self.recovery_residual_activation_weight
        if weight <= 0.0:
            return None
        ungated = getattr(self.policy, "latest_ungated_scaled_residual_action_mean_for_loss", None)
        if not isinstance(ungated, torch.Tensor):
            return None
        gate = residual_action_gate_batch.to(device=ungated.device, dtype=ungated.dtype)
        if gate.dim() == 1:
            gate = gate.unsqueeze(-1)
        gate_scalar = gate.mean(dim=-1)
        residual_norm = torch.sqrt(torch.sum(torch.square(ungated), dim=-1) + 1.0e-8)
        if gate_scalar.shape != residual_norm.shape:
            gate_scalar = gate_scalar.reshape(residual_norm.shape)
        recovery_mask = (gate_scalar >= self.recovery_residual_activation_gate_threshold).to(dtype=residual_norm.dtype)
        recovery_count = recovery_mask.sum()
        if recovery_count <= 0:
            return None
        target = max(self.recovery_residual_activation_target_norm, 0.0)
        shortfall = torch.relu(target - residual_norm)
        activation_loss = (torch.square(shortfall) * recovery_mask).sum() / recovery_count.clamp_min(1.0)
        loss = weight * activation_loss
        max_norm = max(self.recovery_residual_activation_max_norm, 0.0)
        overgrowth_weight = max(self.recovery_residual_activation_overgrowth_weight, 0.0)
        if max_norm > 0.0 and overgrowth_weight > 0.0:
            upper = max(max_norm, target)
            overgrowth = torch.relu(residual_norm - upper)
            overgrowth_loss = (torch.square(overgrowth) * recovery_mask).sum() / recovery_count.clamp_min(1.0)
            loss = loss + overgrowth_weight * overgrowth_loss
        return loss

    def _get_recovery_ppo_sample_weights(self, residual_action_gate_batch, reference):
        weight = self.recovery_ppo_sample_weight
        if weight <= 0.0:
            return None
        if not isinstance(residual_action_gate_batch, torch.Tensor):
            return None
        gate = residual_action_gate_batch.to(device=reference.device, dtype=reference.dtype)
        if gate.dim() == 1:
            gate = gate.unsqueeze(-1)
        gate_scalar = gate.mean(dim=-1)
        if gate_scalar.shape != reference.shape:
            gate_scalar = gate_scalar.reshape(reference.shape)
        threshold = min(max(self.recovery_ppo_sample_gate_threshold, 0.0), 0.999)
        power = max(self.recovery_ppo_sample_power, 0.0)
        active = torch.clamp((gate_scalar - threshold) / max(1.0 - threshold, 1.0e-6), 0.0, 1.0)
        if power != 1.0:
            active = torch.pow(active, power)
        sample_weights = 1.0 + weight * active
        return sample_weights / sample_weights.mean().clamp_min(1.0e-6)

    def _get_recovery_teacher_loss(self, obs_batch, critic_obs_batch, residual_action_gate_batch):
        if not self.recovery_teacher_enabled or self.recovery_teacher_coef <= 0.0:
            return None, {}
        ungated = getattr(self.policy, "latest_ungated_scaled_residual_action_mean_for_loss", None)
        if not isinstance(ungated, torch.Tensor) or ungated.numel() == 0:
            return None, {}

        target, action_weights, teacher_metrics = self._recovery_teacher_target(
            obs_batch,
            critic_obs_batch,
            ungated,
        )
        if target is None or action_weights is None:
            return None, {}

        gate = residual_action_gate_batch.to(device=ungated.device, dtype=ungated.dtype)
        if gate.dim() == 1:
            gate = gate.unsqueeze(-1)
        gate_scalar = gate.mean(dim=-1)
        if gate_scalar.shape[0] != ungated.shape[0]:
            gate_scalar = gate_scalar.reshape(ungated.shape[0])

        score = teacher_metrics["score_tensor"]
        teacher_mask = (
            (gate_scalar >= self.recovery_teacher_gate_threshold)
            & (score >= self.recovery_teacher_score_threshold)
        ).to(dtype=ungated.dtype)
        mask_count = teacher_mask.sum()

        element_loss = F.smooth_l1_loss(ungated, target.detach(), reduction="none")
        weights = action_weights.to(device=ungated.device, dtype=ungated.dtype).view(1, -1)
        per_sample_loss = (element_loss * weights).sum(dim=-1) / weights.sum().clamp_min(1.0e-6)
        if mask_count > 0:
            raw_loss = (per_sample_loss * teacher_mask).sum() / mask_count.clamp_min(1.0)
        else:
            raw_loss = ungated.sum() * 0.0

        metrics = {
            "raw": raw_loss.detach().item(),
            "mask_fraction": teacher_mask.detach().mean().item(),
            "gate_mean": gate_scalar.detach().mean().item(),
            "score": score.detach().mean().item(),
            "target_l2": teacher_metrics["target_l2"],
            "capture_x": teacher_metrics["capture_x"],
            "capture_y": teacher_metrics["capture_y"],
            "force_norm": teacher_metrics.get("force_norm", 0.0),
            "push_delta_norm": teacher_metrics.get("push_delta_norm", 0.0),
        }
        return self.recovery_teacher_coef * raw_loss, metrics

    def _get_privileged_teacher_distill_loss(
        self,
        obs_batch,
        critic_obs_batch,
        history_batch,
        residual_action_gate_batch,
        student_action_mean,
    ):
        if (
            self.train_privileged_teacher
            or not self.privileged_teacher_distill_enabled
            or self.privileged_teacher_distill_coef <= 0.0
        ):
            return None, {}
        target_source = getattr(self.policy, "privileged_teacher_action_target", None)
        if not callable(target_source):
            return None, {}
        if not isinstance(student_action_mean, torch.Tensor) or student_action_mean.numel() == 0:
            return None, {}

        with torch.no_grad():
            target_action_mean = target_source(
                obs_batch,
                critic_obs_batch,
                history=history_batch,
                residual_gate=residual_action_gate_batch,
            )
            score = self._recovery_score_from_batches(obs_batch, critic_obs_batch, student_action_mean.shape[-1])

        gate = residual_action_gate_batch.to(device=student_action_mean.device, dtype=student_action_mean.dtype)
        if gate.dim() == 1:
            gate = gate.unsqueeze(-1)
        gate_scalar = gate.mean(dim=-1)
        if gate_scalar.shape[0] != student_action_mean.shape[0]:
            gate_scalar = gate_scalar.reshape(student_action_mean.shape[0])
        mask = (
            (gate_scalar >= self.privileged_teacher_distill_gate_threshold)
            & (score >= self.privileged_teacher_distill_score_threshold)
        ).to(dtype=student_action_mean.dtype)
        mask_count = mask.sum()

        element_loss = F.smooth_l1_loss(student_action_mean, target_action_mean.detach(), reduction="none")
        per_sample_loss = element_loss.mean(dim=-1)
        if mask_count > 0:
            raw_loss = (per_sample_loss * mask).sum() / mask_count.clamp_min(1.0)
        else:
            raw_loss = student_action_mean.sum() * 0.0
        metrics = {
            "raw": raw_loss.detach().item(),
            "mask_fraction": mask.detach().mean().item(),
            "gate_mean": gate_scalar.detach().mean().item(),
            "score": score.detach().mean().item(),
            "target_l2": torch.sqrt(torch.sum(torch.square(target_action_mean.detach()), dim=-1) + 1.0e-8)
            .mean()
            .item(),
        }
        return self.privileged_teacher_distill_coef * raw_loss, metrics

    def _recovery_score_from_batches(self, obs_batch, critic_obs_batch, action_dim):
        raw_obs_batch = self._inverse_normalized_batch(
            obs_batch.to(device=critic_obs_batch.device, dtype=critic_obs_batch.dtype),
            "obs_normalizer",
        )
        raw_critic_obs_batch = self._inverse_normalized_batch(critic_obs_batch, "privileged_obs_normalizer")
        base_lin_vel, base_ang_vel, projected_gravity, commands, _ = self._recovery_teacher_layout_tensors(
            raw_obs_batch,
            raw_critic_obs_batch,
            action_dim,
        )
        vel_error = base_lin_vel[:, :2] - commands[:, :2]
        return (
            torch.linalg.norm(vel_error, dim=-1)
            + 0.75 * torch.linalg.norm(projected_gravity[:, :2], dim=-1)
            + 0.20 * torch.linalg.norm(base_ang_vel[:, :2], dim=-1)
        ).to(device=critic_obs_batch.device, dtype=critic_obs_batch.dtype)

    def _recovery_teacher_target(self, obs_batch, critic_obs_batch, reference):
        action_dim = reference.shape[-1]
        if action_dim <= 0:
            return None, None, {}
        obs_batch = obs_batch.to(device=reference.device, dtype=reference.dtype)
        critic_obs_batch = critic_obs_batch.to(device=reference.device, dtype=reference.dtype)
        raw_obs_batch = self._inverse_normalized_batch(obs_batch, "obs_normalizer")
        raw_critic_obs_batch = self._inverse_normalized_batch(critic_obs_batch, "privileged_obs_normalizer")

        base_lin_vel, base_ang_vel, projected_gravity, commands, previous_action = self._recovery_teacher_layout_tensors(
            raw_obs_batch,
            raw_critic_obs_batch,
            action_dim,
        )
        force_xy, force_norm, push_delta_xy, push_delta_norm = self._recovery_teacher_privileged_push_tensors(
            raw_critic_obs_batch
        )
        vel_error = base_lin_vel[:, :2] - commands[:, :2]
        capture_x = torch.tanh(
            self.recovery_teacher_velocity_gain * vel_error[:, 0]
            + self.recovery_teacher_tilt_gain * projected_gravity[:, 0]
            + self.recovery_teacher_ang_vel_gain * base_ang_vel[:, 1]
            + self.recovery_teacher_force_gain * force_xy[:, 0]
            + self.recovery_teacher_push_delta_gain * push_delta_xy[:, 0]
        )
        capture_y = torch.tanh(
            self.recovery_teacher_velocity_gain * vel_error[:, 1]
            + self.recovery_teacher_tilt_gain * projected_gravity[:, 1]
            - self.recovery_teacher_ang_vel_gain * base_ang_vel[:, 0]
            + self.recovery_teacher_force_gain * force_xy[:, 1]
            + self.recovery_teacher_push_delta_gain * push_delta_xy[:, 1]
        )
        score = (
            torch.linalg.norm(vel_error, dim=-1)
            + 0.75 * torch.linalg.norm(projected_gravity[:, :2], dim=-1)
            + 0.20 * torch.linalg.norm(base_ang_vel[:, :2], dim=-1)
            + max(self.recovery_teacher_force_gain, 0.0) * force_norm.squeeze(-1)
            + max(self.recovery_teacher_push_delta_gain, 0.0) * push_delta_norm.squeeze(-1)
        )

        target = torch.zeros_like(reference)
        action_weights = reference.new_full((action_dim,), max(self.recovery_teacher_non_leg_weight, 0.0))
        idx = self._recovery_teacher_action_indices(action_dim)
        leg_indices = [value for value in idx.values() if value is not None]
        if not leg_indices:
            return None, None, {}
        action_weights[leg_indices] = 1.0

        max_delta = max(self.recovery_teacher_max_delta, 0.0)
        forward_delta = torch.clamp(0.95 * max_delta * capture_x, min=-max_delta, max=max_delta)
        lateral_delta = torch.clamp(0.90 * max_delta * capture_y, min=-max_delta, max=max_delta)
        capture_norm = torch.sqrt(torch.square(capture_x) + torch.square(capture_y)).clamp(0.0, 1.0)
        knee_delta = torch.clamp(0.55 * max_delta * capture_norm, min=0.0, max=max_delta)
        ankle_pitch_delta = torch.clamp(-0.45 * max_delta * capture_x, min=-max_delta, max=max_delta)
        ankle_roll_delta = torch.clamp(-0.40 * max_delta * capture_y, min=-max_delta, max=max_delta)

        left_hip_pitch = idx.get("left_hip_pitch")
        right_hip_pitch = idx.get("right_hip_pitch")
        if left_hip_pitch is not None and right_hip_pitch is not None:
            phase_left = previous_action[:, left_hip_pitch] <= previous_action[:, right_hip_pitch]
        else:
            phase_left = capture_y >= 0.0
        lateral_dominant = torch.abs(capture_y) > 0.65 * torch.abs(capture_x)
        left_swing = torch.where(lateral_dominant, capture_y >= 0.0, phase_left)
        left_mask = left_swing.to(dtype=reference.dtype)
        right_mask = 1.0 - left_mask

        self._add_teacher_delta(target, idx.get("left_hip_pitch"), left_mask * forward_delta)
        self._add_teacher_delta(target, idx.get("left_hip_roll"), left_mask * lateral_delta)
        self._add_teacher_delta(target, idx.get("left_knee_pitch"), left_mask * knee_delta)
        self._add_teacher_delta(target, idx.get("left_ankle_pitch"), left_mask * ankle_pitch_delta)
        self._add_teacher_delta(target, idx.get("left_ankle_roll"), left_mask * ankle_roll_delta)

        self._add_teacher_delta(target, idx.get("right_hip_pitch"), right_mask * forward_delta)
        self._add_teacher_delta(target, idx.get("right_hip_roll"), right_mask * lateral_delta)
        self._add_teacher_delta(target, idx.get("right_knee_pitch"), right_mask * knee_delta)
        self._add_teacher_delta(target, idx.get("right_ankle_pitch"), right_mask * ankle_pitch_delta)
        self._add_teacher_delta(target, idx.get("right_ankle_roll"), right_mask * ankle_roll_delta)

        support_knee_delta = 0.20 * knee_delta
        self._add_teacher_delta(target, idx.get("left_knee_pitch"), right_mask * support_knee_delta)
        self._add_teacher_delta(target, idx.get("right_knee_pitch"), left_mask * support_knee_delta)
        target = target.clamp(min=-max_delta, max=max_delta)

        metrics = {
            "score_tensor": score,
            "target_l2": torch.sqrt(torch.sum(torch.square(target.detach()), dim=-1) + 1.0e-8).mean().item(),
            "capture_x": capture_x.detach().mean().item(),
            "capture_y": capture_y.detach().mean().item(),
            "force_norm": force_norm.detach().mean().item(),
            "push_delta_norm": push_delta_norm.detach().mean().item(),
        }
        return target, action_weights, metrics

    def _recovery_teacher_layout_tensors(self, obs_batch, critic_obs_batch, action_dim):
        batch_size = obs_batch.shape[0]
        zeros3 = obs_batch.new_zeros(batch_size, 3)
        base_lin_vel = zeros3
        base_ang_vel = obs_batch[:, 0:3] if obs_batch.shape[-1] >= 3 else zeros3
        projected_gravity = obs_batch[:, 3:6] if obs_batch.shape[-1] >= 6 else zeros3
        commands = obs_batch[:, 6:9] if obs_batch.shape[-1] >= 9 else zeros3

        critic_dim = critic_obs_batch.shape[-1]
        if critic_dim >= 12 + action_dim:
            base_lin_vel = critic_obs_batch[:, 0:3]
            base_ang_vel = critic_obs_batch[:, 3:6]
            projected_gravity = critic_obs_batch[:, 6:9]
            commands = critic_obs_batch[:, 9:12]

        previous_action = obs_batch.new_zeros(batch_size, action_dim)
        actor_tail_dim = obs_batch.shape[-1] - 9 - action_dim
        if actor_tail_dim >= 0 and actor_tail_dim % 2 == 0:
            previous_start = 9 + actor_tail_dim
            previous_stop = previous_start + action_dim
            if previous_stop <= obs_batch.shape[-1]:
                previous_action = obs_batch[:, previous_start:previous_stop]
        return base_lin_vel, base_ang_vel, projected_gravity, commands, previous_action

    def _recovery_teacher_privileged_push_tensors(self, critic_obs_batch):
        batch_size = critic_obs_batch.shape[0]
        force_xy = critic_obs_batch.new_zeros(batch_size, 2)
        force_norm = critic_obs_batch.new_zeros(batch_size, 1)
        push_delta_xy = critic_obs_batch.new_zeros(batch_size, 2)
        push_delta_norm = critic_obs_batch.new_zeros(batch_size, 1)
        # track_adapter_recovery_privileged_state appends 18 features:
        # force_xy[2], force_norm, force remaining/elapsed/duration, 9 command
        # force metrics, foot_contact[2], trunk_contact.
        if critic_obs_batch.shape[-1] < 18:
            return force_xy, force_norm, push_delta_xy, push_delta_norm
        privileged = critic_obs_batch[:, -18:]
        force_xy = privileged[:, 0:2]
        force_norm = privileged[:, 2:3].clamp_min(0.0)
        push_delta_xy = privileged[:, 12:14]
        push_delta_norm = privileged[:, 14:15].clamp_min(0.0)
        return force_xy, force_norm, push_delta_xy, push_delta_norm

    def _inverse_normalized_batch(self, batch, normalizer_attr):
        normalizer = getattr(self, normalizer_attr, None)
        inverse = getattr(normalizer, "inverse", None)
        if not callable(inverse):
            return batch
        with torch.no_grad():
            return inverse(batch).to(device=batch.device, dtype=batch.dtype)

    def _recovery_teacher_action_indices(self, action_dim):
        defaults = {
            "left_hip_pitch": 3,
            "right_hip_pitch": 4,
            "left_hip_roll": 8,
            "right_hip_roll": 9,
            "left_hip_yaw": 12,
            "right_hip_yaw": 13,
            "left_knee_pitch": 16,
            "right_knee_pitch": 17,
            "left_ankle_pitch": 18,
            "right_ankle_pitch": 19,
            "left_ankle_roll": 20,
            "right_ankle_roll": 21,
        }
        resolved = {}
        for name, default in defaults.items():
            index = int(self.recovery_teacher_joint_indices.get(name, default))
            resolved[name] = index if 0 <= index < action_dim else None
        return resolved

    @staticmethod
    def _add_teacher_delta(target, index, delta):
        if index is None:
            return
        target[:, index] = target[:, index] + delta

    def _collect_adapter_diagnostics(self):
        diagnostics_source = getattr(self.policy, "get_adapter_diagnostics", None)
        if callable(diagnostics_source):
            diagnostics = diagnostics_source()
        else:
            diagnostics = getattr(self.policy, "adapter_diagnostics", None)
        if not isinstance(diagnostics, dict):
            diagnostics = {}

        scalars = {}
        for name, value in diagnostics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    continue
                scalars[name] = value.detach().mean().item()
            elif isinstance(value, (float, int)):
                scalars[name] = float(value)
        for name, attr in (
            ("base_action_l2", "latest_base_action_mean"),
            ("residual_action_l2", "latest_residual_action_mean"),
            ("scaled_residual_action_l2", "latest_scaled_residual_action_mean"),
        ):
            value = getattr(self.policy, attr, None)
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                squared = torch.square(value.detach())
                if squared.dim() > 1:
                    squared = torch.sum(squared, dim=-1)
                scalars.setdefault(name, squared.mean().item())
        return scalars

    def broadcast_parameters(self):
        model_params = [self.policy.state_dict()]
        if self.train_discriminator:
            model_params.append(self.discriminator.state_dict())
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        offset = 1
        if self.train_discriminator:
            self.discriminator.load_state_dict(model_params[offset])
            offset += 1
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[offset])

    def reduce_parameters(self, parameters=None):
        params = list(self.policy_trainable_parameters if parameters is None else parameters)
        if parameters is None and self.train_discriminator:
            params += self.discriminator_trainable_parameters
        if parameters is None and self.rnd:
            params += [param for param in self.rnd.parameters() if param.requires_grad]
        grads = [param.grad.view(-1) for param in params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

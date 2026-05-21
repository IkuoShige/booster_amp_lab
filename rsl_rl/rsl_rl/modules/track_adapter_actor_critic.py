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

from __future__ import annotations

import copy
from itertools import chain

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_mlp(input_dim, output_dim, hidden_dims, activation):
    layers = []
    dims = [input_dim, *hidden_dims]
    for index in range(len(dims) - 1):
        layers.append(nn.Linear(dims[index], dims[index + 1]))
        layers.append(copy.deepcopy(activation))
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


def _init_last_linear(module, scale: float = 0.0):
    for layer in reversed(module):
        if isinstance(layer, nn.Linear):
            if scale == 0.0:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
            else:
                nn.init.orthogonal_(layer.weight, gain=scale)
                nn.init.zeros_(layer.bias)
            return


def _linear_modules(module):
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


_ACTOR_OBS_TARGET_GROUP_ALIASES = {
    "all": "all",
    "actor": "all",
    "actor_obs": "all",
    "observation": "all",
    "observations": "all",
    "base_ang_vel": "base_ang_vel",
    "base_angular_velocity": "base_ang_vel",
    "angular_velocity": "base_ang_vel",
    "projected_gravity": "projected_gravity",
    "gravity": "projected_gravity",
    "command": "commands",
    "commands": "commands",
    "velocity_command": "commands",
    "velocity_commands": "commands",
    "joint_pos": "joint_pos",
    "joint_position": "joint_pos",
    "joint_positions": "joint_pos",
    "dof_pos": "joint_pos",
    "dof_position": "joint_pos",
    "joint_vel": "joint_vel",
    "joint_velocity": "joint_vel",
    "joint_velocities": "joint_vel",
    "dof_vel": "joint_vel",
    "dof_velocity": "joint_vel",
    "action": "previous_action",
    "actions": "previous_action",
    "last_action": "previous_action",
    "last_actions": "previous_action",
    "prev_action": "previous_action",
    "prev_actions": "previous_action",
    "previous_action": "previous_action",
    "previous_actions": "previous_action",
}


def _canonical_actor_obs_target_group(group):
    key = str(group).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _ACTOR_OBS_TARGET_GROUP_ALIASES:
        raise ValueError(
            f"Unknown world model target group: {group}. "
            f"Expected one of: {sorted(_ACTOR_OBS_TARGET_GROUP_ALIASES)}"
        )
    return _ACTOR_OBS_TARGET_GROUP_ALIASES[key]


def _iter_target_groups(target_groups):
    if target_groups is None:
        return []
    if isinstance(target_groups, str):
        return [group.strip() for group in target_groups.replace("+", ",").split(",") if group.strip()]
    groups = []
    for group in target_groups:
        groups.extend(_iter_target_groups(group))
    return groups


def _actor_obs_group_indices(group, feature_dim, num_actions):
    canonical_group = _canonical_actor_obs_target_group(group)
    if canonical_group == "all":
        return list(range(feature_dim))

    fixed_slices = {
        "base_ang_vel": (0, 3),
        "projected_gravity": (3, 6),
        "commands": (6, 9),
    }
    if canonical_group in fixed_slices:
        start, stop = fixed_slices[canonical_group]
    else:
        tail_dim = feature_dim - 9 - num_actions
        if tail_dim < 0 or tail_dim % 2 != 0:
            raise ValueError(
                "Actor observation layout cannot expose joint/action target groups: "
                f"feature_dim={feature_dim}, num_actions={num_actions}. Expected layout "
                "[base_ang_vel(3), projected_gravity(3), commands(3), joint_pos(N), joint_vel(N), previous_action(A)]."
            )
        joint_dim = tail_dim // 2
        if canonical_group == "joint_pos":
            start, stop = 9, 9 + joint_dim
        elif canonical_group == "joint_vel":
            start, stop = 9 + joint_dim, 9 + 2 * joint_dim
        elif canonical_group == "previous_action":
            start, stop = 9 + 2 * joint_dim, 9 + 2 * joint_dim + num_actions
        else:
            raise ValueError(f"Unhandled world model target group: {group}")

    if start < 0 or stop > feature_dim or stop < start:
        raise ValueError(
            f"World model target group {group} resolves to invalid slice [{start}:{stop}] "
            f"for feature_dim={feature_dim}."
        )
    return list(range(start, stop))


def _is_slice_bounds(value):
    return (
        isinstance(value, (tuple, list))
        and 2 <= len(value) <= 3
        and all(bound is None or isinstance(bound, int) for bound in value)
    )


def _iter_target_slice_specs(target_slices):
    if target_slices is None:
        return []
    if isinstance(target_slices, (int, slice, str, dict)) or _is_slice_bounds(target_slices):
        return [target_slices]
    return list(target_slices)


def _target_slice_indices(slice_spec, feature_dim, num_actions):
    if isinstance(slice_spec, str):
        return _actor_obs_group_indices(slice_spec, feature_dim, num_actions)
    if isinstance(slice_spec, int):
        index = slice_spec + feature_dim if slice_spec < 0 else slice_spec
        if index < 0 or index >= feature_dim:
            raise ValueError(f"World model target index {slice_spec} is out of range for feature_dim={feature_dim}.")
        return [index]
    if isinstance(slice_spec, slice):
        return list(range(*slice_spec.indices(feature_dim)))
    if isinstance(slice_spec, dict):
        if "group" in slice_spec:
            return _actor_obs_group_indices(slice_spec["group"], feature_dim, num_actions)
        if "indices" in slice_spec:
            indices = []
            for index in slice_spec["indices"]:
                indices.extend(_target_slice_indices(index, feature_dim, num_actions))
            return indices
        start = slice_spec.get("start")
        stop = slice_spec.get("stop", slice_spec.get("end"))
        step = slice_spec.get("step")
        return _target_slice_indices(slice(start, stop, step), feature_dim, num_actions)
    if _is_slice_bounds(slice_spec):
        start = slice_spec[0]
        stop = slice_spec[1]
        step = slice_spec[2] if len(slice_spec) == 3 else None
        return _target_slice_indices(slice(start, stop, step), feature_dim, num_actions)
    if isinstance(slice_spec, (tuple, list)):
        indices = []
        for index in slice_spec:
            indices.extend(_target_slice_indices(index, feature_dim, num_actions))
        return indices
    raise ValueError(f"Unsupported world model target slice spec: {slice_spec}")


def _target_feature_indices(feature_dim, num_actions, target_slices=None, target_groups=None):
    indices = []
    for group in _iter_target_groups(target_groups):
        indices.extend(_actor_obs_group_indices(group, feature_dim, num_actions))
    for slice_spec in _iter_target_slice_specs(target_slices):
        indices.extend(_target_slice_indices(slice_spec, feature_dim, num_actions))
    return indices


def _resolve_world_model_target_dim(feature_dim, num_actions, target_dim, target_slices=None, target_groups=None):
    if target_dim is not None:
        target_dim = int(target_dim)
        if target_dim > 0:
            return target_dim
        if target_dim < 0:
            raise ValueError(f"world_model_target_dim must be positive, got {target_dim}.")
    indices = _target_feature_indices(feature_dim, num_actions, target_slices, target_groups)
    if indices:
        return len(indices)
    return feature_dim


class TrackAdapterActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        encoder_hidden_dims=[256, 128],
        residual_hidden_dims=None,
        history_encoder_hidden_dims=None,
        adapter_hidden_dims=None,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        history_length=1,
        history_dim=None,
        history_step_dim=None,
        history_encoder_type="mlp",
        history_latent_dim=64,
        history_embedding_dim=None,
        world_model_hidden_dims=None,
        world_model_target_dim=None,
        world_model_target_groups=None,
        world_model_target_slices=None,
        world_model_horizon=1,
        world_model_output_init_scale=0.01,
        wm_hidden_dims=None,
        wm_target_dim=None,
        wm_target_groups=None,
        wm_target_slices=None,
        wm_horizon=None,
        adapter_mode="action_residual",
        layerwise_adapter_hidden_dims=None,
        layerwise_adapter_output_init_scale=None,
        residual_scale=1.0,
        residual_output_init_scale=0.0,
        privileged_teacher_enabled=False,
        privileged_teacher_hidden_dims=None,
        privileged_teacher_residual_scale=0.30,
        privileged_teacher_output_init_scale=0.0,
        privileged_teacher_use_adapter_base=False,
        freeze_base=True,
        base_checkpoint_path=None,
        base_actor_hidden_dims=None,
        base_critic_hidden_dims=None,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "TrackAdapterActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        if not freeze_base:
            raise ValueError("TrackAdapterActorCritic requires freeze_base=True; the AMP base actor must stay frozen.")

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.history_length = max(1, int(history_length))
        self.history_dim = history_dim if history_dim is not None else num_actor_obs
        self.history_step_dim = int(history_step_dim) if history_step_dim is not None else None
        if self.history_step_dim is None:
            if self.history_dim % self.history_length == 0:
                self.history_step_dim = self.history_dim // self.history_length
            else:
                self.history_step_dim = self.history_dim
        self.history_encoder_type = str(history_encoder_type).lower()
        self.history_latent_dim = history_latent_dim
        self.residual_scale = float(residual_scale)
        self.privileged_teacher_enabled = bool(privileged_teacher_enabled)
        self.privileged_teacher_residual_scale = float(privileged_teacher_residual_scale)
        self.privileged_teacher_use_adapter_base = bool(privileged_teacher_use_adapter_base)
        self.noise_std_type = noise_std_type
        adapter_mode = str(adapter_mode).lower().replace("-", "_")
        if adapter_mode == "layer_wise":
            adapter_mode = "layerwise"
        if adapter_mode not in ("action_residual", "layerwise"):
            raise ValueError(
                f"Unknown Track Adapter mode: {adapter_mode}. Expected 'action_residual' or 'layerwise'."
            )
        self.adapter_mode = adapter_mode

        base_actor_hidden_dims = base_actor_hidden_dims or actor_hidden_dims
        base_critic_hidden_dims = base_critic_hidden_dims or critic_hidden_dims
        if history_encoder_hidden_dims is not None:
            encoder_hidden_dims = history_encoder_hidden_dims
        if adapter_hidden_dims is not None:
            residual_hidden_dims = adapter_hidden_dims
        if history_embedding_dim is not None:
            self.history_latent_dim = history_embedding_dim
            history_latent_dim = history_embedding_dim
        if wm_hidden_dims is not None:
            world_model_hidden_dims = wm_hidden_dims
        if wm_target_dim is not None:
            world_model_target_dim = wm_target_dim
        if wm_target_groups is not None:
            world_model_target_groups = wm_target_groups
        if wm_target_slices is not None:
            world_model_target_slices = wm_target_slices
        if wm_horizon is not None:
            world_model_horizon = wm_horizon
        residual_hidden_dims = residual_hidden_dims or actor_hidden_dims
        layerwise_adapter_hidden_dims = layerwise_adapter_hidden_dims or residual_hidden_dims
        privileged_teacher_hidden_dims = privileged_teacher_hidden_dims or residual_hidden_dims
        if layerwise_adapter_output_init_scale is None:
            layerwise_adapter_output_init_scale = residual_output_init_scale
        if world_model_hidden_dims is None:
            world_model_hidden_dims = [128]
        activation_module = resolve_nn_activation(activation)

        self.base_actor_critic = ActorCritic(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            actor_hidden_dims=base_actor_hidden_dims,
            critic_hidden_dims=base_critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )
        if base_checkpoint_path is not None:
            self._load_base_actor_from_checkpoint(base_checkpoint_path)
        self._freeze_base()

        if self.history_encoder_type == "gru":
            self.history_encoder = nn.GRU(
                input_size=self.history_step_dim,
                hidden_size=history_latent_dim,
                batch_first=True,
            )
        elif self.history_encoder_type == "mlp":
            self.history_encoder = _build_mlp(
                self.history_dim,
                history_latent_dim,
                encoder_hidden_dims,
                activation_module,
            )
        else:
            raise ValueError(
                f"Unknown history_encoder_type: {history_encoder_type}. Expected 'mlp' or 'gru'."
            )
        residual_input_dim = num_actor_obs + history_latent_dim + num_actions
        self.residual_actor = _build_mlp(
            residual_input_dim,
            num_actions,
            residual_hidden_dims,
            activation_module,
        )
        _init_last_linear(self.residual_actor, residual_output_init_scale)
        self.layer_adapters = self._build_layerwise_adapters(
            layerwise_adapter_hidden_dims,
            activation_module,
            float(layerwise_adapter_output_init_scale),
        )
        teacher_input_dim = num_actor_obs + num_critic_obs + num_actions
        self.privileged_teacher_residual_actor = None
        if self.privileged_teacher_enabled:
            self.privileged_teacher_residual_actor = _build_mlp(
                teacher_input_dim,
                num_actions,
                privileged_teacher_hidden_dims,
                activation_module,
            )
            _init_last_linear(self.privileged_teacher_residual_actor, privileged_teacher_output_init_scale)

        critic_input_dim = num_critic_obs + history_latent_dim
        self.critic = _build_mlp(
            critic_input_dim,
            1,
            critic_hidden_dims,
            activation_module,
        )

        self.world_model_horizon = max(1, int(world_model_horizon))
        self.world_model_target_groups = world_model_target_groups
        self.world_model_target_slices = world_model_target_slices
        self.world_model_target_dim = _resolve_world_model_target_dim(
            num_actor_obs,
            num_actions,
            world_model_target_dim,
            world_model_target_slices,
            world_model_target_groups,
        )
        if self.world_model_target_dim <= 0:
            raise ValueError(f"world_model_target_dim must be positive, got {self.world_model_target_dim}.")
        self.world_model_prediction_mode = "autoregressive"
        self.world_model_predictor = _build_mlp(
            history_latent_dim + self.world_model_target_dim + num_actions,
            self.world_model_target_dim,
            world_model_hidden_dims,
            activation_module,
        )
        _init_last_linear(self.world_model_predictor, world_model_output_init_scale)
        self.detach_history_latent_for_policy = False

        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        self.latest_base_action_mean = None
        self.latest_residual_action_mean = None
        self.latest_scaled_residual_action_mean = None
        self.latest_ungated_scaled_residual_action_mean = None
        self.latest_ungated_scaled_residual_action_mean_for_loss = None
        self.latest_residual_action_gate = None
        self.latest_residual_action_gate_for_loss = None
        self.latest_actor_source = None
        self.latest_world_model_metrics = {}
        Normal.set_default_validate_args(False)

        print(f"Base Actor MLP: {self.base_actor_critic.actor}")
        print(f"History Encoder: {self.history_encoder}")
        print(f"Adapter mode: {self.adapter_mode}")
        print(f"Residual Actor MLP: {self.residual_actor}")
        if self.adapter_mode == "layerwise":
            print(f"Layer-wise Adapter MLPs: {self.layer_adapters}")
        if self.privileged_teacher_enabled:
            print(f"Privileged Teacher Residual MLP: {self.privileged_teacher_residual_actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"World Model Predictor MLP: {self.world_model_predictor}")

    def _build_layerwise_adapters(self, hidden_dims, activation_module, output_init_scale):
        adapters = nn.ModuleList()
        for base_layer in self.base_actor_critic.actor:
            if not isinstance(base_layer, nn.Linear):
                continue
            adapters.append(
                _build_mlp(
                    base_layer.in_features + self.history_latent_dim,
                    base_layer.out_features,
                    hidden_dims,
                    activation_module,
                )
            )
            _init_last_linear(adapters[-1], output_init_scale)
        return adapters

    def _freeze_base(self):
        self.base_actor_critic.eval()
        for param in self.base_actor_critic.parameters():
            param.requires_grad_(False)

    def _load_base_actor_from_checkpoint(self, path):
        checkpoint = _load_checkpoint(path)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        actor_state = self._extract_actor_state(state_dict)
        if not actor_state:
            raise ValueError(f"No actor weights found in base checkpoint: {path}")

        base_state = self.base_actor_critic.state_dict()
        expected_actor_keys = {key for key in base_state if key.startswith("actor.")}
        missing = sorted(expected_actor_keys - set(actor_state.keys()))
        if missing:
            raise ValueError(
                "Base checkpoint is missing actor keys: "
                + str(missing[:8])
                + (" ..." if len(missing) > 8 else "")
            )
        incompatible = [
            key
            for key, value in actor_state.items()
            if key not in base_state or base_state[key].shape != value.shape
        ]
        if incompatible:
            raise ValueError(
                "Base checkpoint actor shape mismatch for keys: "
                + str(incompatible[:8])
                + (" ..." if len(incompatible) > 8 else "")
            )
        self.base_actor_critic.load_state_dict(actor_state, strict=False)

    @staticmethod
    def _extract_actor_state(state_dict):
        actor_state = {}
        prefixes = ("module.", "policy.")
        for key, value in state_dict.items():
            normalized_key = key
            for prefix in prefixes:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
            if normalized_key.startswith("base_actor_critic."):
                normalized_key = normalized_key[len("base_actor_critic.") :]
            if normalized_key.startswith("actor."):
                actor_state[normalized_key] = value
        return actor_state

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_base()
        return self

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _pad_or_truncate_history(self, history):
        if history.shape[-1] == self.history_dim:
            return history
        if history.shape[-1] > self.history_dim:
            return history[..., : self.history_dim]
        pad_width = self.history_dim - history.shape[-1]
        padding = history.new_zeros(*history.shape[:-1], pad_width)
        return torch.cat((history, padding), dim=-1)

    def _pad_or_truncate_history_step(self, history):
        if history.shape[-1] == self.history_step_dim:
            return history
        if history.shape[-1] > self.history_step_dim:
            return history[..., : self.history_step_dim]
        pad_width = self.history_step_dim - history.shape[-1]
        padding = history.new_zeros(*history.shape[:-1], pad_width)
        return torch.cat((history, padding), dim=-1)

    def _pack_action_history_for_obs_action_layout(self, history):
        obs_action_dim = self.num_actor_obs + self.num_actions
        if history.dim() != 3 or history.shape[-1] != self.num_actions:
            return None
        if self.history_dim != history.shape[1] * obs_action_dim:
            return None
        observation_slots = history.new_zeros(*history.shape[:-1], self.num_actor_obs)
        return torch.cat((observation_slots, history), dim=-1).flatten(start_dim=1)

    def _pack_action_history_sequence_for_obs_action_layout(self, history):
        obs_action_dim = self.num_actor_obs + self.num_actions
        if history is None or history.dim() != 3 or history.shape[-1] != self.num_actions:
            return None
        if self.history_step_dim != obs_action_dim:
            return None
        observation_slots = history.new_zeros(*history.shape[:-1], self.num_actor_obs)
        return torch.cat((observation_slots, history), dim=-1)

    def _prepare_history_sequence(self, reference, history):
        batch_size = reference.shape[0]
        if history is None:
            return reference.new_zeros(batch_size, self.history_length, self.history_step_dim)
        packed_action_history = self._pack_action_history_sequence_for_obs_action_layout(history)
        if packed_action_history is not None:
            history = packed_action_history
        if history.dim() == 2:
            if history.shape[-1] == self.history_dim:
                history = history.view(batch_size, self.history_length, self.history_step_dim)
            elif history.shape[-1] == self.history_step_dim:
                history = history.unsqueeze(1)
            else:
                history = self._pad_or_truncate_history(history).view(
                    batch_size, self.history_length, self.history_step_dim
                )
        elif history.dim() > 3:
            history = history.flatten(start_dim=2)
        history = self._pad_or_truncate_history_step(history)
        if history.shape[1] > self.history_length:
            history = history[:, -self.history_length :]
        elif history.shape[1] < self.history_length:
            pad_steps = self.history_length - history.shape[1]
            padding = history.new_zeros(batch_size, pad_steps, self.history_step_dim)
            history = torch.cat((padding, history), dim=1)
        return history

    def _prepare_history(self, reference, history):
        if history is None:
            history = reference.new_zeros(reference.shape[0], self.history_dim)
        packed_action_history = self._pack_action_history_for_obs_action_layout(history)
        if packed_action_history is not None:
            return packed_action_history
        if history.dim() > 2:
            history = history.flatten(start_dim=1)
        return self._pad_or_truncate_history(history)

    def _history_latent(self, reference, history):
        if self.history_encoder_type == "gru":
            sequence = self._prepare_history_sequence(reference, history)
            _, hidden = self.history_encoder(sequence)
            return hidden[-1]
        return self.history_encoder(self._prepare_history(reference, history))

    def _prepare_world_model_actions(self, actions, reference):
        if actions.dim() > 2:
            actions = actions.flatten(start_dim=1)
        if actions.shape[-1] == self.num_actions:
            return actions
        if actions.shape[-1] > self.num_actions:
            return actions[..., : self.num_actions]
        pad_width = self.num_actions - actions.shape[-1]
        padding = actions.new_zeros(*actions.shape[:-1], pad_width)
        return torch.cat((actions, padding), dim=-1).to(device=reference.device, dtype=reference.dtype)

    def _prepare_world_model_action_sequence(self, actions, latent):
        actions = actions.to(device=latent.device, dtype=latent.dtype)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if actions.dim() == 2:
            actions = self._prepare_world_model_actions(actions, latent)
            return actions.unsqueeze(-2).expand(actions.shape[0], self.world_model_horizon, self.num_actions)
        if actions.dim() > 3:
            actions = actions.flatten(start_dim=0, end_dim=-3)
        if actions.shape[-1] > self.num_actions:
            actions = actions[..., : self.num_actions]
        elif actions.shape[-1] < self.num_actions:
            pad_width = self.num_actions - actions.shape[-1]
            padding = actions.new_zeros(*actions.shape[:-1], pad_width)
            actions = torch.cat((actions, padding), dim=-1)
        if actions.shape[-2] > self.world_model_horizon:
            return actions[..., : self.world_model_horizon, :]
        if actions.shape[-2] < self.world_model_horizon:
            pad_shape = (*actions.shape[:-2], self.world_model_horizon - actions.shape[-2], self.num_actions)
            actions = torch.cat((actions, actions.new_zeros(pad_shape)), dim=-2)
        return actions

    def _prepare_world_model_initial_state(self, reference, latent):
        if reference is None:
            return latent.new_zeros(latent.shape[0], self.world_model_target_dim)
        reference = reference.to(device=latent.device, dtype=latent.dtype)
        if reference.dim() > 2:
            reference = reference.flatten(start_dim=1)
        if reference.shape[-1] == self.world_model_target_dim:
            return reference
        if reference.shape[-1] > self.world_model_target_dim:
            return reference[..., : self.world_model_target_dim]
        pad_width = self.world_model_target_dim - reference.shape[-1]
        padding = reference.new_zeros(*reference.shape[:-1], pad_width)
        return torch.cat((reference, padding), dim=-1)

    def _prepare_world_model_targets(self, targets, predictions):
        targets = targets.to(device=predictions.device, dtype=predictions.dtype)
        if targets.dim() == predictions.dim() - 1:
            targets = targets.unsqueeze(-2)

        if targets.shape[-2] > self.world_model_horizon:
            targets = targets[..., : self.world_model_horizon, :]
        elif targets.shape[-2] < self.world_model_horizon:
            pad_shape = (*targets.shape[:-2], self.world_model_horizon - targets.shape[-2], targets.shape[-1])
            targets = torch.cat((targets, targets.new_zeros(pad_shape)), dim=-2)

        if targets.shape[-1] > self.world_model_target_dim:
            targets = targets[..., : self.world_model_target_dim]
        elif targets.shape[-1] < self.world_model_target_dim:
            pad_shape = (*targets.shape[:-1], self.world_model_target_dim - targets.shape[-1])
            targets = torch.cat((targets, targets.new_zeros(pad_shape)), dim=-1)
        return targets

    def predict_world_model(self, history, actions, reference=None):
        if reference is None:
            latent_reference = actions[:, 0] if actions.dim() > 2 else actions
        else:
            latent_reference = reference
        latent = self._history_latent(latent_reference, history)
        action_sequence = self._prepare_world_model_action_sequence(actions, latent)
        predicted_state = self._prepare_world_model_initial_state(reference, latent)
        predictions = []
        for step_index in range(self.world_model_horizon):
            step_action = action_sequence[:, step_index]
            transition_input = torch.cat((latent, predicted_state, step_action), dim=-1)
            state_delta = self.world_model_predictor(transition_input)
            predicted_state = predicted_state + state_delta
            predictions.append(predicted_state)
        return torch.stack(predictions, dim=-2)

    def compute_world_model_loss(self, history, actions, targets, reference=None, valid_mask=None):
        predictions = self.predict_world_model(history, actions, reference=reference)
        targets = self._prepare_world_model_targets(targets, predictions).detach()
        absolute_error = torch.abs(predictions - targets)

        if valid_mask is not None:
            mask = valid_mask.to(device=absolute_error.device, dtype=absolute_error.dtype)
            while mask.dim() < absolute_error.dim():
                mask = mask.unsqueeze(-1)
            weighted_error = absolute_error * mask
            denom = mask.expand_as(absolute_error).sum().clamp_min(1.0)
            loss = weighted_error.sum() / denom
            mean_abs_error = weighted_error.sum() / denom
            valid_fraction = mask.mean()
        else:
            loss = absolute_error.mean()
            mean_abs_error = loss
            valid_fraction = absolute_error.new_tensor(1.0)

        self.latest_world_model_metrics = {
            "loss": loss.detach(),
            "abs_error": mean_abs_error.detach(),
            "valid_fraction": valid_fraction.detach(),
        }
        return loss, self.latest_world_model_metrics

    def _base_action_mean(self, observations):
        with torch.no_grad():
            return self.base_actor_critic.actor(observations)

    def _prepare_residual_gate(self, residual_gate, residual):
        if residual_gate is None:
            return residual.new_ones(residual.shape[0], 1)
        gate = residual_gate.to(device=residual.device, dtype=residual.dtype)
        if gate.dim() == 1:
            gate = gate.unsqueeze(-1)
        if gate.shape[0] != residual.shape[0]:
            raise ValueError(
                f"residual_gate batch shape {tuple(gate.shape)} does not match residual shape {tuple(residual.shape)}"
            )
        if gate.shape[-1] not in (1, residual.shape[-1]):
            raise ValueError(
                f"residual_gate width {gate.shape[-1]} must be 1 or num_actions={residual.shape[-1]}"
            )
        return gate.clamp(0.0, 1.0)

    def _prepare_layerwise_gate(self, residual_gate, reference):
        if residual_gate is None:
            return reference.new_ones(reference.shape[0], 1), None
        gate = residual_gate.to(device=reference.device, dtype=reference.dtype)
        if gate.dim() == 1:
            gate = gate.unsqueeze(-1)
        if gate.shape[0] != reference.shape[0]:
            raise ValueError(
                f"residual_gate batch shape {tuple(gate.shape)} does not match observation shape {tuple(reference.shape)}"
            )
        if gate.shape[-1] == 1:
            gate = gate.clamp(0.0, 1.0)
            return gate, None
        if gate.shape[-1] == self.num_actions:
            gate = gate.clamp(0.0, 1.0)
            return gate.mean(dim=-1, keepdim=True), gate
        raise ValueError(f"residual_gate width {gate.shape[-1]} must be 1 or num_actions={self.num_actions}")

    def _layerwise_actor_mean(self, observations, latent, residual_gate=None):
        if len(self.layer_adapters) == 0:
            return self.base_actor_critic.actor(observations)
        gate_scalar, action_gate = self._prepare_layerwise_gate(residual_gate, observations)
        x = observations
        adapter_index = 0
        for layer in self.base_actor_critic.actor:
            if isinstance(layer, nn.Linear):
                base_output = layer(x)
                adapter_input = torch.cat((x, latent), dim=-1)
                adapter_delta = self.layer_adapters[adapter_index](adapter_input)
                if adapter_index == len(self.layer_adapters) - 1 and action_gate is not None:
                    gate = action_gate
                else:
                    gate = gate_scalar
                x = base_output + self.residual_scale * gate * adapter_delta
                adapter_index += 1
            else:
                x = layer(x)
        return x

    def _actor_mean(self, observations, history=None, residual_gate=None, update_latest=True):
        base_mean = self._base_action_mean(observations)
        latent = self._history_latent(observations, history)
        if self.detach_history_latent_for_policy:
            latent = latent.detach()
        if self.adapter_mode == "layerwise":
            if residual_gate is None:
                gate = base_mean.new_ones(base_mean.shape[0], 1)
            else:
                gate, _ = self._prepare_layerwise_gate(residual_gate, base_mean)
            adapted_mean = self._layerwise_actor_mean(observations, latent, residual_gate=residual_gate)
            ungated_adapted_mean = self._layerwise_actor_mean(
                observations,
                latent,
                residual_gate=base_mean.new_ones(base_mean.shape[0], 1),
            )
            scaled_residual_mean = adapted_mean - base_mean
            ungated_scaled_residual_mean = ungated_adapted_mean - base_mean
            residual_mean = ungated_scaled_residual_mean / max(abs(self.residual_scale), 1.0e-8)
        else:
            residual_input = torch.cat((observations, latent, base_mean), dim=-1)
            residual_mean = self.residual_actor(residual_input)
            ungated_scaled_residual_mean = self.residual_scale * residual_mean
            gate = self._prepare_residual_gate(residual_gate, ungated_scaled_residual_mean)
            scaled_residual_mean = gate * ungated_scaled_residual_mean
            adapted_mean = base_mean + scaled_residual_mean
        if update_latest:
            self.latest_base_action_mean = base_mean.detach()
            self.latest_residual_action_mean = residual_mean.detach()
            self.latest_scaled_residual_action_mean = scaled_residual_mean.detach()
            self.latest_ungated_scaled_residual_action_mean = ungated_scaled_residual_mean.detach()
            self.latest_ungated_scaled_residual_action_mean_for_loss = ungated_scaled_residual_mean
            self.latest_residual_action_gate = gate.detach()
            self.latest_residual_action_gate_for_loss = gate
            self.latest_actor_source = "adapter"
        return adapted_mean

    def _privileged_teacher_actor_mean(
        self,
        observations,
        critic_observations,
        history=None,
        residual_gate=None,
        update_latest=True,
    ):
        if not self.privileged_teacher_enabled or self.privileged_teacher_residual_actor is None:
            raise RuntimeError("Privileged teacher actor is not enabled for this TrackAdapterActorCritic.")
        if self.privileged_teacher_use_adapter_base:
            with torch.no_grad():
                base_mean = self._actor_mean(
                    observations,
                    history,
                    residual_gate=residual_gate,
                    update_latest=False,
                )
            base_mean = base_mean.detach()
        else:
            base_mean = self._base_action_mean(observations)
        teacher_input = torch.cat((observations, critic_observations, base_mean), dim=-1)
        residual_mean = self.privileged_teacher_residual_actor(teacher_input)
        ungated_scaled_residual_mean = self.privileged_teacher_residual_scale * residual_mean
        gate = self._prepare_residual_gate(residual_gate, ungated_scaled_residual_mean)
        scaled_residual_mean = gate * ungated_scaled_residual_mean
        adapted_mean = base_mean + scaled_residual_mean

        if update_latest:
            self.latest_base_action_mean = base_mean.detach()
            self.latest_residual_action_mean = residual_mean.detach()
            self.latest_scaled_residual_action_mean = scaled_residual_mean.detach()
            self.latest_ungated_scaled_residual_action_mean = ungated_scaled_residual_mean.detach()
            self.latest_ungated_scaled_residual_action_mean_for_loss = ungated_scaled_residual_mean
            self.latest_residual_action_gate = gate.detach()
            self.latest_residual_action_gate_for_loss = gate
            self.latest_actor_source = "privileged_teacher"
        return adapted_mean

    def update_distribution(self, observations, history=None, residual_gate=None):
        mean = self._actor_mean(observations, history, residual_gate=residual_gate)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def update_privileged_teacher_distribution(self, observations, critic_observations, history=None, residual_gate=None):
        mean = self._privileged_teacher_actor_mean(
            observations,
            critic_observations,
            history=history,
            residual_gate=residual_gate,
        )
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, observations, history=None, residual_gate=None, **kwargs):
        self.update_distribution(observations, history, residual_gate=residual_gate)
        return self.distribution.sample()

    def privileged_teacher_act(self, observations, critic_observations, history=None, residual_gate=None, **kwargs):
        self.update_privileged_teacher_distribution(
            observations,
            critic_observations,
            history=history,
            residual_gate=residual_gate,
        )
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history=None, residual_gate=None):
        return self._actor_mean(observations, history, residual_gate=residual_gate)

    def privileged_teacher_act_inference(self, observations, critic_observations, history=None, residual_gate=None):
        return self._privileged_teacher_actor_mean(
            observations,
            critic_observations,
            history=history,
            residual_gate=residual_gate,
        )

    def privileged_teacher_action_target(self, observations, critic_observations, history=None, residual_gate=None):
        return self._privileged_teacher_actor_mean(
            observations,
            critic_observations,
            history=history,
            residual_gate=residual_gate,
            update_latest=False,
        )

    def evaluate(self, critic_observations, history=None, **kwargs):
        latent = self._history_latent(critic_observations, history)
        if self.detach_history_latent_for_policy:
            latent = latent.detach()
        return self.critic(torch.cat((critic_observations, latent), dim=-1))

    def adapter_parameters(self):
        noise_params = [self.log_std] if self.noise_std_type == "log" else [self.std]
        return chain(
            self.history_encoder.parameters(),
            self.policy_adapter_parameters(),
            self.world_model_predictor.parameters(),
            self.critic.parameters(),
            noise_params,
        )

    def policy_adapter_parameters(self):
        if self.adapter_mode == "layerwise":
            return self.layer_adapters.parameters()
        return self.residual_actor.parameters()

    def ppo_parameters(self, include_history_encoder=True):
        noise_params = [self.log_std] if self.noise_std_type == "log" else [self.std]
        groups = []
        if include_history_encoder:
            groups.append(self.history_encoder.parameters())
        groups.extend(
            (
                self.policy_adapter_parameters(),
                self.critic.parameters(),
                noise_params,
            )
        )
        return chain(*groups)

    def privileged_teacher_ppo_parameters(self):
        if not self.privileged_teacher_enabled or self.privileged_teacher_residual_actor is None:
            raise RuntimeError("Privileged teacher actor is not enabled for this TrackAdapterActorCritic.")
        noise_params = [self.log_std] if self.noise_std_type == "log" else [self.std]
        return chain(self.privileged_teacher_residual_actor.parameters(), self.critic.parameters(), noise_params)

    def reset_privileged_teacher_residual_actor(self, output_init_scale=None):
        if not self.privileged_teacher_enabled or self.privileged_teacher_residual_actor is None:
            raise RuntimeError("Privileged teacher actor is not enabled for this TrackAdapterActorCritic.")
        for layer in _linear_modules(self.privileged_teacher_residual_actor):
            layer.reset_parameters()
        if output_init_scale is None:
            output_init_scale = 0.0
        _init_last_linear(self.privileged_teacher_residual_actor, float(output_init_scale))

    def world_model_parameters(self):
        return chain(self.history_encoder.parameters(), self.world_model_predictor.parameters())

    def get_world_model_diagnostics(self):
        return dict(self.latest_world_model_metrics)

    def get_adapter_diagnostics(self):
        diagnostics = {}
        if self.latest_actor_source == "privileged_teacher":
            diagnostics["privileged_teacher_active"] = 1.0
        elif self.latest_actor_source == "adapter":
            diagnostics["student_adapter_active"] = 1.0
        if isinstance(self.latest_residual_action_gate, torch.Tensor):
            diagnostics["residual_action_gate"] = self.latest_residual_action_gate
        if isinstance(self.latest_ungated_scaled_residual_action_mean, torch.Tensor):
            squared = torch.square(self.latest_ungated_scaled_residual_action_mean.detach())
            if squared.dim() > 1:
                squared = torch.sum(squared, dim=-1)
            diagnostics["ungated_scaled_residual_action_l2"] = squared
            if self.latest_actor_source == "privileged_teacher":
                diagnostics["privileged_teacher_ungated_scaled_residual_action_l2"] = squared
        return diagnostics

    def trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)

    def load_state_dict(self, state_dict, strict=True):
        state_dict = dict(state_dict)
        if self.privileged_teacher_enabled and self.privileged_teacher_residual_actor is not None:
            current_state = super().state_dict()
            for key, value in current_state.items():
                if key.startswith("privileged_teacher_residual_actor.") and key not in state_dict:
                    state_dict[key] = value
        else:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("privileged_teacher_residual_actor.")
            }
        actor_state = self._extract_actor_state(state_dict)
        has_adapter_state = any(
            key.startswith(
                (
                    "history_encoder.",
                    "residual_actor.",
                    "layer_adapters.",
                    "privileged_teacher_residual_actor.",
                    "world_model_predictor.",
                    "critic.",
                    "base_actor_critic.",
                )
            )
            for key in state_dict
        )
        if actor_state and not has_adapter_state:
            self.base_actor_critic.load_state_dict(actor_state, strict=False)
            self._freeze_base()
            return False

        if not strict:
            current_state = super().state_dict()
            skipped = []
            filtered_state = {}
            for key, value in state_dict.items():
                current_value = current_state.get(key)
                if current_value is not None and tuple(current_value.shape) != tuple(value.shape):
                    skipped.append((key, tuple(value.shape), tuple(current_value.shape)))
                    continue
                filtered_state[key] = value
            if skipped:
                preview = ", ".join(
                    f"{name}: checkpoint{old_shape}->current{new_shape}"
                    for name, old_shape, new_shape in skipped[:8]
                )
                suffix = "" if len(skipped) <= 8 else f", ... ({len(skipped)} total)"
                print(f"[TrackAdapterActorCritic] Skipped shape-mismatched checkpoint keys: {preview}{suffix}")
            state_dict = filtered_state

        super().load_state_dict(state_dict, strict=strict)
        self._freeze_base()
        return True

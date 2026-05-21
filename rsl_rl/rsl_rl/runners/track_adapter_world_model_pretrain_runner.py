# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization, TrackAdapterActorCritic
from rsl_rl.modules.track_adapter_actor_critic import _target_feature_indices


@dataclass
class _RolloutBatch:
    observations: torch.Tensor
    next_observations: torch.Tensor
    world_model_states: torch.Tensor | None
    world_model_reference_states: torch.Tensor | None
    histories: torch.Tensor
    actions: torch.Tensor
    dones: torch.Tensor


class TrackAdapterWorldModelPretrainRunner:
    """Stage 3 pretraining runner for Track Adapter world-model dynamics.

    The environment is driven by the frozen AMP base actor only. The residual
    actor is not queried during collection, so pretraining cannot perturb the
    base controller. Optimization is limited to the history encoder and world
    model predictor.
    """

    _NEXT_TARGET_SOURCES = ("next_actor", "next_obs", "next_observation", "next_observations")
    _OBS_TARGET_SOURCES = ("actor", "obs", "observation", "observations", None)

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = dict(train_cfg)
        self.alg_cfg = dict(train_cfg.get("algorithm", {}))
        self.policy_cfg = dict(train_cfg.get("policy", {}))
        self.pretrain_cfg = dict(train_cfg.get("world_model_pretrain", {}))
        self.device = device
        self.env = env
        self.log_dir = log_dir

        self.history_length = int(train_cfg.get("history_length", self.policy_cfg.get("history_length", 20)))
        self.base_checkpoint_path = train_cfg.get(
            "base_checkpoint_path", self.policy_cfg.get("base_checkpoint_path", None)
        )
        if not self.base_checkpoint_path:
            raise ValueError("TrackAdapterWorldModelPretrainRunner requires base_checkpoint_path.")
        self.strict_base_auxiliary_state = bool(train_cfg.get("strict_base_auxiliary_state", True))
        self.freeze_loaded_normalizers = bool(train_cfg.get("freeze_loaded_normalizers", True))

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        self.history_step_dim = num_obs + self.env.num_actions
        self.world_model_state_schema = self._build_world_model_state_schema(num_obs)
        self.world_model_state_dim = sum(width for _, width in self.world_model_state_schema)
        self._world_model_foot_body_ids = None
        self.privileged_obs_type = "critic" if "critic" in extras["observations"] else None
        num_privileged_obs = (
            extras["observations"][self.privileged_obs_type].shape[1]
            if self.privileged_obs_type is not None
            else num_obs
        )
        world_model_cfg = dict(self.alg_cfg.get("world_model_cfg", {}))
        self.world_model_horizon = int(
            self.pretrain_cfg.get(
                "horizon",
                self.pretrain_cfg.get(
                    "world_model_horizon",
                    world_model_cfg.get("horizon", self.policy_cfg.get("world_model_horizon", 1)),
                ),
            )
        )
        self.world_model_target_source = self.pretrain_cfg.get(
            "target_source", world_model_cfg.get("target_source", "next_observations")
        )
        self.world_model_target_groups = self.pretrain_cfg.get(
            "target_groups", world_model_cfg.get("target_groups", world_model_cfg.get("groups", None))
        )
        self.world_model_target_slices = self.pretrain_cfg.get(
            "target_slices", world_model_cfg.get("target_slices", world_model_cfg.get("slices", None))
        )
        target_dim = self.pretrain_cfg.get(
            "target_dim",
            world_model_cfg.get(
                "target_dim",
                self.world_model_state_dim
                if self._world_model_uses_structured_state()
                else self.policy_cfg.get("world_model_target_dim", num_obs),
            ),
        )
        self.world_model_target_dim = int(target_dim) if target_dim is not None else num_obs

        policy_class_name = self.policy_cfg.pop("class_name", "TrackAdapterActorCritic")
        if policy_class_name != "TrackAdapterActorCritic":
            raise ValueError(
                "TrackAdapterWorldModelPretrainRunner requires TrackAdapterActorCritic, "
                f"got {policy_class_name}."
            )
        self.policy_cfg["history_length"] = self.history_length
        self.policy_cfg["history_step_dim"] = self.history_step_dim
        self.policy_cfg["history_dim"] = self.history_length * self.history_step_dim
        self.policy_cfg["world_model_horizon"] = self.world_model_horizon
        self.policy_cfg["world_model_target_dim"] = self.world_model_target_dim
        self.policy_cfg["world_model_target_groups"] = self.world_model_target_groups
        self.policy_cfg["world_model_target_slices"] = self.world_model_target_slices
        self.policy: TrackAdapterActorCritic = TrackAdapterActorCritic(
            num_obs,
            num_privileged_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        self.empirical_normalization = bool(self.cfg.get("empirical_normalization", False))
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = nn.Identity().to(self.device)
            self.privileged_obs_normalizer = nn.Identity().to(self.device)

        self.discriminator_state_dict = None
        self.amp_normalizer = None
        self._load_base_auxiliary_state()

        self.num_steps_per_env = int(
            self.pretrain_cfg.get(
                "num_steps_per_env",
                self.cfg.get("pretrain_num_steps_per_env", self.cfg.get("num_steps_per_env", 24)),
            )
        )
        self.num_learning_epochs = int(
            self.pretrain_cfg.get(
                "num_learning_epochs",
                self.pretrain_cfg.get("epochs", self.alg_cfg.get("num_learning_epochs", 1)),
            )
        )
        self.num_mini_batches = int(
            self.pretrain_cfg.get(
                "num_mini_batches",
                self.pretrain_cfg.get("mini_batches", self.alg_cfg.get("num_mini_batches", 1)),
            )
        )
        self.learning_rate = float(
            self.pretrain_cfg.get(
                "learning_rate",
                self.pretrain_cfg.get("lr", self.alg_cfg.get("learning_rate", 1.0e-3)),
            )
        )
        self.max_grad_norm = float(
            self.pretrain_cfg.get("max_grad_norm", self.alg_cfg.get("max_grad_norm", 1.0))
        )
        self.save_interval = int(self.pretrain_cfg.get("save_interval", self.cfg.get("save_interval", 100)))

        self.world_model_parameters = list(self.policy.world_model_parameters())
        if not self.world_model_parameters:
            raise ValueError("Track Adapter world-model pretraining found no trainable parameters.")
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        for parameter in self.world_model_parameters:
            parameter.requires_grad_(True)
        base_ids = {id(parameter) for parameter in self.policy.base_actor_critic.parameters()}
        if any(id(parameter) in base_ids for parameter in self.world_model_parameters):
            raise ValueError("World-model pretrain optimizer cannot include frozen base actor parameters.")
        self.optimizer = optim.Adam(self.world_model_parameters, lr=self.learning_rate)

        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self._history = None
        self.last_loss_dict = None
        self.eval_mode()

    def _load_base_auxiliary_state(self):
        if not os.path.exists(self.base_checkpoint_path):
            raise FileNotFoundError(f"Track Adapter base checkpoint not found: {self.base_checkpoint_path}")
        loaded = torch.load(self.base_checkpoint_path, map_location=self.device, weights_only=False)

        self.discriminator_state_dict = loaded.get("discriminator_state_dict")
        if self.discriminator_state_dict is None and self.strict_base_auxiliary_state:
            raise RuntimeError("Base checkpoint is missing discriminator_state_dict.")

        self.amp_normalizer = loaded.get("amp_normalizer")
        if self.amp_normalizer is None and self.strict_base_auxiliary_state:
            raise RuntimeError("Base checkpoint is missing amp_normalizer.")

        if not self.empirical_normalization:
            return

        obs_state = loaded.get("obs_norm_state_dict")
        if obs_state is not None:
            try:
                self.obs_normalizer.load_state_dict(obs_state)
                if self.freeze_loaded_normalizers:
                    self.obs_normalizer.eval()
            except RuntimeError as exc:
                message = "[TrackAdapterWorldModelPretrainRunner] Observation normalizer shape mismatch."
                if self.strict_base_auxiliary_state:
                    raise RuntimeError(message) from exc
                print(message + " Skipping load.")
        elif self.strict_base_auxiliary_state:
            raise RuntimeError("Base checkpoint is missing obs_norm_state_dict.")

        priv_state = loaded.get("privileged_obs_norm_state_dict")
        if priv_state is not None:
            try:
                self.privileged_obs_normalizer.load_state_dict(priv_state)
            except RuntimeError as exc:
                message = "[TrackAdapterWorldModelPretrainRunner] Privileged observation normalizer shape mismatch."
                if self.strict_base_auxiliary_state:
                    raise RuntimeError(message) from exc
                print(message + " Skipping load.")
        elif self.strict_base_auxiliary_state:
            raise RuntimeError("Base checkpoint is missing privileged_obs_norm_state_dict.")

    @staticmethod
    def _fit_feature_dim(features: torch.Tensor, target_dim: int | None) -> torch.Tensor:
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

    def _world_model_target_source_name(self):
        source = self.world_model_target_source
        if isinstance(source, dict):
            source = source.get("source", source.get("target_source", "observations"))
        if isinstance(source, str) and "::" in source:
            source = source.split("::", 1)[0]
        return source

    def _world_model_uses_structured_state(self):
        return self._world_model_target_source_name() in ("world_model_state", "wm_state", "structured", "psi_s")

    def _build_world_model_state_schema(self, num_actor_obs):
        return [
            ("normalized_actor_obs", int(num_actor_obs)),
            ("base_lin_vel_b", 3),
            ("base_height", 1),
            ("foot_contacts", 2),
        ]

    @property
    def _base_env(self):
        return self.env.env.env

    def _resolve_world_model_foot_body_ids(self):
        if self._world_model_foot_body_ids is not None:
            return self._world_model_foot_body_ids
        world_model_cfg = dict(self.alg_cfg.get("world_model_cfg", {}))
        sensor_name = world_model_cfg.get("contact_sensor_name", "contact_forces")
        if sensor_name not in self._base_env.scene.sensors:
            raise RuntimeError(f"Track Adapter world-model pretrain contact sensor not found: {sensor_name}")
        contact_sensor = self._base_env.scene.sensors[sensor_name]
        body_names = list(getattr(contact_sensor, "body_names", []))
        if not body_names:
            raise RuntimeError(
                f"Track Adapter world-model pretrain cannot resolve body names for contact sensor {sensor_name}."
            )
        pattern = world_model_cfg.get("foot_body_pattern", r".*_foot_.*")
        foot_ids = [
            index
            for index, name in enumerate(body_names)
            if re.fullmatch(pattern, name) or re.search(pattern, name)
        ]
        if not foot_ids:
            raise RuntimeError(
                "Track Adapter world-model pretrain resolved no foot contact bodies for "
                f"sensor={sensor_name}, pattern={pattern}."
            )
        self._world_model_foot_body_ids = foot_ids
        return foot_ids

    def _compute_world_model_state(self, normalized_actor_obs):
        normalized_actor_obs = normalized_actor_obs.to(self.device)
        world_model_cfg = dict(self.alg_cfg.get("world_model_cfg", {}))
        robot = self._base_env.scene["robot"]
        base_lin_vel_b = robot.data.root_lin_vel_b.to(self.device)
        base_height = robot.data.root_pos_w[:, 2:3].to(self.device)
        foot_ids = self._resolve_world_model_foot_body_ids()
        sensor_name = world_model_cfg.get("contact_sensor_name", "contact_forces")
        contact_sensor = self._base_env.scene.sensors[sensor_name]
        forces = contact_sensor.data.net_forces_w_history.to(self.device)
        contact_norm = torch.max(torch.norm(forces[:, :, foot_ids], dim=-1), dim=1)[0]
        threshold = float(world_model_cfg.get("foot_contact_threshold", 1.0))
        contacts = contact_norm > threshold
        sensor_body_names = list(getattr(contact_sensor, "body_names", []))
        left_ids = [local for local, body_id in enumerate(foot_ids) if "left" in sensor_body_names[body_id].lower()]
        right_ids = [local for local, body_id in enumerate(foot_ids) if "right" in sensor_body_names[body_id].lower()]
        if left_ids and right_ids:
            foot_contacts = torch.stack(
                [torch.any(contacts[:, left_ids], dim=1), torch.any(contacts[:, right_ids], dim=1)],
                dim=-1,
            )
        elif contacts.shape[1] >= 2:
            foot_contacts = contacts[:, :2]
        else:
            foot_contacts = torch.cat((contacts, contacts.new_zeros(contacts.shape[0], 2 - contacts.shape[1])), dim=1)
        return torch.cat(
            (
                normalized_actor_obs,
                base_lin_vel_b,
                base_height,
                foot_contacts.to(dtype=normalized_actor_obs.dtype),
            ),
            dim=-1,
        )

    def _select_world_model_features(self, features):
        indices = _target_feature_indices(
            features.shape[-1],
            self.env.num_actions,
            target_slices=self.world_model_target_slices,
            target_groups=self.world_model_target_groups,
        )
        if not indices:
            return features
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=features.device)
        return features.index_select(-1, index_tensor)

    def world_model_metadata(self):
        return {
            "source": self._world_model_target_source_name(),
            "target_dim": self.world_model_target_dim,
            "target_groups": self.world_model_target_groups,
            "target_slices": self.world_model_target_slices,
            "horizon": self.world_model_horizon,
            "prediction_mode": getattr(self.policy, "world_model_prediction_mode", "autoregressive"),
            "state_schema": list(self.world_model_state_schema),
            "state_dim": self.world_model_state_dim,
            "normalized_actor_obs": True,
        }

    def _normalize_observations(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.to(self.device)
        if not self.empirical_normalization:
            return observations
        return self.obs_normalizer(observations)

    def _append_history(self, history, obs, actions=None, dones=None):
        if actions is None:
            actions = obs.new_zeros(obs.shape[0], self.env.num_actions)
        features = torch.cat((obs, actions.to(device=obs.device, dtype=obs.dtype)), dim=-1)
        if dones is not None:
            reset_env_ids = (dones.to(device=obs.device) > 0).nonzero(as_tuple=False).flatten()
            if len(reset_env_ids) > 0:
                history[reset_env_ids] = 0.0
                features[reset_env_ids] = 0.0
        return torch.cat((history[:, 1:], features.unsqueeze(1)), dim=1)

    def _ensure_history(self, observations: torch.Tensor) -> torch.Tensor:
        if (
            self._history is None
            or self._history.shape[0] != observations.shape[0]
            or self._history.shape[-1] != self.history_step_dim
        ):
            self._history = torch.zeros(
                observations.shape[0],
                self.history_length,
                self.history_step_dim,
                device=observations.device,
            )
        return self._history

    def _base_residual_zero_actions(self, observations: torch.Tensor) -> torch.Tensor:
        return self.policy.base_actor_critic.act_inference(observations)

    def _collect_rollout(self) -> _RolloutBatch:
        obs, _ = self.env.get_observations()
        obs = self._normalize_observations(obs)
        history = self._ensure_history(obs)

        observations = []
        next_observations = []
        world_model_states = []
        world_model_reference_states = []
        histories = []
        actions = []
        dones_list = []

        self.eval_mode()
        with torch.inference_mode():
            for _ in range(self.num_steps_per_env):
                step_history = history.clone()
                step_obs = obs
                if self._world_model_uses_structured_state():
                    world_model_reference_states.append(self._compute_world_model_state(step_obs))
                action = self._base_residual_zero_actions(step_obs)
                next_obs, _, dones, _ = self.env.step(action.to(self.env.device))
                next_obs = self._normalize_observations(next_obs)
                dones = dones.to(self.device)
                if self._world_model_uses_structured_state():
                    world_model_states.append(self._compute_world_model_state(next_obs))

                observations.append(step_obs)
                next_observations.append(next_obs)
                histories.append(step_history)
                actions.append(action)
                dones_list.append(dones)

                history = self._append_history(history, step_obs, action, dones)
                obs = next_obs

        self._history = history.detach()
        return _RolloutBatch(
            observations=torch.stack(observations, dim=0),
            next_observations=torch.stack(next_observations, dim=0),
            world_model_states=torch.stack(world_model_states, dim=0) if world_model_states else None,
            world_model_reference_states=(
                torch.stack(world_model_reference_states, dim=0) if world_model_reference_states else None
            ),
            histories=torch.stack(histories, dim=0),
            actions=torch.stack(actions, dim=0),
            dones=torch.stack(dones_list, dim=0).bool(),
        )

    def _world_model_targets(self, rollout: _RolloutBatch) -> tuple[torch.Tensor, torch.Tensor]:
        horizon = max(1, int(self.world_model_horizon))
        target_source = self._world_model_target_source_name()
        source_is_next = target_source in self._NEXT_TARGET_SOURCES
        if self._world_model_uses_structured_state():
            if rollout.world_model_states is None:
                raise RuntimeError("Structured world-model targets requested but rollout has no world_model_states.")
            source = rollout.world_model_states
            source_is_next = True
        elif source_is_next:
            source = rollout.next_observations
        elif target_source in self._OBS_TARGET_SOURCES:
            source = rollout.observations
        else:
            raise ValueError(f"Unknown world model target source: {self.world_model_target_source}")

        source = self._select_world_model_features(source)
        source = self._fit_feature_dim(source, self.world_model_target_dim)
        num_steps, num_envs = source.shape[:2]
        targets = source.new_zeros(num_steps, num_envs, horizon, source.shape[-1])
        valid_mask = source.new_zeros(num_steps, num_envs, horizon)
        dones = rollout.dones

        for target_index in range(horizon):
            source_start = target_index if source_is_next else target_index + 1
            if source_start >= num_steps:
                continue
            upper = num_steps - source_start
            targets[:upper, :, target_index].copy_(source[source_start:])
            valid = torch.ones(upper, num_envs, dtype=torch.bool, device=source.device)
            for done_offset in range(target_index + 1):
                valid = valid & ~dones[done_offset : upper + done_offset]
            valid_mask[:upper, :, target_index].copy_(valid.to(dtype=valid_mask.dtype))

        return targets, valid_mask

    def _world_model_references(self, rollout: _RolloutBatch) -> torch.Tensor:
        target_source = self._world_model_target_source_name()
        if self._world_model_uses_structured_state():
            if rollout.world_model_reference_states is None:
                raise RuntimeError("Structured world-model references requested but rollout has no reference states.")
            source = rollout.world_model_reference_states
        else:
            source = rollout.observations
        source = self._select_world_model_features(source)
        return self._fit_feature_dim(source, self.world_model_target_dim)

    def _world_model_action_sequences(self, rollout: _RolloutBatch) -> torch.Tensor:
        horizon = max(1, int(self.world_model_horizon))
        num_steps, num_envs = rollout.actions.shape[:2]
        action_sequences = rollout.actions.new_zeros(num_steps, num_envs, horizon, rollout.actions.shape[-1])
        for action_index in range(horizon):
            if action_index >= num_steps:
                continue
            upper = num_steps - action_index
            action_sequences[:upper, :, action_index].copy_(rollout.actions[action_index:])
        return action_sequences

    def update(self, rollout: _RolloutBatch) -> dict[str, float]:
        references = self._world_model_references(rollout)
        targets, valid_mask = self._world_model_targets(rollout)
        histories = rollout.histories.flatten(0, 1)
        action_sequences = self._world_model_action_sequences(rollout).flatten(0, 1)
        references = references.flatten(0, 1)
        targets = targets.flatten(0, 1)
        valid_mask = valid_mask.flatten(0, 1)

        batch_size = histories.shape[0]
        num_mini_batches = max(1, min(int(self.num_mini_batches), batch_size))
        metrics = {
            "wm": 0.0,
            "wm_abs_error": 0.0,
            "wm_valid_fraction": 0.0,
            "wm_grad_norm": 0.0,
        }
        num_updates = 0

        self.train_mode()
        for _ in range(max(1, int(self.num_learning_epochs))):
            indices = torch.randperm(batch_size, device=self.device)
            for batch_indices in torch.chunk(indices, num_mini_batches):
                if batch_indices.numel() == 0:
                    continue
                loss, loss_metrics = self.policy.compute_world_model_loss(
                    history=histories[batch_indices],
                    actions=action_sequences[batch_indices],
                    targets=targets[batch_indices],
                    reference=references[batch_indices],
                    valid_mask=valid_mask[batch_indices],
                )
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.world_model_parameters, self.max_grad_norm)
                self.optimizer.step()

                metrics["wm"] += float(loss.detach().item())
                metrics["wm_abs_error"] += float(loss_metrics["abs_error"].detach().item())
                metrics["wm_valid_fraction"] += float(loss_metrics["valid_fraction"].detach().item())
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.detach().item()
                metrics["wm_grad_norm"] += float(grad_norm)
                num_updates += 1

        if num_updates == 0:
            return metrics
        return {name: value / num_updates for name, value in metrics.items()}

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if (
            init_at_random_ep_len
            and hasattr(self.env, "episode_length_buf")
            and hasattr(self.env, "max_episode_length")
        ):
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + int(num_learning_iterations)
        for it in range(start_iter, tot_iter):
            start = time.time()
            rollout = self._collect_rollout()
            collection_time = time.time() - start

            start = time.time()
            loss_dict = self.update(rollout)
            self.last_loss_dict = dict(loss_dict)
            learn_time = time.time() - start

            self.current_learning_iteration = it
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += collection_time + learn_time
            self.log(it, tot_iter, collection_time, learn_time, loss_dict)

            if self.log_dir is not None and self.save_interval > 0 and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"), infos={"loss_dict": loss_dict})

        if self.log_dir is not None:
            self.save(
                os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"),
                infos={
                    "stage": "track_adapter_world_model_pretrain",
                    "loss_dict": self.last_loss_dict,
                },
            )

    def log(self, iteration: int, total_iterations: int, collection_time: float, learn_time: float, loss_dict: dict):
        elapsed = collection_time + learn_time
        fps = int(self.num_steps_per_env * self.env.num_envs / elapsed) if elapsed > 0 else 0
        metrics = ", ".join(f"{name}={value:.4f}" for name, value in loss_dict.items())
        print(
            "[TrackAdapterWorldModelPretrain] "
            f"iteration {iteration}/{total_iterations} | "
            f"fps={fps} | collection={collection_time:.3f}s | update={learn_time:.3f}s | {metrics}"
        )

    def _disturbance_manifest(self):
        try:
            base_env = self._base_env
            env_cfg = getattr(base_env, "cfg", None)
            command_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
            events_cfg = getattr(env_cfg, "events", None)
            push_event = getattr(events_cfg, "push_robot", None)
            force_push_event = getattr(events_cfg, "external_wrench_push", None)
            physics_material = getattr(events_cfg, "physics_material", None)
            add_base_mass = getattr(events_cfg, "add_base_mass", None)
            base_com = getattr(events_cfg, "base_com", None)
        except Exception:
            return {"available": False}

        def event_enabled(event):
            return event is not None and getattr(event, "func", None) is not None

        def safe_value(value):
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            if isinstance(value, (list, tuple)):
                return [safe_value(item) for item in value]
            if isinstance(value, dict):
                return {str(key): safe_value(item) for key, item in value.items()}
            return repr(value)

        def safe_params(event):
            return safe_value(dict(getattr(event, "params", {}) or {})) if event_enabled(event) else {}

        return {
            "available": True,
            "smoke_mode_env": os.getenv("BOOSTER_TRACK_ADAPTER_SMOKE_MODE"),
            "num_envs": int(getattr(self.env, "num_envs", 0)),
            "push_enabled": event_enabled(push_event),
            "push_params": safe_params(push_event),
            "force_push_enabled": event_enabled(force_push_event),
            "force_push_params": safe_params(force_push_event),
            "material_randomization_enabled": event_enabled(physics_material),
            "base_mass_randomization_enabled": event_enabled(add_base_mass),
            "base_com_randomization_enabled": event_enabled(base_com),
            "failure_mining_enabled": bool(getattr(command_cfg, "failure_mining_enabled", False)),
            "failure_mining_sample_prob": float(getattr(command_cfg, "failure_mining_sample_prob", 0.0)),
            "failure_mining_push_sample_prob": float(
                getattr(command_cfg, "failure_mining_push_sample_prob", 0.0)
            ),
            "sudden_stop_enabled": float(getattr(command_cfg, "rel_sudden_stop_envs", 0.0)) > 0.0,
            "sudden_stop_prob": float(getattr(command_cfg, "rel_sudden_stop_envs", 0.0)),
        }

    def save(self, path: str, infos=None):
        loss_dict = None
        if isinstance(infos, dict):
            loss_dict = infos.get("loss_dict")
        if loss_dict is None:
            loss_dict = self.last_loss_dict
        saved_dict = {
            "model_state_dict": self.policy.state_dict(),
            "discriminator_state_dict": self.discriminator_state_dict,
            "amp_normalizer": self.amp_normalizer,
            "iter": self.current_learning_iteration,
            "infos": infos,
            "base_checkpoint_path": self.base_checkpoint_path,
            "history_length": self.history_length,
            "history_step_dim": self.history_step_dim,
            "track_adapter_stage": "world_model_pretrain",
            "world_model_pretrain": {
                "num_steps_per_env": self.num_steps_per_env,
                "num_learning_epochs": self.num_learning_epochs,
                "num_mini_batches": self.num_mini_batches,
                "learning_rate": self.learning_rate,
                "max_grad_norm": self.max_grad_norm,
                "collection_policy": "base_residual_zero",
                "last_loss_dict": loss_dict,
                "disturbance_manifest": self._disturbance_manifest(),
            },
            "world_model_metadata": self.world_model_metadata(),
            "world_model_target": self.world_model_metadata(),
            "pretrain_optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

    def load(self, path: str, load_optimizer: bool = False):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        if self.strict_base_auxiliary_state:
            ckpt_base_path = loaded_dict.get("base_checkpoint_path")
            if ckpt_base_path is None:
                raise RuntimeError("Track Adapter checkpoint is missing base_checkpoint_path metadata.")
            if os.path.abspath(ckpt_base_path) != os.path.abspath(self.base_checkpoint_path):
                raise RuntimeError(
                    "Track Adapter checkpoint base path mismatch: "
                    f"checkpoint={ckpt_base_path}, current={self.base_checkpoint_path}"
                )
            ckpt_history_length = loaded_dict.get("history_length")
            if ckpt_history_length is None:
                raise RuntimeError("Track Adapter checkpoint is missing history_length metadata.")
            if int(ckpt_history_length) != int(self.history_length):
                raise RuntimeError(
                    "Track Adapter checkpoint history length mismatch: "
                    f"checkpoint={ckpt_history_length}, current={self.history_length}"
                )
            ckpt_metadata = loaded_dict.get("world_model_metadata")
            if ckpt_metadata is not None:
                current_metadata = self.world_model_metadata()
                for key in (
                    "source",
                    "target_dim",
                    "target_groups",
                    "target_slices",
                    "horizon",
                    "prediction_mode",
                    "state_schema",
                    "state_dim",
                ):
                    if ckpt_metadata.get(key) != current_metadata.get(key):
                        raise RuntimeError(
                            f"Track Adapter checkpoint world-model metadata mismatch for {key}: "
                            f"checkpoint={ckpt_metadata.get(key)}, current={current_metadata.get(key)}"
                        )

        self.policy.load_state_dict(loaded_dict["model_state_dict"], strict=self.strict_base_auxiliary_state)
        if loaded_dict.get("discriminator_state_dict") is not None:
            self.discriminator_state_dict = loaded_dict["discriminator_state_dict"]
        elif self.strict_base_auxiliary_state:
            raise RuntimeError("Track Adapter checkpoint is missing discriminator_state_dict.")
        if loaded_dict.get("amp_normalizer") is not None:
            self.amp_normalizer = loaded_dict["amp_normalizer"]
        elif self.strict_base_auxiliary_state:
            raise RuntimeError("Track Adapter checkpoint is missing amp_normalizer.")
        if self.empirical_normalization:
            if "obs_norm_state_dict" in loaded_dict:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            elif self.strict_base_auxiliary_state:
                raise RuntimeError("Track Adapter checkpoint is missing obs_norm_state_dict.")
            if "privileged_obs_norm_state_dict" in loaded_dict:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            elif self.strict_base_auxiliary_state:
                raise RuntimeError("Track Adapter checkpoint is missing privileged_obs_norm_state_dict.")
        if load_optimizer and "pretrain_optimizer_state_dict" in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict["pretrain_optimizer_state_dict"])
        self.current_learning_iteration = int(loaded_dict.get("iter", 0))
        self.eval_mode()
        return loaded_dict.get("infos")

    def train_mode(self):
        self.policy.train()
        self.policy.base_actor_critic.eval()
        if self.empirical_normalization and self.freeze_loaded_normalizers:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def eval_mode(self):
        self.policy.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

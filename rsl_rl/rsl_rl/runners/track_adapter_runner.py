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
import statistics
import time
import math
from collections import deque

import torch

import rsl_rl
from rsl_rl.algorithms import TrackAdapterAMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import Discriminator, EmpiricalNormalization, TrackAdapterActorCritic
from rsl_rl.utils import AMPLoader, Normalizer, store_code_state


def _env_flag(name: str) -> bool:
    return os.getenv(name, "0").lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else float(value)


def _full_training_unlocked() -> bool:
    return (
        _env_flag("BOOSTER_TRACK_ADAPTER_FULL_TRAINING")
        and _env_flag("BOOSTER_TRACK_ADAPTER_SMOKE_PASSED")
        and _env_flag("BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING")
    )


class TrackAdapterRunner:
    """AMP runner for a frozen base actor plus trainable history residual adapter."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.history_length = int(train_cfg.get("history_length", self.policy_cfg.get("history_length", 20)))
        self.freeze_loaded_normalizers = bool(train_cfg.get("freeze_loaded_normalizers", True))
        self.base_checkpoint_path = train_cfg.get(
            "base_checkpoint_path", self.policy_cfg.get("base_checkpoint_path", None)
        )
        if not self.base_checkpoint_path:
            raise ValueError("TrackAdapterRunner requires a frozen AMP base_checkpoint_path.")
        self.strict_base_auxiliary_state = bool(train_cfg.get("strict_base_auxiliary_state", True))
        self.allow_full_training = bool(train_cfg.get("allow_full_training", False)) and _full_training_unlocked()
        self.pretrained_world_model_path = train_cfg.get("pretrained_world_model_path") or os.getenv(
            "BOOSTER_TRACK_ADAPTER_PRETRAINED_WM"
        )
        self.adapter_warm_start_path = train_cfg.get("adapter_warm_start_path") or os.getenv(
            "BOOSTER_TRACK_ADAPTER_ADAPTER_WARM_START"
        )
        self.pretrained_world_model_loaded = False
        self.pretrained_world_model_metadata = None
        self.pretrained_world_model_pretrain = None
        self.pretrained_world_model_iter = None
        self.max_iterations_without_full_training_unlock = int(
            train_cfg.get("max_iterations_without_full_training_unlock", 64)
        )
        self.safety_gate_cfg = dict(train_cfg.get("safety_gate", {}))
        self._safety_gate_strikes = {}

        self.recovery_gated_amp_enabled = bool(train_cfg.get("recovery_gated_amp_enabled", True))
        self.reward_group_names = list(train_cfg.get("reward_group_names", ["task", "style", "recovery", "reg"]))
        self.reward_group_weights = {str(k): float(v) for k, v in train_cfg.get("reward_group_weights", {}).items()}
        self.reward_group_term_prefixes = {
            str(k): tuple(v) for k, v in train_cfg.get("reward_group_term_prefixes", {}).items()
        }
        self.reward_group_term_names = {
            str(k): set(v) for k, v in train_cfg.get("reward_group_term_names", {}).items()
        }
        self.reward_group_default = str(train_cfg.get("reward_group_default", "task"))
        self.disturbance_gate_cfg = dict(train_cfg.get("disturbance_gate", {}))
        self.recovery_command_override_cfg = dict(train_cfg.get("recovery_command_override", {}))
        self.terminal_penalty_cfg = dict(train_cfg.get("terminal_penalty", {}))
        self.residual_penalty_cfg = dict(
            train_cfg.get(
                "residual_penalty_cfg",
                train_cfg.get("residual_penalty", self.alg_cfg.get("residual_penalty_cfg", {})),
            )
        )
        self.residual_action_gate_cfg = dict(train_cfg.get("residual_action_gate", {}))
        self.recovery_controller_cfg = dict(train_cfg.get("recovery_controller", {}))
        self._recovery_phase = None
        self._recovery_phase_time_s = None
        self._recovery_stable_time_s = None
        self._recovery_enter_time_s = None
        self._recovery_command_blend = None
        self._recovery_last_command = None
        self._last_disturbance_gate_info = {}
        self._last_rollout_adapter_diagnostics = {}
        self._rollout_adapter_diag_sums = {}
        self._rollout_adapter_diag_count = 0
        self._runner_scalars = {}
        self.failure_reset_enabled = _env_flag("BOOSTER_TRACK_ADAPTER_FAILURE_RESET_CURRICULUM")
        self.failure_reset_probability = _env_float("BOOSTER_TRACK_ADAPTER_FAILURE_RESET_PROB", 0.0)
        self.failure_reset_delay_steps = _env_int("BOOSTER_TRACK_ADAPTER_FAILURE_RESET_DELAY_STEPS", 6)
        self.failure_reset_history_steps = max(
            self.failure_reset_delay_steps + 2,
            _env_int("BOOSTER_TRACK_ADAPTER_FAILURE_RESET_HISTORY_STEPS", 32),
        )
        self.failure_reset_buffer_size = _env_int("BOOSTER_TRACK_ADAPTER_FAILURE_RESET_BUFFER_SIZE", 4096)
        self.failure_reset_min_samples = _env_int("BOOSTER_TRACK_ADAPTER_FAILURE_RESET_MIN_SAMPLES", 64)
        self.failure_reset_replay_force = _env_flag("BOOSTER_TRACK_ADAPTER_FAILURE_RESET_REPLAY_FORCE")
        self.failure_reset_force_duration_s = _env_float("BOOSTER_TRACK_ADAPTER_FAILURE_RESET_FORCE_DURATION_S", 0.7)
        self._failure_reset_history_cursor = 0
        self._failure_reset_history = {}
        self._failure_reset_replay = {}
        self._failure_reset_replay_step = 0
        self._failure_reset_replay_size = 0
        self.phase_trace_enabled = _env_flag("BOOSTER_TRACK_ADAPTER_PHASE_TRACE")
        self.phase_trace_step_interval = max(1, _env_int("BOOSTER_TRACK_ADAPTER_PHASE_TRACE_STEP_INTERVAL", 10))

        self._configure_multi_gpu()

        if self.alg_cfg["class_name"] in ["TrackAdapterAMPPPO"]:
            self.training_type = "rl"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        self.history_step_dim = num_obs + self.env.num_actions
        self.privileged_obs_type = "critic" if "critic" in extras["observations"] else None
        num_privileged_obs = (
            extras["observations"][self.privileged_obs_type].shape[1]
            if self.privileged_obs_type is not None
            else num_obs
        )

        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            self.alg_cfg["rnd_cfg"]["num_states"] = rnd_state.shape[1]
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        policy_class_name = self.policy_cfg.pop("class_name")
        if policy_class_name != "TrackAdapterActorCritic":
            raise ValueError(f"TrackAdapterRunner requires TrackAdapterActorCritic, got {policy_class_name}.")
        self.policy_cfg.pop("history_length", None)
        self.policy_cfg["history_length"] = self.history_length
        self.policy_cfg["history_step_dim"] = self.history_step_dim
        self.policy_cfg["history_dim"] = self.history_length * self.history_step_dim
        policy: TrackAdapterActorCritic = TrackAdapterActorCritic(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        amp_data = AMPLoader(
            device,
            time_between_frames=self._base_env.step_dt,
            preload_transitions=True,
            num_preload_transitions=train_cfg["amp_num_preload_transitions"],
            motion_files=train_cfg["amp_motion_files"],
        )
        print(self._base_env.step_dt)

        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = Discriminator(
            amp_data.observation_dim * 2,
            train_cfg["amp_reward_coef"],
            train_cfg["amp_discr_hidden_dims"],
            device,
            train_cfg["amp_task_reward_lerp"],
        ).to(self.device)
        min_std = torch.zeros(len(train_cfg["min_normalized_std"]), device=self.device, requires_grad=False)

        alg_class_name = self.alg_cfg.pop("class_name")
        if alg_class_name != "TrackAdapterAMPPPO":
            raise ValueError(f"TrackAdapterRunner requires TrackAdapterAMPPPO, got {alg_class_name}.")
        self.alg: TrackAdapterAMPPPO = TrackAdapterAMPPPO(
            policy,
            discriminator,
            amp_data,
            amp_normalizer,
            device=self.device,
            min_std=min_std,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        self.world_model_state_schema = self._build_world_model_state_schema(num_obs)
        self.world_model_state_dim = sum(width for _, width in self.world_model_state_schema)
        self._world_model_foot_body_ids = None

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

        self._load_base_auxiliary_state()
        self._load_world_model_pretrain()
        self._load_adapter_warm_start()
        self.alg.obs_normalizer = self.obs_normalizer
        self.alg.privileged_obs_normalizer = self.privileged_obs_normalizer

        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
            [self.history_length, self.history_step_dim],
            [self.world_model_state_dim] if self._world_model_uses_structured_state() else None,
        )

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        self._disturbance_contact_body_ids = self._resolve_disturbance_contact_body_ids()
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self._safety_gate_iteration_offset = 0
        self.git_status_repos = [rsl_rl.__file__]

    @property
    def _base_env(self):
        return self.env.env.env

    def _trace_phase(self, phase: str, it: int | None = None, step: int | None = None):
        if not self.phase_trace_enabled:
            return
        parts = [f"[TrackAdapterTrace] phase={phase}", f"t={time.time():.3f}"]
        if it is not None:
            parts.append(f"it={it}")
        if step is not None:
            parts.append(f"step={step}")
        print(" ".join(parts), flush=True)

    def _load_base_auxiliary_state(self):
        if not self.base_checkpoint_path:
            raise ValueError("Track Adapter base checkpoint path is required.")
        if not os.path.exists(self.base_checkpoint_path):
            raise FileNotFoundError(f"Track Adapter base checkpoint not found: {self.base_checkpoint_path}")
        loaded = torch.load(self.base_checkpoint_path, map_location=self.device, weights_only=False)

        ckpt_disc = loaded.get("discriminator_state_dict")
        if ckpt_disc is not None:
            runtime_disc = self.alg.discriminator.state_dict()
            compatible = all(k in runtime_disc and runtime_disc[k].shape == v.shape for k, v in ckpt_disc.items())
            if compatible:
                self.alg.discriminator.load_state_dict(ckpt_disc)
            else:
                message = "[TrackAdapterRunner] Base discriminator shape mismatch."
                if self.strict_base_auxiliary_state:
                    raise RuntimeError(message)
                print(message + " Skipping load.")
        elif self.strict_base_auxiliary_state:
            raise RuntimeError("Base checkpoint is missing discriminator_state_dict.")

        ckpt_norm = loaded.get("amp_normalizer")
        if ckpt_norm is not None and getattr(ckpt_norm, "mean", None) is not None:
            if ckpt_norm.mean.shape == self.alg.amp_normalizer.mean.shape:
                self.alg.amp_normalizer = ckpt_norm
            else:
                message = "[TrackAdapterRunner] Base AMP normalizer shape mismatch."
                if self.strict_base_auxiliary_state:
                    raise RuntimeError(message)
                print(message + " Skipping load.")
        elif self.strict_base_auxiliary_state:
            raise RuntimeError("Base checkpoint is missing a compatible amp_normalizer.")

        if self.empirical_normalization:
            obs_state = loaded.get("obs_norm_state_dict")
            if obs_state is not None:
                try:
                    self.obs_normalizer.load_state_dict(obs_state)
                    if self.freeze_loaded_normalizers:
                        self.obs_normalizer.eval()
                except RuntimeError as exc:
                    message = "[TrackAdapterRunner] Observation normalizer shape mismatch."
                    if self.strict_base_auxiliary_state:
                        raise RuntimeError(message) from exc
                    print(message + " Skipping load.")
            elif self.strict_base_auxiliary_state:
                raise RuntimeError("Base checkpoint is missing obs_norm_state_dict.")
            priv_state = loaded.get("privileged_obs_norm_state_dict")
            if priv_state is not None:
                ckpt_mean = priv_state.get("_mean") if isinstance(priv_state, dict) else None
                runtime_mean = self.privileged_obs_normalizer.state_dict().get("_mean")
                ckpt_shape = tuple(ckpt_mean.shape) if getattr(ckpt_mean, "shape", None) is not None else None
                runtime_shape = tuple(runtime_mean.shape) if getattr(runtime_mean, "shape", None) is not None else None
                if ckpt_shape is not None and runtime_shape is not None and ckpt_shape != runtime_shape:
                    message = (
                        "[TrackAdapterRunner] Base privileged observation normalizer shape mismatch "
                        f"(checkpoint={ckpt_shape}, current={runtime_shape})."
                    )
                    if self.strict_base_auxiliary_state:
                        raise RuntimeError(message)
                    if _env_flag("BOOSTER_TRACK_ADAPTER_VERBOSE_NORMALIZER_MISMATCH"):
                        print(message + " Skipping base normalizer load.")
                else:
                    try:
                        self.privileged_obs_normalizer.load_state_dict(priv_state)
                    except RuntimeError as exc:
                        message = "[TrackAdapterRunner] Base privileged observation normalizer shape mismatch."
                        if self.strict_base_auxiliary_state:
                            raise RuntimeError(message) from exc
                        if _env_flag("BOOSTER_TRACK_ADAPTER_VERBOSE_NORMALIZER_MISMATCH"):
                            print(message + " Skipping base normalizer load.")
            elif self.strict_base_auxiliary_state:
                raise RuntimeError("Base checkpoint is missing privileged_obs_norm_state_dict.")

    def _resolve_disturbance_contact_body_ids(self):
        cfg = self.disturbance_gate_cfg
        if float(cfg.get("contact_weight", 0.0)) <= 0.0:
            return []
        sensor_name = cfg.get("contact_sensor_name", "contact_forces")
        if sensor_name not in self._base_env.scene.sensors:
            raise RuntimeError(f"Track Adapter contact gate sensor not found: {sensor_name}")
        contact_sensor = self._base_env.scene.sensors[sensor_name]
        body_ids = cfg.get("contact_body_ids")
        if body_ids is not None:
            body_ids = [int(body_id) for body_id in body_ids]
        else:
            body_patterns = list(cfg.get("contact_body_names", ["Trunk"]))
            sensor_body_names = list(getattr(contact_sensor, "body_names", []))
            if not sensor_body_names:
                raise RuntimeError(
                    f"Track Adapter contact gate cannot resolve body names for contact sensor {sensor_name}."
                )
            body_ids = [
                index
                for index, name in enumerate(sensor_body_names)
                if any(pattern in name for pattern in body_patterns)
            ]
        if not body_ids:
            raise RuntimeError(
                "Track Adapter contact gate resolved no bodies for "
                f"sensor={sensor_name}, patterns={cfg.get('contact_body_names', ['Trunk'])}."
            )
        return body_ids

    def _world_model_target_source_name(self):
        source = self.alg.world_model_target_source
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

    def _resolve_world_model_foot_body_ids(self):
        if self._world_model_foot_body_ids is not None:
            return self._world_model_foot_body_ids
        sensor_name = self.alg.world_model_cfg.get("contact_sensor_name", "contact_forces")
        if sensor_name not in self._base_env.scene.sensors:
            raise RuntimeError(f"Track Adapter world-model contact sensor not found: {sensor_name}")
        contact_sensor = self._base_env.scene.sensors[sensor_name]
        body_names = list(getattr(contact_sensor, "body_names", []))
        if not body_names:
            raise RuntimeError(f"Track Adapter world-model cannot resolve body names for contact sensor {sensor_name}.")
        pattern = self.alg.world_model_cfg.get("foot_body_pattern", r".*_foot_.*")
        foot_ids = [
            index
            for index, name in enumerate(body_names)
            if re.fullmatch(pattern, name) or re.search(pattern, name)
        ]
        if not foot_ids:
            raise RuntimeError(
                f"Track Adapter world-model resolved no foot contact bodies for sensor={sensor_name}, pattern={pattern}."
            )
        self._world_model_foot_body_ids = foot_ids
        return foot_ids

    def compute_world_model_state(self, normalized_actor_obs):
        normalized_actor_obs = normalized_actor_obs.to(self.device)
        robot = self._base_env.scene["robot"]
        base_lin_vel_b = robot.data.root_lin_vel_b.to(self.device)
        base_height = robot.data.root_pos_w[:, 2:3].to(self.device)
        foot_ids = self._resolve_world_model_foot_body_ids()
        contact_sensor = self._base_env.scene.sensors[self.alg.world_model_cfg.get("contact_sensor_name", "contact_forces")]
        forces = contact_sensor.data.net_forces_w_history.to(self.device)
        contact_norm = torch.max(torch.norm(forces[:, :, foot_ids], dim=-1), dim=1)[0]
        threshold = float(self.alg.world_model_cfg.get("foot_contact_threshold", 1.0))
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

    def world_model_metadata(self):
        return {
            "source": self._world_model_target_source_name(),
            "target_dim": self.alg.world_model_target_dim,
            "target_groups": self.alg.world_model_target_groups,
            "target_slices": self.alg.world_model_target_slices,
            "horizon": self.alg.world_model_horizon,
            "prediction_mode": getattr(self.alg.policy, "world_model_prediction_mode", "autoregressive"),
            "state_schema": list(self.world_model_state_schema),
            "state_dim": self.world_model_state_dim,
            "normalized_actor_obs": True,
        }

    def _validate_pretrain_metadata(self, loaded: dict, path: str):
        if loaded.get("track_adapter_stage") != "world_model_pretrain":
            raise RuntimeError(
                "Track Adapter pretrain path must point to a Stage 3 world-model pretrain checkpoint, "
                f"got stage={loaded.get('track_adapter_stage')} from {path}."
            )
        if "optimizer_state_dict" in loaded or "pretrain_optimizer_state_dict" not in loaded:
            raise RuntimeError(
                "Track Adapter pretrain checkpoint must contain pretrain_optimizer_state_dict and must not be a PPO "
                f"resume checkpoint: {path}"
            )
        pretrain_info = loaded.get("world_model_pretrain")
        if not isinstance(pretrain_info, dict):
            raise RuntimeError(f"Track Adapter pretrain checkpoint is missing world_model_pretrain metadata: {path}")
        if pretrain_info.get("collection_policy") != "base_residual_zero":
            raise RuntimeError(
                "Track Adapter pretrain checkpoint was not collected with the frozen base-only policy: "
                f"{pretrain_info.get('collection_policy')}"
            )
        loss_dict = pretrain_info.get("last_loss_dict")
        infos = loaded.get("infos")
        if loss_dict is None and isinstance(infos, dict):
            loss_dict = infos.get("loss_dict")
        if not isinstance(loss_dict, dict):
            raise RuntimeError(f"Track Adapter pretrain checkpoint is missing finite loss_dict metadata: {path}")
        wm_loss = float(loss_dict.get("wm", float("nan")))
        valid_fraction = float(loss_dict.get("wm_valid_fraction", float("nan")))
        if not math.isfinite(wm_loss) or not math.isfinite(valid_fraction) or valid_fraction <= 0.0:
            raise RuntimeError(
                "Track Adapter pretrain checkpoint has invalid world-model metrics: "
                f"wm={wm_loss}, wm_valid_fraction={valid_fraction}"
            )
        if self.allow_full_training:
            manifest = pretrain_info.get("disturbance_manifest")
            if not isinstance(manifest, dict) or not manifest.get("available", False):
                raise RuntimeError("Track Adapter full training requires pretrain disturbance manifest metadata.")
            required_manifest_flags = (
                "material_randomization_enabled",
                "base_mass_randomization_enabled",
                "base_com_randomization_enabled",
                "sudden_stop_enabled",
            )
            missing = [name for name in required_manifest_flags if not manifest.get(name, False)]
            if not manifest.get("push_enabled", False) and not manifest.get("force_push_enabled", False):
                missing.append("push_enabled_or_force_push_enabled")
            if missing:
                raise RuntimeError(
                    "Track Adapter full training requires disturbance pretraining coverage; missing: "
                    + ", ".join(missing)
                )
        ckpt_base_path = loaded.get("base_checkpoint_path")
        if ckpt_base_path is None:
            raise RuntimeError(f"Track Adapter world-model pretrain checkpoint is missing base_checkpoint_path: {path}")
        if os.path.abspath(ckpt_base_path) != os.path.abspath(self.base_checkpoint_path):
            raise RuntimeError(
                "Track Adapter world-model pretrain base path mismatch: "
                f"checkpoint={ckpt_base_path}, current={self.base_checkpoint_path}"
            )
        ckpt_history_length = loaded.get("history_length")
        if ckpt_history_length is None:
            raise RuntimeError(f"Track Adapter world-model pretrain checkpoint is missing history_length: {path}")
        if int(ckpt_history_length) != int(self.history_length):
            raise RuntimeError(
                "Track Adapter world-model pretrain history length mismatch: "
                f"checkpoint={ckpt_history_length}, current={self.history_length}"
            )
        ckpt_metadata = loaded.get("world_model_metadata")
        if ckpt_metadata is None:
            raise RuntimeError(f"Track Adapter world-model pretrain checkpoint is missing world_model_metadata: {path}")
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
                    f"Track Adapter world-model pretrain metadata mismatch for {key}: "
                    f"checkpoint={ckpt_metadata.get(key)}, current={current_metadata.get(key)}"
                )

    def _load_world_model_pretrain(self):
        if not self.pretrained_world_model_path:
            return
        if not os.path.exists(self.pretrained_world_model_path):
            raise FileNotFoundError(
                f"Track Adapter world-model pretrain checkpoint not found: {self.pretrained_world_model_path}"
            )
        loaded = torch.load(self.pretrained_world_model_path, map_location=self.device, weights_only=False)
        self._validate_pretrain_metadata(loaded, self.pretrained_world_model_path)
        state_dict = loaded.get("model_state_dict", loaded)
        runtime_state = self.alg.policy.state_dict()
        prefixes = ("history_encoder.", "world_model_predictor.")
        pretrain_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(prefixes)
        }
        expected_pretrain_keys = [key for key in runtime_state if key.startswith(prefixes)]
        missing_pretrain_keys = sorted(set(expected_pretrain_keys) - set(pretrain_state))
        if not pretrain_state or missing_pretrain_keys:
            raise RuntimeError(
                "Track Adapter world-model pretrain checkpoint is missing history/world-model state keys: "
                + str(missing_pretrain_keys[:8])
                + (" ..." if len(missing_pretrain_keys) > 8 else "")
            )
        incompatible = [
            key
            for key, value in pretrain_state.items()
            if key not in runtime_state or runtime_state[key].shape != value.shape
        ]
        if incompatible:
            raise RuntimeError(
                "Track Adapter world-model pretrain shape mismatch for keys: "
                + str(incompatible[:8])
                + (" ..." if len(incompatible) > 8 else "")
            )
        runtime_state.update(pretrain_state)
        self.alg.policy.load_state_dict(runtime_state, strict=True)
        self.pretrained_world_model_loaded = True
        self.pretrained_world_model_metadata = loaded.get("world_model_metadata")
        self.pretrained_world_model_pretrain = loaded.get("world_model_pretrain")
        self.pretrained_world_model_iter = loaded.get("iter")
        print(f"[TrackAdapterRunner] Loaded world-model pretrain: {self.pretrained_world_model_path}")

    def _load_adapter_warm_start(self):
        if not self.adapter_warm_start_path:
            return
        if not os.path.exists(self.adapter_warm_start_path):
            raise FileNotFoundError(f"Track Adapter warm-start checkpoint not found: {self.adapter_warm_start_path}")
        loaded = torch.load(self.adapter_warm_start_path, map_location=self.device, weights_only=False)
        ckpt_base_path = loaded.get("base_checkpoint_path")
        if ckpt_base_path is not None and os.path.abspath(ckpt_base_path) != os.path.abspath(self.base_checkpoint_path):
            raise RuntimeError(
                "Track Adapter warm-start checkpoint base path mismatch: "
                f"checkpoint={ckpt_base_path}, current={self.base_checkpoint_path}"
            )
        state_dict = loaded.get("model_state_dict", loaded)
        current = self.alg.policy.state_dict()
        default_prefixes = "layer_adapters.,residual_actor."
        prefixes = tuple(
            value.strip()
            for value in os.getenv("BOOSTER_TRACK_ADAPTER_ADAPTER_WARM_START_PREFIXES", default_prefixes).split(",")
            if value.strip()
        )
        warm_state = {}
        skipped = []
        for key, value in state_dict.items():
            if not key.startswith(prefixes):
                continue
            if key not in current or current[key].shape != value.shape:
                skipped.append(key)
                continue
            warm_state[key] = value.to(device=current[key].device, dtype=current[key].dtype)
        if not warm_state:
            raise RuntimeError(
                f"Track Adapter warm-start checkpoint loaded no compatible adapter keys: {self.adapter_warm_start_path}"
            )
        current.update(warm_state)
        self.alg.policy.load_state_dict(current, strict=True)
        print(
            "[TrackAdapterRunner] Warm-started adapter keys "
            f"from {self.adapter_warm_start_path}: loaded={len(warm_state)}, skipped={len(skipped)}"
        )

    def _reward_manager_step_terms(self, fallback_rewards):
        reward_manager = self._base_env.reward_manager
        term_names = list(getattr(reward_manager, "active_terms", []))
        step_reward = getattr(reward_manager, "_step_reward", None)
        if step_reward is None or len(term_names) == 0:
            return [], None
        return term_names, step_reward.to(self.device) * self._base_env.step_dt

    def _split_reward_groups(self, fallback_rewards):
        term_names, step_reward = self._reward_manager_step_terms(fallback_rewards)
        grouped = {name: torch.zeros_like(fallback_rewards) for name in self.reward_group_names}
        grouped.setdefault(self.reward_group_default, torch.zeros_like(fallback_rewards))
        if step_reward is None:
            grouped[self.reward_group_default] = fallback_rewards
            return grouped

        assigned = torch.zeros(len(term_names), device=self.device, dtype=torch.bool)
        for group_name in self.reward_group_names:
            explicit_names = self.reward_group_term_names.get(group_name, set())
            prefixes = self.reward_group_term_prefixes.get(group_name, ())
            mask = torch.tensor(
                [name in explicit_names or name.startswith(prefixes) for name in term_names],
                device=self.device,
                dtype=torch.bool,
            )
            if torch.any(mask):
                grouped[group_name] = grouped[group_name] + step_reward[:, mask].sum(dim=1)
                assigned |= mask
        if torch.any(~assigned):
            grouped[self.reward_group_default] = grouped[self.reward_group_default] + step_reward[:, ~assigned].sum(
                dim=1
            )
        return grouped

    def _compute_disturbance_gates(self):
        cfg = self.disturbance_gate_cfg
        command_name = cfg.get("command_name", "base_velocity")
        yaw_scale = float(cfg.get("yaw_scale", 0.30))
        target_height = float(cfg.get("target_height", 0.57))
        beta = float(cfg.get("beta", 1.0))
        min_style_gate = float(cfg.get("min_style_gate", 0.0))
        command_min_weight = float(cfg.get("command_min_weight", 0.35))

        robot = self._base_env.scene["robot"]
        command = self._base_env.command_manager.get_command(command_name).to(self.device)
        projected_gravity = robot.data.projected_gravity_b.to(self.device)
        root_ang_vel_b = robot.data.root_ang_vel_b.to(self.device)
        root_lin_vel_b = robot.data.root_lin_vel_b.to(self.device)
        root_pos_w = robot.data.root_pos_w.to(self.device)

        tilt = torch.norm(projected_gravity[:, :2], dim=1)
        omega_xy = torch.norm(root_ang_vel_b[:, :2], dim=1)
        actual = torch.stack([root_lin_vel_b[:, 0], root_lin_vel_b[:, 1], yaw_scale * root_ang_vel_b[:, 2]], dim=-1)
        target = torch.stack([command[:, 0], command[:, 1], yaw_scale * command[:, 2]], dim=-1)
        velocity_error = torch.norm(actual - target, dim=1)
        height_error = torch.abs(root_pos_w[:, 2] - target_height)
        contact_score = torch.zeros_like(height_error)
        contact_weight = float(cfg.get("contact_weight", 0.0))
        if contact_weight > 0.0:
            contact_sensor = self._base_env.scene.sensors[cfg.get("contact_sensor_name", "contact_forces")]
            forces = contact_sensor.data.net_forces_w_history.to(self.device)
            contact_norm = torch.max(torch.norm(forces[:, :, self._disturbance_contact_body_ids], dim=-1), dim=1)[0]
            contact_score = torch.any(contact_norm > float(cfg.get("contact_threshold", 1.0)), dim=1).float()

        score = (
            float(cfg.get("tilt_weight", 1.8)) * tilt
            + float(cfg.get("ang_vel_weight", 0.25)) * omega_xy
            + float(cfg.get("velocity_weight", 0.75)) * velocity_error
            + float(cfg.get("height_weight", 3.0)) * height_error
            + contact_weight * contact_score
        )
        score = torch.nan_to_num(score, nan=8.0, posinf=8.0, neginf=0.0).clamp(0.0, float(cfg.get("max_score", 8.0)))
        style_gate = torch.exp(-beta * score).clamp(min_style_gate, 1.0)

        zero_command_scale = self.cfg.get("amp_zero_command_scale", 1.0)
        if zero_command_scale != 1.0:
            threshold = self.cfg.get("amp_zero_command_threshold", 1.0e-6)
            zero_command = torch.norm(command, dim=1) < threshold
            style_gate = style_gate.clone()
            style_gate[zero_command] *= float(zero_command_scale)
        raw_style_gate = style_gate.clone()

        push_active = torch.zeros_like(style_gate)
        push_recovery_gate_min = float(cfg.get("push_recovery_gate_min", 0.0))
        push_style_gate_max = cfg.get("push_style_gate_max")
        push_command_gate_min = float(cfg.get("push_command_gate_min", 0.0))
        push_command_gate_max = float(cfg.get("push_command_gate_max", 1.0))
        push_command_gate_late_max = float(cfg.get("push_command_gate_late_max", push_command_gate_max))
        push_command_gate_full_s = float(cfg.get("push_command_gate_full_s", 0.0))
        push_command_gate_taper_s = float(cfg.get("push_command_gate_taper_s", push_command_gate_full_s))
        if push_command_gate_max < push_command_gate_min:
            raise RuntimeError(
                "Track Adapter push command gate max must be greater than or equal to "
                f"push command gate min ({push_command_gate_max} < {push_command_gate_min})."
            )
        if push_command_gate_late_max < push_command_gate_min:
            raise RuntimeError(
                "Track Adapter push command gate late max must be greater than or equal to "
                f"push command gate min ({push_command_gate_late_max} < {push_command_gate_min})."
            )
        if (
            push_recovery_gate_min > 0.0
            or push_style_gate_max is not None
            or push_command_gate_min > 0.0
            or push_command_gate_max < 1.0
            or push_command_gate_late_max < 1.0
        ):
            try:
                command_term = self._base_env.command_manager.get_term(command_name)
                metric_name = cfg.get("push_active_metric_name", "failure_push_active")
                metric = command_term.metrics.get(metric_name) if hasattr(command_term, "metrics") else None
            except Exception as exc:
                raise RuntimeError(
                    f"Track Adapter push-active gates require command term '{command_name}' "
                    "to expose push activity metrics."
                ) from exc
            if metric is not None:
                push_active = metric.to(self.device, dtype=style_gate.dtype).clamp(0.0, 1.0)
                if push_active.shape != style_gate.shape:
                    raise RuntimeError(
                        f"Push-active metric shape {tuple(push_active.shape)} does not match "
                        f"num envs shape {tuple(style_gate.shape)}."
                    )
                if push_style_gate_max is not None:
                    style_gate = torch.where(
                        push_active > 0.0,
                        torch.minimum(style_gate, torch.full_like(style_gate, float(push_style_gate_max))),
                        style_gate,
                    )
            else:
                raise RuntimeError(
                    f"Track Adapter push-active gates require metric '{metric_name}' on command term '{command_name}'."
                )

        recovery_gate = (1.0 - style_gate).clamp(0.0, 1.0)
        raw_recovery_gate = (1.0 - raw_style_gate).clamp(0.0, 1.0)
        if push_recovery_gate_min > 0.0:
            recovery_gate = torch.maximum(recovery_gate, push_active * push_recovery_gate_min)
        command_gate = command_min_weight + (1.0 - command_min_weight) * style_gate
        command_gate_cap = None
        if push_command_gate_max < 1.0 or push_command_gate_late_max < 1.0:
            command_gate_cap = torch.full_like(command_gate, push_command_gate_max)
            if push_command_gate_late_max != push_command_gate_max:
                try:
                    command_term = self._base_env.command_manager.get_term(command_name)
                    elapsed_metric = command_term.metrics.get("failure_push_elapsed_s") if hasattr(command_term, "metrics") else None
                except Exception as exc:
                    raise RuntimeError(
                        f"Track Adapter phase-aware push command gate requires command term '{command_name}' "
                        "to expose push elapsed metrics."
                    ) from exc
                if not isinstance(elapsed_metric, torch.Tensor):
                    raise RuntimeError(
                        "Track Adapter phase-aware push command gate requires metric "
                        f"'failure_push_elapsed_s' on command term '{command_name}'."
                    )
                push_elapsed = elapsed_metric.to(self.device, dtype=command_gate.dtype)
                if push_elapsed.shape != command_gate.shape:
                    raise RuntimeError(
                        f"Push elapsed metric shape {tuple(push_elapsed.shape)} does not match "
                        f"num envs shape {tuple(command_gate.shape)}."
                    )
                if push_command_gate_taper_s > push_command_gate_full_s:
                    phase = (
                        (push_elapsed - push_command_gate_full_s)
                        / max(push_command_gate_taper_s - push_command_gate_full_s, 1.0e-6)
                    ).clamp(0.0, 1.0)
                else:
                    phase = (push_elapsed >= push_command_gate_full_s).to(command_gate.dtype)
                command_gate_cap = command_gate_cap + phase * (push_command_gate_late_max - push_command_gate_max)
                command_gate_cap = command_gate_cap.clamp(0.0, 1.0)
            capped_command_gate = torch.minimum(
                command_gate,
                command_gate_cap,
            )
            command_gate = torch.where(push_active > 0.0, capped_command_gate, command_gate)
        if push_command_gate_min > 0.0:
            command_gate = torch.maximum(command_gate, push_active * push_command_gate_min)

        self._last_disturbance_gate_info = {
            "score": score.detach(),
            "push_active": push_active.detach(),
            "command": command.detach(),
            "raw_style_gate": raw_style_gate.detach(),
            "raw_recovery_gate": raw_recovery_gate.detach(),
        }
        style_gate, recovery_gate, command_gate = self._update_recovery_controller_gates(
            style_gate, recovery_gate, command_gate
        )
        self._runner_scalars.update(
            {
                "Gate/style_mean": style_gate.mean().item(),
                "Gate/recovery_mean": recovery_gate.mean().item(),
                "Gate/command_mean": command_gate.mean().item(),
                "Gate/push_active_mean": push_active.mean().item(),
                "Gate/push_command_cap_mean": command_gate_cap.mean().item() if command_gate_cap is not None else 1.0,
                "Disturbance/score_mean": score.mean().item(),
                "Disturbance/score_max": score.max().item(),
                "Disturbance/tilt_mean": tilt.mean().item(),
                "Disturbance/velocity_error_mean": velocity_error.mean().item(),
                "Disturbance/contact_mean": contact_score.mean().item(),
            }
        )
        return style_gate, recovery_gate, command_gate

    def _recovery_controller_enabled(self) -> bool:
        return bool(self.recovery_controller_cfg.get("enabled", False))

    def _ensure_recovery_controller_state(self, reference: torch.Tensor):
        if not self._recovery_controller_enabled():
            return
        num_envs = int(reference.shape[0])
        device = reference.device
        needs_init = (
            self._recovery_phase is None
            or self._recovery_phase.shape[0] != num_envs
            or self._recovery_phase.device != device
        )
        if not needs_init:
            return
        self._recovery_phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._recovery_phase_time_s = torch.zeros(num_envs, device=device, dtype=reference.dtype)
        self._recovery_stable_time_s = torch.zeros(num_envs, device=device, dtype=reference.dtype)
        self._recovery_enter_time_s = torch.zeros(num_envs, device=device, dtype=reference.dtype)
        self._recovery_command_blend = torch.zeros(num_envs, device=device, dtype=reference.dtype)
        self._recovery_last_command = torch.zeros(num_envs, 3, device=device, dtype=reference.dtype)

    def _reset_recovery_controller_state(self, dones: torch.Tensor | None = None):
        if not self._recovery_controller_enabled() or self._recovery_phase is None:
            return
        if dones is None:
            reset_ids = torch.arange(self._recovery_phase.shape[0], device=self._recovery_phase.device)
        else:
            reset_ids = (dones.to(self._recovery_phase.device).reshape(-1) > 0).nonzero(as_tuple=False).flatten()
        if reset_ids.numel() == 0:
            return
        self._recovery_phase[reset_ids] = 0
        self._recovery_phase_time_s[reset_ids] = 0.0
        self._recovery_stable_time_s[reset_ids] = 0.0
        self._recovery_enter_time_s[reset_ids] = 0.0
        self._recovery_command_blend[reset_ids] = 0.0
        self._recovery_last_command[reset_ids] = 0.0

    def _update_recovery_controller_gates(self, style_gate, recovery_gate, command_gate):
        if not self._recovery_controller_enabled():
            return style_gate, recovery_gate, command_gate
        self._ensure_recovery_controller_state(style_gate)
        cfg = self.recovery_controller_cfg
        command_name = self.disturbance_gate_cfg.get("command_name", "base_velocity")
        score = self._last_disturbance_gate_info.get("score")
        if not isinstance(score, torch.Tensor) or score.shape != style_gate.shape:
            score = recovery_gate.detach()
        score = score.to(self.device, dtype=style_gate.dtype)

        external_active, sudden_stop_active = self._command_recovery_activity(command_name, style_gate)
        active_threshold = float(cfg.get("active_threshold", 0.05))
        enter_score = float(cfg.get("enter_score", 0.95))
        enter_recovery_gate = float(cfg.get("enter_recovery_gate", 0.45))
        exit_score = float(cfg.get("exit_score", 0.45))
        stable_score = float(cfg.get("stable_score", exit_score))
        min_recovery_s = float(cfg.get("min_recovery_s", 0.80))
        min_stable_s = float(cfg.get("min_stable_s", 0.25))
        return_stable_s = float(cfg.get("return_stable_s", 0.30))
        return_decay_s = max(float(cfg.get("return_decay_s", 1.00)), 1.0e-6)
        step_dt = float(getattr(self._base_env, "step_dt", 0.02))
        enter_delay_s = max(float(cfg.get("enter_delay_s", step_dt)), 0.0)

        phase = self._recovery_phase
        phase_time = self._recovery_phase_time_s + step_dt
        enter_condition = (
            (score >= enter_score)
            | (recovery_gate >= enter_recovery_gate)
            | (external_active > active_threshold)
        )
        stable_condition = (score <= stable_score) & (external_active <= active_threshold)
        enter_time = torch.where(
            enter_condition,
            self._recovery_enter_time_s + step_dt,
            torch.zeros_like(self._recovery_enter_time_s),
        )
        enter_signal = enter_condition & (enter_time >= enter_delay_s)
        stable_time = torch.where(
            stable_condition,
            self._recovery_stable_time_s + step_dt,
            torch.zeros_like(self._recovery_stable_time_s),
        )

        old_phase = phase.clone()
        next_phase = phase.clone()
        to_recovery = enter_signal & (phase != 1)
        next_phase = torch.where(to_recovery, torch.ones_like(next_phase), next_phase)
        phase_time = torch.where(to_recovery, torch.zeros_like(phase_time), phase_time)
        stable_time = torch.where(to_recovery, torch.zeros_like(stable_time), stable_time)

        to_return = (
            (next_phase == 1)
            & (~enter_condition)
            & (score <= exit_score)
            & (stable_time >= min_stable_s)
            & (phase_time >= min_recovery_s)
        )
        next_phase = torch.where(to_return, torch.full_like(next_phase, 2), next_phase)
        phase_time = torch.where(to_return, torch.zeros_like(phase_time), phase_time)

        to_nominal = (
            (next_phase == 2)
            & (~enter_condition)
            & (phase_time >= return_decay_s)
            & (stable_time >= return_stable_s)
        )
        next_phase = torch.where(to_nominal, torch.zeros_like(next_phase), next_phase)
        phase_time = torch.where(to_nominal, torch.zeros_like(phase_time), phase_time)
        stable_time = torch.where(to_nominal, torch.zeros_like(stable_time), stable_time)

        self._recovery_phase = next_phase
        self._recovery_phase_time_s = phase_time
        self._recovery_stable_time_s = stable_time
        self._recovery_enter_time_s = enter_time

        recovery_mask = next_phase == 1
        return_mask = next_phase == 2
        nominal_mask = next_phase == 0
        active_mask = recovery_mask | return_mask

        ramp_s = max(float(cfg.get("ramp_s", self.recovery_command_override_cfg.get("ramp_s", 0.20))), 1.0e-6)
        recovery_blend = (phase_time / ramp_s).clamp(0.0, 1.0)
        return_weight = (1.0 - phase_time / return_decay_s).clamp(0.0, 1.0)
        blend = torch.where(
            recovery_mask,
            recovery_blend,
            torch.where(return_mask, return_weight, torch.zeros_like(recovery_blend)),
        )
        self._recovery_command_blend = blend

        recovery_style_gate_max = float(cfg.get("recovery_style_gate_max", 0.05))
        return_style_floor = float(cfg.get("return_style_floor", 0.10))
        return_progress = (1.0 - return_weight).clamp(0.0, 1.0)
        return_style_cap = (return_style_floor + (1.0 - return_style_floor) * return_progress).clamp(0.0, 1.0)
        style_gate = torch.where(
            recovery_mask,
            torch.minimum(style_gate, torch.full_like(style_gate, recovery_style_gate_max)),
            style_gate,
        )
        style_gate = torch.where(return_mask, torch.minimum(style_gate, return_style_cap), style_gate)

        recovery_min_gate = float(cfg.get("recovery_gate_min", 0.85))
        return_min_gate = float(cfg.get("return_recovery_gate_min", 0.05))
        return_gate = return_min_gate + (recovery_min_gate - return_min_gate) * return_weight
        recovery_gate = torch.maximum(recovery_gate, recovery_mask.to(recovery_gate.dtype) * recovery_min_gate)
        recovery_gate = torch.maximum(recovery_gate, return_mask.to(recovery_gate.dtype) * return_gate)

        recovery_command_reward_scale = float(cfg.get("recovery_command_reward_scale", 0.05))
        return_command_floor = float(cfg.get("return_command_reward_floor", 0.35))
        return_command_scale = return_command_floor + (1.0 - return_command_floor) * return_progress
        command_scale = torch.where(
            recovery_mask,
            torch.full_like(command_gate, recovery_command_reward_scale),
            torch.where(return_mask, return_command_scale, torch.ones_like(command_gate)),
        )
        command_gate = command_gate * command_scale

        transitions = old_phase != next_phase
        self._last_disturbance_gate_info.update(
            {
                "recovery_phase": next_phase.detach(),
                "recovery_phase_time_s": phase_time.detach(),
                "recovery_stable_time_s": stable_time.detach(),
                "recovery_command_blend": blend.detach(),
                "recovery_return_weight": return_weight.detach(),
                "recovery_controller_active": active_mask.to(style_gate.dtype).detach(),
                "recovery_phase_nominal": nominal_mask.to(style_gate.dtype).detach(),
                "recovery_phase_recovery": recovery_mask.to(style_gate.dtype).detach(),
                "recovery_phase_return": return_mask.to(style_gate.dtype).detach(),
                "sudden_stop_active": sudden_stop_active.detach(),
            }
        )
        self._runner_scalars.update(
            {
                "RecoveryMode/active_mean": active_mask.float().mean().item(),
                "RecoveryMode/phase_nominal_mean": nominal_mask.float().mean().item(),
                "RecoveryMode/phase_recovery_mean": recovery_mask.float().mean().item(),
                "RecoveryMode/phase_return_mean": return_mask.float().mean().item(),
                "RecoveryMode/transition_mean": transitions.float().mean().item(),
                "RecoveryMode/stable_time_mean": stable_time.mean().item(),
                "RecoveryMode/command_blend_mean": blend.mean().item(),
            }
        )
        return style_gate, recovery_gate, command_gate

    def _command_recovery_activity(self, command_name: str, reference: torch.Tensor):
        active = torch.zeros_like(reference)
        sudden_stop_active = torch.zeros_like(reference)
        push_active = getattr(self, "_last_disturbance_gate_info", {}).get("push_active")
        if isinstance(push_active, torch.Tensor) and push_active.shape == reference.shape:
            active = torch.maximum(active, push_active.to(self.device, dtype=reference.dtype))
        try:
            command_term = self._base_env.command_manager.get_term(command_name)
            timer = getattr(command_term, "sudden_stop_timer", None)
            if timer is not None:
                sudden_stop_active = (timer.to(self.device) > 0.0).to(dtype=reference.dtype)
                active = torch.maximum(active, sudden_stop_active)
            metric = command_term.metrics.get("failure_push_active") if hasattr(command_term, "metrics") else None
            if metric is not None:
                push_metric = metric.to(self.device, dtype=reference.dtype).clamp(0.0, 1.0)
                active = torch.maximum(active, push_metric)
            force_metric = command_term.metrics.get("force_push_active") if hasattr(command_term, "metrics") else None
            if force_metric is not None:
                force_metric = force_metric.to(self.device, dtype=reference.dtype).clamp(0.0, 1.0)
                active = torch.maximum(active, force_metric)
        except Exception:
            pass
        try:
            command = self._base_env.command_manager.get_command(command_name).to(self.device)
            robot = self._base_env.scene["robot"]
            yaw_scale = float(self.disturbance_gate_cfg.get("yaw_scale", 0.30))
            zero_threshold = float(
                self.residual_action_gate_cfg.get(
                    "zero_command_threshold",
                    self.residual_penalty_cfg.get("zero_command_threshold", 0.08),
                )
            )
            stop_speed_threshold = float(self.residual_action_gate_cfg.get("manual_stop_speed_threshold", 0.10))
            actual_velocity = torch.stack(
                [
                    robot.data.root_lin_vel_b[:, 0],
                    robot.data.root_lin_vel_b[:, 1],
                    yaw_scale * robot.data.root_ang_vel_b[:, 2],
                ],
                dim=-1,
            ).to(self.device)
            manual_stop_active = (
                (torch.norm(command, dim=1) < zero_threshold)
                & (torch.norm(actual_velocity, dim=1) > stop_speed_threshold)
            ).to(dtype=reference.dtype)
            active = torch.maximum(active, manual_stop_active)
            sudden_stop_active = torch.maximum(sudden_stop_active, manual_stop_active)
        except Exception:
            pass
        return active.clamp(0.0, 1.0), sudden_stop_active.clamp(0.0, 1.0)

    @staticmethod
    def _clone_gate_info(gate_info):
        cloned = {}
        for name, value in gate_info.items():
            cloned[name] = value.detach().clone() if isinstance(value, torch.Tensor) else value
        return cloned

    def _normalize_actor_obs_without_update(self, raw_obs: torch.Tensor) -> torch.Tensor:
        normalizer = self.obs_normalizer
        if hasattr(normalizer, "_mean") and hasattr(normalizer, "_std"):
            mean = normalizer._mean.to(device=raw_obs.device, dtype=raw_obs.dtype)
            std = normalizer._std.to(device=raw_obs.device, dtype=raw_obs.dtype)
            eps = float(getattr(normalizer, "eps", 1.0e-2))
            return (raw_obs - mean) / (std + eps)
        return raw_obs

    def _raw_actor_obs_without_update(self, normalized_obs: torch.Tensor) -> torch.Tensor:
        normalizer = self.obs_normalizer
        if hasattr(normalizer, "inverse"):
            return normalizer.inverse(normalized_obs)
        return normalized_obs

    def _metric_vector(self, command_term, name: str, reference: torch.Tensor) -> torch.Tensor:
        metric = command_term.metrics.get(name) if hasattr(command_term, "metrics") else None
        if isinstance(metric, torch.Tensor):
            metric = metric.to(device=reference.device, dtype=reference.dtype)
            if metric.shape == reference.shape:
                return metric
            if metric.numel() == 1:
                return metric.reshape(()).expand_as(reference)
        return torch.zeros_like(reference)

    def _policy_obs_with_recovery_command(self, normalized_obs: torch.Tensor, gate_info: dict | None):
        cfg = self.recovery_command_override_cfg
        if not bool(cfg.get("enabled", False)):
            return normalized_obs
        if normalized_obs.shape[-1] < 9:
            return normalized_obs

        command_name = str(cfg.get("command_name", self.disturbance_gate_cfg.get("command_name", "base_velocity")))
        try:
            command_term = self._base_env.command_manager.get_term(command_name)
            command = self._base_env.command_manager.get_command(command_name).to(normalized_obs.device)
            robot = self._base_env.scene["robot"]
        except Exception:
            return normalized_obs

        dtype = normalized_obs.dtype
        command = command[:, :3].to(dtype=dtype)
        command_norm = torch.linalg.norm(command, dim=1)
        reference = command_norm

        push_delta_xy = torch.stack(
            (
                self._metric_vector(command_term, "failure_push_delta_x", reference),
                self._metric_vector(command_term, "failure_push_delta_y", reference),
            ),
            dim=-1,
        )
        push_active = self._metric_vector(command_term, "failure_push_active", reference).clamp(0.0, 1.0)
        force_active = self._metric_vector(command_term, "force_push_active", reference).clamp(0.0, 1.0)
        elapsed = self._metric_vector(command_term, "failure_push_elapsed_s", reference).clamp_min(0.0)
        active = torch.maximum(push_active, force_active)
        if isinstance(gate_info, dict):
            for name in ("active_recovery", "push_active", "recovery_controller_active"):
                value = gate_info.get(name)
                if isinstance(value, torch.Tensor) and value.shape == reference.shape:
                    active = torch.maximum(active, value.to(device=normalized_obs.device, dtype=dtype).clamp(0.0, 1.0))

        active_threshold = float(cfg.get("active_threshold", 0.05))
        zero_only = bool(cfg.get("zero_command_only", True))
        zero_threshold = float(
            cfg.get(
                "zero_command_threshold",
                self.residual_action_gate_cfg.get(
                    "zero_command_threshold",
                    self.residual_penalty_cfg.get("zero_command_threshold", 0.08),
                ),
            )
        )
        controller_enabled = self._recovery_controller_enabled()
        controller_blend = None
        recovery_phase_mask = None
        if controller_enabled and isinstance(gate_info, dict):
            controller_blend = gate_info.get("recovery_command_blend")
            recovery_phase_mask = gate_info.get("recovery_phase_recovery")
            if isinstance(controller_blend, torch.Tensor) and controller_blend.shape == reference.shape:
                controller_blend = controller_blend.to(device=normalized_obs.device, dtype=dtype).clamp(0.0, 1.0)
            else:
                controller_blend = None
            if isinstance(recovery_phase_mask, torch.Tensor) and recovery_phase_mask.shape == reference.shape:
                recovery_phase_mask = recovery_phase_mask.to(device=normalized_obs.device, dtype=torch.bool)
            else:
                recovery_phase_mask = None

        active_mask = active > active_threshold
        if controller_blend is not None:
            active_mask = active_mask | (controller_blend > active_threshold)
        if zero_only:
            active_mask = active_mask & (command_norm <= zero_threshold)
        if not bool(active_mask.any()):
            return normalized_obs

        root_lin_vel_b = robot.data.root_lin_vel_b.to(device=normalized_obs.device, dtype=dtype)
        root_ang_vel_b = robot.data.root_ang_vel_b.to(device=normalized_obs.device, dtype=dtype)
        projected_gravity_b = robot.data.projected_gravity_b.to(device=normalized_obs.device, dtype=dtype)
        velocity_xy = root_lin_vel_b[:, :2]
        ang_xy_projected = torch.stack((root_ang_vel_b[:, 1], -root_ang_vel_b[:, 0]), dim=-1)
        direction_signal = (
            float(cfg.get("push_direction_gain", 1.0)) * push_delta_xy
            + float(cfg.get("velocity_direction_gain", 0.35)) * velocity_xy
            + float(cfg.get("tilt_direction_gain", 0.0)) * projected_gravity_b[:, :2]
            + float(cfg.get("ang_vel_direction_gain", 0.0)) * ang_xy_projected
        )
        direction_norm = torch.linalg.norm(direction_signal, dim=1).clamp_min(1.0e-6)
        direction = direction_signal / direction_norm.unsqueeze(-1)

        push_speed = torch.linalg.norm(push_delta_xy, dim=1) * float(cfg.get("push_speed_gain", 0.85))
        velocity_speed = torch.linalg.norm(velocity_xy, dim=1) * float(cfg.get("velocity_speed_gain", 0.35))
        signal_speed = direction_norm * float(cfg.get("signal_speed_gain", 0.0))
        speed = (push_speed + velocity_speed + signal_speed).clamp(
            min=float(cfg.get("min_speed", 0.25)),
            max=float(cfg.get("max_speed", 1.60)),
        )
        direction_threshold = float(cfg.get("direction_threshold", 0.03))
        direction_valid = direction_norm > direction_threshold
        reuse_last_command = torch.zeros_like(active_mask)
        if controller_enabled and self._recovery_last_command is not None and controller_blend is not None:
            last_command = self._recovery_last_command.to(device=normalized_obs.device, dtype=dtype)
            reuse_last_command = (controller_blend > active_threshold) & (
                torch.linalg.norm(last_command[:, :2], dim=1) > direction_threshold
            )
        if not bool(((active_mask & direction_valid) | reuse_last_command).any()):
            return normalized_obs

        yaw_cmd = (-float(cfg.get("yaw_damping_gain", 0.0)) * root_ang_vel_b[:, 2]).clamp(
            -float(cfg.get("max_yaw", 0.0)),
            float(cfg.get("max_yaw", 0.0)),
        )
        recovery_command = torch.cat((direction * speed.unsqueeze(-1), yaw_cmd.unsqueeze(-1)), dim=-1)
        recovery_command = recovery_command.clamp(
            min=torch.tensor(
                [
                    -float(cfg.get("max_abs_x", cfg.get("max_speed", 1.60))),
                    -float(cfg.get("max_abs_y", cfg.get("max_speed", 1.60))),
                    -float(cfg.get("max_yaw", 0.0)),
                ],
                device=normalized_obs.device,
                dtype=dtype,
            ),
            max=torch.tensor(
                [
                    float(cfg.get("max_abs_x", cfg.get("max_speed", 1.60))),
                    float(cfg.get("max_abs_y", cfg.get("max_speed", 1.60))),
                    float(cfg.get("max_yaw", 0.0)),
                ],
                device=normalized_obs.device,
                dtype=dtype,
            ),
        )

        if controller_enabled and self._recovery_last_command is not None:
            self._ensure_recovery_controller_state(reference)
            update_mask = active_mask & direction_valid
            if recovery_phase_mask is not None:
                update_mask = update_mask & recovery_phase_mask
            if bool(update_mask.any()):
                self._recovery_last_command[update_mask] = recovery_command[update_mask].detach()
            recovery_command = torch.where(
                (active_mask & direction_valid).unsqueeze(-1),
                recovery_command,
                self._recovery_last_command.to(device=normalized_obs.device, dtype=dtype),
            )

        if controller_blend is None:
            ramp_s = max(float(cfg.get("ramp_s", 0.20)), 1.0e-6)
            ramp = (elapsed / ramp_s).clamp(0.0, 1.0)
            blend = active * ramp * float(cfg.get("blend", 1.0))
        else:
            blend = controller_blend * float(cfg.get("blend", 1.0))
        active_mask = active_mask & (torch.linalg.norm(recovery_command[:, :2], dim=1) > direction_threshold)
        blend = torch.where(active_mask, blend.clamp(0.0, 1.0), torch.zeros_like(blend))

        raw_obs = self._raw_actor_obs_without_update(normalized_obs).clone()
        start = int(cfg.get("command_slice_start", 6))
        stop = start + 3
        if stop > raw_obs.shape[-1]:
            return normalized_obs
        raw_command = raw_obs[:, start:stop]
        raw_obs[:, start:stop] = raw_command * (1.0 - blend.unsqueeze(-1)) + recovery_command * blend.unsqueeze(-1)

        self._runner_scalars.update(
            {
                "RecoveryCommand/active_mean": active_mask.to(dtype=dtype).mean().item(),
                "RecoveryCommand/blend_mean": blend.mean().item(),
                "RecoveryCommand/speed_mean": (speed * active_mask.to(dtype=dtype)).mean().item(),
                "RecoveryCommand/x_mean": (recovery_command[:, 0] * blend).mean().item(),
                "RecoveryCommand/y_mean": (recovery_command[:, 1] * blend).mean().item(),
            }
        )
        return self._normalize_actor_obs_without_update(raw_obs)

    def _compute_residual_action_gate(self, style_gate=None, recovery_gate=None):
        cfg = self.residual_action_gate_cfg
        if not bool(cfg.get("enabled", True)):
            if style_gate is not None:
                return torch.ones_like(style_gate).unsqueeze(-1)
            return None
        if style_gate is None or recovery_gate is None:
            style_gate, recovery_gate, _ = self._compute_disturbance_gates()

        command_name = self.disturbance_gate_cfg.get("command_name", "base_velocity")
        command = self._base_env.command_manager.get_command(command_name).to(self.device)
        command_norm = torch.norm(command, dim=1)
        active_recovery, sudden_stop_active = self._command_recovery_activity(command_name, style_gate)
        controller_active = self._last_disturbance_gate_info.get("recovery_controller_active")
        if isinstance(controller_active, torch.Tensor) and controller_active.shape == active_recovery.shape:
            active_recovery = torch.maximum(
                active_recovery,
                controller_active.to(self.device, dtype=active_recovery.dtype).clamp(0.0, 1.0),
            )
        self._last_disturbance_gate_info.update(
            {
                "active_recovery": active_recovery.detach(),
                "sudden_stop_active": sudden_stop_active.detach(),
            }
        )

        stable_floor = float(cfg.get("stable_floor", 0.02))
        stable_max_gate = float(cfg.get("stable_max_gate", 0.10))
        recovery_scale = float(cfg.get("recovery_scale", 1.0))
        recovery_power = float(cfg.get("recovery_power", 1.0))
        recovery_gate_for_action = recovery_gate
        push_active = self._last_disturbance_gate_info.get("push_active")
        push_phase_enabled = bool(cfg.get("push_phase_enabled", False))
        if push_phase_enabled and isinstance(push_active, torch.Tensor) and push_active.shape == recovery_gate.shape:
            raw_recovery_gate = self._last_disturbance_gate_info.get("raw_recovery_gate")
            if isinstance(raw_recovery_gate, torch.Tensor) and raw_recovery_gate.shape == recovery_gate.shape:
                recovery_gate_for_action = torch.where(
                    push_active.to(self.device, dtype=torch.bool),
                    raw_recovery_gate.to(self.device, dtype=recovery_gate.dtype),
                    recovery_gate,
                )
        recovery_component = recovery_scale * torch.pow(recovery_gate_for_action.clamp(0.0, 1.0), recovery_power)
        gate = (stable_floor * style_gate + recovery_component).clamp(0.0, 1.0)

        stable_style_threshold = float(cfg.get("stable_style_threshold", 0.75))
        stable_recovery_threshold = float(cfg.get("stable_recovery_threshold", 0.25))
        stable_mask = (
            (style_gate >= stable_style_threshold)
            & (recovery_gate <= stable_recovery_threshold)
            & (active_recovery <= 0.0)
        )
        gate = torch.where(stable_mask, torch.minimum(gate, torch.full_like(gate, stable_max_gate)), gate)

        zero_command_gate = cfg.get("zero_command_gate")
        if zero_command_gate is not None:
            zero_threshold = float(
                cfg.get(
                    "zero_command_threshold",
                    self.residual_penalty_cfg.get(
                        "zero_command_threshold",
                        getattr(self, "cfg", {}).get("amp_zero_command_threshold", 0.08),
                    ),
                )
            )
            zero_stable = (command_norm < zero_threshold) & (active_recovery <= 0.0)
            gate = torch.where(
                zero_stable,
                torch.minimum(gate, torch.full_like(gate, float(zero_command_gate))),
                gate,
            )

        if isinstance(push_active, torch.Tensor) and push_active.shape == gate.shape:
            push_min_gate = float(cfg.get("push_min_gate", 0.65))
            push_active_gate = push_active.to(self.device, dtype=gate.dtype).clamp(0.0, 1.0)
            if push_phase_enabled:
                push_elapsed = torch.zeros_like(gate)
                try:
                    command_term = self._base_env.command_manager.get_term(command_name)
                    metric = command_term.metrics.get("failure_push_elapsed_s") if hasattr(command_term, "metrics") else None
                    if isinstance(metric, torch.Tensor) and metric.shape == gate.shape:
                        push_elapsed = metric.to(self.device, dtype=gate.dtype)
                except Exception:
                    pass
                full_s = max(float(cfg.get("push_full_gate_s", 0.80)), 0.0)
                taper_s = max(float(cfg.get("push_taper_s", 1.00)), 1.0e-6)
                late_min_gate = float(cfg.get("push_late_min_gate", 0.35))
                late_min_gate = max(min(late_min_gate, push_min_gate), 0.0)
                elapsed_after_full = torch.clamp(push_elapsed - full_s, min=0.0)
                phase_floor = late_min_gate + (push_min_gate - late_min_gate) * torch.exp(-elapsed_after_full / taper_s)
                phase_floor = torch.where(push_elapsed <= full_s, torch.full_like(gate, push_min_gate), phase_floor)

                score = self._last_disturbance_gate_info.get("score")
                if isinstance(score, torch.Tensor) and score.shape == gate.shape:
                    threshold = float(cfg.get("push_high_score_threshold", 1.10))
                    width = max(float(cfg.get("push_high_score_width", 0.45)), 1.0e-6)
                    high_min_gate = float(cfg.get("push_high_score_min_gate", 0.92))
                    score_blend = torch.clamp((score.to(self.device, dtype=gate.dtype) - threshold) / width, 0.0, 1.0)
                    high_score_floor = late_min_gate + score_blend * (high_min_gate - late_min_gate)
                    phase_floor = torch.maximum(phase_floor, high_score_floor)

                push_floor = push_active_gate * phase_floor.clamp(0.0, 1.0)
                gate = torch.maximum(gate, push_floor)
                self._runner_scalars.update(
                    {
                        "Gate/residual_action_push_floor_mean": push_floor.mean().item(),
                        "Gate/residual_action_push_elapsed_mean": (push_elapsed * push_active_gate).mean().item(),
                    }
                )
            else:
                gate = torch.maximum(gate, push_active_gate * push_min_gate)

        sudden_stop_min_gate = float(cfg.get("sudden_stop_min_gate", 0.55))
        if self._recovery_controller_enabled():
            phase_recovery = self._last_disturbance_gate_info.get("recovery_phase_recovery")
            phase_return = self._last_disturbance_gate_info.get("recovery_phase_return")
            return_weight = self._last_disturbance_gate_info.get("recovery_return_weight")
            if isinstance(phase_recovery, torch.Tensor) and phase_recovery.shape == gate.shape:
                phase_recovery = phase_recovery.to(self.device, dtype=gate.dtype).clamp(0.0, 1.0)
                phase_return = (
                    phase_return.to(self.device, dtype=gate.dtype).clamp(0.0, 1.0)
                    if isinstance(phase_return, torch.Tensor) and phase_return.shape == gate.shape
                    else torch.zeros_like(gate)
                )
                return_weight = (
                    return_weight.to(self.device, dtype=gate.dtype).clamp(0.0, 1.0)
                    if isinstance(return_weight, torch.Tensor) and return_weight.shape == gate.shape
                    else torch.zeros_like(gate)
                )
                phase_recovery_floor = float(cfg.get("phase_recovery_min_gate", cfg.get("push_min_gate", 0.80)))
                phase_return_start = float(cfg.get("phase_return_start_gate", phase_recovery_floor))
                phase_return_end = float(cfg.get("phase_return_end_gate", stable_floor))
                phase_return_floor = phase_return_end + (phase_return_start - phase_return_end) * return_weight
                phase_floor = phase_recovery * phase_recovery_floor + phase_return * phase_return_floor
                gate = torch.maximum(gate, phase_floor)
                self._runner_scalars["Gate/recovery_phase_floor_mean"] = phase_floor.mean().item()
        gate = torch.maximum(gate, sudden_stop_active * sudden_stop_min_gate)
        gate = torch.nan_to_num(gate, nan=stable_floor, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        self._runner_scalars.update(
            {
                "Gate/residual_action_mean": gate.mean().item(),
                "Gate/residual_action_min": gate.min().item(),
                "Gate/residual_action_max": gate.max().item(),
                "Gate/residual_action_stable_limited": stable_mask.float().mean().item(),
                "Gate/sudden_stop_active_mean": sudden_stop_active.mean().item(),
            }
        )
        return gate.unsqueeze(-1)

    def _compute_residual_penalty(self, style_gate, recovery_gate, gate_info=None):
        if not bool(self.residual_penalty_cfg.get("enabled", True)):
            return torch.zeros_like(style_gate)
        residual = getattr(self.alg.policy, "latest_scaled_residual_action_mean", None)
        if residual is None:
            residual = getattr(self.alg.policy, "latest_residual_action_mean", None)
        if residual is None:
            return torch.zeros_like(style_gate)
        residual = residual.to(self.device)
        stable_coef = float(
            self.residual_penalty_cfg.get("stable_coef", self.residual_penalty_cfg.get("stable_weight", 0.02))
        )
        recovery_coef = float(
            self.residual_penalty_cfg.get("recovery_coef", self.residual_penalty_cfg.get("recovery_weight", 0.002))
        )
        coef = stable_coef * style_gate + recovery_coef * recovery_gate
        zero_command_weight = float(self.residual_penalty_cfg.get("zero_command_weight", 0.0))
        zero_command_mean = torch.zeros((), device=self.device)
        push_stable_mean = torch.zeros((), device=self.device)
        push_stable_mask = None
        if zero_command_weight > 0.0:
            command_name = self.disturbance_gate_cfg.get("command_name", "base_velocity")
            command = None
            if isinstance(gate_info, dict):
                command = gate_info.get("command")
            if not isinstance(command, torch.Tensor):
                command = self._base_env.command_manager.get_command(command_name)
            command = command.to(self.device)
            zero_threshold = float(
                self.residual_penalty_cfg.get(
                    "zero_command_threshold",
                    self.cfg.get("amp_zero_command_threshold", 1.0e-6),
                )
            )
            zero_command = (torch.norm(command, dim=1) < zero_threshold).float()
            active_recovery = None
            if isinstance(gate_info, dict):
                active_recovery = gate_info.get("active_recovery")
            if isinstance(active_recovery, torch.Tensor) and active_recovery.shape == zero_command.shape:
                active_recovery = active_recovery.to(self.device, dtype=zero_command.dtype)
            else:
                active_recovery, _ = self._command_recovery_activity(command_name, zero_command)
                active_recovery = active_recovery.to(dtype=zero_command.dtype)
            stable_zero_command = zero_command * (1.0 - active_recovery.clamp(0.0, 1.0))
            zero_command_mean = stable_zero_command.mean()
            coef = coef + zero_command_weight * stable_zero_command * style_gate
        residual_l2 = torch.sum(torch.square(residual), dim=-1)
        push_stable_weight = float(self.residual_penalty_cfg.get("push_stable_weight", 0.0))
        if push_stable_weight > 0.0 and isinstance(gate_info, dict):
            push_active = gate_info.get("push_active")
            score = gate_info.get("score")
            if isinstance(push_active, torch.Tensor) and isinstance(score, torch.Tensor):
                push_active = push_active.to(self.device, dtype=residual_l2.dtype).clamp(0.0, 1.0)
                score = score.to(self.device, dtype=residual_l2.dtype)
                if push_active.shape == residual_l2.shape and score.shape == residual_l2.shape:
                    score_threshold = float(self.residual_penalty_cfg.get("push_stable_score_threshold", 0.75))
                    threshold = max(score_threshold, 1.0e-6)
                    stability = ((threshold - score) / threshold).clamp(0.0, 1.0)
                    push_stable_mask = push_active * stability
                    push_stable_mean = push_stable_mask.mean()
                    coef = coef + push_stable_weight * push_stable_mask
        penalty = coef * residual_l2
        ungated = getattr(self.alg.policy, "latest_ungated_scaled_residual_action_mean", None)
        ungated_stable_weight = float(self.residual_penalty_cfg.get("ungated_stable_weight", 0.0))
        if ungated_stable_weight > 0.0 and isinstance(ungated, torch.Tensor):
            ungated = ungated.to(self.device)
            ungated_l2 = torch.sum(torch.square(ungated), dim=-1)
            penalty = penalty + ungated_stable_weight * style_gate * ungated_l2
            push_stable_ungated_weight = float(self.residual_penalty_cfg.get("push_stable_ungated_weight", 0.0))
            if push_stable_ungated_weight > 0.0 and isinstance(push_stable_mask, torch.Tensor):
                penalty = penalty + push_stable_ungated_weight * push_stable_mask * ungated_l2
            self._runner_scalars["Adapter/ungated_residual_l2_mean"] = ungated_l2.mean().item()
        max_penalty = self.residual_penalty_cfg.get("max_penalty")
        if max_penalty is not None:
            penalty = torch.clamp(penalty, max=float(max_penalty))
        self._runner_scalars.update(
            {
                "Adapter/residual_l2_mean": residual_l2.mean().item(),
                "Adapter/residual_penalty_mean": penalty.mean().item(),
                "Adapter/zero_command_residual_penalty_active": zero_command_mean.item(),
                "Adapter/push_stable_residual_penalty_active": push_stable_mean.item(),
            }
        )
        return penalty

    def _compute_track_adapter_reward(
        self,
        amp_obs,
        next_amp_obs,
        task_rewards,
        style_gate=None,
        recovery_gate=None,
        command_gate=None,
        gate_info=None,
    ):
        groups = self._split_reward_groups(task_rewards)
        if style_gate is None or recovery_gate is None or command_gate is None:
            style_gate, recovery_gate, command_gate = self._compute_disturbance_gates()
            gate_info = self._last_disturbance_gate_info
        amp_rewards = self.alg.discriminator.predict_amp_reward(
            amp_obs,
            next_amp_obs,
            torch.zeros_like(task_rewards),
            normalizer=self.alg.amp_normalizer,
        )[0]

        reg_stable_scale = float(self.disturbance_gate_cfg.get("reg_stable_scale", 1.0))
        reg_recovery_scale = float(self.disturbance_gate_cfg.get("reg_recovery_scale", 0.35))
        reg_gate = reg_stable_scale * style_gate + reg_recovery_scale * recovery_gate

        weighted = []
        reward_parts = {
            "task": command_gate * groups.get("task", task_rewards),
            "style": style_gate * amp_rewards,
            "recovery": recovery_gate * groups.get("recovery", torch.zeros_like(task_rewards)),
            "reg": reg_gate * groups.get("reg", torch.zeros_like(task_rewards)),
        }
        for name in self.reward_group_names:
            reward = reward_parts.get(name, groups.get(name, torch.zeros_like(task_rewards)))
            weight = self.reward_group_weights.get(name, 1.0)
            weighted.append(weight * reward)
            self._runner_scalars[f"RewardGroup/{name}_mean"] = reward.mean().item()

        rewards = torch.stack(weighted, dim=-1).sum(dim=-1)
        rewards = rewards - self._compute_residual_penalty(style_gate, recovery_gate, gate_info=gate_info)
        self._runner_scalars["RewardGroup/total_mean"] = rewards.mean().item()
        return rewards

    def _compute_terminal_penalty(self, rewards: torch.Tensor) -> torch.Tensor:
        cfg = self.terminal_penalty_cfg
        if not bool(cfg.get("enabled", False)):
            return rewards.new_zeros(rewards.shape)
        manager = getattr(self._base_env, "termination_manager", None)
        if manager is None or not hasattr(manager, "get_term"):
            return rewards.new_zeros(rewards.shape)

        active_terms = set(getattr(manager, "active_terms", []))
        terms = cfg.get("terms")
        if terms is None:
            terms = {"base_contact": cfg.get("base_contact", -100.0)}
        penalty = rewards.new_zeros(rewards.shape)
        for name, weight in terms.items():
            if name not in active_terms:
                continue
            if name == "base_contact":
                term = getattr(manager, "terminated", None)
                if not isinstance(term, torch.Tensor):
                    continue
            else:
                try:
                    term = manager.get_term(name)
                except Exception:
                    continue
            term = term.to(device=rewards.device, dtype=rewards.dtype).reshape(rewards.shape)
            penalty = penalty + term * float(weight)
            self._runner_scalars[f"TerminalPenalty/{name}_mean"] = term.mean().item()
        self._runner_scalars["TerminalPenalty/mean"] = penalty.mean().item()
        return penalty

    def _failure_reset_external_wrench_state(self):
        states = getattr(self._base_env, "_booster_external_wrench_pulse_states", None)
        if not isinstance(states, dict):
            return None
        for state in states.values():
            if isinstance(state, dict) and isinstance(state.get("forces"), torch.Tensor):
                return state
        return None

    def _ensure_failure_reset_buffers(self):
        if not self.failure_reset_enabled:
            return False
        asset = self._base_env.scene["robot"]
        num_envs = int(self.env.num_envs)
        device = self.device
        joint_dim = int(asset.data.joint_pos.shape[-1])
        if self._failure_reset_history:
            return True
        history_shape = (self.failure_reset_history_steps, num_envs)
        self._failure_reset_history = {
            "root_state_rel": torch.zeros(*history_shape, 13, device=device),
            "joint_pos": torch.zeros(*history_shape, joint_dim, device=device),
            "joint_vel": torch.zeros(*history_shape, joint_dim, device=device),
            "command": torch.zeros(*history_shape, 3, device=device),
            "force": torch.zeros(*history_shape, 3, device=device),
        }
        self._failure_reset_replay = {
            "root_state_rel": torch.zeros(self.failure_reset_buffer_size, 13, device=device),
            "joint_pos": torch.zeros(self.failure_reset_buffer_size, joint_dim, device=device),
            "joint_vel": torch.zeros(self.failure_reset_buffer_size, joint_dim, device=device),
            "command": torch.zeros(self.failure_reset_buffer_size, 3, device=device),
            "force": torch.zeros(self.failure_reset_buffer_size, 3, device=device),
        }
        return True

    def _capture_failure_reset_history(self):
        if not self._ensure_failure_reset_buffers():
            return
        asset = self._base_env.scene["robot"]
        cursor = self._failure_reset_history_cursor
        root_state = asset.data.root_state_w.to(self.device).clone()
        origins = self._base_env.scene.env_origins.to(self.device)
        root_state[:, :3] -= origins
        self._failure_reset_history["root_state_rel"][cursor].copy_(root_state)
        self._failure_reset_history["joint_pos"][cursor].copy_(asset.data.joint_pos.to(self.device))
        self._failure_reset_history["joint_vel"][cursor].copy_(asset.data.joint_vel.to(self.device))
        command = self._base_env.command_manager.get_command("base_velocity").to(self.device)
        self._failure_reset_history["command"][cursor].copy_(command[:, :3])
        force_state = self._failure_reset_external_wrench_state()
        force = torch.zeros(self.env.num_envs, 3, device=self.device)
        if isinstance(force_state, dict):
            forces = force_state.get("forces")
            if isinstance(forces, torch.Tensor) and forces.shape[0] == self.env.num_envs:
                force.copy_(forces.to(self.device).sum(dim=1))
        self._failure_reset_history["force"][cursor].copy_(force)
        self._failure_reset_history_cursor = (cursor + 1) % self.failure_reset_history_steps

    def _record_failure_reset_samples(self, dones):
        if not self._ensure_failure_reset_buffers():
            return 0
        if self.failure_reset_buffer_size <= 0:
            return 0
        terminated = getattr(getattr(self._base_env, "termination_manager", None), "terminated", None)
        if isinstance(terminated, torch.Tensor):
            failure_mask = terminated.to(self.device, dtype=torch.bool)
        else:
            failure_mask = dones.to(self.device, dtype=torch.bool).reshape(-1)
        env_ids = torch.nonzero(failure_mask, as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return 0
        source_cursor = (self._failure_reset_history_cursor - self.failure_reset_delay_steps - 1) % (
            self.failure_reset_history_steps
        )
        num_samples = int(env_ids.numel())
        slots = (torch.arange(num_samples, device=self.device) + self._failure_reset_replay_step) % (
            self.failure_reset_buffer_size
        )
        for name, history_tensor in self._failure_reset_history.items():
            self._failure_reset_replay[name][slots].copy_(history_tensor[source_cursor, env_ids])
        self._failure_reset_replay_step = (self._failure_reset_replay_step + num_samples) % self.failure_reset_buffer_size
        self._failure_reset_replay_size = min(
            self.failure_reset_buffer_size,
            self._failure_reset_replay_size + num_samples,
        )
        self._runner_scalars["FailureReset/recorded"] = float(num_samples)
        self._runner_scalars["FailureReset/replay_size"] = float(self._failure_reset_replay_size)
        return num_samples

    def _apply_failure_reset_curriculum(self, dones):
        if not self._ensure_failure_reset_buffers():
            return 0
        if self.failure_reset_probability <= 0.0 or self._failure_reset_replay_size < self.failure_reset_min_samples:
            return 0
        reset_ids = torch.nonzero(dones.to(self.device, dtype=torch.bool).reshape(-1), as_tuple=False).flatten()
        if reset_ids.numel() == 0:
            return 0
        selected = torch.rand(reset_ids.numel(), device=self.device) <= self.failure_reset_probability
        target_ids = reset_ids[selected]
        if target_ids.numel() == 0:
            return 0
        sample_ids = torch.randint(self._failure_reset_replay_size, (target_ids.numel(),), device=self.device)
        asset = self._base_env.scene["robot"]
        root_state = self._failure_reset_replay["root_state_rel"][sample_ids].clone()
        root_state[:, :3] += self._base_env.scene.env_origins.to(self.device)[target_ids]
        asset.write_root_state_to_sim(root_state.to(asset.device), env_ids=target_ids.to(asset.device))
        asset.write_joint_state_to_sim(
            self._failure_reset_replay["joint_pos"][sample_ids].to(asset.device),
            self._failure_reset_replay["joint_vel"][sample_ids].to(asset.device),
            env_ids=target_ids.to(asset.device),
        )
        try:
            command_term = self._base_env.command_manager.get_term("base_velocity")
            command_term.vel_command_b[target_ids.to(command_term.device)] = self._failure_reset_replay["command"][
                sample_ids
            ].to(command_term.device)
        except Exception:
            pass
        if self.failure_reset_replay_force:
            force_state = self._failure_reset_external_wrench_state()
            if isinstance(force_state, dict) and isinstance(force_state.get("forces"), torch.Tensor):
                force = self._failure_reset_replay["force"][sample_ids].to(force_state["forces"].device)
                target_force_ids = target_ids.to(force_state["forces"].device)
                force_state["forces"][target_force_ids] = force.unsqueeze(1).expand_as(
                    force_state["forces"][target_force_ids]
                )
                force_state["torques"][target_ids.to(force_state["torques"].device)] = 0.0
                force_state["remaining_s"][target_ids.to(force_state["remaining_s"].device)] = self.failure_reset_force_duration_s
                force_state["duration_s"][target_ids.to(force_state["duration_s"].device)] = self.failure_reset_force_duration_s
                force_state["ramp_up_s"][target_ids.to(force_state["ramp_up_s"].device)] = 0.0
        episode_length_buf = getattr(self.env, "episode_length_buf", None)
        if isinstance(episode_length_buf, torch.Tensor):
            episode_length_buf[target_ids.to(episode_length_buf.device)] = 2
        self._runner_scalars["FailureReset/applied"] = float(target_ids.numel())
        return int(target_ids.numel())

    def _reset_rollout_adapter_diagnostics(self):
        self._rollout_adapter_diag_sums = {}
        self._rollout_adapter_diag_count = 0

    def _record_rollout_adapter_diagnostics(self):
        values = {}
        per_env_l2 = {}
        for name, attr in (
            ("scaled_residual_action_l2", "latest_scaled_residual_action_mean"),
            ("ungated_scaled_residual_action_l2", "latest_ungated_scaled_residual_action_mean"),
        ):
            value = getattr(self.alg.policy, attr, None)
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                squared = torch.square(value.detach())
                if squared.dim() > 1:
                    squared = torch.sum(squared, dim=-1)
                values[name] = squared.mean().item()
                per_env_l2[name] = squared
        gate = getattr(self.alg.policy, "latest_residual_action_gate", None)
        if isinstance(gate, torch.Tensor) and gate.numel() > 0:
            gate = gate.detach()
            values["residual_action_gate_mean"] = gate.mean().item()
            values["residual_action_gate_min"] = gate.min().item()
            values["residual_action_gate_max"] = gate.max().item()
        gate_info = getattr(self, "_last_disturbance_gate_info", {})
        if per_env_l2:
            masks = {
                "push_active": gate_info.get("push_active"),
                "active_recovery": gate_info.get("active_recovery"),
                "sudden_stop": gate_info.get("sudden_stop_active"),
            }
            if "active_recovery" in masks and isinstance(masks["active_recovery"], torch.Tensor):
                stable_mask = masks["active_recovery"].detach() <= 0.0
                masks["stable"] = stable_mask
            for residual_name, residual_l2 in per_env_l2.items():
                for mask_name, mask in masks.items():
                    if not isinstance(mask, torch.Tensor):
                        continue
                    mask = mask.detach().to(device=residual_l2.device)
                    if mask.shape != residual_l2.shape:
                        continue
                    selected = mask > 0.0
                    if torch.any(selected):
                        values[f"{residual_name}_{mask_name}"] = residual_l2[selected].mean().item()
                    else:
                        values[f"{residual_name}_{mask_name}"] = 0.0
            if isinstance(gate, torch.Tensor):
                gate_scalar = gate.squeeze(-1) if gate.dim() > 1 else gate
                for mask_name, mask in masks.items():
                    if not isinstance(mask, torch.Tensor):
                        continue
                    mask = mask.detach().to(device=gate_scalar.device)
                    if mask.shape != gate_scalar.shape:
                        continue
                    selected = mask > 0.0
                    if torch.any(selected):
                        values[f"residual_action_gate_{mask_name}"] = gate_scalar[selected].mean().item()
                    else:
                        values[f"residual_action_gate_{mask_name}"] = 0.0
        if not values:
            return
        for name, value in values.items():
            self._rollout_adapter_diag_sums[name] = self._rollout_adapter_diag_sums.get(name, 0.0) + value
        self._rollout_adapter_diag_count += 1

    def _finalize_rollout_adapter_diagnostics(self):
        if self._rollout_adapter_diag_count <= 0:
            self._last_rollout_adapter_diagnostics = {}
            return
        diagnostics = {
            name: value / self._rollout_adapter_diag_count
            for name, value in self._rollout_adapter_diag_sums.items()
        }
        self._last_rollout_adapter_diagnostics = diagnostics
        for name, value in diagnostics.items():
            self._runner_scalars[f"AdapterRollout/{name}"] = value

    def _append_history(self, history, obs, actions=None, dones=None):
        if actions is None:
            actions = obs.new_zeros(obs.shape[0], self.env.num_actions)
        features = torch.cat((obs, actions.to(device=obs.device, dtype=obs.dtype)), dim=-1)
        if dones is not None:
            reset_env_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            if len(reset_env_ids) > 0:
                history[reset_env_ids] = 0.0
                features[reset_env_ids] = 0.0
        return torch.cat((history[:, 1:], features.unsqueeze(1)), dim=1)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        if (
            num_learning_iterations > self.max_iterations_without_full_training_unlock
            and not self.allow_full_training
        ):
            raise RuntimeError(
                "Refusing Track Adapter full training: set BOOSTER_TRACK_ADAPTER_FULL_TRAINING=1, "
                "BOOSTER_TRACK_ADAPTER_SMOKE_PASSED=1, and BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING=1 only after "
                "implementation, smoke, contact, residual, style, and sudden-stop gates pass. "
                f"Requested {num_learning_iterations} iterations; current unlocked limit is "
                f"{self.max_iterations_without_full_training_unlock}."
            )
        if (
            num_learning_iterations > self.max_iterations_without_full_training_unlock
            and self.allow_full_training
            and not self.pretrained_world_model_loaded
        ):
            raise RuntimeError(
                "Refusing Track Adapter full training without a Stage 3 world-model pretrain checkpoint. "
                "Set BOOSTER_TRACK_ADAPTER_PRETRAINED_WM to a verified pretrain checkpoint."
            )
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
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
        obs = self.obs_normalizer(obs)
        privileged_obs = self.privileged_obs_normalizer(privileged_obs)
        history = torch.zeros(self.env.num_envs, self.history_length, self.history_step_dim, device=self.device)
        self.train_mode()
        warm_start_teacher_replay = getattr(self.alg, "warm_start_privileged_teacher_distill_replay", None)
        if callable(warm_start_teacher_replay):
            warm_updates, warm_loss = warm_start_teacher_replay()
            if warm_updates > 0:
                print(
                    "[TrackAdapterRunner] Privileged-teacher replay warm start: "
                    f"{warm_updates} updates, mean BC loss {warm_loss / max(warm_updates, 1):.6f}"
                )

        ep_infos = []
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
        self._safety_gate_iteration_offset = (
            start_iter if bool(self.safety_gate_cfg.get("min_iteration_relative", False)) else 0
        )
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            self._trace_phase("iteration_start", it=it)
            start = time.time()
            self._reset_rollout_adapter_diagnostics()
            with torch.inference_mode():
                for rollout_step in range(self.num_steps_per_env):
                    if rollout_step % self.phase_trace_step_interval == 0:
                        self._trace_phase("rollout_step_start", it=it, step=rollout_step)
                    self._capture_failure_reset_history()
                    style_gate, recovery_gate, command_gate = self._compute_disturbance_gates()
                    residual_action_gate = self._compute_residual_action_gate(style_gate, recovery_gate)
                    pre_action_gate_info = self._clone_gate_info(self._last_disturbance_gate_info)
                    policy_obs = self._policy_obs_with_recovery_command(obs, pre_action_gate_info)
                    history_obs = policy_obs
                    world_model_reference_state = (
                        self.compute_world_model_state(history_obs) if self._world_model_uses_structured_state() else None
                    )
                    actions = self.alg.act(
                        policy_obs,
                        privileged_obs,
                        amp_obs,
                        history,
                        residual_action_gate=residual_action_gate,
                    )
                    self._record_rollout_adapter_diagnostics()
                    if rollout_step % self.phase_trace_step_interval == 0:
                        self._trace_phase("before_env_step", it=it, step=rollout_step)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    if rollout_step % self.phase_trace_step_interval == 0:
                        self._trace_phase("after_env_step", it=it, step=rollout_step)
                    step_observations = infos.get("observations", {}) if isinstance(infos, dict) else {}
                    terminal_amp_obs = step_observations.get("amp_observations")
                    self._record_failure_reset_samples(dones)
                    if rollout_step % self.phase_trace_step_interval == 0:
                        self._trace_phase("before_failure_reset", it=it, step=rollout_step)
                    self._apply_failure_reset_curriculum(dones)
                    if rollout_step % self.phase_trace_step_interval == 0:
                        self._trace_phase("after_failure_reset", it=it, step=rollout_step)
                    obs, extras = self.env.get_observations()
                    if rollout_step % self.phase_trace_step_interval == 0:
                        self._trace_phase("after_get_observations", it=it, step=rollout_step)
                    refreshed_observations = extras["observations"]
                    if isinstance(infos, dict):
                        infos["observations"] = refreshed_observations
                    next_amp_obs = refreshed_observations.get("amp_observations")

                    obs, rewards, dones, next_amp_obs = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                        next_amp_obs.to(self.device),
                    )
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs_source = refreshed_observations.get(self.privileged_obs_type)
                        if privileged_obs_source is None:
                            privileged_obs_source = step_observations[self.privileged_obs_type]
                        privileged_obs = self.privileged_obs_normalizer(
                            privileged_obs_source.to(self.device)
                        )
                    else:
                        privileged_obs = obs
                    world_model_state = (
                        self.compute_world_model_state(obs) if self._world_model_uses_structured_state() else None
                    )

                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    reset_env_ids = self._base_env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                    terminal_amp_source = terminal_amp_obs if terminal_amp_obs is not None else next_amp_obs
                    terminal_amp_states = terminal_amp_source.to(self.device)[reset_env_ids]
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    if self.recovery_gated_amp_enabled:
                        rewards = self._compute_track_adapter_reward(
                            amp_obs,
                            next_amp_obs_with_term,
                            rewards,
                            style_gate=style_gate,
                            recovery_gate=recovery_gate,
                            command_gate=command_gate,
                            gate_info=pre_action_gate_info,
                        )
                    else:
                        rewards = self.alg.discriminator.predict_amp_reward(
                            amp_obs,
                            next_amp_obs_with_term,
                            rewards,
                            normalizer=self.alg.amp_normalizer,
                        )[0]
                    rewards = rewards + self._compute_terminal_penalty(rewards)

                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(
                        rewards,
                        dones,
                        infos,
                        next_amp_obs_with_term,
                        next_observations=obs,
                        world_model_state=world_model_state,
                        world_model_reference_state=world_model_reference_state,
                    )
                    self._reset_recovery_controller_state(dones)
                    if rollout_step % self.phase_trace_step_interval == 0:
                        self._trace_phase("after_process_env_step", it=it, step=rollout_step)
                    history = self._append_history(history, history_obs, actions, dones)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
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
                self._finalize_rollout_adapter_diagnostics()
                self._trace_phase("before_compute_returns", it=it)
                self.alg.compute_returns(privileged_obs, history)
                self._trace_phase("after_compute_returns", it=it)

            self._trace_phase("before_update", it=it)
            loss_dict = self.alg.update()
            self._trace_phase("after_update", it=it)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self._trace_phase("before_save", it=it)
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                    self._trace_phase("after_save", it=it)
            self._publish_training_feedback(ep_infos, lenbuffer, it)
            if self._check_safety_gates(ep_infos, it):
                if self.log_dir is not None and not self.disable_logs:
                    self.save(os.path.join(self.log_dir, f"model_{it}_aborted.pt"))
                raise RuntimeError("Track Adapter safety gate failed; training stopped before promotion to full run.")

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _episode_info_mean(self, ep_infos, key_suffix):
        values = []
        for ep_info in ep_infos:
            for key, value in ep_info.items():
                if key == key_suffix or key.endswith("/" + key_suffix):
                    if not isinstance(value, torch.Tensor):
                        value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
                    values.append(value.to(self.device, dtype=torch.float32).reshape(-1))
        if not values:
            return None
        return torch.cat(values).mean().item()

    def _publish_training_feedback(self, ep_infos, lenbuffer, iteration: int):
        base_env = self._base_env
        base_contact = self._episode_info_mean(ep_infos, self.safety_gate_cfg.get("base_contact_key", "base_contact"))
        push_failure = self._episode_info_mean(ep_infos, "failure_push_mining_recorded")
        push_active = self._episode_info_mean(ep_infos, "failure_push_active")
        if lenbuffer:
            episode_length = float(statistics.mean(lenbuffer))
        else:
            episode_length = None
        setattr(
            base_env,
            "_booster_track_adapter_training_stats",
            {
                "iteration": int(iteration),
                "base_contact": base_contact,
                "push_failure": push_failure,
                "push_active": push_active,
                "episode_length": episode_length,
            },
        )

    def _safety_gate_strike(self, name: str, value: float, threshold: float, patience: int) -> bool:
        self._runner_scalars[f"Safety/{name}"] = value
        if value <= threshold:
            self._safety_gate_strikes[name] = 0
            return False
        strikes = self._safety_gate_strikes.get(name, 0) + 1
        self._safety_gate_strikes[name] = strikes
        print(
            "[TrackAdapterRunner] Safety gate strike "
            f"{strikes}/{patience}: {name}={value:.4f} > {threshold:.4f}"
        )
        return strikes >= patience

    def _check_safety_gates(self, ep_infos, iteration):
        cfg = self.safety_gate_cfg
        if not bool(cfg.get("enabled", True)):
            return False
        gate_iteration = int(iteration) - int(getattr(self, "_safety_gate_iteration_offset", 0))
        self._runner_scalars["Safety/gate_iteration"] = float(gate_iteration)
        if gate_iteration < int(cfg.get("min_iteration", 5)):
            return False
        patience = int(cfg.get("patience", 2))
        should_abort = False
        base_contact = self._episode_info_mean(ep_infos, cfg.get("base_contact_key", "base_contact"))
        if base_contact is not None:
            should_abort = self._safety_gate_strike(
                "base_contact",
                base_contact,
                float(cfg.get("max_base_contact", 0.20)),
                patience,
            ) or should_abort

        residual_threshold = cfg.get("max_scaled_residual_action_l2")
        residual_l2 = self._last_rollout_adapter_diagnostics.get("scaled_residual_action_l2")
        if residual_threshold is not None and residual_l2 is not None:
            should_abort = self._safety_gate_strike(
                "scaled_residual_action_l2",
                residual_l2,
                float(residual_threshold),
                patience,
            ) or should_abort
        return should_abort

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        for key, value in self._runner_scalars.items():
            self.writer.add_scalar(key, value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        title = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        log_string = (
            f"""{'#' * width}\n"""
            f"""{title.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
        )
        for key, value in locs["loss_dict"].items():
            log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
        if len(locs["rewbuffer"]) > 0:
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "ppo_optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "amp_normalizer": self.alg.amp_normalizer,
            "iter": self.current_learning_iteration,
            "infos": infos,
            "base_checkpoint_path": self.base_checkpoint_path,
            "history_length": self.history_length,
            "history_step_dim": self.history_step_dim,
            "track_adapter_stage": "ppo",
            "pretrained_world_model_path": self.pretrained_world_model_path,
            "pretrained_world_model_loaded": self.pretrained_world_model_loaded,
            "pretrained_world_model_iter": self.pretrained_world_model_iter,
            "pretrained_world_model_metadata": self.pretrained_world_model_metadata,
            "pretrained_world_model_pretrain": self.pretrained_world_model_pretrain,
            "world_model_metadata": self.world_model_metadata(),
        }
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        if getattr(self.alg, "world_model_optimizer", None) is not None:
            saved_dict["world_model_optimizer_state_dict"] = self.alg.world_model_optimizer.state_dict()
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        replay_save = getattr(self.alg, "save_privileged_teacher_distill_replay", None)
        if callable(replay_save):
            replay_path = replay_save(force=True)
            if replay_path:
                saved_dict["privileged_teacher_distill_replay_path"] = replay_path
        torch.save(saved_dict, path)
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        keep_requested_pretrain = os.getenv("BOOSTER_TRACK_ADAPTER_KEEP_PRETRAINED_WM_ON_RESUME", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        requested_pretrain_path = self.pretrained_world_model_path
        if loaded_dict.get("track_adapter_stage") == "world_model_pretrain":
            raise RuntimeError(
                "TrackAdapterRunner.load() received a Stage 3 pretrain checkpoint. "
                "Use BOOSTER_TRACK_ADAPTER_PRETRAINED_WM for pretrain weights and --checkpoint only for PPO resumes."
            )
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
            ckpt_world_model_metadata = loaded_dict.get("world_model_metadata")
            if ckpt_world_model_metadata is None:
                raise RuntimeError("Track Adapter checkpoint is missing world_model_metadata.")
            current_world_model_metadata = self.world_model_metadata()
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
                if ckpt_world_model_metadata.get(key) != current_world_model_metadata.get(key):
                    raise RuntimeError(
                        f"Track Adapter checkpoint world-model metadata mismatch for {key}: "
                        f"checkpoint={ckpt_world_model_metadata.get(key)}, "
                        f"current={current_world_model_metadata.get(key)}"
                    )

        if keep_requested_pretrain:
            self.pretrained_world_model_path = requested_pretrain_path
        else:
            self.pretrained_world_model_path = loaded_dict.get(
                "pretrained_world_model_path", self.pretrained_world_model_path
            )
            self.pretrained_world_model_loaded = bool(
                loaded_dict.get("pretrained_world_model_loaded", self.pretrained_world_model_loaded)
            )
            self.pretrained_world_model_iter = loaded_dict.get(
                "pretrained_world_model_iter", self.pretrained_world_model_iter
            )
            self.pretrained_world_model_metadata = loaded_dict.get(
                "pretrained_world_model_metadata", loaded_dict.get("world_model_metadata")
            )
            self.pretrained_world_model_pretrain = loaded_dict.get("pretrained_world_model_pretrain")

        resumed_training = self.alg.policy.load_state_dict(
            loaded_dict["model_state_dict"],
            strict=self.strict_base_auxiliary_state,
        )
        if keep_requested_pretrain:
            if not self.pretrained_world_model_path:
                raise RuntimeError(
                    "BOOSTER_TRACK_ADAPTER_KEEP_PRETRAINED_WM_ON_RESUME requires "
                    "BOOSTER_TRACK_ADAPTER_PRETRAINED_WM."
                )
            self._load_world_model_pretrain()
            print(
                "[TrackAdapterRunner] Reapplied requested Stage 3 world-model pretrain after PPO resume; "
                "skipping resumed world-model optimizer state."
        )
        self._override_action_std_after_resume()
        residual_output_perturbed = self._perturb_residual_output_after_resume()
        privileged_teacher_reset = self._reset_privileged_teacher_after_resume()

        ckpt_disc = loaded_dict.get("discriminator_state_dict")
        if ckpt_disc is not None:
            runtime_disc = self.alg.discriminator.state_dict()
            compatible = all(k in runtime_disc and runtime_disc[k].shape == v.shape for k, v in ckpt_disc.items())
            if compatible:
                self.alg.discriminator.load_state_dict(ckpt_disc)
            elif self.strict_base_auxiliary_state:
                raise RuntimeError("[TrackAdapterRunner] Checkpoint discriminator shape mismatch.")
        elif self.strict_base_auxiliary_state:
            raise RuntimeError("Track Adapter checkpoint is missing discriminator_state_dict.")
        if loaded_dict.get("amp_normalizer") is not None:
            ckpt_norm = loaded_dict["amp_normalizer"]
            if getattr(ckpt_norm, "mean", None) is not None and ckpt_norm.mean.shape == self.alg.amp_normalizer.mean.shape:
                self.alg.amp_normalizer = ckpt_norm
            elif self.strict_base_auxiliary_state:
                raise RuntimeError("[TrackAdapterRunner] Checkpoint AMP normalizer shape mismatch.")
        elif self.strict_base_auxiliary_state:
            raise RuntimeError("Track Adapter checkpoint is missing amp_normalizer.")
        if self.alg.rnd and "rnd_state_dict" in loaded_dict:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if self.empirical_normalization:
            if "obs_norm_state_dict" in loaded_dict:
                try:
                    self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                except RuntimeError as exc:
                    message = "[TrackAdapterRunner] Checkpoint observation normalizer shape mismatch."
                    if self.strict_base_auxiliary_state:
                        raise RuntimeError(message) from exc
                    print(message + " Skipping load.")
            elif self.strict_base_auxiliary_state:
                raise RuntimeError("Track Adapter checkpoint is missing obs_norm_state_dict.")
            if "privileged_obs_norm_state_dict" in loaded_dict:
                try:
                    self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
                except RuntimeError as exc:
                    message = "[TrackAdapterRunner] Checkpoint privileged observation normalizer shape mismatch."
                    if self.strict_base_auxiliary_state:
                        raise RuntimeError(message) from exc
                    print(message + " Skipping load.")
            elif self.strict_base_auxiliary_state:
                raise RuntimeError("Track Adapter checkpoint is missing privileged_obs_norm_state_dict.")
        if load_optimizer and resumed_training and (residual_output_perturbed or privileged_teacher_reset):
            reason = "residual output layer was perturbed" if residual_output_perturbed else "privileged teacher was reset"
            print(
                "[TrackAdapterRunner] Skipped optimizer state load because the resumed "
                f"{reason}."
            )
        elif load_optimizer and resumed_training:
            ppo_optimizer_state = loaded_dict.get("ppo_optimizer_state_dict", loaded_dict.get("optimizer_state_dict"))
            if ppo_optimizer_state is not None:
                self.alg.optimizer.load_state_dict(ppo_optimizer_state)
            world_model_optimizer_state = loaded_dict.get("world_model_optimizer_state_dict")
            if (
                world_model_optimizer_state is not None
                and getattr(self.alg, "world_model_optimizer", None) is not None
                and not keep_requested_pretrain
            ):
                self.alg.world_model_optimizer.load_state_dict(world_model_optimizer_state)
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def _override_action_std_after_resume(self):
        requested_std = os.getenv("BOOSTER_TRACK_ADAPTER_OVERRIDE_ACTION_STD")
        if requested_std is None or requested_std.strip() == "":
            return
        action_std = float(requested_std)
        if not math.isfinite(action_std) or action_std <= 0.0:
            raise ValueError(f"BOOSTER_TRACK_ADAPTER_OVERRIDE_ACTION_STD must be positive and finite, got {requested_std}")
        policy = self.alg.policy
        noise_std_type = getattr(policy, "noise_std_type", "scalar")
        with torch.no_grad():
            if noise_std_type == "scalar" and hasattr(policy, "std"):
                policy.std.fill_(action_std)
            elif noise_std_type == "log" and hasattr(policy, "log_std"):
                policy.log_std.fill_(math.log(action_std))
            else:
                raise RuntimeError(f"Cannot override Track Adapter action std for noise_std_type={noise_std_type!r}.")
        print(f"[TrackAdapterRunner] Overrode resumed action std to {action_std:.4f}.")

    def _perturb_residual_output_after_resume(self):
        requested_weight_std = os.getenv("BOOSTER_TRACK_ADAPTER_RESIDUAL_RESUME_OUTPUT_NOISE_STD")
        requested_bias_std = os.getenv("BOOSTER_TRACK_ADAPTER_RESIDUAL_RESUME_BIAS_NOISE_STD")
        if (
            (requested_weight_std is None or requested_weight_std.strip() == "")
            and (requested_bias_std is None or requested_bias_std.strip() == "")
        ):
            return False
        weight_std = (
            float(requested_weight_std)
            if requested_weight_std is not None and requested_weight_std.strip() != ""
            else 0.0
        )
        bias_std = (
            float(requested_bias_std)
            if requested_bias_std is not None and requested_bias_std.strip() != ""
            else 0.0
        )
        if not math.isfinite(weight_std) or weight_std < 0.0:
            raise ValueError(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_RESUME_OUTPUT_NOISE_STD must be non-negative and finite, "
                f"got {requested_weight_std}"
            )
        if not math.isfinite(bias_std) or bias_std < 0.0:
            raise ValueError(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_RESUME_BIAS_NOISE_STD must be non-negative and finite, "
                f"got {requested_bias_std}"
        )
        if weight_std == 0.0 and bias_std == 0.0:
            return False

        policy = self.alg.policy
        adapter_mode = getattr(policy, "adapter_mode", "action_residual")
        if adapter_mode == "layerwise":
            adapter_modules = list(getattr(policy, "layer_adapters", []))
            if not adapter_modules:
                raise RuntimeError("Cannot perturb residual output after resume: policy has no layer_adapters.")
        else:
            residual_actor = getattr(policy, "residual_actor", None)
            if residual_actor is None:
                raise RuntimeError("Cannot perturb residual output after resume: policy has no residual_actor.")
            adapter_modules = [residual_actor]

        perturbed_layers = 0
        with torch.no_grad():
            for adapter_module in adapter_modules:
                last_linear = None
                for module in adapter_module.modules():
                    if isinstance(module, torch.nn.Linear):
                        last_linear = module
                if last_linear is None:
                    continue
                if weight_std > 0.0:
                    last_linear.weight.add_(torch.randn_like(last_linear.weight) * weight_std)
                if bias_std > 0.0 and last_linear.bias is not None:
                    last_linear.bias.add_(torch.randn_like(last_linear.bias) * bias_std)
                perturbed_layers += 1
        if perturbed_layers == 0:
            raise RuntimeError("Cannot perturb residual output after resume: adapter has no Linear output layer.")
        print(
            "[TrackAdapterRunner] Perturbed resumed adapter output layers "
            f"(mode={adapter_mode}, layers={perturbed_layers}, weight_std={weight_std:.6f}, bias_std={bias_std:.6f})."
        )
        return True

    def _reset_privileged_teacher_after_resume(self):
        if not _env_flag("BOOSTER_TRACK_ADAPTER_RESET_PRIVILEGED_TEACHER_ON_RESUME"):
            return False
        reset_teacher = getattr(self.alg.policy, "reset_privileged_teacher_residual_actor", None)
        if not callable(reset_teacher):
            raise RuntimeError("Cannot reset privileged teacher: policy has no reset method.")
        output_init_scale = _env_float(
            "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_OUTPUT_INIT_SCALE",
            getattr(self.alg.policy, "privileged_teacher_output_init_scale", 0.0),
        )
        reset_teacher(output_init_scale=output_init_scale)
        print(
            "[TrackAdapterRunner] Reset privileged teacher residual actor after resume "
            f"(output_init_scale={output_init_scale:.6g})."
        )
        return True

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        inference_device = device if device is not None else self.device
        if self.cfg["empirical_normalization"] and device is not None:
            self.obs_normalizer.to(device)
        history = None

        def policy(obs, dones=None):
            nonlocal history
            obs = obs.to(inference_device)
            obs_input = self.obs_normalizer(obs) if self.cfg["empirical_normalization"] else obs
            if (
                history is None
                or history.shape[0] != obs_input.shape[0]
                or history.shape[-1] != self.history_step_dim
            ):
                history = torch.zeros(
                    obs_input.shape[0], self.history_length, self.history_step_dim, device=obs_input.device
                )
            if dones is not None:
                reset_env_ids = (dones.to(obs_input.device).reshape(-1) > 0).nonzero(as_tuple=False).flatten()
                if len(reset_env_ids) > 0:
                    history[reset_env_ids] = 0.0
                self._reset_recovery_controller_state(dones)
            style_gate, recovery_gate, _ = self._compute_disturbance_gates()
            residual_action_gate = self._compute_residual_action_gate(style_gate, recovery_gate)
            gate_info = self._clone_gate_info(self._last_disturbance_gate_info)
            policy_obs = self._policy_obs_with_recovery_command(obs_input, gate_info)
            if residual_action_gate is not None:
                residual_action_gate = residual_action_gate.to(device=policy_obs.device)
            actions = self.alg.policy.act_inference(
                policy_obs,
                history=history,
                residual_gate=residual_action_gate,
            )
            history = self._append_history(history, policy_obs, actions)
            return actions

        def reset_history():
            nonlocal history
            history = None
            self._reset_recovery_controller_state()

        policy.reset_history = reset_history  # type: ignore[attr-defined]
        return policy

    def train_mode(self):
        self.alg.policy.train()
        if self.alg.freeze_discriminator:
            self.alg.discriminator.eval()
        else:
            self.alg.discriminator.train()
        if self.alg.rnd:
            self.alg.rnd.train()
        if self.empirical_normalization:
            if self.freeze_loaded_normalizers:
                self.obs_normalizer.eval()
                self.privileged_obs_normalizer.eval()
            else:
                self.obs_normalizer.train()
                self.privileged_obs_normalizer.train()

    def eval_mode(self):
        self.alg.policy.eval()
        self.alg.discriminator.eval()
        if self.alg.rnd:
            self.alg.rnd.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self):
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)

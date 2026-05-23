# Copyright (c) 2025-2026, The Booster Soccer Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""V4 Stage 5 — Encoder-augmented multi-critic AMP on-policy runner.

This runner is :class:`MultiCriticAmpOnPolicyRunner` with two substitutions:

* The policy class defaults to :class:`EncoderActorCritic` and is wired with
  the ``encoder_cfg`` block read from the agent cfg.
* The algorithm class defaults to :class:`EncoderMultiCriticAMPPPO` and is
  passed ``decoder_target_slice`` / ``decoder_loss_coef`` from the
  ``encoder_cfg`` block.

Everything else (per-term reward partitioning, AMP discriminator, etc.) is
inherited unchanged.
"""
from __future__ import annotations

import os
import statistics
import time
from collections import deque
from typing import Sequence

import torch

import rsl_rl
from rsl_rl.algorithms import EncoderMultiCriticAMPPPO, MultiCriticAMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    Discriminator,
    EmpiricalNormalization,
    EncoderActorCritic,
    MultiCriticActorCritic,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.runners.multi_critic_amp_runner import MultiCriticAmpOnPolicyRunner
from rsl_rl.utils import AMPLoader, Normalizer, store_code_state


class EncoderMultiCriticAmpRunner(MultiCriticAmpOnPolicyRunner):
    """Stage 5 runner — wires up encoder + decoder aux loss."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # Mirror ``MultiCriticAmpOnPolicyRunner.__init__`` but build the
        # EncoderActorCritic policy + EncoderMultiCriticAMPPPO algorithm.
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        self._configure_multi_gpu()

        if self.alg_cfg["class_name"] not in (
            "PPO",
            "AMPPPO",
            "MultiCriticAMPPPO",
            "EncoderMultiCriticAMPPPO",
        ):
            raise ValueError(
                "Unsupported algorithm class for EncoderMultiCriticAmpRunner: "
                f"{self.alg_cfg['class_name']}"
            )
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

        # Critic-group configuration (same as MultiCriticAmpOnPolicyRunner).
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
        amp_group_name = str(self.alg_cfg.get("amp_critic_group", self.critic_group_names[-1]))
        if amp_group_name not in self.critic_group_names:
            raise ValueError(
                f"amp_critic_group '{amp_group_name}' not in critic_group_names {self.critic_group_names}."
            )
        self._amp_group_idx = self.critic_group_names.index(amp_group_name)

        # ---- read encoder cfg ------------------------------------------------
        encoder_cfg_obj = train_cfg.get("encoder_cfg") or self.alg_cfg.get("encoder_cfg")
        if encoder_cfg_obj is None:
            raise ValueError(
                "EncoderMultiCriticAmpRunner requires an 'encoder_cfg' block in the agent cfg."
            )
        encoder_cfg = dict(encoder_cfg_obj)

        # ---- build encoder policy -------------------------------------------
        policy_class_name = self.policy_cfg.pop("class_name", "EncoderActorCritic")
        try:
            policy_class = eval(policy_class_name)
        except NameError as exc:
            raise ValueError(
                f"Unknown policy class '{policy_class_name}' for EncoderMultiCriticAmpRunner."
            ) from exc
        policy_cfg = dict(self.policy_cfg)
        policy_cfg.setdefault("num_critic_groups", self.num_critic_groups)
        policy_cfg.setdefault("critic_group_names", self.critic_group_names)
        # Splice encoder-cfg fields that EncoderActorCritic consumes directly.
        for key in (
            "history_slice",
            "critic_history_slice",
            "history_len",
            "history_dim",
            "latent_dim",
            "encoder_hidden_dims",
            "decoder_target_dim",
            "decoder_hidden_dims",
        ):
            if key in encoder_cfg:
                policy_cfg.setdefault(key, encoder_cfg[key])
        policy = policy_class(num_obs, num_privileged_obs, self.env.num_actions, **policy_cfg).to(self.device)

        # AMP data + discriminator + normalizer (same as parent).
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

        # ---- build the algorithm --------------------------------------------
        alg_class_name = self.alg_cfg.pop("class_name")
        try:
            alg_class = eval(alg_class_name)
        except NameError as exc:
            raise ValueError(
                f"Unknown algorithm class '{alg_class_name}' for EncoderMultiCriticAmpRunner."
            ) from exc
        alg_kwargs = dict(self.alg_cfg)
        alg_kwargs.pop("num_critic_groups", None)
        alg_kwargs.pop("critic_group_names", None)
        alg_kwargs.pop("critic_group_weights", None)
        alg_kwargs.pop("amp_critic_group", None)
        alg_kwargs.pop("encoder_cfg", None)

        # Decoder-aux-loss config goes directly to EncoderMultiCriticAMPPPO.
        decoder_target_slice = encoder_cfg.get("decoder_target_slice")
        if decoder_target_slice is None:
            raise ValueError("encoder_cfg.decoder_target_slice is required.")
        decoder_loss_coef = float(encoder_cfg.get("decoder_loss_coef", 1.0))

        self.alg: EncoderMultiCriticAMPPPO = alg_class(
            policy,
            discriminator,
            amp_data,
            amp_normalizer,
            device=self.device,
            min_std=min_std,
            num_critic_groups=self.num_critic_groups,
            critic_group_names=self.critic_group_names,
            critic_group_weights=self.critic_group_weights,
            decoder_target_slice=decoder_target_slice,
            decoder_loss_coef=decoder_loss_coef,
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

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])

        ckpt_disc = loaded_dict["discriminator_state_dict"]
        runtime_disc = self.alg.discriminator.state_dict()
        disc_compatible = all(
            k in runtime_disc and runtime_disc[k].shape == v.shape
            for k, v in ckpt_disc.items()
        )
        if disc_compatible:
            self.alg.discriminator.load_state_dict(ckpt_disc)
        else:
            print(
                f"[EncoderMultiCriticAmpRunner] Skipping discriminator load: shape mismatch with {path}."
            )

        ckpt_norm = loaded_dict.get("amp_normalizer")
        if (
            ckpt_norm is not None
            and getattr(ckpt_norm, "mean", None) is not None
            and ckpt_norm.mean.shape == self.alg.amp_normalizer.mean.shape
        ):
            self.alg.amp_normalizer = ckpt_norm
        else:
            print("[EncoderMultiCriticAmpRunner] Skipping amp_normalizer load: shape mismatch.")

        if self.alg.rnd and "rnd_state_dict" in loaded_dict:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        if self.empirical_normalization:
            obs_state = loaded_dict.get("obs_norm_state_dict")
            priv_state = loaded_dict.get("privileged_obs_norm_state_dict")
            obs_to_load = self._adapt_normalizer_state(
                obs_state,
                self.obs_normalizer,
                history_start=self.alg.policy.history_slice[0],
                history_end=self.alg.policy.history_slice[1],
            )
            priv_to_load = self._adapt_normalizer_state(
                priv_state,
                self.privileged_obs_normalizer,
                history_start=self.alg.policy.critic_history_slice[0],
                history_end=self.alg.policy.critic_history_slice[1],
            )
            if obs_to_load is not None and priv_to_load is not None:
                self.obs_normalizer.load_state_dict(obs_to_load)
                self.privileged_obs_normalizer.load_state_dict(priv_to_load)
                if not resumed_training:
                    print("[EncoderMultiCriticAmpRunner] Warm-started adapted observation normalizers.")
            else:
                print("[EncoderMultiCriticAmpRunner] Skipping observation normalizers: shape mismatch.")

        if load_optimizer and resumed_training and disc_compatible:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if self.alg.rnd and "rnd_optimizer_state_dict" in loaded_dict:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        elif load_optimizer and not resumed_training:
            print("[EncoderMultiCriticAmpRunner] Warm-start load: optimizer and iteration are reset.")

        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos", {})

    def _adapt_normalizer_state(
        self,
        state: dict | None,
        normalizer,
        *,
        history_start: int,
        history_end: int,
    ) -> dict | None:
        if state is None:
            return None
        runtime = normalizer.state_dict()
        if state["_mean"].shape == runtime["_mean"].shape:
            return state

        old_width = state["_mean"].shape[1]
        new_width = runtime["_mean"].shape[1]
        new_history_width = int(history_end) - int(history_start)
        suffix_width = max(new_width - int(history_end), 0)
        old_history_width = old_width - int(history_start) - suffix_width
        if old_history_width <= 0 or old_width < int(history_start):
            return None

        adapted = {k: v.clone() for k, v in runtime.items()}
        prefix_width = min(int(history_start), old_width, new_width)
        copy_history_width = min(old_history_width, new_history_width)
        dst_hist_start = int(history_end) - copy_history_width
        src_hist_start = int(history_start)
        for key in ("_mean", "_var", "_std"):
            adapted[key][:, :prefix_width] = state[key][:, :prefix_width]
            adapted[key][:, dst_hist_start:history_end] = state[
                key
            ][:, src_hist_start : src_hist_start + copy_history_width]
            if suffix_width > 0:
                adapted[key][:, -suffix_width:] = state[key][:, -suffix_width:]
        adapted["count"] = state.get("count", adapted["count"]).clone()
        return adapted

# Copyright (c) 2025-2026, The Booster Soccer Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""V4 Stage 5 — Encoder-augmented multi-critic actor-critic.

This module extends :class:`MultiCriticActorCritic` with two side-channel
components used in Stage 5 of the soccer AMP skill library design
(``docs/soccer_amp_skill_library_design.md`` §5.2):

* **Encoder** — an MLP that consumes a contiguous slice of the actor
  observation (the ``ball_history`` segment, ``history_len * history_dim``
  wide) and produces a ``latent_dim``-dimensional latent.
* **Decoder** — an MLP that consumes the latent and predicts a privileged
  target slice (e.g. ``[GT_ball_pos_b, GT_ball_vel_b, GT_base_lin_vel]``)
  drawn from the critic observation. The decoder is only used at training
  time, where its MSE loss is added to the PPO update by
  :class:`rsl_rl.algorithms.EncoderMultiCriticAMPPPO`.

Both actor and critic MLPs are rebuilt with an input width of
``(num_obs - history_width + latent_dim)`` so that the latent replaces the
raw history slice in the network input.

The encoder/decoder are dormant for any existing task — only configs that
inherit :class:`BaseEncoderMultiCriticAMPAgentCfg` and use the
``EncoderMultiCriticAmpRunner`` instantiate this class.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from rsl_rl.modules.multi_critic_actor_critic import MultiCriticActorCritic
from rsl_rl.utils import resolve_nn_activation


class EncoderActorCritic(MultiCriticActorCritic):
    """Multi-critic actor-critic with a history encoder + privileged decoder.

    See module docstring for the architectural sketch. The class is wire-
    compatible with the existing ``MultiCriticAmpOnPolicyRunner`` plumbing:
    ``act``, ``act_inference``, and ``evaluate`` all accept the raw flat
    observation tensors the runner already produces.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        history_slice: Sequence[int],
        critic_history_slice: Sequence[int] | None = None,
        history_len: int = 10,
        history_dim: int = 5,
        latent_dim: int = 64,
        encoder_hidden_dims: Sequence[int] = (128, 64),
        decoder_target_dim: int = 9,
        decoder_hidden_dims: Sequence[int] = (64, 64),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        num_critic_groups: int = 2,
        critic_group_names: Sequence[str] = ("goal", "aux"),
        **kwargs,
    ) -> None:
        # ---- validate slicing ------------------------------------------------
        if len(history_slice) != 2:
            raise ValueError(
                f"history_slice must be a 2-tuple (start, end); got {history_slice!r}."
            )
        start_a, end_a = int(history_slice[0]), int(history_slice[1])
        if not (0 <= start_a < end_a <= num_actor_obs):
            raise ValueError(
                f"history_slice [{start_a}, {end_a}) is out of bounds for "
                f"num_actor_obs={num_actor_obs}."
            )
        history_width = end_a - start_a
        expected_history_width = int(history_len) * int(history_dim)
        if history_width != expected_history_width:
            raise ValueError(
                f"history_slice width ({history_width}) does not match "
                f"history_len*history_dim ({history_len}*{history_dim}={expected_history_width})."
            )

        if critic_history_slice is None:
            # Mirror actor-side slice on the critic obs by default. The runner
            # produces critic obs that prefix the actor obs in most tasks.
            critic_start, critic_end = start_a, end_a
        else:
            if len(critic_history_slice) != 2:
                raise ValueError(
                    f"critic_history_slice must be a 2-tuple; got {critic_history_slice!r}."
                )
            critic_start, critic_end = int(critic_history_slice[0]), int(critic_history_slice[1])
        if not (0 <= critic_start < critic_end <= num_critic_obs):
            raise ValueError(
                f"critic_history_slice [{critic_start}, {critic_end}) is out of bounds for "
                f"num_critic_obs={num_critic_obs}."
            )
        critic_history_width = critic_end - critic_start
        if critic_history_width != history_width:
            raise ValueError(
                f"critic_history_slice width ({critic_history_width}) must equal "
                f"actor history_slice width ({history_width})."
            )

        # ---- build parent with reduced obs widths ---------------------------
        # The encoder replaces the history slice with a `latent_dim` chunk, so
        # the actor MLP must accept (num_actor_obs - history_width + latent_dim)
        # inputs. We pass that reduced width to the parent constructor so the
        # actor/critic MLPs are sized correctly and the inherited noise/dist
        # bookkeeping stays untouched.
        latent_dim = int(latent_dim)
        actor_input_dim = num_actor_obs - history_width + latent_dim
        critic_input_dim = num_critic_obs - critic_history_width + latent_dim

        super().__init__(
            num_actor_obs=actor_input_dim,
            num_critic_obs=critic_input_dim,
            num_actions=num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            num_critic_groups=num_critic_groups,
            critic_group_names=critic_group_names,
            **kwargs,
        )

        # ---- record slicing config -----------------------------------------
        self.history_slice = (start_a, end_a)
        self.critic_history_slice = (critic_start, critic_end)
        self.history_len = int(history_len)
        self.history_dim = int(history_dim)
        self.latent_dim = latent_dim
        self.history_width = history_width
        self.expected_actor_obs = int(num_actor_obs)
        self.expected_critic_obs = int(num_critic_obs)

        # ---- build encoder + decoder ---------------------------------------
        activation_module = resolve_nn_activation(activation)
        self.encoder = self._build_aux_mlp(history_width, list(encoder_hidden_dims), activation_module,
                                           out_dim=latent_dim)
        self.decoder = self._build_aux_mlp(latent_dim, list(decoder_hidden_dims), activation_module,
                                           out_dim=int(decoder_target_dim))
        self.decoder_target_dim = int(decoder_target_dim)
        self._init_linear_encoder_as_recent_history()

        # Latest latent produced by ``act``/``act_inference``/``evaluate``.
        # Cached on the module so the algorithm can compute the decoder loss
        # without re-running the encoder over the mini-batch obs.
        self.latest_latent: torch.Tensor | None = None

        print(
            f"EncoderActorCritic: history_slice={self.history_slice} ({self.history_width}d) "
            f"-> latent_dim={self.latent_dim}; decoder_target_dim={self.decoder_target_dim}; "
            f"actor_input_dim={actor_input_dim}, critic_input_dim={critic_input_dim}."
        )

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _build_aux_mlp(
        input_dim: int,
        hidden_dims: Sequence[int],
        activation_module: nn.Module,
        *,
        out_dim: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(activation_module)
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def _init_linear_encoder_as_recent_history(self) -> None:
        """Warm-start helper: linear encoder initially exposes recent history.

        With ``encoder_hidden_dims=()`` the encoder is a single Linear layer.
        Initialize its first channels as an identity over the most recent
        history frames so legacy policies that consumed a 10-frame raw history
        can be partially transferred into the latent policy.
        """
        if len(self.encoder) != 1 or not isinstance(self.encoder[0], nn.Linear):
            return
        linear = self.encoder[0]
        recent_width = min(self.latent_dim, min(self.history_len, 10) * self.history_dim)
        source_start = self.history_width - recent_width
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.zero_()
            eye = torch.eye(recent_width, device=linear.weight.device, dtype=linear.weight.dtype)
            linear.weight[:recent_width, source_start : source_start + recent_width] = eye

    def _project_obs(self, obs: torch.Tensor, slice_range: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace ``obs[..., slice_range]`` with the encoder latent.

        Returns ``(combined, latent)`` where ``combined`` is the tensor fed to
        the downstream MLP and ``latent`` is the raw encoder output (for
        caching). Slicing is the same regardless of leading batch dims; only
        the last dim is touched.
        """
        start, end = slice_range
        history_slice = obs[..., start:end]
        latent = self.encoder(history_slice)
        prefix = obs[..., :start]
        suffix = obs[..., end:]
        combined = torch.cat([prefix, latent, suffix], dim=-1)
        return combined, latent

    def _adapt_first_layer_weight(self, key: str, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
            return None
        if key == "actor.0.weight":
            history_start = self.history_slice[0]
        elif key.startswith("critics.") and key.endswith(".0.weight"):
            history_start = self.critic_history_slice[0]
        else:
            return None
        if source.shape[1] <= history_start or target.shape[1] <= history_start:
            return None

        adapted = target.clone()
        adapted[:, :history_start] = source[:, :history_start]

        suffix_width = max(target.shape[1] - (history_start + self.latent_dim), 0)
        old_history_width = source.shape[1] - history_start - suffix_width
        if old_history_width <= 0:
            return None
        copy_width = min(old_history_width, self.latent_dim)
        adapted[:, history_start : history_start + copy_width] = source[
            :, history_start : history_start + copy_width
        ]
        if suffix_width > 0:
            adapted[:, -suffix_width:] = source[:, -suffix_width:]
        return adapted

    def load_state_dict(self, state_dict, strict=True):  # type: ignore[override]
        current = self.state_dict()
        exact = set(state_dict.keys()) == set(current.keys()) and all(
            k in current and current[k].shape == v.shape for k, v in state_dict.items()
        )
        if exact:
            super().load_state_dict(state_dict, strict=strict)
            return True

        merged = {k: v.clone() for k, v in current.items()}
        loaded: list[str] = []
        adapted: list[str] = []
        skipped: list[str] = []
        for key, value in state_dict.items():
            if key not in current:
                skipped.append(key)
                continue
            target = current[key]
            if target.shape == value.shape:
                merged[key] = value.to(device=target.device, dtype=target.dtype)
                loaded.append(key)
                continue
            adapted_weight = self._adapt_first_layer_weight(
                key,
                value.to(device=target.device, dtype=target.dtype),
                target,
            )
            if adapted_weight is not None:
                merged[key] = adapted_weight
                adapted.append(key)
            else:
                skipped.append(key)

        nn.Module.load_state_dict(self, merged, strict=False)
        print(
            "[EncoderActorCritic] Warm-started from shape-mismatched checkpoint: "
            f"loaded={len(loaded)}, adapted={adapted}, skipped={len(skipped)}."
        )
        return False

    # ------------------------------------------------------ public encoder API
    def encode_actor_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(projected_obs, latent)`` for an actor-side observation."""
        return self._project_obs(obs, self.history_slice)

    def encode_critic_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(projected_obs, latent)`` for a critic-side observation."""
        return self._project_obs(obs, self.critic_history_slice)

    # ---------------------------------------------------------- forward paths
    def update_distribution(self, observations: torch.Tensor) -> None:  # type: ignore[override]
        projected, latent = self.encode_actor_obs(observations)
        self.latest_latent = latent
        super().update_distribution(projected)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        projected, latent = self.encode_actor_obs(observations)
        self.latest_latent = latent
        return self.actor(projected)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:  # type: ignore[override]
        projected, _ = self.encode_critic_obs(critic_observations)
        # Each critic returns shape (N, 1); concatenate over the last dim -> (N, G)
        values = [critic(projected) for critic in self.critics]
        return torch.cat(values, dim=-1)

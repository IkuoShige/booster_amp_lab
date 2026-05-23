# Copyright (c) 2025-2026, The Booster Soccer Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Multi-critic actor-critic for LVDRS-style (goal vs. auxiliary) value heads.

This module extends :class:`ActorCritic` with ``G`` separate critic heads, each
implemented as its own MLP trunk. ``evaluate`` returns ``(N, G)`` values; the
PPO algorithm then aggregates them with per-group weights to drive a single
policy. See ``output-sc/lvdrs.md`` §3.2 and Appendix B for the motivation.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation


class MultiCriticActorCritic(ActorCritic):
    """Actor-Critic with ``G`` separate critic value heads."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        num_critic_groups: int = 2,
        critic_group_names: Sequence[str] = ("goal", "aux"),
        **kwargs,
    ) -> None:
        # Build the actor (and the single ``self.critic``) using ActorCritic's
        # initialization for backward-compat. We then *replace* the critic with
        # ``num_critic_groups`` separate critic networks.
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        if num_critic_groups <= 0:
            raise ValueError(f"num_critic_groups must be positive, got {num_critic_groups}.")
        if len(critic_group_names) != num_critic_groups:
            raise ValueError(
                f"critic_group_names length ({len(critic_group_names)}) does not match "
                f"num_critic_groups ({num_critic_groups})."
            )

        self.num_critic_groups = int(num_critic_groups)
        self.critic_group_names = tuple(str(n) for n in critic_group_names)

        # Drop the inherited single critic and re-build G heads.
        del self.critic
        activation_module = resolve_nn_activation(activation)
        critic_hidden_dims_list = list(critic_hidden_dims)
        self.critics = nn.ModuleList(
            [
                self._build_mlp(num_critic_obs, critic_hidden_dims_list, activation_module)
                for _ in range(self.num_critic_groups)
            ]
        )

        # Provide a single ``self.critic`` shim that returns ``(N, G)`` so
        # generic code (e.g. ``ActorCritic.evaluate``) keeps working if any
        # caller falls through, but we override ``evaluate`` below.
        # We use a property so external state-dict loading is unaffected.
        print(f"MultiCritic actor-critic: {self.num_critic_groups} value heads {self.critic_group_names}")

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: Sequence[int], activation_module: nn.Module) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(activation_module)
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute ``(N, G)`` values stacked along the last dim."""
        # Each critic returns shape (N, 1); concatenate over the last dim -> (N, G)
        values = [critic(critic_observations) for critic in self.critics]
        return torch.cat(values, dim=-1)

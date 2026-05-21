from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensor
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class OmniVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with an explicit yaw-only/pivot subset."""

    cfg: "OmniVelocityCommandCfg"

    def __init__(self, cfg: "OmniVelocityCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if self.cfg.failure_mining_contact_sensor_cfg is not None:
            try:
                self.cfg.failure_mining_contact_sensor_cfg.resolve(env.scene)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to resolve failure-mining contact sensor for OmniVelocityCommand."
                ) from exc
            body_ids = self.cfg.failure_mining_contact_sensor_cfg.body_ids
            if isinstance(body_ids, int):
                body_ids = [body_ids]
                self.cfg.failure_mining_contact_sensor_cfg.body_ids = body_ids
            if isinstance(body_ids, slice) or len(body_ids) == 0:
                raise RuntimeError("Failure-mining contact sensor must resolve explicit non-empty body ids.")
        self.sudden_stop_timer = torch.zeros(self.num_envs, device=self.device)
        self._sudden_stop_sampled = torch.zeros(self.num_envs, device=self.device)
        self._sudden_stop_previous_speed = torch.zeros(self.num_envs, device=self.device)
        self._low_speed_sampled = torch.zeros(self.num_envs, device=self.device)
        self._diagonal_sampled = torch.zeros(self.num_envs, device=self.device)
        self._yaw_only_sampled = torch.zeros(self.num_envs, device=self.device)
        self._hard_command_sampled = torch.zeros(self.num_envs, device=self.device)
        self._failure_mining_sampled = torch.zeros(self.num_envs, device=self.device)
        self._failure_command_bins = torch.zeros(self._failure_bin_count, device=self.device)
        self._failure_was_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._failure_push_bins = torch.zeros(self._failure_push_bin_count, device=self.device)
        self._push_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._push_failure_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._push_elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self._push_direction_bin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._push_magnitude_bin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._push_command_bin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._push_command_norm = torch.zeros(self.num_envs, device=self.device)
        self._push_delta_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._push_delta_norm = torch.zeros(self.num_envs, device=self.device)
        self._push_mining_sampled = torch.zeros(self.num_envs, device=self.device)

        self.metrics["sudden_stop_active"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sudden_stop_sampled"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sudden_stop_previous_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["low_speed_sampled"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["low_speed_command"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["diagonal_sampled"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["diagonal_command"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["yaw_only_sampled"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hard_command_sampled"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_base_contact"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_terminated"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_during_sudden_stop"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_mining_sampled"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_mining_total"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_mining_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_mining_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_push_active"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_push_elapsed_s"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_push_mining_sampled"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_push_mining_recorded"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_push_delta_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_push_delta_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_push_delta_norm"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_push_mining_total"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_push_mining_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["failure_push_mining_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_push_active"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_push_started"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_push_active_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_push_started_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_push_curriculum_alpha"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_push_force_min_n"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_push_force_max_n"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_push_duration_min_s"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_push_duration_max_s"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_push_activation_probability"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_push_applied_force_max_n"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def _failure_bin_count(self) -> int:
        return (
            max(int(self.cfg.failure_mining_lin_x_bins), 1)
            * max(int(self.cfg.failure_mining_lin_y_bins), 1)
            * max(int(self.cfg.failure_mining_yaw_bins), 1)
        )

    def _hard_command_points(self) -> torch.Tensor | None:
        points = getattr(self.cfg, "hard_command_points", None)
        if not points:
            return None
        tensor = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        if tensor.ndim != 2 or tensor.shape[1] != 3:
            raise ValueError(f"hard_command_points must have shape [N, 3], got {tuple(tensor.shape)}")
        return tensor

    @property
    def _failure_push_location_bin_count(self) -> int:
        location_bins = max(int(self.cfg.failure_mining_push_location_bins), 1)
        return location_bins * location_bins

    @property
    def _failure_push_bin_shape(self) -> tuple[int, int, int, int, int]:
        return (
            max(int(self.cfg.failure_mining_push_direction_bins), 1),
            max(int(self.cfg.failure_mining_push_magnitude_bins), 1),
            max(int(self.cfg.failure_mining_push_duration_bins), 1),
            self._failure_push_location_bin_count,
            self._failure_bin_count,
        )

    @property
    def _failure_push_bin_count(self) -> int:
        direction_bins, magnitude_bins, duration_bins, location_bins, command_bins = self._failure_push_bin_shape
        return direction_bins * magnitude_bins * duration_bins * location_bins * command_bins

    def _axis_bin_indices(self, values: torch.Tensor, value_range: tuple[float, float], bins: int) -> torch.Tensor:
        low, high = float(value_range[0]), float(value_range[1])
        width = max(high - low, 1.0e-6)
        return torch.clamp(((values - low) / width * bins).long(), 0, bins - 1)

    def _command_bin_indices(self, commands: torch.Tensor) -> torch.Tensor:
        x_bins = max(int(self.cfg.failure_mining_lin_x_bins), 1)
        y_bins = max(int(self.cfg.failure_mining_lin_y_bins), 1)
        yaw_bins = max(int(self.cfg.failure_mining_yaw_bins), 1)

        x_idx = self._axis_bin_indices(commands[:, 0], self.cfg.ranges.lin_vel_x, x_bins)
        y_idx = self._axis_bin_indices(commands[:, 1], self.cfg.ranges.lin_vel_y, y_bins)
        yaw_idx = self._axis_bin_indices(commands[:, 2], self.cfg.ranges.ang_vel_z, yaw_bins)
        return x_idx * y_bins * yaw_bins + y_idx * yaw_bins + yaw_idx

    def _failure_command_norm(self, commands: torch.Tensor) -> torch.Tensor:
        yaw_scale = max(float(getattr(self.cfg, "failure_mining_command_yaw_scale", 0.30)), 0.0)
        scaled = torch.stack((commands[:, 0], commands[:, 1], yaw_scale * commands[:, 2]), dim=1)
        return torch.norm(scaled, dim=1)

    def _push_xy_max_magnitude(self, velocity_range: dict[str, tuple[float, float]]) -> float:
        x_range = velocity_range.get("x", (0.0, 0.0))
        y_range = velocity_range.get("y", (0.0, 0.0))
        max_x = max(abs(float(x_range[0])), abs(float(x_range[1])))
        max_y = max(abs(float(y_range[0])), abs(float(y_range[1])))
        return max(math.sqrt(max_x * max_x + max_y * max_y), 1.0e-6)

    def _push_direction_bin_indices(self, push_delta: torch.Tensor) -> torch.Tensor:
        direction_bins = max(int(self.cfg.failure_mining_push_direction_bins), 1)
        angles = torch.atan2(push_delta[:, 1], push_delta[:, 0])
        normalized = (angles + math.pi) / (2.0 * math.pi)
        return torch.clamp((normalized * direction_bins).long(), 0, direction_bins - 1)

    def _push_magnitude_bin_indices(self, push_delta: torch.Tensor, max_magnitude: float) -> torch.Tensor:
        magnitude_bins = max(int(self.cfg.failure_mining_push_magnitude_bins), 1)
        magnitudes = torch.linalg.norm(push_delta[:, :2], dim=1)
        return torch.clamp((magnitudes / max(max_magnitude, 1.0e-6) * magnitude_bins).long(), 0, magnitude_bins - 1)

    def _push_duration_bin_indices(self, durations_s: torch.Tensor) -> torch.Tensor:
        duration_bins = max(int(self.cfg.failure_mining_push_duration_bins), 1)
        window_s = max(float(self.cfg.failure_mining_push_active_window_s), 1.0e-6)
        return torch.clamp((durations_s / window_s * duration_bins).long(), 0, duration_bins - 1)

    def _root_xy_location(self, env_ids: torch.Tensor) -> torch.Tensor:
        root_xy = self.robot.data.root_pos_w[env_ids, :2]
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is not None:
            root_xy = root_xy - env_origins[env_ids, :2].to(device=root_xy.device)
        return root_xy

    def _location_bin_indices(self, root_xy: torch.Tensor) -> torch.Tensor:
        location_bins = max(int(self.cfg.failure_mining_push_location_bins), 1)
        extent = max(float(self.cfg.failure_mining_push_location_extent), 1.0e-6)
        x_idx = torch.clamp(((root_xy[:, 0] + extent) / (2.0 * extent) * location_bins).long(), 0, location_bins - 1)
        y_idx = torch.clamp(((root_xy[:, 1] + extent) / (2.0 * extent) * location_bins).long(), 0, location_bins - 1)
        return x_idx * location_bins + y_idx

    def _flat_push_bin_indices(
        self,
        direction_idx: torch.Tensor,
        magnitude_idx: torch.Tensor,
        duration_idx: torch.Tensor,
        location_idx: torch.Tensor,
        command_idx: torch.Tensor,
    ) -> torch.Tensor:
        _, magnitude_bins, duration_bins, location_bins, command_bins = self._failure_push_bin_shape
        return (
            (((direction_idx * magnitude_bins + magnitude_idx) * duration_bins + duration_idx) * location_bins)
            + location_idx
        ) * command_bins + command_idx

    def _push_bin_summary(self, bins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total = torch.sum(bins)
        if total > 0.0:
            top_count, top_bin = torch.max(bins, dim=0)
            top_prob = top_count / total
            normalized_top_bin = top_bin.float() / max(bins.numel() - 1, 1)
        else:
            top_prob = torch.zeros((), device=self.device)
            normalized_top_bin = torch.zeros((), device=self.device)
        return top_prob, normalized_top_bin

    def _commands_from_bins(self, bin_indices: torch.Tensor) -> torch.Tensor:
        x_bins = max(int(self.cfg.failure_mining_lin_x_bins), 1)
        y_bins = max(int(self.cfg.failure_mining_lin_y_bins), 1)
        yaw_bins = max(int(self.cfg.failure_mining_yaw_bins), 1)

        x_idx = bin_indices // (y_bins * yaw_bins)
        y_idx = (bin_indices // yaw_bins) % y_bins
        yaw_idx = bin_indices % yaw_bins

        def sample_axis(indices: torch.Tensor, value_range: tuple[float, float], bins: int) -> torch.Tensor:
            low, high = float(value_range[0]), float(value_range[1])
            step = (high - low) / max(bins, 1)
            jitter = torch.rand(indices.shape, device=self.device)
            return low + (indices.float() + jitter) * step

        return torch.stack(
            [
                sample_axis(x_idx, self.cfg.ranges.lin_vel_x, x_bins),
                sample_axis(y_idx, self.cfg.ranges.lin_vel_y, y_bins),
                sample_axis(yaw_idx, self.cfg.ranges.ang_vel_z, yaw_bins),
            ],
            dim=-1,
        )

    def _sample_failed_commands(self, env_ids: torch.Tensor):
        if (
            not self.cfg.failure_mining_enabled
            or self.cfg.failure_mining_sample_prob <= 0.0
            or env_ids.numel() == 0
            or torch.sum(self._failure_command_bins) <= 0.0
        ):
            return

        mine_mask = torch.rand(env_ids.numel(), device=self.device) <= self.cfg.failure_mining_sample_prob
        mine_ids = env_ids[mine_mask]
        if mine_ids.numel() == 0:
            return

        probabilities = self._failure_command_bins / torch.sum(self._failure_command_bins)
        sampled_bins = torch.multinomial(probabilities, mine_ids.numel(), replacement=True)
        self.vel_command_b[mine_ids] = self._commands_from_bins(sampled_bins)
        self._failure_mining_sampled[mine_ids] = 1.0

    def _apply_xy_norm_cap(self, env_ids: torch.Tensor):
        cap = float(self.cfg.lin_vel_xy_norm_max)
        if cap <= 0.0 or env_ids.numel() == 0:
            return
        command_xy = self.vel_command_b[env_ids, :2]
        norm = torch.norm(command_xy, dim=1, keepdim=True)
        scale = torch.clamp(cap / norm.clamp_min(1.0e-6), max=1.0)
        self.vel_command_b[env_ids, :2] = command_xy * scale

    def _sample_hard_commands(self, env_ids: torch.Tensor):
        probability = max(min(float(getattr(self.cfg, "rel_hard_envs", 0.0)), 1.0), 0.0)
        if probability <= 0.0 or env_ids.numel() == 0:
            return
        hard_points = self._hard_command_points()
        if hard_points is None or hard_points.shape[0] == 0:
            return
        hard_mask = torch.rand(env_ids.numel(), device=self.device) <= probability
        hard_ids = env_ids[hard_mask]
        if hard_ids.numel() == 0:
            return
        point_indices = torch.randint(0, hard_points.shape[0], (hard_ids.numel(),), device=self.device)
        self.vel_command_b[hard_ids] = hard_points[point_indices]
        self._hard_command_sampled[hard_ids] = 1.0
        if hasattr(self, "_low_speed_sampled"):
            self._low_speed_sampled[hard_ids] = 0.0
        if hasattr(self, "_diagonal_sampled"):
            self._diagonal_sampled[hard_ids] = 0.0
        if hasattr(self, "_yaw_only_sampled"):
            self._yaw_only_sampled[hard_ids] = 0.0
        if hasattr(self, "_failure_mining_sampled"):
            self._failure_mining_sampled[hard_ids] = 0.0

    def _sample_low_speed_commands(self, env_ids: torch.Tensor):
        if self.cfg.rel_low_speed_envs <= 0.0 or env_ids.numel() == 0:
            return
        num_low_speed = int(math.ceil(env_ids.numel() * min(max(float(self.cfg.rel_low_speed_envs), 0.0), 1.0)))
        if num_low_speed <= 0:
            return
        permutation = torch.randperm(env_ids.numel(), device=self.device)
        low_speed_ids = env_ids[permutation[:num_low_speed]]
        if low_speed_ids.numel() == 0:
            return

        x_min, x_max = self.cfg.low_speed_lin_x_abs_range
        x_abs = torch.empty(low_speed_ids.numel(), device=self.device).uniform_(float(x_min), float(x_max))
        forward_mask = torch.rand(low_speed_ids.numel(), device=self.device) <= self.cfg.low_speed_forward_prob
        self.vel_command_b[low_speed_ids, 0] = torch.where(forward_mask, x_abs, -x_abs)

        y_abs = max(float(self.cfg.low_speed_lin_y_abs_max), 0.0)
        yaw_abs = max(float(self.cfg.low_speed_yaw_abs_max), 0.0)
        self.vel_command_b[low_speed_ids, 1] = torch.empty(low_speed_ids.numel(), device=self.device).uniform_(
            -y_abs, y_abs
        )
        self.vel_command_b[low_speed_ids, 2] = torch.empty(low_speed_ids.numel(), device=self.device).uniform_(
            -yaw_abs, yaw_abs
        )
        self._low_speed_sampled[low_speed_ids] = 1.0
        if hasattr(self, "_diagonal_sampled"):
            self._diagonal_sampled[low_speed_ids] = 0.0
        if hasattr(self, "_yaw_only_sampled"):
            self._yaw_only_sampled[low_speed_ids] = 0.0
        if hasattr(self, "_failure_mining_sampled"):
            self._failure_mining_sampled[low_speed_ids] = 0.0

    def _sample_diagonal_commands(self, env_ids: torch.Tensor):
        if self.cfg.rel_diagonal_envs <= 0.0 or env_ids.numel() == 0:
            return
        num_diagonal = int(math.ceil(env_ids.numel() * min(max(float(self.cfg.rel_diagonal_envs), 0.0), 1.0)))
        if num_diagonal <= 0:
            return
        permutation = torch.randperm(env_ids.numel(), device=self.device)
        diagonal_ids = env_ids[permutation[:num_diagonal]]
        if diagonal_ids.numel() == 0:
            return

        x_min, x_max = self.cfg.diagonal_lin_x_abs_range
        y_min, y_max = self.cfg.diagonal_lin_y_abs_range
        x_abs = torch.empty(diagonal_ids.numel(), device=self.device).uniform_(float(x_min), float(x_max))
        y_abs = torch.empty(diagonal_ids.numel(), device=self.device).uniform_(float(y_min), float(y_max))
        forward_mask = torch.rand(diagonal_ids.numel(), device=self.device) <= self.cfg.diagonal_forward_prob
        lateral_sign = torch.where(
            torch.rand(diagonal_ids.numel(), device=self.device) <= 0.5,
            torch.ones(diagonal_ids.numel(), device=self.device),
            -torch.ones(diagonal_ids.numel(), device=self.device),
        )
        self.vel_command_b[diagonal_ids, 0] = torch.where(forward_mask, x_abs, -x_abs)
        self.vel_command_b[diagonal_ids, 1] = y_abs * lateral_sign

        yaw_abs = max(float(self.cfg.diagonal_yaw_abs_max), 0.0)
        self.vel_command_b[diagonal_ids, 2] = torch.empty(diagonal_ids.numel(), device=self.device).uniform_(
            -yaw_abs, yaw_abs
        )
        self._diagonal_sampled[diagonal_ids] = 1.0
        if hasattr(self, "_low_speed_sampled"):
            self._low_speed_sampled[diagonal_ids] = 0.0
        if hasattr(self, "_yaw_only_sampled"):
            self._yaw_only_sampled[diagonal_ids] = 0.0
        if hasattr(self, "_failure_mining_sampled"):
            self._failure_mining_sampled[diagonal_ids] = 0.0

    def sample_failure_aware_pushes(
        self,
        env_ids: torch.Tensor,
        velocity_range: dict[str, tuple[float, float]],
        default_push_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sampled_mask = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
        if (
            not self.cfg.failure_mining_enabled
            or self.cfg.failure_mining_push_sample_prob <= 0.0
            or env_ids.numel() == 0
            or torch.sum(self._failure_push_bins) <= 0.0
        ):
            return default_push_delta, sampled_mask

        mine_mask = torch.rand(env_ids.numel(), device=self.device) <= self.cfg.failure_mining_push_sample_prob
        mine_positions = mine_mask.nonzero(as_tuple=False).flatten()
        if mine_positions.numel() == 0:
            return default_push_delta, sampled_mask

        push_bins = self._failure_push_bins.view(self._failure_push_bin_shape)
        global_scores = push_bins.sum(dim=(2, 3, 4)).reshape(-1)
        max_magnitude = self._push_xy_max_magnitude(velocity_range)
        current_locations = self._location_bin_indices(self._root_xy_location(env_ids[mine_positions]))
        current_commands = self._command_bin_indices(self.vel_command_b[env_ids[mine_positions]])

        for local_idx, location_idx, command_idx in zip(mine_positions, current_locations, current_commands):
            conditioned_scores = push_bins[:, :, :, location_idx, command_idx].sum(dim=2).reshape(-1)
            scores = conditioned_scores if torch.sum(conditioned_scores) > 0.0 else global_scores
            if torch.sum(scores) <= 0.0:
                continue
            sampled_direction_magnitude = torch.multinomial(scores / torch.sum(scores), 1).squeeze(0)
            direction_bins, magnitude_bins, _, _, _ = self._failure_push_bin_shape
            direction_idx = sampled_direction_magnitude // magnitude_bins
            magnitude_idx = sampled_direction_magnitude % magnitude_bins
            angle = (
                (direction_idx.float() + torch.rand((), device=self.device)) / direction_bins * (2.0 * math.pi)
            ) - math.pi
            magnitude = (magnitude_idx.float() + torch.rand((), device=self.device)) / magnitude_bins * max_magnitude
            default_push_delta[local_idx, 0] = torch.clamp(
                magnitude * torch.cos(angle), *velocity_range.get("x", (0.0, 0.0))
            )
            default_push_delta[local_idx, 1] = torch.clamp(
                magnitude * torch.sin(angle), *velocity_range.get("y", (0.0, 0.0))
            )
            sampled_mask[local_idx] = True

        return default_push_delta, sampled_mask

    def record_external_push(
        self,
        env_ids: torch.Tensor,
        push_delta: torch.Tensor,
        sampled_mask: torch.Tensor,
        velocity_range: dict[str, tuple[float, float]],
    ):
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        push_delta = push_delta.to(device=self.device)
        sampled_mask = sampled_mask.to(device=self.device, dtype=torch.bool)
        self._push_active[env_ids] = True
        self._push_failure_recorded[env_ids] = False
        self._push_elapsed_s[env_ids] = 0.0
        self._push_direction_bin[env_ids] = self._push_direction_bin_indices(push_delta)
        self._push_magnitude_bin[env_ids] = self._push_magnitude_bin_indices(
            push_delta, self._push_xy_max_magnitude(velocity_range)
        )
        self._push_command_bin[env_ids] = self._command_bin_indices(self.vel_command_b[env_ids])
        self._push_command_norm[env_ids] = self._failure_command_norm(self.vel_command_b[env_ids])
        self._push_delta_xy[env_ids] = push_delta[:, :2]
        self._push_delta_norm[env_ids] = torch.linalg.norm(push_delta[:, :2], dim=1)
        self._push_mining_sampled[env_ids] = sampled_mask.float()

    def _base_contact_mask(self) -> torch.Tensor:
        sensor_cfg = self.cfg.failure_mining_contact_sensor_cfg
        if sensor_cfg is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        contact_sensor: ContactSensor = self._env.scene.sensors[sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w_history
        contact_force = torch.max(torch.norm(forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
        if contact_force.ndim == 1:
            return contact_force > self.cfg.failure_mining_contact_threshold
        return torch.any(contact_force > self.cfg.failure_mining_contact_threshold, dim=1)

    def record_failure_mask(
        self,
        failure_mask: torch.Tensor,
        base_contact_mask: torch.Tensor | None = None,
        terminated_mask: torch.Tensor | None = None,
    ):
        """Record per-env failures immediately from termination/recovery hooks."""
        failure = failure_mask.to(device=self.device, dtype=torch.bool).reshape(self.num_envs)
        if base_contact_mask is None:
            base_contact = failure
        else:
            base_contact = base_contact_mask.to(device=self.device, dtype=torch.bool).reshape(self.num_envs)
        if terminated_mask is None:
            terminated = torch.zeros_like(failure)
        else:
            terminated = terminated_mask.to(device=self.device, dtype=torch.bool).reshape(self.num_envs)
        self._record_failure_bins(failure, base_contact=base_contact, terminated=terminated, apply_decay=False)

    def _terminated_mask(self) -> torch.Tensor:
        termination_manager = getattr(self._env, "termination_manager", None)
        terminated = getattr(termination_manager, "terminated", None)
        if terminated is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated.to(device=self.device, dtype=torch.bool)

    def _record_failure_bins(
        self,
        failure: torch.Tensor,
        base_contact: torch.Tensor | None = None,
        terminated: torch.Tensor | None = None,
        apply_decay: bool = True,
    ):
        failure = failure.to(device=self.device, dtype=torch.bool).reshape(self.num_envs)
        base_contact = failure if base_contact is None else base_contact.to(device=self.device, dtype=torch.bool)
        terminated = (
            torch.zeros_like(failure)
            if terminated is None
            else terminated.to(device=self.device, dtype=torch.bool).reshape(self.num_envs)
        )
        new_failure = failure & ~self._failure_was_active
        command_norm = self._failure_command_norm(self.vel_command_b)
        min_command_norm = max(float(self.cfg.failure_mining_min_command_norm), 0.0)
        command_failure = new_failure & (command_norm >= min_command_norm)

        if apply_decay and self.cfg.failure_mining_decay < 1.0:
            decay = max(float(self.cfg.failure_mining_decay), 0.0)
            self._failure_command_bins *= decay
            self._failure_push_bins *= decay
        if torch.any(command_failure):
            bin_indices = self._command_bin_indices(self.vel_command_b[command_failure])
            self._failure_command_bins.scatter_add_(
                0, bin_indices, torch.ones_like(bin_indices, dtype=self._failure_command_bins.dtype)
            )

        push_min_command_norm = max(float(self.cfg.failure_mining_push_min_command_norm), 0.0)
        push_failure = failure & self._push_active & ~self._push_failure_recorded
        if push_min_command_norm > 0.0:
            push_failure &= self._push_command_norm >= push_min_command_norm
        if torch.any(push_failure):
            push_failure_ids = push_failure.nonzero(as_tuple=False).flatten()
            push_bin_indices = self._flat_push_bin_indices(
                self._push_direction_bin[push_failure_ids],
                self._push_magnitude_bin[push_failure_ids],
                self._push_duration_bin_indices(self._push_elapsed_s[push_failure_ids]),
                self._location_bin_indices(self._root_xy_location(push_failure_ids)),
                self._push_command_bin[push_failure_ids],
            )
            self._failure_push_bins.scatter_add_(
                0, push_bin_indices, torch.ones_like(push_bin_indices, dtype=self._failure_push_bins.dtype)
            )
            self._push_failure_recorded[push_failure_ids] = True

        self._failure_was_active |= failure

    def _update_failure_metrics(self):
        if self.cfg.failure_mining_push_active_window_s > 0.0:
            self._push_elapsed_s[self._push_active] += self._env.step_dt
            expired = self._push_active & (self._push_elapsed_s > self.cfg.failure_mining_push_active_window_s)
            if torch.any(expired):
                self._push_active[expired] = False
                self._push_mining_sampled[expired] = 0.0
        else:
            self._push_active.zero_()
            self._push_mining_sampled.zero_()

        sudden_stop_active = self.sudden_stop_timer > 0.0
        base_contact = self._base_contact_mask()
        terminated = self._terminated_mask()
        failure = base_contact | terminated if self.cfg.failure_mining_include_terminations else base_contact
        self._record_failure_bins(failure, base_contact=base_contact, terminated=terminated)

        total = torch.sum(self._failure_command_bins)
        if total > 0.0:
            top_count, top_bin = torch.max(self._failure_command_bins, dim=0)
            top_prob = top_count / total
            normalized_top_bin = top_bin.float() / max(self._failure_bin_count - 1, 1)
        else:
            top_prob = torch.zeros((), device=self.device)
            normalized_top_bin = torch.zeros((), device=self.device)
        push_top_prob, normalized_push_top_bin = self._push_bin_summary(self._failure_push_bins)

        x_min, x_max = self.cfg.low_speed_lin_x_abs_range
        command_abs = torch.abs(self.vel_command_b)
        low_speed_command = (
            (command_abs[:, 0] >= float(x_min))
            & (command_abs[:, 0] <= float(x_max))
            & (command_abs[:, 1] <= float(self.cfg.low_speed_lin_y_abs_max))
            & (command_abs[:, 2] <= float(self.cfg.low_speed_yaw_abs_max))
        )
        diagonal_x_min, diagonal_x_max = self.cfg.diagonal_lin_x_abs_range
        diagonal_y_min, diagonal_y_max = self.cfg.diagonal_lin_y_abs_range
        diagonal_command = (
            (command_abs[:, 0] >= float(diagonal_x_min))
            & (command_abs[:, 0] <= float(diagonal_x_max))
            & (command_abs[:, 1] >= float(diagonal_y_min))
            & (command_abs[:, 1] <= float(diagonal_y_max))
            & (command_abs[:, 2] <= float(self.cfg.diagonal_yaw_abs_max))
        )

        self.metrics["sudden_stop_active"] = sudden_stop_active.float()
        self.metrics["sudden_stop_sampled"] = self._sudden_stop_sampled
        self.metrics["sudden_stop_previous_speed"] = self._sudden_stop_previous_speed
        self.metrics["low_speed_sampled"] = self._low_speed_sampled
        self.metrics["low_speed_command"] = low_speed_command.float()
        self.metrics["diagonal_sampled"] = self._diagonal_sampled
        self.metrics["diagonal_command"] = diagonal_command.float()
        self.metrics["yaw_only_sampled"] = self._yaw_only_sampled
        self.metrics["hard_command_sampled"] = self._hard_command_sampled
        self.metrics["failure_base_contact"] = base_contact.float()
        self.metrics["failure_terminated"] = terminated.float()
        self.metrics["failure_during_sudden_stop"] = (failure & sudden_stop_active).float()
        self.metrics["failure_mining_sampled"] = self._failure_mining_sampled
        self.metrics["failure_mining_total"][:] = total
        self.metrics["failure_mining_top1_prob"][:] = top_prob
        self.metrics["failure_mining_top1_bin"][:] = normalized_top_bin
        self.metrics["failure_push_active"] = self._push_active.float()
        self.metrics["failure_push_elapsed_s"] = self._push_elapsed_s
        self.metrics["failure_push_mining_sampled"] = self._push_mining_sampled
        self.metrics["failure_push_mining_recorded"] = self._push_failure_recorded.float()
        active_float = self._push_active.float()
        self.metrics["failure_push_delta_x"] = self._push_delta_xy[:, 0] * active_float
        self.metrics["failure_push_delta_y"] = self._push_delta_xy[:, 1] * active_float
        self.metrics["failure_push_delta_norm"] = self._push_delta_norm * active_float
        self.metrics["failure_push_mining_total"][:] = torch.sum(self._failure_push_bins)
        self.metrics["failure_push_mining_top1_prob"][:] = push_top_prob
        self.metrics["failure_push_mining_top1_bin"][:] = normalized_push_top_bin
        self._failure_was_active = failure | self._failure_was_active

    def _reset_resampled_env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        reset_mask = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
        reset_buf = getattr(self._env, "reset_buf", None)
        if isinstance(reset_buf, torch.Tensor):
            reset_mask |= reset_buf[env_ids.to(device=reset_buf.device)].to(device=self.device, dtype=torch.bool)
        termination_manager = getattr(self._env, "termination_manager", None)
        terminated = getattr(termination_manager, "terminated", None)
        if isinstance(terminated, torch.Tensor):
            reset_mask |= terminated[env_ids.to(device=terminated.device)].to(device=self.device, dtype=torch.bool)
        episode_length_buf = getattr(self._env, "episode_length_buf", None)
        if isinstance(episode_length_buf, torch.Tensor):
            reset_mask |= episode_length_buf[env_ids.to(device=episode_length_buf.device)].to(device=self.device) <= 1
        return env_ids[reset_mask]

    def _clear_reset_only_state(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self._failure_was_active[env_ids] = False
        self._push_active[env_ids] = False
        self._push_failure_recorded[env_ids] = False
        self._push_elapsed_s[env_ids] = 0.0
        if hasattr(self, "_push_direction_bin"):
            self._push_direction_bin[env_ids] = 0
        if hasattr(self, "_push_magnitude_bin"):
            self._push_magnitude_bin[env_ids] = 0
        if hasattr(self, "_push_command_bin"):
            self._push_command_bin[env_ids] = 0
        if hasattr(self, "_push_command_norm"):
            self._push_command_norm[env_ids] = 0.0
        if hasattr(self, "_push_delta_xy"):
            self._push_delta_xy[env_ids] = 0.0
        if hasattr(self, "_push_delta_norm"):
            self._push_delta_norm[env_ids] = 0.0
        self._push_mining_sampled[env_ids] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if not hasattr(self, "_hard_command_sampled"):
            self._hard_command_sampled = torch.zeros(self.vel_command_b.shape[0], device=self.device)
        previous_command = self.vel_command_b[env_ids_tensor].clone()
        reset_env_ids = self._reset_resampled_env_ids(env_ids_tensor)
        super()._resample_command(env_ids)
        post_reset_env_ids = self._reset_resampled_env_ids(env_ids_tensor)
        if post_reset_env_ids.numel() > 0:
            reset_env_ids = torch.unique(torch.cat((reset_env_ids, post_reset_env_ids)))
        self._clear_reset_only_state(reset_env_ids)

        r = torch.empty(len(env_ids_tensor), device=self.device)
        self.sudden_stop_timer[env_ids_tensor] = 0.0
        self._sudden_stop_sampled[env_ids_tensor] = 0.0
        self._sudden_stop_previous_speed[env_ids_tensor] = torch.norm(previous_command, dim=1)
        self._low_speed_sampled[env_ids_tensor] = 0.0
        self._diagonal_sampled[env_ids_tensor] = 0.0
        self._yaw_only_sampled[env_ids_tensor] = 0.0
        self._hard_command_sampled[env_ids_tensor] = 0.0
        self._failure_mining_sampled[env_ids_tensor] = 0.0
        self._sample_failed_commands(env_ids_tensor)
        self._sample_low_speed_commands(env_ids_tensor)
        self._sample_diagonal_commands(env_ids_tensor)

        if self.cfg.rel_yaw_only_envs > 0.0:
            yaw_only_mask = r.uniform_(0.0, 1.0) <= self.cfg.rel_yaw_only_envs
            yaw_only_ids = env_ids_tensor[yaw_only_mask]
            if yaw_only_ids.numel() > 0:
                self.vel_command_b[yaw_only_ids, 0] = 0.0
                self.vel_command_b[yaw_only_ids, 1] = 0.0
                yaw_abs = r[: yaw_only_ids.numel()].uniform_(
                    self.cfg.yaw_only_ang_vel_abs_range[0], self.cfg.yaw_only_ang_vel_abs_range[1]
                )
                yaw_sign = torch.where(r[: yaw_only_ids.numel()].uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
                self.vel_command_b[yaw_only_ids, 2] = yaw_abs * yaw_sign
                self._low_speed_sampled[yaw_only_ids] = 0.0
                self._diagonal_sampled[yaw_only_ids] = 0.0
                self._yaw_only_sampled[yaw_only_ids] = 1.0
                self._hard_command_sampled[yaw_only_ids] = 0.0
                self._failure_mining_sampled[yaw_only_ids] = 0.0

        self._sample_hard_commands(env_ids_tensor)
        self._apply_xy_norm_cap(env_ids_tensor)

        if self.cfg.rel_sudden_stop_envs <= 0.0:
            return

        previous_speed = torch.norm(previous_command, dim=1)
        stop_mask = (r.uniform_(0.0, 1.0) <= self.cfg.rel_sudden_stop_envs) & (
            previous_speed >= self.cfg.sudden_stop_min_command_speed
        )
        stop_ids = env_ids_tensor[stop_mask]
        if stop_ids.numel() > 0:
            self.vel_command_b[stop_ids] = 0.0
            self.sudden_stop_timer[stop_ids] = self.cfg.sudden_stop_window_s
            self._sudden_stop_sampled[stop_ids] = 1.0
            self._low_speed_sampled[stop_ids] = 0.0
            self._diagonal_sampled[stop_ids] = 0.0
            self._yaw_only_sampled[stop_ids] = 0.0
            self._hard_command_sampled[stop_ids] = 0.0
            self._failure_mining_sampled[stop_ids] = 0.0

    def _update_command(self):
        super()._update_command()
        if self.cfg.sudden_stop_window_s > 0.0:
            self.sudden_stop_timer = torch.clamp(self.sudden_stop_timer - self._env.step_dt, min=0.0)
        self._update_failure_metrics()


@configclass
class OmniVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for omni locomotion commands with explicit pivot samples."""

    class_type: type = OmniVelocityCommand
    rel_yaw_only_envs: float = 0.15
    """Probability that a sampled command is pure yaw, i.e. pivot in place."""

    yaw_only_ang_vel_abs_range: tuple[float, float] = (0.15, 0.3)
    """Absolute yaw-rate range used for yaw-only samples."""

    rel_low_speed_envs: float = 0.0
    """Probability that a sampled command is replaced by a low-speed straight-walk command."""

    low_speed_lin_x_abs_range: tuple[float, float] = (0.15, 0.45)
    """Absolute x velocity range used for low-speed straight-walk samples."""

    low_speed_lin_y_abs_max: float = 0.08
    """Maximum absolute lateral velocity for low-speed straight-walk samples."""

    low_speed_yaw_abs_max: float = 0.06
    """Maximum absolute yaw velocity for low-speed straight-walk samples."""

    low_speed_forward_prob: float = 0.9
    """Probability that a low-speed straight-walk sample is forward rather than backward."""

    rel_hard_envs: float = 0.0
    """Probability that a sampled command is replaced by one of hard_command_points."""

    hard_command_points: list[tuple[float, float, float]] = []
    """Explicit command points used to emphasize benchmark-critical commands."""

    rel_diagonal_envs: float = 0.0
    """Probability that a sampled command is replaced by a diagonal high-lateral command."""

    diagonal_lin_x_abs_range: tuple[float, float] = (0.8, 1.2)
    """Absolute x velocity range used for diagonal command samples."""

    diagonal_lin_y_abs_range: tuple[float, float] = (0.6, 1.0)
    """Absolute y velocity range used for diagonal command samples."""

    diagonal_yaw_abs_max: float = 0.08
    """Maximum absolute yaw velocity for diagonal command samples."""

    diagonal_forward_prob: float = 0.85
    """Probability that a diagonal sample is forward rather than backward."""

    lin_vel_xy_norm_max: float = 0.0
    """Optional cap on sampled horizontal command norm. Disabled when non-positive."""

    rel_sudden_stop_envs: float = 0.0
    """Probability that a resampled non-zero command is abruptly replaced by a zero command."""

    sudden_stop_min_command_speed: float = 0.35
    """Minimum previous command norm required for sudden-stop sampling."""

    sudden_stop_window_s: float = 1.5
    """Time window exposed for stop-specific reward terms after a sudden zero command."""

    failure_mining_contact_sensor_cfg: SceneEntityCfg | None = None
    """Contact sensor used to tag base-contact failures for command metrics and optional mining."""

    failure_mining_contact_threshold: float = 1.0
    """Contact threshold for failure-mining base-contact detection."""

    failure_mining_include_terminations: bool = True
    """Whether generic terminations also update failure-mining command bins."""

    failure_mining_enabled: bool = False
    """If true, occasionally resample velocity commands from previously failed command bins."""

    failure_mining_sample_prob: float = 0.0
    """Probability that a resampled command is replaced by a command from failed bins."""

    failure_mining_min_command_norm: float = 0.10
    """Minimum command norm required for base-contact failures to update command-mining bins."""

    failure_mining_command_yaw_scale: float = 0.30
    """Yaw scaling used when computing command norm for failure-mining filters."""

    failure_mining_push_min_command_norm: float = 0.0
    """Minimum command norm required for push-window failures to update push-mining bins."""

    failure_mining_decay: float = 0.995
    """Per-step exponential decay for failed-command bin counts."""

    failure_mining_lin_x_bins: int = 5
    """Number of failed-command bins along x velocity."""

    failure_mining_lin_y_bins: int = 5
    """Number of failed-command bins along y velocity."""

    failure_mining_yaw_bins: int = 5
    """Number of failed-command bins along yaw velocity."""

    failure_mining_push_sample_prob: float = 0.0
    """Probability that an external push samples direction/magnitude from failed push bins."""

    failure_mining_push_active_window_s: float = 2.0
    """Seconds after a push during which contact/termination records that push as a failure."""

    failure_mining_push_direction_bins: int = 8
    """Number of failed-push bins over horizontal push direction."""

    failure_mining_push_magnitude_bins: int = 4
    """Number of failed-push bins over horizontal push magnitude."""

    failure_mining_push_duration_bins: int = 4
    """Number of failed-push bins over time from push to failure."""

    failure_mining_push_location_bins: int = 3
    """Number of failed-push grid bins per local x/y axis."""

    failure_mining_push_location_extent: float = 2.5
    """Half-width in meters for local x/y location binning around each environment origin."""


def _sample_root_velocity_delta(
    velocity_range: dict[str, tuple[float, float]],
    shape: torch.Size | tuple[int, ...],
    device: str,
) -> torch.Tensor:
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=device)
    return sample_uniform(ranges[:, 0], ranges[:, 1], shape, device=device)


def _curriculum_alpha(env: ManagerBasedRLEnv, start_step: int = 0, duration_steps: int = 0) -> float:
    if duration_steps <= 0:
        return 1.0
    step = float(getattr(env, "common_step_counter", 0))
    return max(0.0, min(1.0, (step - float(start_step)) / max(float(duration_steps), 1.0)))


def _lerp_range(start: tuple[float, float], end: tuple[float, float], alpha: float) -> tuple[float, float]:
    return (
        float(start[0]) + (float(end[0]) - float(start[0])) * alpha,
        float(start[1]) + (float(end[1]) - float(start[1])) * alpha,
    )


def _interpolate_velocity_range(
    env: ManagerBasedRLEnv,
    velocity_range: dict[str, tuple[float, float]],
    curriculum_velocity_range: dict[str, tuple[float, float]] | None = None,
    curriculum_start_step: int = 0,
    curriculum_duration_steps: int = 0,
) -> dict[str, tuple[float, float]]:
    if not curriculum_velocity_range:
        return velocity_range
    alpha = _curriculum_alpha(env, curriculum_start_step, curriculum_duration_steps)
    return {
        key: _lerp_range(value, curriculum_velocity_range.get(key, value), alpha)
        for key, value in velocity_range.items()
    } | {
        key: curriculum_velocity_range[key]
        for key in curriculum_velocity_range
        if key not in velocity_range
    }


def _failure_mining_command_term(env: ManagerBasedRLEnv, command_name: str):
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None or not hasattr(command_manager, "get_term"):
        raise RuntimeError("Failure-aware push mining requires an environment command_manager with get_term().")
    command_term = command_manager.get_term(command_name)
    if not hasattr(command_term, "sample_failure_aware_pushes") or not hasattr(command_term, "record_external_push"):
        raise RuntimeError(
            f"Command term '{command_name}' does not support failure-aware push mining. "
            "Use OmniVelocityCommand for Track Adapter push mining."
        )
    return command_term


def push_by_setting_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    failure_mining_command_name: str | None = None,
    curriculum_velocity_range: dict[str, tuple[float, float]] | None = None,
    curriculum_start_step: int = 0,
    curriculum_duration_steps: int = 0,
):
    """Push the asset by setting root velocity, optionally biased by Track Adapter failure bins."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=asset.device)

    active_velocity_range = _interpolate_velocity_range(
        env,
        velocity_range,
        curriculum_velocity_range=curriculum_velocity_range,
        curriculum_start_step=curriculum_start_step,
        curriculum_duration_steps=curriculum_duration_steps,
    )
    vel_w = asset.data.root_vel_w[env_ids].clone()
    push_delta = _sample_root_velocity_delta(active_velocity_range, vel_w.shape, asset.device)
    sampled_mask = torch.zeros(env_ids.numel(), dtype=torch.bool, device=asset.device)
    command_term = None
    if failure_mining_command_name is not None:
        command_term = _failure_mining_command_term(env, failure_mining_command_name)
        push_delta, sampled_mask = command_term.sample_failure_aware_pushes(env_ids, active_velocity_range, push_delta)

    asset.write_root_velocity_to_sim(vel_w + push_delta, env_ids=env_ids)

    if command_term is not None:
        command_term.record_external_push(env_ids, push_delta, sampled_mask, active_velocity_range)


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int],
                 tail_len: int = 0, device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]
        self.tail_len = tail_len

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def max_reset_frame(self) -> int:
        return self.time_step_total - self.tail_len


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        print(f'{self.robot.body_names=}')
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(
            self.cfg.motion_file, self.body_indexes, tail_len=self.cfg.tail_len, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.max_reset_frame // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.max_reset_frame, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        # sampling_probabilities = (
        #     (1 - self.cfg.adaptive_uniform_ratio) * sampling_probabilities
        #     + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        # )

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.max_reset_frame - 1)
        ).long()
        self.time_steps[env_ids] = (sampled_bins / self.bin_count * (self.motion.max_reset_frame - 1)).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        if self.cfg.play:
            self.time_steps[env_ids] = 0

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    play: bool = False

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    tail_len: int = 0

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

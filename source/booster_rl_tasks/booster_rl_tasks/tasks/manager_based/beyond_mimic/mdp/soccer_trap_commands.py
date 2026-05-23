"""Command term for the V4 Stage 2 soccer trap-skill task.

Single K1 (the *receiver*) + 1 ball. On reset the ball is spawned 1.5–3 m
away from the robot in a forward cone and given an initial velocity (2–5 m/s)
pointing roughly toward the robot (±20° spread, height 0.11–0.5 m). The robot
must time its foot position to deaden the ball — trap success is defined as
ball within ``trap_success_radius`` of a robot foot AND ball xy-speed below
``trap_success_ball_speed`` (m/s).

This module mirrors :mod:`soccer_commands.SoccerKickCommand` where it
helps observation / reward functions resolve common attributes
(``ball_pos_b_perceived``, ``ball_mask_perceived``, ``last_seen_dt``,
``ball_history``, ``robot_yaw_quat``, ``num_envs``, ``device``, ``role``).
There is no kicker / pass / shoot mode here — the receiver IS the robot.

Pure-tensor implementation; no Python loops over envs.
"""
from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, yaw_quat

try:
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import (
        BLUE_ARROW_X_MARKER_CFG,
        GREEN_ARROW_X_MARKER_CFG,
        RED_ARROW_X_MARKER_CFG,
    )

    _MARKERS_AVAILABLE = True
except Exception:  # pragma: no cover — headless / minimal envs
    VisualizationMarkers = None  # type: ignore[assignment]
    BLUE_ARROW_X_MARKER_CFG = None  # type: ignore[assignment]
    GREEN_ARROW_X_MARKER_CFG = None  # type: ignore[assignment]
    RED_ARROW_X_MARKER_CFG = None  # type: ignore[assignment]
    _MARKERS_AVAILABLE = False

from booster_rl_tasks.assets.objects.soccer import (
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    SOCCER_BALL_RADIUS,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_perception import (
    VirtualPerception,
    VirtualPerceptionCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SoccerTrapCommand(CommandTerm):
    """Per-env trap-skill state for the V4 Stage 2 task.

    Randomization (robot pose, ball spawn pose + initial velocity) happens in
    :meth:`_resample_command`. Per-step bookkeeping (body-yaw frame ball state,
    perception update + history buffer push, trap-success latch, predicted
    contact distance) lives in :meth:`_update_command`.
    """

    cfg: "SoccerTrapCommandCfg"

    def __init__(self, cfg: "SoccerTrapCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._env = env
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._ball: RigidObject = env.scene[cfg.ball_name]

        # Resolve foot body indices once.
        foot_ids, _ = self._robot.find_bodies(list(cfg.foot_body_names), preserve_order=True)
        self._foot_ids = torch.as_tensor(foot_ids, dtype=torch.long, device=self.device)

        N = self.num_envs
        d = self.device

        # Cached per-step world quantities (updated each compute()) -----------
        self._ball_pos_w = torch.zeros(N, 3, device=d)
        self._ball_vel_w = torch.zeros(N, 3, device=d)
        self._robot_pos_w = torch.zeros(N, 3, device=d)
        self._robot_yaw_quat = torch.zeros(N, 4, device=d)
        self._robot_yaw_quat[:, 0] = 1.0  # identity wxyz

        # Privileged / yaw-frame quantities -----------------------------------
        self._ball_pos_b = torch.zeros(N, 3, device=d)
        self._ball_vel_b = torch.zeros(N, 3, device=d)

        # Trap-detection latch (one-shot per episode) -------------------------
        self._trap_success_awarded = torch.zeros(N, dtype=torch.bool, device=d)
        # Predicted closest-approach distance over the next horizon (m).
        # Updated each step; consumed by the ``trap_anticipation`` reward.
        self._predicted_contact_distance = torch.full((N,), float("inf"), device=d)

        # Metrics for logging -------------------------------------------------
        self.metrics["trap_success_rate"] = torch.zeros(N, device=d)
        self.metrics["ball_xy_speed_at_episode_end"] = torch.zeros(N, device=d)
        self.metrics["time_to_trap"] = torch.zeros(N, device=d)
        # Internal accumulator: step counter since ball was launched.
        self._steps_since_launch = torch.zeros(N, dtype=torch.long, device=d)
        self._time_to_trap = torch.zeros(N, device=d)

        # Virtual perception (optional). When ``cfg.perception`` is None the
        # observations fall back to ground-truth values. When set, the actor
        # sees noisy/intermittent detections matching the head-camera pipeline.
        self._perception: VirtualPerception | None = None
        if cfg.perception is not None:
            self._perception = VirtualPerception(
                cfg=cfg.perception,
                robot=self._robot,
                num_envs=N,
                dt=float(env.step_dt),
                device=d,
            )

        # Perceived ball state buffers (read by observations).
        self._ball_pos_b_perceived = torch.zeros(N, 3, device=d)
        self._ball_mask_perceived = torch.ones(N, device=d)
        self._last_seen_dt_buf = torch.zeros(N, device=d)

        # V4 Step A — ball observation history buffer. Same layout as the
        # kick variant so :func:`soccer_role_obs.ball_history_flat` can be
        # reused without modification.
        self._ball_history_len: int = int(cfg.ball_history_len)
        self._ball_history_dim: int = 5
        self._ball_history_buf = torch.zeros(
            self._ball_history_len, N, self._ball_history_dim, device=d
        )

    # --- Public properties -------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        """Trap task has no per-env target signal — return an empty (N, 0) tensor."""
        return torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def ball_pos_w(self) -> torch.Tensor:
        return self._ball_pos_w

    @property
    def ball_vel_w(self) -> torch.Tensor:
        return self._ball_vel_w

    @property
    def ball_pos_b(self) -> torch.Tensor:
        return self._ball_pos_b

    @property
    def ball_vel_b(self) -> torch.Tensor:
        return self._ball_vel_b

    @property
    def robot(self) -> Articulation:
        return self._robot

    @property
    def ball(self) -> RigidObject:
        return self._ball

    @property
    def foot_ids(self) -> torch.Tensor:
        return self._foot_ids

    @property
    def trap_success_awarded(self) -> torch.Tensor:
        """One-shot per-env latch — True once the ball has been trapped."""
        return self._trap_success_awarded

    @property
    def predicted_contact_distance(self) -> torch.Tensor:
        """Predicted closest foot↔ball xy distance over the next
        ``cfg.predicted_contact_horizon_s`` seconds, assuming straight-line
        ball motion at current velocity. Shape: (num_envs,).
        """
        return self._predicted_contact_distance

    @property
    def robot_yaw_quat(self) -> torch.Tensor:
        return self._robot_yaw_quat

    @property
    def ball_pos_b_perceived(self) -> torch.Tensor:
        return self._ball_pos_b_perceived

    @property
    def ball_mask_perceived(self) -> torch.Tensor:
        return self._ball_mask_perceived

    @property
    def last_seen_dt(self) -> torch.Tensor:
        return self._last_seen_dt_buf

    @property
    def perception(self) -> "VirtualPerception | None":
        return self._perception

    @property
    def ball_history(self) -> torch.Tensor:
        """Ball observation history (read by ``soccer_role_obs.ball_history_flat``).

        Shape: ``(history_len, num_envs, 5)``.
        """
        return self._ball_history_buf

    @property
    def ball_history_len(self) -> int:
        return self._ball_history_len

    @property
    def role(self) -> str:
        """Per-agent role string (read by ``soccer_role_obs.role_one_hot``)."""
        return str(self.cfg.role)

    # --- Required overrides -----------------------------------------------
    def _update_metrics(self):
        self.metrics["trap_success_rate"][:] = self._trap_success_awarded.float()
        # Final ball xy speed — updated each step; observers read the last
        # value at episode-end logging.
        self.metrics["ball_xy_speed_at_episode_end"][:] = torch.linalg.norm(
            self._ball_vel_w[:, :2], dim=-1
        )
        # Time-to-trap: latched at trap moment (zero otherwise).
        self.metrics["time_to_trap"][:] = self._time_to_trap

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids_t = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        n = int(env_ids_t.numel())
        if n == 0:
            return
        d = self.device

        # ----- Robot pose --------------------------------------------------
        spawn_x = _uniform(*self.cfg.robot_spawn_x_range, n=n, device=d)
        spawn_y = _uniform(*self.cfg.robot_spawn_y_range, n=n, device=d)
        spawn_yaw = _uniform(*self.cfg.robot_spawn_yaw_range, n=n, device=d)

        env_origins = self._env.scene.env_origins[env_ids_t]

        robot_pose = torch.zeros(n, 7, device=d)
        robot_pose[:, 0] = env_origins[:, 0] + spawn_x
        robot_pose[:, 1] = env_origins[:, 1] + spawn_y
        robot_pose[:, 2] = env_origins[:, 2] + self.cfg.robot_spawn_z
        robot_pose[:, 3:7] = quat_from_euler_xyz(
            torch.zeros_like(spawn_yaw),
            torch.zeros_like(spawn_yaw),
            spawn_yaw,
        )
        self._robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids_t)
        self._robot.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=d), env_ids=env_ids_t
        )

        default_jp = self._robot.data.default_joint_pos[env_ids_t]
        default_jv = self._robot.data.default_joint_vel[env_ids_t]
        self._robot.write_joint_state_to_sim(default_jp, default_jv, env_ids=env_ids_t)

        # ----- Ball spawn pose (virtual kicker position) -------------------
        # Spawn ball 1.5–3 m from receiver in a forward cone (configurable
        # half-angle around receiver +x in body-yaw frame), random height.
        ball_dist = _uniform(*self.cfg.ball_spawn_distance_range, n=n, device=d)
        ball_angle = _uniform(*self.cfg.ball_spawn_angle_range, n=n, device=d)
        ball_height = _uniform(*self.cfg.ball_spawn_height_range, n=n, device=d)

        local_x = ball_dist * torch.cos(ball_angle)
        local_y = ball_dist * torch.sin(ball_angle)
        cos_y = torch.cos(spawn_yaw)
        sin_y = torch.sin(spawn_yaw)
        offset_x_w = cos_y * local_x - sin_y * local_y
        offset_y_w = sin_y * local_x + cos_y * local_y

        ball_pose = torch.zeros(n, 7, device=d)
        ball_pose[:, 0] = robot_pose[:, 0] + offset_x_w
        ball_pose[:, 1] = robot_pose[:, 1] + offset_y_w
        ball_pose[:, 2] = env_origins[:, 2] + ball_height
        ball_pose[:, 3] = 1.0
        self._ball.write_root_pose_to_sim(ball_pose, env_ids=env_ids_t)

        # ----- Ball initial velocity (toward receiver, with ±20° noise) ----
        # Direction = (robot_xy - ball_xy) / |...| rotated by uniform angular
        # noise. Magnitude sampled in cfg.ball_initial_speed_range.
        speed = _uniform(*self.cfg.ball_initial_speed_range, n=n, device=d)
        angle_noise = _uniform(*self.cfg.ball_initial_angle_noise, n=n, device=d)

        vec_x = robot_pose[:, 0] - ball_pose[:, 0]
        vec_y = robot_pose[:, 1] - ball_pose[:, 1]
        norm = torch.sqrt(vec_x * vec_x + vec_y * vec_y).clamp_min(1e-6)
        dir_x = vec_x / norm
        dir_y = vec_y / norm
        # Rotate (dir_x, dir_y) by angle_noise.
        c = torch.cos(angle_noise)
        s = torch.sin(angle_noise)
        v_x = (c * dir_x - s * dir_y) * speed
        v_y = (s * dir_x + c * dir_y) * speed

        ball_vel = torch.zeros(n, 6, device=d)
        ball_vel[:, 0] = v_x
        ball_vel[:, 1] = v_y
        # Tiny positive z component so the ball isn't immediately stuck on
        # the floor friction when it spawns at non-zero height. Keep it
        # small so the trajectory is still roughly straight.
        ball_vel[:, 2] = 0.0
        self._ball.write_root_velocity_to_sim(ball_vel, env_ids=env_ids_t)

        # ----- Reset latches ----------------------------------------------
        self._trap_success_awarded[env_ids_t] = False
        self._predicted_contact_distance[env_ids_t] = float("inf")
        self._steps_since_launch[env_ids_t] = 0
        self._time_to_trap[env_ids_t] = 0.0

        # ----- Perception reset (resamples per-env DR coefficients) -------
        if self._perception is not None:
            self._perception.reset(env_ids_t)
            self._ball_pos_b_perceived[env_ids_t] = 0.0
            self._ball_mask_perceived[env_ids_t] = 0.0
            self._last_seen_dt_buf[env_ids_t] = 0.0

        # ----- V4 Step A — clear ball history buffer for reset envs ------
        self._ball_history_buf[:, env_ids_t, :] = 0.0

    def _update_command(self):
        d = self.device

        # ----- Cache world state -----------------------------------------
        self._ball_pos_w = self._ball.data.root_pos_w.clone()
        self._ball_vel_w = self._ball.data.root_lin_vel_w.clone()
        self._robot_pos_w = self._robot.data.root_pos_w.clone()
        self._robot_yaw_quat = yaw_quat(self._robot.data.root_quat_w)

        # ----- ball xyz in body-yaw frame --------------------------------
        rel_ball_w = self._ball_pos_w - self._robot_pos_w
        self._ball_pos_b = quat_apply_inverse(self._robot_yaw_quat, rel_ball_w)
        self._ball_vel_b = quat_apply_inverse(self._robot_yaw_quat, self._ball_vel_w)

        # ----- Virtual perception update --------------------------------
        if self._perception is not None:
            self._perception.update(self._robot, self._ball_pos_w)
            self._ball_pos_b_perceived[:, :2] = self._perception.ball_pos_b
            # Constant z fill: ball spawn radius (camera projection assumed
            # to live on the ground plane, identical to the kick command).
            self._ball_pos_b_perceived[:, 2] = SOCCER_BALL_RADIUS
            self._ball_mask_perceived = self._perception.ball_mask
            self._last_seen_dt_buf = self._perception.last_seen_dt
        else:
            self._ball_pos_b_perceived = self._ball_pos_b
            self._ball_mask_perceived = torch.ones(self.num_envs, device=d)
            self._last_seen_dt_buf = torch.zeros(self.num_envs, device=d)

        # ----- V4 Step A — push current perceived state into history ----
        prev_pos = self._ball_history_buf[-1, :, :2]
        prev_mask = self._ball_history_buf[-1, :, 2]
        cur_pos = self._ball_pos_b_perceived[:, :2]
        cur_mask = self._ball_mask_perceived
        dt = float(self._env.step_dt)
        valid_diff = prev_mask * cur_mask
        ball_speed_b = (
            torch.linalg.norm(cur_pos - prev_pos, dim=-1) / max(dt, 1e-6) * valid_diff
        )
        new_slot = torch.stack(
            [
                cur_pos[:, 0],
                cur_pos[:, 1],
                cur_mask,
                self._last_seen_dt_buf,
                ball_speed_b.clamp_max(20.0),
            ],
            dim=-1,
        )
        if self._ball_history_len > 1:
            self._ball_history_buf[:-1] = self._ball_history_buf[1:].clone()
        self._ball_history_buf[-1] = new_slot

        # ----- Foot world positions (reused below) ----------------------
        foot_pos_w = self._robot.data.body_pos_w[:, self._foot_ids, :]  # (N, 2, 3)

        # ----- Trap success latching ------------------------------------
        ball_p = self._ball_pos_w.unsqueeze(1)
        foot_ball_d = torch.linalg.norm(foot_pos_w - ball_p, dim=-1)  # (N, 2)
        min_foot_ball_d = foot_ball_d.amin(dim=-1)
        ball_xy_speed = torch.linalg.norm(self._ball_vel_w[:, :2], dim=-1)
        trap_now = (min_foot_ball_d < float(self.cfg.trap_success_radius)) & (
            ball_xy_speed < float(self.cfg.trap_success_ball_speed)
        )
        new_trap = trap_now & ~self._trap_success_awarded
        # Record time-to-trap on the rising edge.
        steps = self._steps_since_launch.float() * dt
        self._time_to_trap = torch.where(new_trap, steps, self._time_to_trap)
        self._trap_success_awarded = self._trap_success_awarded | new_trap
        # Step counter increments every step (used only for time-to-trap).
        self._steps_since_launch = self._steps_since_launch + 1

        # ----- Predicted contact distance (anticipation reward) ---------
        # Project the ball's xy position forward by horizon seconds assuming
        # straight-line motion. Sample 5 points along the projection and take
        # the smallest foot↔ball xy distance encountered.
        horizon_s = float(self.cfg.predicted_contact_horizon_s)
        n_samples = 5
        ts = torch.linspace(0.0, horizon_s, steps=n_samples, device=d)
        # Shape: (n_samples, N, 2). Project ball xy along its current vel.
        ball_xy = self._ball_pos_w[:, :2]
        ball_vxy = self._ball_vel_w[:, :2]
        proj_pos = ball_xy.unsqueeze(0) + ts.unsqueeze(-1).unsqueeze(-1) * ball_vxy.unsqueeze(0)
        # Foot xy: (N, 2_feet, 2).
        foot_xy = foot_pos_w[:, :, :2]
        # Distance per (sample, env, foot). Expand foot to (1, N, 2_feet, 2).
        dist = torch.linalg.norm(
            proj_pos.unsqueeze(2) - foot_xy.unsqueeze(0), dim=-1
        )  # (n_samples, N, 2)
        # Min over feet, then min over samples -> (N,).
        self._predicted_contact_distance = dist.amin(dim=-1).amin(dim=0)

    # --- Debug visualization ----------------------------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Show ball velocity arrow + receiver yaw marker (optional)."""
        if debug_vis:
            if not _MARKERS_AVAILABLE:
                return
            if not hasattr(self, "_ball_vel_marker"):
                try:
                    vel_cfg = RED_ARROW_X_MARKER_CFG.replace(
                        prim_path="/Visuals/Command/soccer_trap_ball_vel"
                    )
                    self._ball_vel_marker = VisualizationMarkers(vel_cfg)
                except Exception:  # pragma: no cover
                    self._ball_vel_marker = None
                    return
            for m in (getattr(self, "_ball_vel_marker", None),):
                if m is not None:
                    try:
                        m.set_visibility(True)
                    except Exception:  # pragma: no cover
                        pass
        else:
            for m in (getattr(self, "_ball_vel_marker", None),):
                if m is not None:
                    try:
                        m.set_visibility(False)
                    except Exception:  # pragma: no cover
                        pass

    def _debug_vis_callback(self, event):
        if not getattr(self._robot, "is_initialized", True):
            return
        marker = getattr(self, "_ball_vel_marker", None)
        if marker is None:
            return
        d = self.device
        N = self.num_envs
        zeros = torch.zeros(N, device=d)
        try:
            pos = self._ball_pos_w.clone()
            pos[:, 2] = pos[:, 2] + 0.15
            vyaw = torch.atan2(self._ball_vel_w[:, 1], self._ball_vel_w[:, 0])
            quat = quat_from_euler_xyz(zeros, zeros, vyaw)
            marker.visualize(translations=pos, orientations=quat)
        except Exception:  # pragma: no cover
            pass


@configclass
class SoccerTrapCommandCfg(CommandTermCfg):
    """Configuration for :class:`SoccerTrapCommand`."""

    class_type: type = SoccerTrapCommand
    asset_name: str = "robot"
    ball_name: str = "ball"

    # Robot spawn pose (world frame, relative to env_origin).
    robot_spawn_x_range: tuple[float, float] = (-FIELD_HALF_LENGTH + 2.0, FIELD_HALF_LENGTH - 4.0)
    robot_spawn_y_range: tuple[float, float] = (-FIELD_HALF_WIDTH + 1.5, FIELD_HALF_WIDTH - 1.5)
    robot_spawn_yaw_range: tuple[float, float] = (-math.pi, math.pi)
    robot_spawn_z: float = 0.57  # matches BOOSTER_K1_CFG.init_state.pos.z

    # Ball spawn: polar coordinates relative to receiver's body-yaw frame.
    ball_spawn_distance_range: tuple[float, float] = (1.5, 3.0)
    ball_spawn_angle_range: tuple[float, float] = (-math.pi / 4, math.pi / 4)
    ball_spawn_height_range: tuple[float, float] = (SOCCER_BALL_RADIUS, 0.5)

    # Initial ball velocity (toward receiver, with angular noise).
    ball_initial_speed_range: tuple[float, float] = (2.0, 5.0)
    ball_initial_angle_noise: tuple[float, float] = (-math.pi / 9, math.pi / 9)  # ±20°

    # Foot bodies used for trap detection.
    foot_body_names: tuple[str, str] = ("left_foot_link", "right_foot_link")

    # Trap-success thresholds.
    trap_success_radius: float = 0.30
    trap_success_ball_speed: float = 0.4

    # Optional virtual perception. When None, observations use ground-truth.
    perception: VirtualPerceptionCfg | None = None

    # V4 Step A — ball observation history length (number of past frames the
    # actor sees). 10 frames at 50 Hz ≈ 200 ms.
    ball_history_len: int = 10

    # V4 — per-agent role. For Stage 2 this is fixed to "receiver".
    role: str = "receiver"

    # Anticipation reward: forward-projection horizon (seconds).
    predicted_contact_horizon_s: float = 0.5

    # Episode driver — disable timed resampling so resets come only from env reset.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)


def _uniform(lo: float, hi: float, *, n: int, device: torch.device) -> torch.Tensor:
    return torch.rand(n, device=device) * (hi - lo) + lo

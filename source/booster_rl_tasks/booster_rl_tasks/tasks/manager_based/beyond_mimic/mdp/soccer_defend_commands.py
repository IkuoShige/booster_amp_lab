"""Command term for the V4 Stage 3 Soccer Defend Skill task.

Owns:
  * randomized defender pose at episode reset (close to its own-goal)
  * randomized ball spawn at a "kicker" position 4-8 m in front of the
    defender, with an initial velocity (2-8 m/s) pointed toward the own-goal
  * one-shot latches for ``block_success`` (non-foot body contact),
    ``steal_success`` (foot near ball + ball decelerates) and
    ``own_goal_scored`` (ball crosses own-goal line)
  * per-step ``is_in_lane`` flag: defender between ball and own-goal center
  * per-step ``ball_pushed_away_flag``: post-contact ball moves away from
    own-goal
  * privileged ball state in body-yaw frame (read by critic obs / rewards)

The command intentionally mirrors :class:`SoccerKickCommand`'s public API
(``ball_pos_w``, ``ball_pos_b``, ``ball_history``, ``role``, etc.) so the
shared :mod:`soccer_role_obs` and :mod:`soccer_observations` helpers can be
reused without modification.
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

from booster_rl_tasks.assets.objects.soccer import (
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    GOAL_HALF_WIDTH,
    GOAL_LINE_X,
    SOCCER_BALL_RADIUS,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_perception import (
    VirtualPerception,
    VirtualPerceptionCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SoccerDefendCommand(CommandTerm):
    """Holds per-env soccer defend state."""

    cfg: "SoccerDefendCommandCfg"

    def __init__(self, cfg: "SoccerDefendCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._env = env
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._ball: RigidObject = env.scene[cfg.ball_name]

        N = self.num_envs
        d = self.device

        # ---- Body id resolution ------------------------------------------
        # Feet (used for steal-detection).
        foot_ids, _ = self._robot.find_bodies(list(cfg.foot_body_names), preserve_order=True)
        self._foot_ids = torch.as_tensor(foot_ids, dtype=torch.long, device=d)

        # Non-foot bodies (used for block-detection). Resolved by taking the
        # full body list and removing the foot bodies. Using a regex-style
        # find_bodies(".*") here keeps this URDF-agnostic.
        all_ids, _ = self._robot.find_bodies(".*", preserve_order=True)
        foot_set = set(int(x) for x in foot_ids)
        block_ids = [int(i) for i in all_ids if int(i) not in foot_set]
        # Fall back to all-bodies if somehow we filter everything out, so we
        # don't crash with a 0-d index tensor on degenerate URDFs.
        if len(block_ids) == 0:
            block_ids = [int(i) for i in all_ids]
        self._block_body_ids = torch.as_tensor(block_ids, dtype=torch.long, device=d)

        # ---- Cached per-step world quantities ----------------------------
        self._ball_pos_w = torch.zeros(N, 3, device=d)
        self._ball_vel_w = torch.zeros(N, 3, device=d)
        self._robot_pos_w = torch.zeros(N, 3, device=d)
        self._robot_yaw_quat = torch.zeros(N, 4, device=d)
        self._robot_yaw_quat[:, 0] = 1.0  # identity wxyz

        # ---- Privileged / yaw-frame quantities ---------------------------
        self._ball_pos_b = torch.zeros(N, 3, device=d)
        self._ball_vel_b = torch.zeros(N, 3, device=d)
        self._own_goal_pos_b = torch.zeros(N, 2, device=d)
        self._own_goal_dir_b = torch.zeros(N, 2, device=d)

        # ---- Latches -----------------------------------------------------
        self._block_success_awarded = torch.zeros(N, dtype=torch.bool, device=d)
        self._steal_success_awarded = torch.zeros(N, dtype=torch.bool, device=d)
        self._own_goal_scored = torch.zeros(N, dtype=torch.bool, device=d)
        self._ball_pushed_away_flag = torch.zeros(N, dtype=torch.bool, device=d)
        self._is_in_lane = torch.zeros(N, dtype=torch.bool, device=d)

        # V4.1 — continuous intercept signal. ``_intercept_distance`` is the
        # perpendicular distance from the defender to the ball→own-goal line
        # segment (capped by projection onto the segment for "between"
        # configurations, else distance to the nearest segment endpoint).
        # Used by ``ball_intercept_proximity`` for continuous shaping.
        self._intercept_distance = torch.zeros(N, device=d)

        # ---- Ball-speed history for steal-deceleration detection ---------
        # Stored as a rolling buffer of shape (lookback, N).
        self._steal_lookback = int(cfg.steal_frame_lookback)
        self._ball_xy_speed_hist = torch.zeros(self._steal_lookback, N, device=d)

        # ---- Optional virtual perception ---------------------------------
        self._perception: VirtualPerception | None = None
        if cfg.perception is not None:
            self._perception = VirtualPerception(
                cfg=cfg.perception,
                robot=self._robot,
                num_envs=N,
                dt=float(env.step_dt),
                device=d,
            )

        # Perceived ball state buffers (read by observations / debug viz).
        self._ball_pos_b_perceived = torch.zeros(N, 3, device=d)
        self._ball_mask_perceived = torch.ones(N, device=d)
        self._last_seen_dt_buf = torch.zeros(N, device=d)

        # ---- Ball history buffer (Step A) --------------------------------
        # Same layout as SoccerKickCommand so the shared
        # :func:`soccer_role_obs.ball_history_flat` helper works unchanged.
        self._ball_history_len: int = int(cfg.ball_history_len)
        self._ball_history_dim: int = 5
        self._ball_history_buf = torch.zeros(
            self._ball_history_len, N, self._ball_history_dim, device=d
        )

        # ---- Metrics -----------------------------------------------------
        # V4.1: per-episode EMAs for terminal-style latches (see
        # SoccerKickCommand for rationale). Block / steal are not terminal
        # but happen ~75 % through the episode, so a raw latch average is a
        # weak signal too — EMA gives a cleaner per-episode rate.
        self._block_success_rate_ema = torch.zeros(N, device=d)
        self._steal_success_rate_ema = torch.zeros(N, device=d)
        self._own_goal_scored_rate_ema = torch.zeros(N, device=d)
        self._rate_ema_alpha: float = 0.03

        self.metrics["block_success_rate"] = torch.zeros(N, device=d)
        self.metrics["steal_success_rate"] = torch.zeros(N, device=d)
        self.metrics["own_goal_scored_rate"] = torch.zeros(N, device=d)
        self.metrics["lane_hold_frac"] = torch.zeros(N, device=d)
        self.metrics["peak_ball_speed"] = torch.zeros(N, device=d)
        self.metrics["intercept_distance_avg"] = torch.zeros(N, device=d)
        self._peak_ball_speed_buf = torch.zeros(N, device=d)

    # =====================================================================
    # Public properties (mirror :class:`SoccerKickCommand` shape so the
    # shared role/history obs helpers can be reused.)
    # =====================================================================

    @property
    def command(self) -> torch.Tensor:
        """Concatenated command vector — for defender we expose
        ``[own_goal_dir_b (cos, sin), is_in_lane]``. Shape (N, 3)."""
        return torch.cat(
            [self._own_goal_dir_b, self._is_in_lane.float().unsqueeze(-1)],
            dim=-1,
        )

    @property
    def robot(self) -> Articulation:
        return self._robot

    @property
    def ball(self) -> RigidObject:
        return self._ball

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
    def ball_pos_b_perceived(self) -> torch.Tensor:
        return self._ball_pos_b_perceived

    @property
    def ball_mask_perceived(self) -> torch.Tensor:
        return self._ball_mask_perceived

    @property
    def last_seen_dt(self) -> torch.Tensor:
        return self._last_seen_dt_buf

    @property
    def robot_yaw_quat(self) -> torch.Tensor:
        return self._robot_yaw_quat

    @property
    def ball_history(self) -> torch.Tensor:
        return self._ball_history_buf

    @property
    def ball_history_len(self) -> int:
        return self._ball_history_len

    @property
    def perception(self) -> "VirtualPerception | None":
        return self._perception

    @property
    def role(self) -> str:
        return str(self.cfg.role)

    # ---- Defend-specific latches / quantities -------------------------------

    @property
    def block_success_awarded(self) -> torch.Tensor:
        return self._block_success_awarded

    @property
    def steal_success_awarded(self) -> torch.Tensor:
        return self._steal_success_awarded

    @property
    def own_goal_scored(self) -> torch.Tensor:
        return self._own_goal_scored

    @property
    def is_in_lane(self) -> torch.Tensor:
        return self._is_in_lane

    @property
    def intercept_distance(self) -> torch.Tensor:
        """V4.1: defender's distance to the ball→own-goal segment (m).

        Continuous signal; ~0 means the defender is exactly on the line
        between the ball and the own-goal — the position most likely to
        result in a block. Used by :func:`ball_intercept_proximity`.

        Shape: ``(num_envs,)``.
        """
        return self._intercept_distance

    @property
    def ball_pushed_away_flag(self) -> torch.Tensor:
        return self._ball_pushed_away_flag

    @property
    def own_goal_pos_b(self) -> torch.Tensor:
        return self._own_goal_pos_b

    @property
    def own_goal_dir_b(self) -> torch.Tensor:
        return self._own_goal_dir_b

    # =====================================================================
    # Required overrides
    # =====================================================================

    def _update_metrics(self):
        # V4.1: report per-episode EMA rates instead of per-step latch
        # averages. ``lane_hold_frac`` stays a per-step average because the
        # latch is naturally per-step (defender is in lane each frame).
        self.metrics["block_success_rate"][:] = self._block_success_rate_ema
        self.metrics["steal_success_rate"][:] = self._steal_success_rate_ema
        self.metrics["own_goal_scored_rate"][:] = self._own_goal_scored_rate_ema
        self.metrics["lane_hold_frac"][:] = self._is_in_lane.float()
        self.metrics["peak_ball_speed"][:] = self._peak_ball_speed_buf
        self.metrics["intercept_distance_avg"][:] = self._intercept_distance

    def _resample_command(self, env_ids: Sequence[int]):
        # Normalize env_ids to a 1-D long tensor.
        if isinstance(env_ids, slice):
            env_ids_t = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        n = int(env_ids_t.numel())
        if n == 0:
            return
        d = self.device

        # ---- Defender pose ----------------------------------------------
        spawn_x = _uniform(*self.cfg.defender_spawn_x_range, n=n, device=d)
        spawn_y = _uniform(*self.cfg.defender_spawn_y_range, n=n, device=d)
        spawn_yaw = _uniform(*self.cfg.defender_spawn_yaw_range, n=n, device=d)

        env_origins = self._env.scene.env_origins[env_ids_t]

        robot_pose = torch.zeros(n, 7, device=d)
        robot_pose[:, 0] = env_origins[:, 0] + spawn_x
        robot_pose[:, 1] = env_origins[:, 1] + spawn_y
        robot_pose[:, 2] = env_origins[:, 2] + self.cfg.defender_spawn_z
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

        # ---- Ball spawn at a "kicker" position in front of the defender --
        # "In front" = +x direction in env frame (toward the shoot goal), so
        # an inbound shot directed at the own-goal travels in -x.
        ball_dist = _uniform(*self.cfg.ball_spawn_distance_range, n=n, device=d)
        ball_y_offset = _uniform(*self.cfg.ball_spawn_y_range, n=n, device=d)

        ball_pose = torch.zeros(n, 7, device=d)
        # Robot world xy + forward offset (we ignore yaw here so the ball is
        # always east of the defender, mirroring the §4.3 "ball at random
        # kicker position in front of the defender" spec).
        ball_pose[:, 0] = robot_pose[:, 0] + ball_dist
        ball_pose[:, 1] = robot_pose[:, 1] + ball_y_offset
        # Clamp ball into the field so a wide spawn doesn't immediately fly out.
        max_x = env_origins[:, 0] + float(FIELD_HALF_LENGTH) - 0.5
        ball_pose[:, 0] = torch.minimum(ball_pose[:, 0], max_x)
        ball_pose[:, 2] = env_origins[:, 2] + self.cfg.ball_spawn_height
        ball_pose[:, 3] = 1.0
        self._ball.write_root_pose_to_sim(ball_pose, env_ids=env_ids_t)

        # ---- Ball initial velocity, pointed at own-goal center -----------
        own_goal_x_w = env_origins[:, 0] + float(self.cfg.own_goal_line_x)
        own_goal_y_w = env_origins[:, 1] + 0.0
        vec_x = own_goal_x_w - ball_pose[:, 0]
        vec_y = own_goal_y_w - ball_pose[:, 1]
        base_yaw = torch.atan2(vec_y, vec_x)
        # ±15° angular noise around the own-goal direction.
        noise = _uniform(*self.cfg.ball_initial_angle_noise, n=n, device=d)
        shot_yaw = base_yaw + noise
        speed = _uniform(*self.cfg.ball_initial_speed_range, n=n, device=d)
        vel = torch.zeros(n, 6, device=d)
        vel[:, 0] = speed * torch.cos(shot_yaw)
        vel[:, 1] = speed * torch.sin(shot_yaw)
        # Small upward kick to keep the ball from sticking immediately.
        vel[:, 2] = 0.0
        self._ball.write_root_velocity_to_sim(vel, env_ids=env_ids_t)

        # ---- V4.1 — Capture per-episode EMAs BEFORE clearing latches -----
        alpha = float(self._rate_ema_alpha)
        one_minus = 1.0 - alpha
        just_block = self._block_success_awarded[env_ids_t].float()
        just_steal = self._steal_success_awarded[env_ids_t].float()
        just_own_goal = self._own_goal_scored[env_ids_t].float()
        self._block_success_rate_ema[env_ids_t] = (
            alpha * just_block + one_minus * self._block_success_rate_ema[env_ids_t]
        )
        self._steal_success_rate_ema[env_ids_t] = (
            alpha * just_steal + one_minus * self._steal_success_rate_ema[env_ids_t]
        )
        self._own_goal_scored_rate_ema[env_ids_t] = (
            alpha * just_own_goal + one_minus * self._own_goal_scored_rate_ema[env_ids_t]
        )

        # ---- Reset latches ----------------------------------------------
        self._block_success_awarded[env_ids_t] = False
        self._steal_success_awarded[env_ids_t] = False
        self._own_goal_scored[env_ids_t] = False
        self._ball_pushed_away_flag[env_ids_t] = False
        self._is_in_lane[env_ids_t] = False
        self._intercept_distance[env_ids_t] = 0.0
        self._ball_xy_speed_hist[:, env_ids_t] = 0.0
        self._peak_ball_speed_buf[env_ids_t] = 0.0

        # ---- Perception reset -------------------------------------------
        if self._perception is not None:
            self._perception.reset(env_ids_t)
            self._ball_pos_b_perceived[env_ids_t] = 0.0
            self._ball_mask_perceived[env_ids_t] = 0.0
            self._last_seen_dt_buf[env_ids_t] = 0.0

        # ---- Clear ball history -----------------------------------------
        self._ball_history_buf[:, env_ids_t, :] = 0.0

    def _update_command(self):
        d = self.device
        N = self.num_envs
        env_origins = self._env.scene.env_origins

        # ---- Cache world state -------------------------------------------
        self._ball_pos_w = self._ball.data.root_pos_w.clone()
        self._ball_vel_w = self._ball.data.root_lin_vel_w.clone()
        self._robot_pos_w = self._robot.data.root_pos_w.clone()
        self._robot_yaw_quat = yaw_quat(self._robot.data.root_quat_w)

        # ---- Ball xyz in body-yaw frame ----------------------------------
        rel_ball_w = self._ball_pos_w - self._robot_pos_w
        self._ball_pos_b = quat_apply_inverse(self._robot_yaw_quat, rel_ball_w)
        self._ball_vel_b = quat_apply_inverse(self._robot_yaw_quat, self._ball_vel_w)

        # ---- Virtual perception update ----------------------------------
        if self._perception is not None:
            self._perception.update(self._robot, self._ball_pos_w)
            self._ball_pos_b_perceived[:, :2] = self._perception.ball_pos_b
            self._ball_pos_b_perceived[:, 2] = self.cfg.ball_spawn_height
            self._ball_mask_perceived = self._perception.ball_mask
            self._last_seen_dt_buf = self._perception.last_seen_dt
        else:
            self._ball_pos_b_perceived = self._ball_pos_b
            self._ball_mask_perceived = torch.ones(N, device=d)
            self._last_seen_dt_buf = torch.zeros(N, device=d)

        # ---- Push current perceived state into the history buffer -------
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

        # ---- Own-goal direction in body-yaw frame ------------------------
        own_goal_world_xy = torch.zeros(N, 2, device=d)
        own_goal_world_xy[:, 0] = env_origins[:, 0] + float(self.cfg.own_goal_line_x)
        own_goal_world_xy[:, 1] = env_origins[:, 1] + 0.0
        rel_g_w = torch.zeros(N, 3, device=d)
        rel_g_w[:, 0:2] = own_goal_world_xy - self._robot_pos_w[:, :2]
        rel_g_b = quat_apply_inverse(self._robot_yaw_quat, rel_g_w)
        self._own_goal_pos_b = rel_g_b[:, :2]
        gn = torch.linalg.norm(rel_g_b[:, :2], dim=-1, keepdim=True).clamp_min(1e-6)
        self._own_goal_dir_b = rel_g_b[:, :2] / gn

        # ---- Ball-speed history (steal deceleration) --------------------
        ball_xy_speed = torch.linalg.norm(self._ball_vel_w[:, :2], dim=-1)
        if self._steal_lookback > 1:
            self._ball_xy_speed_hist[:-1] = self._ball_xy_speed_hist[1:].clone()
        self._ball_xy_speed_hist[-1] = ball_xy_speed
        self._peak_ball_speed_buf = torch.maximum(
            self._peak_ball_speed_buf, ball_xy_speed
        )

        # ---- Block detection (any non-foot body within radius) ----------
        block_pos_w = self._robot.data.body_pos_w[:, self._block_body_ids, :]
        ball_p = self._ball_pos_w.unsqueeze(1)
        block_d = torch.linalg.norm(block_pos_w - ball_p, dim=-1)
        min_block_d = block_d.amin(dim=-1)
        block_now = min_block_d < float(self.cfg.block_radius)
        new_block = block_now & ~self._block_success_awarded
        self._block_success_awarded = self._block_success_awarded | new_block

        # ---- Steal detection (foot near ball AND ball decelerates) ------
        foot_pos_w = self._robot.data.body_pos_w[:, self._foot_ids, :]
        foot_d = torch.linalg.norm(foot_pos_w - ball_p, dim=-1)
        min_foot_d = foot_d.amin(dim=-1)
        foot_near = min_foot_d < float(self.cfg.steal_radius)
        # ball deceleration vs ``steal_frame_lookback`` frames ago: positive
        # value means ball xy speed dropped by that much.
        past_speed = self._ball_xy_speed_hist[0]
        deceleration = past_speed - ball_xy_speed
        decel_strong = deceleration > float(self.cfg.steal_deceleration_thresh)
        steal_now = foot_near & decel_strong
        new_steal = steal_now & ~self._steal_success_awarded
        self._steal_success_awarded = self._steal_success_awarded | new_steal

        # ---- Lane-hold detection ----------------------------------------
        # Defender is "in lane" when it lies between the ball and the
        # own-goal, within a corridor 0.8 m wide. Specifically:
        #   project (defender - own_goal) onto (ball - own_goal)
        #   in_lane iff:
        #     - perpendicular distance < 0.8 m
        #     - projection scalar in [0.1, 0.9] (between, not behind)
        ball_xy = self._ball_pos_w[:, :2]
        og_xy = own_goal_world_xy  # already (N, 2)
        defender_xy = self._robot_pos_w[:, :2]
        lane_vec = ball_xy - og_xy  # (N, 2)
        lane_len2 = (lane_vec * lane_vec).sum(-1).clamp_min(1e-6)
        d_vec = defender_xy - og_xy
        proj_scalar = (d_vec * lane_vec).sum(-1) / lane_len2  # in [0, 1] when between
        proj_point = og_xy + proj_scalar.unsqueeze(-1) * lane_vec
        perp = torch.linalg.norm(defender_xy - proj_point, dim=-1)
        self._is_in_lane = (
            (perp < float(self.cfg.lane_perp_thresh))
            & (proj_scalar > float(self.cfg.lane_proj_lo))
            & (proj_scalar < float(self.cfg.lane_proj_hi))
        )

        # V4.1 — continuous intercept distance for ``ball_intercept_proximity``.
        # Distance from the defender to the closest point ON the ball→own-goal
        # *segment*: when the projection falls within [0, 1] it's the
        # perpendicular distance ``perp``; outside that range it's the
        # distance to the nearest segment endpoint. This makes the signal
        # well-defined even when the defender stands behind the ball or
        # past the own-goal.
        proj_clamped = proj_scalar.clamp(0.0, 1.0)
        proj_point_seg = og_xy + proj_clamped.unsqueeze(-1) * lane_vec
        self._intercept_distance = torch.linalg.norm(
            defender_xy - proj_point_seg, dim=-1
        )

        # ---- Ball-pushed-away flag --------------------------------------
        # Post-contact (block OR steal awarded), AND ball velocity projection
        # onto (own_goal - ball) is negative (ball moving away from own goal).
        post_contact = self._block_success_awarded | self._steal_success_awarded
        clearing_vec = og_xy - ball_xy  # (N, 2)
        cn = torch.linalg.norm(clearing_vec, dim=-1).clamp_min(1e-6)
        clearing_dir = clearing_vec / cn.unsqueeze(-1)
        ball_proj = (self._ball_vel_w[:, :2] * clearing_dir).sum(-1)
        # ball is moving toward goal when ball_proj > 0; we want < 0.
        pushed = post_contact & (ball_proj < 0.0)
        self._ball_pushed_away_flag = pushed

        # ---- Own-goal-scored detection ----------------------------------
        bx_local = self._ball_pos_w[:, 0] - env_origins[:, 0]
        by_local = self._ball_pos_w[:, 1] - env_origins[:, 1]
        # Ball "crossed the line" when bx_local <= own_goal_line_x + small
        # margin AND y is within the post width.
        scored = (
            (bx_local < (float(self.cfg.own_goal_line_x) + float(self.cfg.own_goal_margin)))
            & (torch.abs(by_local) < float(self.cfg.own_goal_half_width))
        )
        self._own_goal_scored = self._own_goal_scored | scored

    # =====================================================================
    # Debug viz (no-op in headless; mirrors SoccerKickCommand)
    # =====================================================================

    def _set_debug_vis_impl(self, debug_vis: bool):  # pragma: no cover — viewer only
        pass

    def _debug_vis_callback(self, event):  # pragma: no cover — viewer only
        pass


@configclass
class SoccerDefendCommandCfg(CommandTermCfg):
    """Configuration for :class:`SoccerDefendCommand`."""

    class_type: type = SoccerDefendCommand
    asset_name: str = "robot"
    ball_name: str = "ball"

    # Defender spawn: positioned 1-4 m in front of the own-goal (env-frame).
    defender_spawn_x_range: tuple[float, float] = (
        -FIELD_HALF_LENGTH + 1.0,
        -FIELD_HALF_LENGTH + 4.0,
    )
    defender_spawn_y_range: tuple[float, float] = (-2.0, 2.0)
    defender_spawn_yaw_range: tuple[float, float] = (-math.pi / 4.0, math.pi / 4.0)
    defender_spawn_z: float = 0.57  # matches BOOSTER_K1_CFG.init_state.pos.z

    # Ball spawn at a "kicker" position 4-8 m in front of the defender (+x).
    ball_spawn_distance_range: tuple[float, float] = (4.0, 8.0)
    ball_spawn_y_range: tuple[float, float] = (-1.5, 1.5)
    ball_spawn_height: float = SOCCER_BALL_RADIUS

    # Ball initial velocity (pointed at own-goal center, with angular noise).
    ball_initial_speed_range: tuple[float, float] = (2.0, 8.0)
    ball_initial_angle_noise: tuple[float, float] = (-math.pi / 12.0, math.pi / 12.0)

    # Block / steal detection thresholds.
    block_radius: float = 0.30
    steal_radius: float = 0.25
    steal_deceleration_thresh: float = 0.5
    steal_frame_lookback: int = 5

    foot_body_names: tuple[str, str] = ("left_foot_link", "right_foot_link")

    # Own-goal geometry (mirror of the shoot goal at x = -GOAL_LINE_X).
    own_goal_line_x: float = -GOAL_LINE_X
    own_goal_half_width: float = GOAL_HALF_WIDTH
    own_goal_margin: float = 0.10  # how far inside the line counts as "scored"

    # Lane-hold corridor: defender is "in lane" when its perpendicular
    # distance to the ball↔own-goal segment is < ``lane_perp_thresh`` and
    # the projection scalar lies in (lane_proj_lo, lane_proj_hi).
    lane_perp_thresh: float = 0.8
    lane_proj_lo: float = 0.1
    lane_proj_hi: float = 0.9

    # Optional virtual perception (None → fall back to GT, matching
    # :class:`SoccerKickCommand`'s default).
    perception: VirtualPerceptionCfg | None = None

    # Ball history length (Step A).
    ball_history_len: int = 10

    # Per-agent role string (constant per skill task — Stage 3 = "defender").
    role: str = "defender"

    # Disable timed resampling so resets come only from env reset.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)


def _uniform(lo: float, hi: float, *, n: int, device: torch.device) -> torch.Tensor:
    return torch.rand(n, device=device) * (hi - lo) + lo

from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
import math
import os
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from booster_assets import BOOSTER_ASSETS_DIR
from booster_rl_tasks.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE
from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
import booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp as mdp
from .tracking_env_cfg import TrackingEnvCfg


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_force_magnitude_mixture(name: str) -> tuple[tuple[float, float, float], ...] | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    components: list[tuple[float, float, float]] = []
    for index, raw_component in enumerate(value.replace(";", ",").split(",")):
        component = raw_component.strip()
        if not component:
            continue
        parts = [part.strip() for part in component.split(":")]
        if len(parts) != 3:
            raise ValueError(
                f"{name} component #{index + 1} must use 'min:max:weight' format, got {component!r}."
            )
        low, high, weight = (float(parts[0]), float(parts[1]), float(parts[2]))
        if not all(math.isfinite(value) for value in (low, high, weight)):
            raise ValueError(f"{name} component #{index + 1} must contain finite values, got {component!r}.")
        if high <= low:
            raise ValueError(f"{name} component #{index + 1} must have max > min, got {component!r}.")
        if weight <= 0.0:
            raise ValueError(f"{name} component #{index + 1} must have positive weight, got {component!r}.")
        components.append((low, high, weight))
    if not components:
        raise ValueError(f"{name} did not contain any valid force mixture components.")
    return tuple(components)


def _env_force_duration_mixture(name: str) -> tuple[tuple[float, float, float, float, float, float, float], ...] | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    components: list[tuple[float, float, float, float, float, float, float]] = []
    for index, raw_component in enumerate(value.replace(";", ",").split(",")):
        component = raw_component.strip()
        if not component:
            continue
        parts = [part.strip() for part in component.split(":")]
        if len(parts) != 7:
            raise ValueError(
                f"{name} component #{index + 1} must use "
                "'force_min:force_max:duration_min:duration_max:ramp_min:ramp_max:weight' format, "
                f"got {component!r}."
            )
        force_low, force_high, duration_low, duration_high, ramp_low, ramp_high, weight = (
            float(parts[0]),
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
            float(parts[5]),
            float(parts[6]),
        )
        values = (force_low, force_high, duration_low, duration_high, ramp_low, ramp_high, weight)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} component #{index + 1} must contain finite values, got {component!r}.")
        if force_high <= force_low:
            raise ValueError(f"{name} component #{index + 1} must have force_max > force_min, got {component!r}.")
        if duration_high <= duration_low:
            raise ValueError(f"{name} component #{index + 1} must have duration_max > duration_min, got {component!r}.")
        if weight <= 0.0:
            raise ValueError(f"{name} component #{index + 1} must have positive weight, got {component!r}.")
        components.append((force_low, force_high, duration_low, duration_high, ramp_low, ramp_high, weight))
    if not components:
        raise ValueError(f"{name} did not contain any valid force-duration mixture components.")
    return tuple(components)


def _full_training_enabled() -> bool:
    return (
        _env_bool("BOOSTER_TRACK_ADAPTER_FULL_TRAINING", False)
        and _env_bool("BOOSTER_TRACK_ADAPTER_SMOKE_PASSED", False)
        and _env_bool("BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING", False)
    )


def _amp_ft_hard_command_points(include_axis_yaw: bool = True) -> list[tuple[float, float, float]]:
    # Duplicate primary single-axis points so uniform hard-point sampling still
    # respects the user's benchmark priorities.
    primary: list[tuple[float, float, float]] = []
    for _ in range(3):
        primary.extend(
            [
                (0.0, 0.0, 0.0),
                (0.2, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.5, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (0.0, 0.2, 0.0),
                (0.0, -0.2, 0.0),
                (0.0, 0.4, 0.0),
                (0.0, -0.4, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
            ]
        )
    for _ in range(2):
        primary.extend(
            [
                (0.0, 0.0, 0.3),
                (0.0, 0.0, -0.3),
                (0.0, 0.0, 0.6),
                (0.0, 0.0, -0.6),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 1.5),
                (0.0, 0.0, -1.5),
            ]
        )
    if not include_axis_yaw:
        return primary
    secondary = [
        (0.5, 0.0, 0.6),
        (0.5, 0.0, -0.6),
        (1.0, 0.0, 0.6),
        (1.0, 0.0, -0.6),
        (1.5, 0.0, 1.0),
        (1.5, 0.0, -1.0),
        (0.0, 0.4, 0.6),
        (0.0, -0.4, -0.6),
        (0.0, 1.0, 0.6),
        (0.0, -1.0, -0.6),
    ]
    return primary + secondary


@configclass
class FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = K1_ACTION_SCALE
        # self.actions.joint_pos.scale = 0.25



@configclass
class FlatWoStateEstimationEnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        
@configclass
class RoughWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.debug_vis = False        # 设为True可视化地形分布
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(10.0, 10.0),            # 每个地形块尺寸（米）
            border_width=20.0,            # 边界宽度（米）
            num_rows=5,                   # 地形网格行数
            num_cols=10,                  # 地形网格列数
            horizontal_scale=0.1,         # 水平分辨率
            vertical_scale=0.005,         # 垂直分辨率
            slope_threshold=0.75,         # 网格简化阈值
            use_cache=False,              # 每次重新生成地形
            curriculum=False,              # 启用课程学习
            sub_terrains={
                # 80%接近平面的地形（非常平滑）
                "nearly_flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.8,
                    noise_range=(0.0, 0.005),    # 高度波动0-0.5cm（几乎平坦）
                    noise_step=0.005,            # 噪声步长0.5cm
                    border_width=0.25,
                ),
                # 20%随机粗糙地形
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(-0.015, 0.015),    # 高度波动±1.5cm
                    noise_step=0.005,               # 噪声步长0.5cm
                    border_width=0.25,
                ),
            },
        )
        if _env_bool("BOOSTER_AMP_FT_ENABLE", False):
            self._configure_amp_axis_finetune()

    def _configure_amp_axis_finetune(self):
        cmd = self.commands.base_velocity
        cmd.resampling_time_range = (
            _env_float("BOOSTER_AMP_FT_COMMAND_RESAMPLE_MIN", 6.0),
            _env_float("BOOSTER_AMP_FT_COMMAND_RESAMPLE_MAX", 10.0),
        )
        cmd.rel_standing_envs = _env_float("BOOSTER_AMP_FT_STANDING_ENVS", 0.12)
        cmd.rel_low_speed_envs = _env_float("BOOSTER_AMP_FT_LOW_SPEED_ENVS", 0.16)
        cmd.low_speed_lin_x_abs_range = (
            _env_float("BOOSTER_AMP_FT_LOW_SPEED_X_MIN", 0.15),
            _env_float("BOOSTER_AMP_FT_LOW_SPEED_X_MAX", 0.55),
        )
        cmd.low_speed_lin_y_abs_max = _env_float("BOOSTER_AMP_FT_LOW_SPEED_Y_ABS_MAX", 0.02)
        cmd.low_speed_yaw_abs_max = _env_float("BOOSTER_AMP_FT_LOW_SPEED_YAW_ABS_MAX", 0.02)
        cmd.low_speed_forward_prob = _env_float("BOOSTER_AMP_FT_LOW_SPEED_FORWARD_PROB", 0.96)
        cmd.rel_yaw_only_envs = _env_float("BOOSTER_AMP_FT_YAW_ONLY_ENVS", 0.20)
        cmd.yaw_only_ang_vel_abs_range = (
            _env_float("BOOSTER_AMP_FT_YAW_ONLY_MIN", 0.25),
            _env_float("BOOSTER_AMP_FT_YAW_ONLY_MAX", 1.50),
        )
        cmd.rel_hard_envs = _env_float("BOOSTER_AMP_FT_HARD_ENVS", 0.55)
        cmd.hard_command_points = _amp_ft_hard_command_points(
            include_axis_yaw=_env_bool("BOOSTER_AMP_FT_INCLUDE_AXIS_YAW_HARD_POINTS", True)
        )
        cmd.rel_diagonal_envs = _env_float("BOOSTER_AMP_FT_DIAGONAL_ENVS", 0.0)
        cmd.lin_vel_xy_norm_max = _env_float("BOOSTER_AMP_FT_LIN_XY_NORM_MAX", 0.0)
        cmd.rel_sudden_stop_envs = _env_float("BOOSTER_AMP_FT_SUDDEN_STOP_ENVS", 0.0)
        cmd.failure_mining_push_active_window_s = _env_float("BOOSTER_AMP_FT_PUSH_ACTIVE_WINDOW_S", 2.5)
        cmd.failure_mining_push_sample_prob = _env_float("BOOSTER_AMP_FT_FAILURE_PUSH_SAMPLE_PROB", 0.20)
        cmd.failure_mining_push_min_command_norm = _env_float("BOOSTER_AMP_FT_FAILURE_PUSH_MIN_COMMAND_NORM", 0.0)
        cmd.ranges.lin_vel_x = (
            _env_float("BOOSTER_AMP_FT_LIN_X_MIN", -0.60),
            _env_float("BOOSTER_AMP_FT_LIN_X_MAX", 2.00),
        )
        cmd.ranges.lin_vel_y = (
            _env_float("BOOSTER_AMP_FT_LIN_Y_MIN", -1.00),
            _env_float("BOOSTER_AMP_FT_LIN_Y_MAX", 1.00),
        )
        cmd.ranges.ang_vel_z = (
            _env_float("BOOSTER_AMP_FT_YAW_MIN", -1.50),
            _env_float("BOOSTER_AMP_FT_YAW_MAX", 1.50),
        )
        cmd.debug_vis = _env_bool("BOOSTER_AMP_FT_COMMAND_DEBUG_VIS", False)
        self.scene.contact_forces.debug_vis = _env_bool("BOOSTER_AMP_FT_CONTACT_DEBUG_VIS", False)

        if _env_bool("BOOSTER_AMP_FT_PUSH", True):
            push_xy = _env_float("BOOSTER_AMP_FT_PUSH_XY", 0.40)
            push_yaw = _env_float("BOOSTER_AMP_FT_PUSH_YAW", 0.25)
            self.events.push_robot.interval_range_s = (
                _env_float("BOOSTER_AMP_FT_PUSH_INTERVAL_MIN", 5.0),
                _env_float("BOOSTER_AMP_FT_PUSH_INTERVAL_MAX", 8.0),
            )
            self.events.push_robot.params["velocity_range"] = {
                "x": (-push_xy, push_xy),
                "y": (-push_xy, push_xy),
                "yaw": (-push_yaw, push_yaw),
            }
            final_push_xy = _env_float("BOOSTER_TRACK_ADAPTER_PUSH_XY_FINAL", push_xy)
            final_push_yaw = _env_float("BOOSTER_TRACK_ADAPTER_PUSH_YAW_FINAL", push_yaw)
            self.events.push_robot.params["curriculum_velocity_range"] = {
                "x": (-final_push_xy, final_push_xy),
                "y": (-final_push_xy, final_push_xy),
                "yaw": (-final_push_yaw, final_push_yaw),
            }
            self.events.push_robot.params["curriculum_start_step"] = _env_int(
                "BOOSTER_TRACK_ADAPTER_PUSH_CURRICULUM_START_STEP", 0
            )
            self.events.push_robot.params["curriculum_duration_steps"] = _env_int(
                "BOOSTER_TRACK_ADAPTER_PUSH_CURRICULUM_DURATION_STEPS", 0
            )
            self.events.push_robot.params["failure_mining_command_name"] = "base_velocity"
        else:
            self.events.push_robot = None

        self.events.physics_material.params["static_friction_range"] = (
            _env_float("BOOSTER_AMP_FT_STATIC_FRICTION_MIN", 0.35),
            _env_float("BOOSTER_AMP_FT_STATIC_FRICTION_MAX", 0.85),
        )
        self.events.physics_material.params["dynamic_friction_range"] = (
            _env_float("BOOSTER_AMP_FT_DYNAMIC_FRICTION_MIN", 0.35),
            _env_float("BOOSTER_AMP_FT_DYNAMIC_FRICTION_MAX", 0.85),
        )
        self.events.add_base_mass.params["mass_distribution_params"] = (
            _env_float("BOOSTER_AMP_FT_BASE_MASS_MIN", -0.25),
            _env_float("BOOSTER_AMP_FT_BASE_MASS_MAX", 0.60),
        )
        com_xy = _env_float("BOOSTER_AMP_FT_COM_XY_ABS", 0.03)
        self.events.base_com.params["com_range"] = {
            "x": (-com_xy, com_xy),
            "y": (-com_xy, com_xy),
            "z": (-_env_float("BOOSTER_AMP_FT_COM_Z_ABS", 0.01), _env_float("BOOSTER_AMP_FT_COM_Z_ABS", 0.01)),
        }
        reset_force = _env_float("BOOSTER_AMP_FT_RESET_FORCE_ABS", 1.0)
        reset_torque = _env_float("BOOSTER_AMP_FT_RESET_TORQUE_ABS", 1.0)
        self.events.base_external_force_torque.params["force_range"] = (-reset_force, reset_force)
        self.events.base_external_force_torque.params["torque_range"] = (-reset_torque, reset_torque)

        command_name = "base_velocity"
        self.rewards.vx_only_track_vx_exp = RewTerm(
            func=mdp.single_axis_lin_vel_track_exp,
            weight=_env_float("BOOSTER_AMP_FT_VX_TRACK_WEIGHT", 4.0),
            params={"command_name": command_name, "axis": "x", "std": 0.20, "max_cross_command": 0.05, "max_yaw_command": 0.05},
        )
        self.rewards.vx_only_crosstalk_l2 = RewTerm(
            func=mdp.single_axis_lin_crosstalk_l2,
            weight=_env_float("BOOSTER_AMP_FT_VX_CROSSTALK_WEIGHT", -3.0),
            params={"command_name": command_name, "axis": "x", "max_cross_command": 0.05, "max_yaw_command": 0.05},
        )
        self.rewards.vy_only_track_vy_exp = RewTerm(
            func=mdp.single_axis_lin_vel_track_exp,
            weight=_env_float("BOOSTER_AMP_FT_VY_TRACK_WEIGHT", 3.0),
            params={"command_name": command_name, "axis": "y", "std": 0.22, "max_cross_command": 0.05, "max_yaw_command": 0.05},
        )
        self.rewards.vy_only_crosstalk_l2 = RewTerm(
            func=mdp.single_axis_lin_crosstalk_l2,
            weight=_env_float("BOOSTER_AMP_FT_VY_CROSSTALK_WEIGHT", -2.4),
            params={"command_name": command_name, "axis": "y", "max_cross_command": 0.05, "max_yaw_command": 0.05},
        )
        self.rewards.yaw_only_track_ang_vel_z_rel_exp = RewTerm(
            func=mdp.yaw_only_track_ang_vel_z_rel_exp,
            weight=_env_float("BOOSTER_AMP_FT_YAW_TRACK_REL_WEIGHT", 2.4),
            params={"command_name": command_name, "std": 0.50, "lin_threshold": 0.05, "yaw_threshold": 0.10, "min_command": 0.30},
        )
        self.rewards.yaw_only_translation_l2 = RewTerm(
            func=mdp.yaw_only_translation_l2,
            weight=_env_float("BOOSTER_AMP_FT_YAW_TRANSLATION_WEIGHT", -1.8),
            params={"command_name": command_name, "lin_threshold": 0.05, "yaw_threshold": 0.10},
        )
        self.rewards.zero_command_velocity_l2 = RewTerm(
            func=mdp.zero_command_velocity_l2,
            weight=_env_float("BOOSTER_AMP_FT_ZERO_VELOCITY_WEIGHT", -4.0),
            params={"command_name": command_name, "command_threshold": 0.05, "skip_push_active": True},
        )
        self.rewards.zero_command_joint_vel_l2 = RewTerm(
            func=mdp.zero_command_joint_vel_l2,
            weight=_env_float("BOOSTER_AMP_FT_ZERO_JOINT_VEL_WEIGHT", -0.05),
            params={"command_name": command_name, "command_threshold": 0.05, "skip_push_active": True},
        )
        self.rewards.vx_yaw_track_exp = RewTerm(
            func=mdp.axis_yaw_track_exp,
            weight=_env_float("BOOSTER_AMP_FT_VX_YAW_TRACK_WEIGHT", 0.9),
            params={"command_name": command_name, "axis": "x", "lin_std": 0.30, "yaw_std": 0.55},
        )
        self.rewards.vx_yaw_crosstalk_l2 = RewTerm(
            func=mdp.axis_yaw_crosstalk_l2,
            weight=_env_float("BOOSTER_AMP_FT_VX_YAW_CROSSTALK_WEIGHT", -0.8),
            params={"command_name": command_name, "axis": "x"},
        )
        self.rewards.vy_yaw_track_exp = RewTerm(
            func=mdp.axis_yaw_track_exp,
            weight=_env_float("BOOSTER_AMP_FT_VY_YAW_TRACK_WEIGHT", 0.8),
            params={"command_name": command_name, "axis": "y", "lin_std": 0.32, "yaw_std": 0.55},
        )
        self.rewards.vy_yaw_crosstalk_l2 = RewTerm(
            func=mdp.axis_yaw_crosstalk_l2,
            weight=_env_float("BOOSTER_AMP_FT_VY_YAW_CROSSTALK_WEIGHT", -0.7),
            params={"command_name": command_name, "axis": "y"},
        )
        self.rewards.push_recovery_velocity_track_exp = RewTerm(
            func=mdp.push_recovery_velocity_track_exp,
            weight=_env_float("BOOSTER_AMP_FT_PUSH_RETURN_CMD_WEIGHT", 1.1),
            params={"command_name": command_name, "std": 0.65},
        )

        self.rewards.stand_still.weight = _env_float("BOOSTER_AMP_FT_STAND_STILL_WEIGHT", -0.8)
        self.rewards.track_lin_vel_xy_exp.weight = _env_float("BOOSTER_AMP_FT_TRACK_LIN_WEIGHT", 5.0)
        self.rewards.track_ang_vel_z_exp.weight = _env_float("BOOSTER_AMP_FT_TRACK_YAW_WEIGHT", 2.2)
        self.rewards.yaw_only_track_ang_vel_z_exp.weight = _env_float("BOOSTER_AMP_FT_YAW_TRACK_WEIGHT", 1.6)
        self.rewards.feet_air_time.weight = _env_float("BOOSTER_AMP_FT_FEET_AIR_TIME_WEIGHT", 0.45)
        self.rewards.feet_slide.weight = _env_float("BOOSTER_AMP_FT_FEET_SLIDE_WEIGHT", -1.1)
        self.rewards.action_rate_l2.weight = _env_float("BOOSTER_AMP_FT_ACTION_RATE_WEIGHT", -0.012)
        self.rewards.ang_vel_xy_l2.weight = _env_float("BOOSTER_AMP_FT_ANG_VEL_XY_WEIGHT", -0.15)


@configclass
class TrackAdapterRoughWoStateEstimationEnvCfg(RoughWoStateEstimationEnvCfg):
    """Track Adapter task with 75-dim policy observations for the frozen AMP checkpoint."""

    def __post_init__(self):
        super().__post_init__()
        full_training_enabled = _full_training_enabled()
        smoke_mode = _env_bool("BOOSTER_TRACK_ADAPTER_SMOKE_MODE", not full_training_enabled)
        if not full_training_enabled and not smoke_mode:
            raise ValueError(
                "BOOSTER_TRACK_ADAPTER_SMOKE_MODE=0 requires BOOSTER_TRACK_ADAPTER_FULL_TRAINING=1, "
                "BOOSTER_TRACK_ADAPTER_SMOKE_PASSED=1, and BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING=1."
            )
        if smoke_mode:
            self.scene.num_envs = _env_int("BOOSTER_TRACK_ADAPTER_SMOKE_NUM_ENVS", min(self.scene.num_envs, 1024))

        # Keep the existing policy observation group unchanged:
        # base_ang_vel[3] + projected_gravity[3] + command[3]
        # + joint_pos[22] + joint_vel[22] + previous_action[22] = 75.
        if _env_bool("BOOSTER_TRACK_ADAPTER_PRIVILEGED_RECOVERY_OBS", False):
            self.observations.critic.recovery_privileged = ObsTerm(
                func=mdp.track_adapter_recovery_privileged_state,
                clip=(-100.0, 100.0),
                scale=1.0,
                params={
                    "command_name": "base_velocity",
                    "force_scale": _env_float("BOOSTER_TRACK_ADAPTER_PRIVILEGED_FORCE_SCALE", 300.0),
                    "duration_scale": _env_float("BOOSTER_TRACK_ADAPTER_PRIVILEGED_DURATION_SCALE", 3.0),
                    "foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot_.*"]),
                    "trunk_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Trunk"]),
                    "contact_threshold": _env_float("BOOSTER_TRACK_ADAPTER_PRIVILEGED_CONTACT_THRESHOLD", 1.0),
                },
            )
        cmd = self.commands.base_velocity
        cmd.resampling_time_range = (
            _env_float("BOOSTER_TRACK_ADAPTER_COMMAND_RESAMPLE_MIN", 8.0 if smoke_mode else 7.0),
            _env_float("BOOSTER_TRACK_ADAPTER_COMMAND_RESAMPLE_MAX", 12.0 if smoke_mode else 11.0),
        )
        cmd.rel_standing_envs = _env_float("BOOSTER_TRACK_ADAPTER_STANDING_ENVS", 0.08 if smoke_mode else 0.06)
        cmd.rel_yaw_only_envs = _env_float("BOOSTER_TRACK_ADAPTER_YAW_ONLY_ENVS", 0.08 if smoke_mode else 0.12)
        cmd.yaw_only_ang_vel_abs_range = (
            _env_float("BOOSTER_TRACK_ADAPTER_YAW_ONLY_MIN", 0.15),
            _env_float("BOOSTER_TRACK_ADAPTER_YAW_ONLY_MAX", 0.60 if smoke_mode else 1.50),
        )
        cmd.rel_hard_envs = _env_float("BOOSTER_TRACK_ADAPTER_HARD_ENVS", 0.12 if smoke_mode else 0.45)
        cmd.hard_command_points = _amp_ft_hard_command_points(
            include_axis_yaw=_env_bool("BOOSTER_TRACK_ADAPTER_INCLUDE_AXIS_YAW_HARD_POINTS", True)
        )
        cmd.rel_low_speed_envs = _env_float("BOOSTER_TRACK_ADAPTER_LOW_SPEED_ENVS", 0.10 if smoke_mode else 0.22)
        cmd.low_speed_lin_x_abs_range = (
            _env_float("BOOSTER_TRACK_ADAPTER_LOW_SPEED_X_MIN", 0.15),
            _env_float("BOOSTER_TRACK_ADAPTER_LOW_SPEED_X_MAX", 0.45),
        )
        cmd.low_speed_lin_y_abs_max = _env_float("BOOSTER_TRACK_ADAPTER_LOW_SPEED_Y_ABS_MAX", 0.08)
        cmd.low_speed_yaw_abs_max = _env_float("BOOSTER_TRACK_ADAPTER_LOW_SPEED_YAW_ABS_MAX", 0.06)
        cmd.low_speed_forward_prob = _env_float("BOOSTER_TRACK_ADAPTER_LOW_SPEED_FORWARD_PROB", 0.90)
        cmd.rel_diagonal_envs = _env_float("BOOSTER_TRACK_ADAPTER_DIAGONAL_ENVS", 0.02 if smoke_mode else 0.08)
        cmd.diagonal_lin_x_abs_range = (
            _env_float("BOOSTER_TRACK_ADAPTER_DIAGONAL_X_MIN", 0.75),
            _env_float("BOOSTER_TRACK_ADAPTER_DIAGONAL_X_MAX", 1.20),
        )
        cmd.diagonal_lin_y_abs_range = (
            _env_float("BOOSTER_TRACK_ADAPTER_DIAGONAL_Y_MIN", 0.60),
            _env_float("BOOSTER_TRACK_ADAPTER_DIAGONAL_Y_MAX", 1.00),
        )
        cmd.diagonal_yaw_abs_max = _env_float("BOOSTER_TRACK_ADAPTER_DIAGONAL_YAW_ABS_MAX", 0.08)
        cmd.diagonal_forward_prob = _env_float("BOOSTER_TRACK_ADAPTER_DIAGONAL_FORWARD_PROB", 0.85)
        cmd.lin_vel_xy_norm_max = _env_float("BOOSTER_TRACK_ADAPTER_LIN_XY_NORM_MAX", 0.0)
        cmd.rel_sudden_stop_envs = _env_float("BOOSTER_TRACK_ADAPTER_SUDDEN_STOP_ENVS", 0.02 if smoke_mode else 0.08)
        cmd.sudden_stop_min_command_speed = _env_float("BOOSTER_TRACK_ADAPTER_SUDDEN_STOP_MIN_SPEED", 0.80)
        cmd.sudden_stop_window_s = _env_float("BOOSTER_TRACK_ADAPTER_SUDDEN_STOP_WINDOW_S", 0.75)
        cmd.ranges.lin_vel_x = (
            _env_float("BOOSTER_TRACK_ADAPTER_LIN_X_MIN", -0.60 if smoke_mode else -0.80),
            _env_float("BOOSTER_TRACK_ADAPTER_LIN_X_MAX", 1.40 if smoke_mode else 2.00),
        )
        cmd.ranges.lin_vel_y = (
            _env_float("BOOSTER_TRACK_ADAPTER_LIN_Y_MIN", -0.50 if smoke_mode else -0.70),
            _env_float("BOOSTER_TRACK_ADAPTER_LIN_Y_MAX", 0.50 if smoke_mode else 1.00),
        )
        cmd.ranges.ang_vel_z = (
            _env_float("BOOSTER_TRACK_ADAPTER_YAW_MIN", -0.60 if smoke_mode else -1.50),
            _env_float("BOOSTER_TRACK_ADAPTER_YAW_MAX", 0.60 if smoke_mode else 1.50),
        )
        cmd.failure_mining_contact_sensor_cfg = SceneEntityCfg("contact_forces", body_names=["Trunk"])
        cmd.failure_mining_contact_threshold = _env_float("BOOSTER_TRACK_ADAPTER_BASE_CONTACT_THRESHOLD", 2.0)
        cmd.failure_mining_enabled = _env_bool("BOOSTER_TRACK_ADAPTER_FAILURE_MINING", not smoke_mode)
        cmd.failure_mining_sample_prob = _env_float(
            "BOOSTER_TRACK_ADAPTER_FAILURE_MINING_SAMPLE_PROB", 0.0 if smoke_mode else 0.08
        )
        cmd.failure_mining_min_command_norm = _env_float("BOOSTER_TRACK_ADAPTER_FAILURE_MINING_MIN_COMMAND_NORM", 0.10)
        cmd.failure_mining_command_yaw_scale = _env_float("BOOSTER_TRACK_ADAPTER_FAILURE_COMMAND_YAW_SCALE", 0.30)
        cmd.failure_mining_push_min_command_norm = _env_float(
            "BOOSTER_TRACK_ADAPTER_FAILURE_PUSH_MIN_COMMAND_NORM", 0.0
        )
        cmd.failure_mining_decay = _env_float("BOOSTER_TRACK_ADAPTER_FAILURE_MINING_DECAY", 0.995)
        cmd.failure_mining_push_sample_prob = _env_float(
            "BOOSTER_TRACK_ADAPTER_FAILURE_PUSH_MINING_SAMPLE_PROB", 0.0 if smoke_mode else 0.30
        )
        cmd.failure_mining_push_active_window_s = _env_float(
            "BOOSTER_TRACK_ADAPTER_FAILURE_PUSH_WINDOW_S", 1.25 if smoke_mode else 2.5
        )
        cmd.failure_mining_push_direction_bins = _env_int("BOOSTER_TRACK_ADAPTER_FAILURE_PUSH_DIRECTION_BINS", 8)
        cmd.failure_mining_push_magnitude_bins = _env_int("BOOSTER_TRACK_ADAPTER_FAILURE_PUSH_MAGNITUDE_BINS", 4)
        cmd.failure_mining_push_duration_bins = _env_int("BOOSTER_TRACK_ADAPTER_FAILURE_PUSH_DURATION_BINS", 4)
        cmd.failure_mining_push_location_bins = _env_int("BOOSTER_TRACK_ADAPTER_FAILURE_PUSH_LOCATION_BINS", 3)
        cmd.failure_mining_push_location_extent = _env_float("BOOSTER_TRACK_ADAPTER_FAILURE_PUSH_LOCATION_EXTENT", 2.5)
        cmd.debug_vis = _env_bool("BOOSTER_TRACK_ADAPTER_COMMAND_DEBUG_VIS", False)
        self.scene.contact_forces.debug_vis = _env_bool("BOOSTER_TRACK_ADAPTER_CONTACT_DEBUG_VIS", False)

        self.events.physics_material.params["static_friction_range"] = (
            _env_float("BOOSTER_TRACK_ADAPTER_STATIC_FRICTION_MIN", 0.35 if smoke_mode else 0.30),
            _env_float("BOOSTER_TRACK_ADAPTER_STATIC_FRICTION_MAX", 0.80 if smoke_mode else 0.90),
        )
        self.events.physics_material.params["dynamic_friction_range"] = (
            _env_float("BOOSTER_TRACK_ADAPTER_DYNAMIC_FRICTION_MIN", 0.35 if smoke_mode else 0.30),
            _env_float("BOOSTER_TRACK_ADAPTER_DYNAMIC_FRICTION_MAX", 0.80 if smoke_mode else 0.90),
        )
        self.events.physics_material.params["restitution_range"] = (
            0.0,
            _env_float("BOOSTER_TRACK_ADAPTER_RESTITUTION_MAX", 0.03 if smoke_mode else 0.05),
        )
        self.events.add_base_mass.params["mass_distribution_params"] = (
            _env_float("BOOSTER_TRACK_ADAPTER_BASE_MASS_MIN", -0.20 if smoke_mode else -0.35),
            _env_float("BOOSTER_TRACK_ADAPTER_BASE_MASS_MAX", 0.50 if smoke_mode else 0.80),
        )
        self.events.base_com.params["com_range"] = {
            "x": (
                -_env_float("BOOSTER_TRACK_ADAPTER_COM_X_ABS", 0.025 if smoke_mode else 0.04),
                _env_float("BOOSTER_TRACK_ADAPTER_COM_X_ABS", 0.025 if smoke_mode else 0.04),
            ),
            "y": (
                -_env_float("BOOSTER_TRACK_ADAPTER_COM_Y_ABS", 0.025 if smoke_mode else 0.04),
                _env_float("BOOSTER_TRACK_ADAPTER_COM_Y_ABS", 0.025 if smoke_mode else 0.04),
            ),
            "z": (
                -_env_float("BOOSTER_TRACK_ADAPTER_COM_Z_ABS", 0.010),
                _env_float("BOOSTER_TRACK_ADAPTER_COM_Z_ABS", 0.010),
            ),
        }
        self.events.base_external_force_torque.params["force_range"] = (
            -_env_float("BOOSTER_TRACK_ADAPTER_RESET_FORCE_ABS", 1.0 if smoke_mode else 2.0),
            _env_float("BOOSTER_TRACK_ADAPTER_RESET_FORCE_ABS", 1.0 if smoke_mode else 2.0),
        )
        self.events.base_external_force_torque.params["torque_range"] = (
            -_env_float("BOOSTER_TRACK_ADAPTER_RESET_TORQUE_ABS", 1.0 if smoke_mode else 2.0),
            _env_float("BOOSTER_TRACK_ADAPTER_RESET_TORQUE_ABS", 1.0 if smoke_mode else 2.0),
        )

        delay_min = os.getenv("BOOSTER_TRACK_ADAPTER_ACTUATOR_DELAY_MIN")
        delay_max = os.getenv("BOOSTER_TRACK_ADAPTER_ACTUATOR_DELAY_MAX")
        if _env_bool("BOOSTER_TRACK_ADAPTER_ACTUATOR_DELAY", False) or delay_min is not None or delay_max is not None:
            min_delay = int(delay_min) if delay_min is not None else 1
            max_delay = int(delay_max) if delay_max is not None else 4
            for actuator_cfg in self.scene.robot.actuators.values():
                if hasattr(actuator_cfg, "min_delay") and hasattr(actuator_cfg, "max_delay"):
                    actuator_cfg.min_delay = min_delay
                    actuator_cfg.max_delay = max_delay

        if _env_bool("BOOSTER_TRACK_ADAPTER_PUSH", not smoke_mode):
            push_xy = _env_float("BOOSTER_TRACK_ADAPTER_PUSH_XY", 0.20 if smoke_mode else 0.40)
            push_yaw = _env_float("BOOSTER_TRACK_ADAPTER_PUSH_YAW", 0.10 if smoke_mode else 0.22)
            self.events.push_robot.interval_range_s = (
                _env_float("BOOSTER_TRACK_ADAPTER_PUSH_INTERVAL_MIN", 10.0 if smoke_mode else 4.0),
                _env_float("BOOSTER_TRACK_ADAPTER_PUSH_INTERVAL_MAX", 14.0 if smoke_mode else 7.0),
            )
            self.events.push_robot.params["velocity_range"] = {
                "x": (-push_xy, push_xy),
                "y": (-push_xy, push_xy),
                "yaw": (-push_yaw, push_yaw),
            }
            final_push_xy = _env_float("BOOSTER_TRACK_ADAPTER_PUSH_XY_FINAL", push_xy)
            final_push_yaw = _env_float("BOOSTER_TRACK_ADAPTER_PUSH_YAW_FINAL", push_yaw)
            self.events.push_robot.params["curriculum_velocity_range"] = {
                "x": (-final_push_xy, final_push_xy),
                "y": (-final_push_xy, final_push_xy),
                "yaw": (-final_push_yaw, final_push_yaw),
            }
            self.events.push_robot.params["curriculum_start_step"] = _env_int(
                "BOOSTER_TRACK_ADAPTER_PUSH_CURRICULUM_START_STEP", 0
            )
            self.events.push_robot.params["curriculum_duration_steps"] = _env_int(
                "BOOSTER_TRACK_ADAPTER_PUSH_CURRICULUM_DURATION_STEPS", 0
            )
            self.events.push_robot.params["failure_mining_command_name"] = "base_velocity"
        else:
            self.events.push_robot = None

        if _env_bool("BOOSTER_TRACK_ADAPTER_FORCE_PUSH", False):
            self.events.external_wrench_push.interval_range_s = (
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_STEP_INTERVAL", 0.02),
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_STEP_INTERVAL", 0.02),
            )
            self.events.external_wrench_push.params["force_magnitude_range"] = (
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_MIN_N", 40.0 if smoke_mode else 180.0),
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_MAX_N", 120.0 if smoke_mode else 520.0),
            )
            self.events.external_wrench_push.params["force_magnitude_range_final"] = (
                _env_float(
                    "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_MIN_N",
                    _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_MIN_N", 40.0 if smoke_mode else 180.0),
                ),
                _env_float(
                    "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_MAX_N",
                    _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_MAX_N", 120.0 if smoke_mode else 520.0),
                ),
            )
            force_magnitude_mixture = _env_force_magnitude_mixture("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_MIXTURE")
            if force_magnitude_mixture is not None:
                self.events.external_wrench_push.params["force_magnitude_mixture"] = force_magnitude_mixture
            force_duration_mixture = _env_force_duration_mixture("BOOSTER_TRACK_ADAPTER_FORCE_DURATION_MIXTURE")
            if force_duration_mixture is not None:
                self.events.external_wrench_push.params["force_duration_mixture"] = force_duration_mixture
            self.events.external_wrench_push.params["force_duration_mixture_scale_with_curriculum"] = _env_bool(
                "BOOSTER_TRACK_ADAPTER_FORCE_DURATION_MIXTURE_SCALE_WITH_CURRICULUM", False
            )
            self.events.external_wrench_push.params["force_duration_mixture_force_floor"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_DURATION_MIXTURE_FORCE_FLOOR", 0.0
            )
            fixed_direction_probability = _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FIXED_DIRECTION_PROB", 0.0)
            if fixed_direction_probability > 0.0:
                self.events.external_wrench_push.params["fixed_direction_xy"] = (
                    _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FIXED_DIRECTION_X", 1.0),
                    _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FIXED_DIRECTION_Y", 0.0),
                )
                self.events.external_wrench_push.params["fixed_direction_probability"] = fixed_direction_probability
                self.events.external_wrench_push.params["fixed_direction_jitter_rad"] = _env_float(
                    "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FIXED_DIRECTION_JITTER_RAD", 0.0
                )
            self.events.external_wrench_push.params["duration_range_s"] = (
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_DURATION_MIN_S", 0.04 if smoke_mode else 0.06),
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_DURATION_MAX_S", 0.08 if smoke_mode else 0.14),
            )
            self.events.external_wrench_push.params["duration_range_s_final"] = (
                _env_float(
                    "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_DURATION_MIN_S",
                    _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_DURATION_MIN_S", 0.04 if smoke_mode else 0.06),
                ),
                _env_float(
                    "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_DURATION_MAX_S",
                    _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_DURATION_MAX_S", 0.08 if smoke_mode else 0.14),
                ),
            )
            self.events.external_wrench_push.params["ramp_up_range_s"] = (
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_RAMP_UP_MIN_S", 0.0),
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_RAMP_UP_MAX_S", 0.0),
            )
            self.events.external_wrench_push.params["pulse_interval_range_s"] = (
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_INTERVAL_MIN_S", 8.0 if smoke_mode else 3.5),
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_INTERVAL_MAX_S", 12.0 if smoke_mode else 6.0),
            )
            self.events.external_wrench_push.params["pulse_interval_range_s_final"] = (
                _env_float(
                    "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_INTERVAL_MIN_S",
                    _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_INTERVAL_MIN_S", 8.0 if smoke_mode else 3.5),
                ),
                _env_float(
                    "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_INTERVAL_MAX_S",
                    _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_INTERVAL_MAX_S", 12.0 if smoke_mode else 6.0),
                ),
            )
            self.events.external_wrench_push.params["position_range"] = {
                "x": (
                    -_env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_POS_X_ABS", 0.08),
                    _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_POS_X_ABS", 0.08),
                ),
                "y": (
                    -_env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_POS_Y_ABS", 0.14),
                    _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_POS_Y_ABS", 0.14),
                ),
                "z": (
                    _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_POS_Z_MIN", 0.05),
                    _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_POS_Z_MAX", 0.22),
                ),
            }
            self.events.external_wrench_push.params["torque_z_range"] = (
                -_env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_TORQUE_Z_ABS", 8.0 if smoke_mode else 32.0),
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_TORQUE_Z_ABS", 8.0 if smoke_mode else 32.0),
            )
            final_torque_z_abs = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_TORQUE_Z_ABS",
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_TORQUE_Z_ABS", 8.0 if smoke_mode else 32.0),
            )
            self.events.external_wrench_push.params["torque_z_range_final"] = (
                -final_torque_z_abs,
                final_torque_z_abs,
            )
            self.events.external_wrench_push.params["force_z_range"] = (
                -_env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FORCE_Z_ABS", 10.0 if smoke_mode else 35.0),
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FORCE_Z_ABS", 10.0 if smoke_mode else 35.0),
            )
            self.events.external_wrench_push.params["command_name"] = "base_velocity"
            self.events.external_wrench_push.params["start_command_norm_max"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_START_COMMAND_NORM_MAX", 0.0
            )
            self.events.external_wrench_push.params["start_command_yaw_scale"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_START_COMMAND_YAW_SCALE", 0.30
            )
            self.events.external_wrench_push.params["equivalent_velocity_abs"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_EQUIV_VEL_ABS", 0.35 if smoke_mode else 1.0
            )
            self.events.external_wrench_push.params["equivalent_velocity_abs_final"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_EQUIV_VEL_ABS",
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_EQUIV_VEL_ABS", 0.35 if smoke_mode else 1.0),
            )
            self.events.external_wrench_push.params["equivalent_yaw_abs"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_EQUIV_YAW_ABS", 0.08 if smoke_mode else 0.30
            )
            self.events.external_wrench_push.params["equivalent_yaw_abs_final"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_EQUIV_YAW_ABS",
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_EQUIV_YAW_ABS", 0.08 if smoke_mode else 0.30),
            )
            self.events.external_wrench_push.params["failure_mining"] = _env_bool(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FAILURE_MINING", True
            )
            self.events.external_wrench_push.params["is_global"] = _env_bool(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_GLOBAL_FRAME", True
            )
            self.events.external_wrench_push.params["activation_probability"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_PROB", 1.0
            )
            self.events.external_wrench_push.params["activation_probability_final"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_FINAL_PROB",
                _env_float("BOOSTER_TRACK_ADAPTER_FORCE_PUSH_PROB", 1.0),
            )
            self.events.external_wrench_push.params["curriculum_start_step"] = _env_int(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_CURRICULUM_START_STEP", 0
            )
            self.events.external_wrench_push.params["curriculum_duration_steps"] = _env_int(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_CURRICULUM_DURATION_STEPS", 0
            )
            self.events.external_wrench_push.params["adaptive_curriculum"] = _env_bool(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_ADAPTIVE_CURRICULUM", False
            )
            self.events.external_wrench_push.params["adaptive_alpha_initial"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_ADAPTIVE_ALPHA_INITIAL", 0.0
            )
            self.events.external_wrench_push.params["adaptive_alpha_min"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_ADAPTIVE_ALPHA_MIN", 0.0
            )
            self.events.external_wrench_push.params["adaptive_alpha_max"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_ADAPTIVE_ALPHA_MAX", 1.0
            )
            self.events.external_wrench_push.params["adaptive_alpha_step_up"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_ADAPTIVE_STEP_UP", 0.02
            )
            self.events.external_wrench_push.params["adaptive_alpha_step_down"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_ADAPTIVE_STEP_DOWN", 0.08
            )
            self.events.external_wrench_push.params["adaptive_promote_base_contact"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_ADAPTIVE_PROMOTE_BASE_CONTACT", 0.05
            )
            self.events.external_wrench_push.params["adaptive_demote_base_contact"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_ADAPTIVE_DEMOTE_BASE_CONTACT", 0.12
            )
            self.events.external_wrench_push.params["adaptive_promote_episode_length"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_ADAPTIVE_PROMOTE_EPISODE_LENGTH", 940.0
            )
            self.events.external_wrench_push.params["adaptive_promote_push_failure"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_ADAPTIVE_PROMOTE_PUSH_FAILURE", 0.04
            )
            self.events.external_wrench_push.params["adaptive_demote_push_failure"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_ADAPTIVE_DEMOTE_PUSH_FAILURE", 0.10
            )
            self.events.external_wrench_push.params["adaptive_min_push_active"] = _env_float(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_ADAPTIVE_MIN_PUSH_ACTIVE", 0.05
            )
            self.events.external_wrench_push.params["max_pulses_per_episode"] = _env_int(
                "BOOSTER_TRACK_ADAPTER_FORCE_PUSH_MAX_PULSES_PER_EPISODE", 0
            )
        else:
            self.events.external_wrench_push = None

        foot_sensor = SceneEntityCfg("contact_forces", body_names=[".*_foot_.*"])
        foot_asset = SceneEntityCfg("robot", body_names=[".*_foot_.*"])
        undesired_sensor = SceneEntityCfg(
            "contact_forces", body_names=[".*_Shank", ".*Hip.*", ".*hand.*", ".*Arm.*", "Head_.*", "Trunk"]
        )
        self.rewards.low_speed_track_lin_vel_xy_exp = RewTerm(
            func=mdp.low_speed_track_lin_vel_xy_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_LOW_SPEED_TRACK_WEIGHT", 0.8 if smoke_mode else 1.2),
            params={
                "command_name": "base_velocity",
                "std": _env_float("BOOSTER_TRACK_ADAPTER_LOW_SPEED_TRACK_STD", 0.18),
                "min_command": _env_float("BOOSTER_TRACK_ADAPTER_LOW_SPEED_TRACK_MIN_COMMAND", 0.10),
                "max_command": _env_float("BOOSTER_TRACK_ADAPTER_LOW_SPEED_TRACK_MAX_COMMAND", 0.45),
                "max_lateral": _env_float("BOOSTER_TRACK_ADAPTER_LOW_SPEED_TRACK_MAX_LATERAL", 0.08),
                "max_yaw": _env_float("BOOSTER_TRACK_ADAPTER_LOW_SPEED_TRACK_MAX_YAW", 0.08),
            },
        )
        self.rewards.vx_only_track_vx_exp = RewTerm(
            func=mdp.single_axis_lin_vel_track_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_VX_TRACK_WEIGHT", 1.0 if smoke_mode else 2.0),
            params={
                "command_name": "base_velocity",
                "axis": "x",
                "std": _env_float("BOOSTER_TRACK_ADAPTER_VX_TRACK_STD", 0.20),
                "max_cross_command": 0.05,
                "max_yaw_command": 0.05,
            },
        )
        self.rewards.vx_only_crosstalk_l2 = RewTerm(
            func=mdp.single_axis_lin_crosstalk_l2,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_VX_CROSSTALK_WEIGHT", -0.8 if smoke_mode else -1.6),
            params={"command_name": "base_velocity", "axis": "x", "max_cross_command": 0.05, "max_yaw_command": 0.05},
        )
        self.rewards.vy_only_track_vy_exp = RewTerm(
            func=mdp.single_axis_lin_vel_track_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_VY_TRACK_WEIGHT", 0.9 if smoke_mode else 1.6),
            params={
                "command_name": "base_velocity",
                "axis": "y",
                "std": _env_float("BOOSTER_TRACK_ADAPTER_VY_TRACK_STD", 0.22),
                "max_cross_command": 0.05,
                "max_yaw_command": 0.05,
            },
        )
        self.rewards.vy_only_crosstalk_l2 = RewTerm(
            func=mdp.single_axis_lin_crosstalk_l2,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_VY_CROSSTALK_WEIGHT", -0.7 if smoke_mode else -1.3),
            params={"command_name": "base_velocity", "axis": "y", "max_cross_command": 0.05, "max_yaw_command": 0.05},
        )
        self.rewards.yaw_only_track_ang_vel_z_rel_exp = RewTerm(
            func=mdp.yaw_only_track_ang_vel_z_rel_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_YAW_TRACK_REL_WEIGHT", 0.8 if smoke_mode else 1.4),
            params={
                "command_name": "base_velocity",
                "std": _env_float("BOOSTER_TRACK_ADAPTER_YAW_TRACK_REL_STD", 0.50),
                "lin_threshold": 0.05,
                "yaw_threshold": 0.10,
                "min_command": 0.30,
            },
        )
        self.rewards.yaw_only_translation_l2 = RewTerm(
            func=mdp.yaw_only_translation_l2,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_YAW_TRANSLATION_WEIGHT", -0.5 if smoke_mode else -1.0),
            params={"command_name": "base_velocity", "lin_threshold": 0.05, "yaw_threshold": 0.10},
        )
        self.rewards.zero_command_velocity_l2 = RewTerm(
            func=mdp.zero_command_velocity_l2,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_ZERO_VEL_WEIGHT", -0.4 if smoke_mode else -0.8),
            params={"command_name": "base_velocity", "command_threshold": 0.05, "skip_push_active": True},
        )
        self.rewards.zero_command_joint_vel_l2 = RewTerm(
            func=mdp.zero_command_joint_vel_l2,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_ZERO_JOINT_VEL_WEIGHT", -0.02 if smoke_mode else -0.05),
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.05,
                "skip_push_active": True,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            },
        )
        self.rewards.axis_yaw_x_track_exp = RewTerm(
            func=mdp.axis_yaw_track_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_AXIS_YAW_X_TRACK_WEIGHT", 0.4 if smoke_mode else 0.8),
            params={"command_name": "base_velocity", "axis": "x", "lin_std": 0.28, "yaw_std": 0.45},
        )
        self.rewards.axis_yaw_x_crosstalk_l2 = RewTerm(
            func=mdp.axis_yaw_crosstalk_l2,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_AXIS_YAW_X_CROSSTALK_WEIGHT", -0.2 if smoke_mode else -0.4),
            params={"command_name": "base_velocity", "axis": "x"},
        )
        self.rewards.axis_yaw_y_track_exp = RewTerm(
            func=mdp.axis_yaw_track_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_AXIS_YAW_Y_TRACK_WEIGHT", 0.3 if smoke_mode else 0.6),
            params={"command_name": "base_velocity", "axis": "y", "lin_std": 0.30, "yaw_std": 0.45},
        )
        self.rewards.axis_yaw_y_crosstalk_l2 = RewTerm(
            func=mdp.axis_yaw_crosstalk_l2,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_AXIS_YAW_Y_CROSSTALK_WEIGHT", -0.15 if smoke_mode else -0.3),
            params={"command_name": "base_velocity", "axis": "y"},
        )
        self.rewards.recovery_push_velocity_track_exp = RewTerm(
            func=mdp.push_recovery_velocity_track_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_PUSH_RECOVERY_TRACK_WEIGHT", 0.8 if smoke_mode else 1.4),
            params={"command_name": "base_velocity", "std": 0.65, "yaw_scale": 0.30},
        )
        self.rewards.recovery_delayed_velocity_track_exp = RewTerm(
            func=mdp.push_recovery_delayed_velocity_track_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_PUSH_RECOVERY_DELAYED_TRACK_WEIGHT", 0.0),
            params={
                "command_name": "base_velocity",
                "min_elapsed_s": _env_float("BOOSTER_TRACK_ADAPTER_PUSH_RECOVERY_DELAYED_MIN_S", 0.35),
                "max_elapsed_s": _env_float("BOOSTER_TRACK_ADAPTER_PUSH_RECOVERY_DELAYED_MAX_S", 0.0),
                "std": 0.65,
                "yaw_scale": 0.30,
            },
        )
        self.rewards.recovery_push_velocity_cancel_exp = RewTerm(
            func=mdp.recovery_push_velocity_cancel_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_PUSH_VELOCITY_CANCEL_WEIGHT", 0.0),
            params={
                "command_name": "base_velocity",
                "min_elapsed_s": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_PUSH_VELOCITY_CANCEL_MIN_S", 0.0),
                "max_elapsed_s": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_PUSH_VELOCITY_CANCEL_MAX_S", 0.60),
                "min_push_norm": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_PUSH_VELOCITY_CANCEL_MIN_PUSH", 0.05),
                "std": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_PUSH_VELOCITY_CANCEL_STD", 0.35),
                "yaw_std": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_PUSH_VELOCITY_CANCEL_YAW_STD", 1.20),
            },
        )
        self.rewards.recovery_upright_exp = RewTerm(
            func=mdp.recovery_upright_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_UPRIGHT_WEIGHT", 1.2 if smoke_mode else 1.7),
            params={"std": 0.35},
        )
        self.rewards.recovery_base_height_exp = RewTerm(
            func=mdp.recovery_base_height_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_HEIGHT_WEIGHT", 0.8 if smoke_mode else 1.0),
            params={"target_height": 0.57, "std": 0.20},
        )
        self.rewards.recovery_ang_vel_xy_damp_exp = RewTerm(
            func=mdp.recovery_ang_vel_xy_damp_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_ANGVEL_WEIGHT", 0.6 if smoke_mode else 0.8),
            params={"std": 2.0},
        )
        self.rewards.recovery_no_fall_alive = RewTerm(
            func=mdp.recovery_no_fall_alive,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_NO_FALL_WEIGHT", 1.0 if smoke_mode else 1.4),
            params={"undesired_sensor_cfg": undesired_sensor, "min_height": 0.35, "max_tilt": 0.75},
        )
        self.rewards.recovery_no_base_contact = RewTerm(
            func=mdp.recovery_no_base_contact,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_NO_BASE_CONTACT_WEIGHT", 0.0),
            params={"sensor_cfg": undesired_sensor},
        )
        self.rewards.recovery_base_contact_penalty = RewTerm(
            func=mdp.recovery_base_contact_penalty,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_BASE_CONTACT_PENALTY_WEIGHT", 0.0),
            params={
                "command_name": "base_velocity",
                "sensor_cfg": undesired_sensor,
                "contact_threshold": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_BASE_CONTACT_THRESHOLD", 1.0),
                "grace_s": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_BASE_CONTACT_GRACE_S", 0.05),
            },
        )
        self.rewards.recovery_foot_placement_exp = RewTerm(
            func=mdp.recovery_foot_placement_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_FOOT_PLACE_WEIGHT", 0.35 if smoke_mode else 0.50),
            params={"foot_asset_cfg": foot_asset, "target_height": 0.57, "std": 0.35},
        )
        self.rewards.recovery_push_capture_point_exp = RewTerm(
            func=mdp.recovery_push_capture_point_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_PUSH_CAPTURE_POINT_WEIGHT", 0.0),
            params={
                "command_name": "base_velocity",
                "foot_asset_cfg": foot_asset,
                "target_height": 0.57,
                "std": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_PUSH_CAPTURE_POINT_STD", 0.32),
                "min_elapsed_s": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_PUSH_CAPTURE_POINT_MIN_S", 0.0),
                "max_elapsed_s": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_PUSH_CAPTURE_POINT_MAX_S", 1.8),
            },
        )
        self.rewards.recovery_directional_step_exp = RewTerm(
            func=mdp.recovery_directional_step_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_STEP_WEIGHT", 0.0),
            params={
                "command_name": "base_velocity",
                "foot_asset_cfg": foot_asset,
                "min_projection": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_STEP_MIN_PROJ", 0.10),
                "target_projection": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_STEP_TARGET_PROJ", 0.26),
                "projection_std": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_STEP_PROJ_STD", 0.20),
                "lateral_std": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_STEP_LAT_STD", 0.35),
                "min_push_norm": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_STEP_MIN_PUSH", 0.05),
                "min_elapsed_s": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_STEP_MIN_S", 0.0),
                "max_elapsed_s": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_STEP_MAX_S", 1.5),
            },
        )
        self.rewards.recovery_directional_support_contact_exp = RewTerm(
            func=mdp.recovery_directional_support_contact_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_SUPPORT_WEIGHT", 0.0),
            params={
                "command_name": "base_velocity",
                "foot_asset_cfg": foot_asset,
                "sensor_cfg": foot_sensor,
                "min_projection": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_SUPPORT_MIN_PROJ", 0.08),
                "target_projection": _env_float(
                    "BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_SUPPORT_TARGET_PROJ", 0.30
                ),
                "projection_std": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_SUPPORT_PROJ_STD", 0.18),
                "lateral_std": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_SUPPORT_LAT_STD", 0.32),
                "contact_threshold": _env_float(
                    "BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_SUPPORT_CONTACT_THRESHOLD", 1.0
                ),
                "min_push_norm": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_SUPPORT_MIN_PUSH", 0.05),
                "min_elapsed_s": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_SUPPORT_MIN_S", 0.04),
                "max_elapsed_s": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_SUPPORT_MAX_S", 1.60),
            },
        )
        self.rewards.recovery_directional_first_contact_step_exp = RewTerm(
            func=mdp.recovery_directional_first_contact_step_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_DIRECTIONAL_FIRST_CONTACT_WEIGHT", 0.0),
            params={
                "command_name": "base_velocity",
                "foot_asset_cfg": foot_asset,
                "sensor_cfg": foot_sensor,
                "min_projection": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_FIRST_CONTACT_MIN_PROJ", 0.08),
                "target_projection": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_FIRST_CONTACT_TARGET_PROJ", 0.22),
                "projection_std": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_FIRST_CONTACT_PROJ_STD", 0.16),
                "lateral_std": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_FIRST_CONTACT_LAT_STD", 0.28),
                "min_push_norm": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_FIRST_CONTACT_MIN_PUSH", 0.02),
                "min_elapsed_s": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_FIRST_CONTACT_MIN_S", 0.10),
                "max_elapsed_s": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_FIRST_CONTACT_MAX_S", 2.40),
                "min_air_time_s": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_FIRST_CONTACT_MIN_AIR_S", 0.06),
                "max_command_norm": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_FIRST_CONTACT_MAX_COMMAND_NORM", 0.10),
                "command_yaw_scale": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_FIRST_CONTACT_YAW_SCALE", 0.30),
                "require_force_active": _env_bool("BOOSTER_TRACK_ADAPTER_RECOVERY_FIRST_CONTACT_FORCE_ACTIVE", True),
            },
        )
        self.rewards.recovery_return_to_command_exp = RewTerm(
            func=mdp.recovery_return_to_command_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_RETURN_CMD_WEIGHT", 0.8 if smoke_mode else 1.0),
            params={"command_name": "base_velocity", "std": 0.6, "stable_score_threshold": 0.75},
        )
        self.rewards.recovery_feet_support_contact = RewTerm(
            func=mdp.recovery_feet_support_contact,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_SUPPORT_WEIGHT", 0.4 if smoke_mode else 0.75),
            params={"sensor_cfg": foot_sensor, "min_contacts": 1},
        )
        self.rewards.recovery_lateral_step_allowance = RewTerm(
            func=mdp.recovery_lateral_step_allowance,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_STEP_WIDTH_WEIGHT", 0.2),
            params={"foot_asset_cfg": foot_asset, "min_width": 0.12, "max_width": 0.55},
        )
        self.rewards.recovery_body_frame_stance_width = RewTerm(
            func=mdp.recovery_body_frame_stance_width,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_BODY_FRAME_STANCE_WEIGHT", 0.0),
            params={
                "foot_asset_cfg": foot_asset,
                "min_width": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_BODY_FRAME_STANCE_MIN", 0.12),
                "target_width": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_BODY_FRAME_STANCE_TARGET", 0.26),
                "max_width": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_BODY_FRAME_STANCE_MAX", 0.55),
                "std": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_BODY_FRAME_STANCE_STD", 0.16),
            },
        )
        self.rewards.recovery_sudden_stop_upright_exp = RewTerm(
            func=mdp.recovery_sudden_stop_upright_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_STOP_UPRIGHT_WEIGHT", 0.7 if smoke_mode else 1.0),
            params={"command_name": "base_velocity", "std": 0.30},
        )
        self.rewards.recovery_sudden_stop_velocity_damp_exp = RewTerm(
            func=mdp.recovery_sudden_stop_velocity_damp_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_STOP_VEL_DAMP_WEIGHT", 0.9 if smoke_mode else 1.2),
            params={"command_name": "base_velocity", "std": 0.45},
        )
        self.rewards.recovery_sudden_stop_default_pose_exp = RewTerm(
            func=mdp.recovery_sudden_stop_default_pose_exp,
            weight=_env_float("BOOSTER_TRACK_ADAPTER_STOP_DEFAULT_POSE_WEIGHT", 0.2 if smoke_mode else 0.3),
            params={"command_name": "base_velocity", "std": 1.6},
        )

        self.terminations.base_contact.params["threshold"] = _env_float(
            "BOOSTER_TRACK_ADAPTER_BASE_CONTACT_THRESHOLD", 2.0
        )
        self.terminations.base_contact.func = mdp.illegal_contact_after_duration
        self.terminations.base_contact.params["duration_s"] = _env_float(
            "BOOSTER_TRACK_ADAPTER_BASE_CONTACT_DURATION_S", 0.08
        )
        self.terminations.base_contact.params["command_name"] = "base_velocity"

        self.rewards.ang_vel_xy_l2.weight = _env_float("BOOSTER_TRACK_ADAPTER_ANG_VEL_XY_WEIGHT", -0.12)
        self.rewards.action_rate_l2.weight = _env_float("BOOSTER_TRACK_ADAPTER_ACTION_RATE_WEIGHT", -0.008)
        self.rewards.dof_torques_l2.weight = _env_float("BOOSTER_TRACK_ADAPTER_TORQUE_WEIGHT", -8.0e-6)
        self.rewards.dof_acc_l2.weight = _env_float("BOOSTER_TRACK_ADAPTER_DOF_ACC_WEIGHT", -2.0e-7)


@configclass
class FlatLowFreqEnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE

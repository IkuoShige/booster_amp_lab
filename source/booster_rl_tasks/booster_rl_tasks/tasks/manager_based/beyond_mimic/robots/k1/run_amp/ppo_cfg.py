import os
from pathlib import Path

from isaaclab.utils import configclass
from booster_assets import BOOSTER_ASSETS_DIR
from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BaseAMPAgentCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg, RslRlRndCfg
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp import symmetry


_AMP_ROOT = os.path.join(BOOSTER_ASSETS_DIR, "motions", "K1", "motion_amp_expert")
def _repo_root_from_file() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "rsl_rl").exists() and (parent / "logs").exists():
            return parent
    return Path.cwd()


_TRACK_ADAPTER_BASE_CHECKPOINT_PATH = os.getenv(
    "BOOSTER_TRACK_ADAPTER_BASE_CHECKPOINT",
    str(
        _repo_root_from_file()
        / "logs/rsl_rl/run_amp_y/2026-05-04_19-10-18_resume56_yawlow_zero_stand/model_70000.pt"
    ),
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _full_training_enabled() -> bool:
    return (
        _env_bool("BOOSTER_TRACK_ADAPTER_FULL_TRAINING", False)
        and _env_bool("BOOSTER_TRACK_ADAPTER_SMOKE_PASSED", False)
        and _env_bool("BOOSTER_TRACK_ADAPTER_ALLOW_FULL_TRAINING", False)
    )


@configclass
class TrackAdapterActorCriticCfg(RslRlPpoActorCriticCfg):
    base_checkpoint_path: str = _TRACK_ADAPTER_BASE_CHECKPOINT_PATH
    history_length: int = 79
    history_encoder_type: str = "gru"
    history_encoder_hidden_dims: list[int] = [256, 128]
    adapter_hidden_dims: list[int] = [256, 128]
    adapter_mode: str = "action_residual"
    layerwise_adapter_hidden_dims: list[int] | None = None
    layerwise_adapter_output_init_scale: float | None = None
    history_embedding_dim: int = 64
    residual_scale: float = 0.25
    residual_output_init_scale: float = 0.0
    privileged_teacher_enabled: bool = False
    privileged_teacher_hidden_dims: list[int] | None = None
    privileged_teacher_residual_scale: float = 0.30
    privileged_teacher_output_init_scale: float = 0.0
    privileged_teacher_use_adapter_base: bool = False
    world_model_hidden_dims: list[int] = [256, 128]
    world_model_target_dim: int = 81
    world_model_horizon: int = 20
    world_model_output_init_scale: float = 0.001
    freeze_base: bool = True


@configclass
class TrackAdapterAlgorithmCfg(RslRlPpoAlgorithmCfg):
    freeze_discriminator: bool = True
    residual_penalty_cfg: dict = {}
    recovery_teacher_cfg: dict = {}
    privileged_teacher_cfg: dict = {}
    privileged_teacher_distill_cfg: dict = {}
    world_model_loss_coef: float = 0.10
    world_model_cfg: dict = {}


@configclass
class PPORunnerCfg(BaseAMPAgentCfg):
    max_iterations = 50000
    experiment_name = "run_amp_y"

    # Resume from run_amp_y and keep the legacy 56-col AMP layout:
    #   joint_pos[22] + joint_vel[22] + EE_pos_b[12].
    # The first resume experiment should preserve the good command response of
    # run_amp_y while widening the command range and adding explicit stability
    # rewards. 62-col/root-velocity AMP remains a later low-weight fine-tune
    # ablation, not the baseline for this run.
    amp_reward_coef = 0.2

    # 56-col forward + lateral corpus used by the run_amp_y checkpoint, plus
    # 56-col backward clips derived from the 62-col omni corpus (root velocity
    # columns stripped). Pivot clips are intentionally excluded in this resume
    # run; yaw-only stepping is encouraged through commands/rewards instead of
    # an AMP pivot prior.
    amp_motion_files = [
        # forward
        os.path.join(_AMP_ROOT, "walk.txt"),
        os.path.join(_AMP_ROOT, "walk2run.txt"),
        os.path.join(_AMP_ROOT, "run.txt"),
        os.path.join(_AMP_ROOT, "run2walk.txt"),
        # lateral
        os.path.join(_AMP_ROOT, "lateral", "strafe_walk_left.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_walk_right.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_walk2run_left.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_walk2run_right.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_run_left.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_run_right.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_run2walk_left.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_run2walk_right.txt"),
        # backward
        os.path.join(_AMP_ROOT, "backward", "backward_walk.txt"),
        os.path.join(_AMP_ROOT, "backward", "backward_walk2run.txt"),
        os.path.join(_AMP_ROOT, "backward", "backward_run.txt"),
        os.path.join(_AMP_ROOT, "backward", "backward_run2walk.txt"),
    ]
    amp_num_preload_transitions = 200000
    # NOTE: amp_task_reward_lerp is read but the vendored discriminator just
    # uses r = disc_r + task_r when >0. Kept for signature compatibility.
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 22

    amp_command_gating = False

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        if not _env_bool("BOOSTER_AMP_FT_ENABLE", False):
            return
        self.run_name = os.getenv("BOOSTER_AMP_FT_RUN_NAME", self.run_name or "amp_axis_ft")
        self.max_iterations = _env_int("BOOSTER_AMP_FT_MAX_ITERATIONS", self.max_iterations)
        self.save_interval = _env_int("BOOSTER_AMP_FT_SAVE_INTERVAL", 100)
        self.num_steps_per_env = _env_int("BOOSTER_AMP_FT_STEPS_PER_ENV", 32)
        self.amp_reward_coef = _env_float("BOOSTER_AMP_FT_AMP_REWARD_COEF", 0.18)
        self.algorithm.learning_rate = _env_float("BOOSTER_AMP_FT_LR", 2.0e-4)
        self.algorithm.entropy_coef = _env_float("BOOSTER_AMP_FT_ENTROPY", 0.003)
        self.algorithm.num_learning_epochs = _env_int("BOOSTER_AMP_FT_NUM_LEARNING_EPOCHS", 4)
        self.algorithm.num_mini_batches = _env_int("BOOSTER_AMP_FT_NUM_MINI_BATCHES", 8)
        self.algorithm.max_grad_norm = _env_float("BOOSTER_AMP_FT_MAX_GRAD_NORM", 0.8)
        self.algorithm.desired_kl = _env_float("BOOSTER_AMP_FT_DESIRED_KL", 0.01)


@configclass
class TrackAdapterPPORunnerCfg(PPORunnerCfg):
    """Frozen AMP base plus history-informed residual adapter task config."""

    num_steps_per_env = 48
    smoke_max_iterations = 300
    full_max_iterations = 12000
    max_iterations = smoke_max_iterations
    experiment_name = "run_amp_track_adapter"
    run_name = "track_adapter_recovery_smoke"
    runner_class_name = "TrackAdapterRunner"
    resume = False

    base_checkpoint_path = _TRACK_ADAPTER_BASE_CHECKPOINT_PATH
    history_length = 79
    freeze_discriminator = True
    recovery_gated_amp_enabled = True
    full_training_gate_open = False
    allow_full_training = False
    max_iterations_without_full_training_unlock = smoke_max_iterations
    strict_base_auxiliary_state = True
    pretrained_world_model_path = None
    gated_requested_max_iterations = 0
    amp_zero_command_scale = 0.85
    amp_zero_command_threshold = 0.08

    policy = TrackAdapterActorCriticCfg(
        class_name="TrackAdapterActorCritic",
        init_noise_std=0.25,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        base_checkpoint_path=_TRACK_ADAPTER_BASE_CHECKPOINT_PATH,
        history_length=79,
        history_encoder_type="gru",
        history_encoder_hidden_dims=[256, 128],
        adapter_hidden_dims=[256, 128],
        history_embedding_dim=64,
        residual_scale=0.08,
        residual_output_init_scale=0.0,
        privileged_teacher_enabled=False,
        privileged_teacher_hidden_dims=[256, 256, 128],
        privileged_teacher_residual_scale=0.30,
        privileged_teacher_output_init_scale=0.0,
        privileged_teacher_use_adapter_base=False,
        world_model_hidden_dims=[256, 128],
        world_model_target_dim=81,
        world_model_horizon=20,
        world_model_output_init_scale=0.001,
        freeze_base=True,
    )
    algorithm = TrackAdapterAlgorithmCfg(
        class_name="TrackAdapterAMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,
        num_learning_epochs=2,
        num_mini_batches=32,
        learning_rate=2.0e-6,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
        freeze_discriminator=True,
        world_model_loss_coef=0.10,
        world_model_cfg={
            "enabled": True,
            "loss_coef": 0.10,
            "horizon": 20,
            "target_source": "psi_s",
            "target_dim": 81,
            "alternating_updates": True,
            "joint_loss": False,
            "updates_per_iteration": 2,
            "mini_batch_size": 1024,
            "replay_buffer_size": 65536,
            "replay_device": "cpu",
            "replay_dtype": "float16",
            "ppo_updates_history_encoder": False,
            "learning_rate": 2.0e-6,
            "contact_sensor_name": "contact_forces",
            "foot_body_pattern": r".*_foot_.*",
            "foot_contact_threshold": 1.0,
        },
        residual_penalty_cfg={
            "enabled": True,
            "stable_weight": 0.18,
            "recovery_weight": 0.02,
            "zero_command_weight": 0.90,
            "zero_command_threshold": 0.08,
            "ungated_stable_weight": 0.03,
            "ungated_policy_loss_weight": 0.03,
            "ungated_policy_gate_threshold": 0.10,
            "recovery_activation_weight": 0.0,
            "recovery_activation_gate_threshold": 0.65,
            "recovery_activation_target_norm": 0.02,
            "recovery_activation_max_norm": 0.0,
            "recovery_activation_overgrowth_weight": 0.0,
            "recovery_ppo_sample_weight": 0.0,
            "recovery_ppo_sample_gate_threshold": 0.45,
            "recovery_ppo_sample_power": 1.0,
            "max_penalty": 3.0,
            "gate": "style_gate",
        },
        recovery_teacher_cfg={
            "enabled": False,
            "coef": 0.0,
            "gate_threshold": 0.55,
            "score_threshold": 0.18,
            "velocity_gain": 0.55,
            "tilt_gain": 0.35,
            "ang_vel_gain": 0.10,
            "max_delta": 0.12,
            "non_leg_weight": 0.02,
            "separate_actor_grad_clip": False,
            "actor_max_grad_norm": 1.0,
        },
        privileged_teacher_cfg={
            "enabled": False,
            "train": False,
        },
        privileged_teacher_distill_cfg={
            "enabled": False,
            "coef": 0.0,
            "gate_threshold": 0.55,
            "score_threshold": 0.12,
        },
    )

    reward_group_names = ["task", "style", "recovery", "reg"]
    reward_group_default = "task"
    reward_group_weights = {
        "task": 1.30,
        "style": 0.45,
        "recovery": 1.15,
        "reg": 0.35,
    }
    reward_group_term_prefixes = {
        "recovery": ["recovery_"],
    }
    reward_group_term_names = {
        "reg": [
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "head_ang_vel_xy_l2",
            "dof_torques_l2",
            "dof_acc_l2",
            "action_rate_l2",
            "feet_stumble",
            "feet_slide",
            "undesired_contacts",
            "dof_pos_limits",
        ],
    }
    disturbance_gate = {
        "command_name": "base_velocity",
        "yaw_scale": 0.30,
        "target_height": 0.57,
        "beta": 0.55,
        "min_style_gate": 0.20,
        "command_min_weight": 0.55,
        "tilt_weight": 1.8,
        "ang_vel_weight": 0.25,
        "velocity_weight": 0.50,
        "height_weight": 2.0,
        "contact_weight": 2.0,
        "contact_sensor_name": "contact_forces",
        "contact_body_names": ["Trunk"],
        "contact_threshold": 1.0,
        "max_score": 6.0,
        "reg_stable_scale": 1.0,
        "reg_recovery_scale": 0.65,
        "push_active_metric_name": "failure_push_active",
        "push_recovery_gate_min": 0.45,
        "push_style_gate_max": 0.60,
        "push_command_gate_min": 0.85,
        "push_command_gate_max": 1.0,
        "push_command_gate_late_max": 1.0,
        "push_command_gate_full_s": 0.0,
        "push_command_gate_taper_s": 0.0,
    }
    residual_penalty_cfg = {
        "enabled": True,
        "stable_weight": 0.18,
        "recovery_weight": 0.02,
        "zero_command_weight": 0.90,
        "zero_command_threshold": 0.08,
        "ungated_stable_weight": 0.03,
        "ungated_policy_loss_weight": 0.03,
        "ungated_policy_gate_threshold": 0.10,
        "push_stable_weight": 0.0,
        "push_stable_ungated_weight": 0.0,
        "push_stable_score_threshold": 0.75,
        "recovery_activation_weight": 0.0,
        "recovery_activation_gate_threshold": 0.65,
        "recovery_activation_target_norm": 0.02,
        "recovery_activation_max_norm": 0.0,
        "recovery_activation_overgrowth_weight": 0.0,
        "recovery_ppo_sample_weight": 0.0,
        "recovery_ppo_sample_gate_threshold": 0.45,
        "recovery_ppo_sample_power": 1.0,
        "max_penalty": 3.0,
        "gate": "style_gate",
    }
    residual_action_gate = {
        "enabled": True,
        "stable_floor": 0.02,
        "stable_max_gate": 0.08,
        "stable_style_threshold": 0.75,
        "stable_recovery_threshold": 0.25,
        "zero_command_gate": 0.0,
        "zero_command_threshold": 0.08,
        "recovery_scale": 1.0,
        "recovery_power": 1.0,
        "push_min_gate": 0.70,
        "push_phase_enabled": False,
        "push_full_gate_s": 0.80,
        "push_taper_s": 1.00,
        "push_late_min_gate": 0.35,
        "push_high_score_threshold": 1.10,
        "push_high_score_width": 0.45,
        "push_high_score_min_gate": 0.92,
        "sudden_stop_min_gate": 0.60,
        "manual_stop_speed_threshold": 0.10,
    }
    recovery_controller = {
        "enabled": False,
        "active_threshold": 0.05,
        "enter_score": 0.95,
        "enter_recovery_gate": 0.45,
        "exit_score": 0.45,
        "stable_score": 0.35,
        "enter_delay_s": 0.04,
        "min_recovery_s": 0.80,
        "min_stable_s": 0.25,
        "return_stable_s": 0.30,
        "return_decay_s": 1.00,
        "ramp_s": 0.18,
        "recovery_style_gate_max": 0.05,
        "return_style_floor": 0.10,
        "recovery_gate_min": 0.85,
        "return_recovery_gate_min": 0.05,
        "recovery_command_reward_scale": 0.05,
        "return_command_reward_floor": 0.35,
    }
    recovery_command_override = {
        "enabled": False,
        "command_name": "base_velocity",
        "command_slice_start": 6,
        "zero_command_only": True,
        "zero_command_threshold": 0.08,
        "active_threshold": 0.05,
        "direction_threshold": 0.03,
        "push_direction_gain": 1.0,
        "velocity_direction_gain": 0.35,
        "tilt_direction_gain": 0.0,
        "ang_vel_direction_gain": 0.0,
        "push_speed_gain": 0.85,
        "velocity_speed_gain": 0.35,
        "signal_speed_gain": 0.0,
        "min_speed": 0.25,
        "max_speed": 1.60,
        "max_abs_x": 1.60,
        "max_abs_y": 1.20,
        "yaw_damping_gain": 0.0,
        "max_yaw": 0.0,
        "ramp_s": 0.20,
        "blend": 1.0,
    }
    terminal_penalty = {
        "enabled": False,
        "terms": {
            "base_contact": -100.0,
        },
    }
    safety_gate = {
        "enabled": True,
        "min_iteration": 8,
        "min_iteration_relative": False,
        "max_base_contact": 0.20,
        "max_scaled_residual_action_l2": 1.25,
        "patience": 2,
        "base_contact_key": "base_contact",
    }

    def __post_init__(self):
        super().__post_init__()
        gate_open = _full_training_enabled()
        requested_max_iterations = os.getenv("BOOSTER_TRACK_ADAPTER_MAX_ITERATIONS")
        smoke_iterations = min(
            _env_int("BOOSTER_TRACK_ADAPTER_SMOKE_ITERATIONS", self.smoke_max_iterations),
            int(self.smoke_max_iterations),
        )
        full_iterations = _env_int("BOOSTER_TRACK_ADAPTER_FULL_ITERATIONS", self.full_max_iterations)

        self.full_training_gate_open = gate_open
        self.allow_full_training = gate_open
        self.max_iterations_without_full_training_unlock = smoke_iterations
        self.strict_base_auxiliary_state = _env_bool(
            "BOOSTER_TRACK_ADAPTER_STRICT_BASE_AUXILIARY_STATE",
            bool(self.strict_base_auxiliary_state),
        )
        if gate_open:
            self.max_iterations = int(requested_max_iterations) if requested_max_iterations else full_iterations
        else:
            requested = int(requested_max_iterations) if requested_max_iterations else smoke_iterations
            self.gated_requested_max_iterations = requested
            self.max_iterations = min(requested, smoke_iterations)
            self.save_interval = min(self.save_interval, 25)
        self.save_interval = _env_int("BOOSTER_TRACK_ADAPTER_SAVE_INTERVAL", self.save_interval)

        self.base_checkpoint_path = os.getenv("BOOSTER_TRACK_ADAPTER_BASE_CHECKPOINT", self.base_checkpoint_path)
        self.pretrained_world_model_path = os.getenv(
            "BOOSTER_TRACK_ADAPTER_PRETRAINED_WM", self.pretrained_world_model_path
        )
        self.history_length = int(os.getenv("BOOSTER_TRACK_ADAPTER_HISTORY_LENGTH", self.history_length))
        self.num_steps_per_env = _env_int("BOOSTER_TRACK_ADAPTER_PPO_STEPS_PER_ENV", self.num_steps_per_env)
        self.freeze_discriminator = _env_bool("BOOSTER_TRACK_ADAPTER_FREEZE_DISCRIMINATOR", True)
        env_run_name = os.getenv("BOOSTER_TRACK_ADAPTER_RUN_NAME")
        if env_run_name:
            self.run_name = env_run_name
        elif gate_open:
            self.run_name = "track_adapter_recovery_full"
        self.amp_zero_command_scale = _env_float(
            "BOOSTER_TRACK_ADAPTER_AMP_ZERO_COMMAND_SCALE", self.amp_zero_command_scale
        )
        self.amp_zero_command_threshold = _env_float(
            "BOOSTER_TRACK_ADAPTER_AMP_ZERO_COMMAND_THRESHOLD", self.amp_zero_command_threshold
        )

        self.policy.base_checkpoint_path = self.base_checkpoint_path
        self.policy.history_length = self.history_length
        self.policy.adapter_mode = os.getenv("BOOSTER_TRACK_ADAPTER_MODE", self.policy.adapter_mode)
        layerwise_dims = os.getenv("BOOSTER_TRACK_ADAPTER_LAYERWISE_HIDDEN_DIMS")
        if layerwise_dims:
            self.policy.layerwise_adapter_hidden_dims = [
                int(value.strip()) for value in layerwise_dims.split(",") if value.strip()
            ]
        layerwise_output_init = os.getenv("BOOSTER_TRACK_ADAPTER_LAYERWISE_OUTPUT_INIT_SCALE")
        if layerwise_output_init is not None and layerwise_output_init.strip() != "":
            self.policy.layerwise_adapter_output_init_scale = float(layerwise_output_init)
        self.policy.residual_scale = _env_float("BOOSTER_TRACK_ADAPTER_RESIDUAL_SCALE", self.policy.residual_scale)
        self.policy.residual_output_init_scale = _env_float(
            "BOOSTER_TRACK_ADAPTER_RESIDUAL_OUTPUT_INIT_SCALE",
            self.policy.residual_output_init_scale,
        )
        self.policy.privileged_teacher_enabled = _env_bool(
            "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER",
            bool(self.policy.privileged_teacher_enabled),
        )
        teacher_dims = os.getenv("BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_HIDDEN_DIMS")
        if teacher_dims:
            self.policy.privileged_teacher_hidden_dims = [
                int(value.strip()) for value in teacher_dims.split(",") if value.strip()
            ]
        self.policy.privileged_teacher_residual_scale = _env_float(
            "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_RESIDUAL_SCALE",
            self.policy.privileged_teacher_residual_scale,
        )
        self.policy.privileged_teacher_output_init_scale = _env_float(
            "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_OUTPUT_INIT_SCALE",
            self.policy.privileged_teacher_output_init_scale,
        )
        self.policy.privileged_teacher_use_adapter_base = _env_bool(
            "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_USE_ADAPTER_BASE",
            bool(self.policy.privileged_teacher_use_adapter_base),
        )
        self.policy.init_noise_std = _env_float("BOOSTER_TRACK_ADAPTER_INIT_NOISE_STD", self.policy.init_noise_std)
        self.algorithm.freeze_discriminator = self.freeze_discriminator
        self.algorithm.learning_rate = _env_float("BOOSTER_TRACK_ADAPTER_LR", self.algorithm.learning_rate)
        self.algorithm.entropy_coef = _env_float("BOOSTER_TRACK_ADAPTER_ENTROPY", self.algorithm.entropy_coef)
        self.algorithm.max_grad_norm = _env_float("BOOSTER_TRACK_ADAPTER_MAX_GRAD_NORM", self.algorithm.max_grad_norm)
        self.algorithm.num_learning_epochs = _env_int(
            "BOOSTER_TRACK_ADAPTER_NUM_LEARNING_EPOCHS", self.algorithm.num_learning_epochs
        )
        self.algorithm.num_mini_batches = _env_int(
            "BOOSTER_TRACK_ADAPTER_NUM_MINI_BATCHES", self.algorithm.num_mini_batches
        )
        self.algorithm.world_model_loss_coef = _env_float(
            "BOOSTER_TRACK_ADAPTER_WM_LOSS_COEF", self.algorithm.world_model_loss_coef
        )
        self.algorithm.world_model_cfg = {
            **self.algorithm.world_model_cfg,
            "loss_coef": self.algorithm.world_model_loss_coef,
            "horizon": _env_int("BOOSTER_TRACK_ADAPTER_WM_HORIZON", self.algorithm.world_model_cfg["horizon"]),
            "target_source": os.getenv(
                "BOOSTER_TRACK_ADAPTER_WM_TARGET_SOURCE",
                self.algorithm.world_model_cfg.get("target_source", "psi_s"),
            ),
            "target_dim": _env_int(
                "BOOSTER_TRACK_ADAPTER_WM_TARGET_DIM",
                int(self.algorithm.world_model_cfg.get("target_dim", self.policy.world_model_target_dim)),
            ),
            "alternating_updates": _env_bool(
                "BOOSTER_TRACK_ADAPTER_WM_ALTERNATING_UPDATES",
                bool(self.algorithm.world_model_cfg.get("alternating_updates", True)),
            ),
            "joint_loss": _env_bool(
                "BOOSTER_TRACK_ADAPTER_WM_JOINT_LOSS",
                bool(self.algorithm.world_model_cfg.get("joint_loss", False)),
            ),
            "updates_per_iteration": _env_int(
                "BOOSTER_TRACK_ADAPTER_WM_UPDATES_PER_ITERATION",
                int(self.algorithm.world_model_cfg.get("updates_per_iteration", 2)),
            ),
            "mini_batch_size": _env_int(
                "BOOSTER_TRACK_ADAPTER_WM_MINI_BATCH_SIZE",
                int(self.algorithm.world_model_cfg.get("mini_batch_size", 1024)),
            ),
            "replay_buffer_size": _env_int(
                "BOOSTER_TRACK_ADAPTER_WM_REPLAY_BUFFER_SIZE",
                int(self.algorithm.world_model_cfg.get("replay_buffer_size", 65536)),
            ),
            "replay_device": os.getenv(
                "BOOSTER_TRACK_ADAPTER_WM_REPLAY_DEVICE",
                self.algorithm.world_model_cfg.get("replay_device", "cpu"),
            ),
            "replay_dtype": os.getenv(
                "BOOSTER_TRACK_ADAPTER_WM_REPLAY_DTYPE",
                self.algorithm.world_model_cfg.get("replay_dtype", "float16"),
            ),
            "ppo_updates_history_encoder": _env_bool(
                "BOOSTER_TRACK_ADAPTER_WM_PPO_UPDATES_HISTORY_ENCODER",
                bool(self.algorithm.world_model_cfg.get("ppo_updates_history_encoder", False)),
            ),
            "learning_rate": _env_float(
                "BOOSTER_TRACK_ADAPTER_WM_LEARNING_RATE",
                float(self.algorithm.world_model_cfg.get("learning_rate", self.algorithm.learning_rate)),
            ),
            "contact_sensor_name": os.getenv(
                "BOOSTER_TRACK_ADAPTER_WM_CONTACT_SENSOR",
                self.algorithm.world_model_cfg.get("contact_sensor_name", "contact_forces"),
            ),
            "foot_body_pattern": os.getenv(
                "BOOSTER_TRACK_ADAPTER_WM_FOOT_BODY_PATTERN",
                self.algorithm.world_model_cfg.get("foot_body_pattern", r".*_foot_.*"),
            ),
            "foot_contact_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_WM_FOOT_CONTACT_THRESHOLD",
                float(self.algorithm.world_model_cfg.get("foot_contact_threshold", 1.0)),
            ),
        }
        self.policy.world_model_target_dim = int(self.algorithm.world_model_cfg["target_dim"])
        self.policy.world_model_horizon = int(self.algorithm.world_model_cfg["horizon"])
        self.algorithm.residual_penalty_cfg = {
            **self.residual_penalty_cfg,
            "stable_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_STABLE_WEIGHT", self.residual_penalty_cfg["stable_weight"]
            ),
            "recovery_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_RECOVERY_WEIGHT", self.residual_penalty_cfg["recovery_weight"]
            ),
            "max_penalty": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_MAX_PENALTY", self.residual_penalty_cfg["max_penalty"]
            ),
            "zero_command_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_ZERO_COMMAND_WEIGHT",
                self.residual_penalty_cfg["zero_command_weight"],
            ),
            "zero_command_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_ZERO_COMMAND_THRESHOLD",
                self.residual_penalty_cfg["zero_command_threshold"],
            ),
            "ungated_stable_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_UNGATED_STABLE_WEIGHT",
                self.residual_penalty_cfg["ungated_stable_weight"],
            ),
            "ungated_policy_loss_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_UNGATED_POLICY_LOSS_WEIGHT",
                self.residual_penalty_cfg["ungated_policy_loss_weight"],
            ),
            "ungated_policy_gate_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_UNGATED_POLICY_GATE_THRESHOLD",
                self.residual_penalty_cfg["ungated_policy_gate_threshold"],
            ),
            "push_stable_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_PUSH_STABLE_WEIGHT",
                self.residual_penalty_cfg["push_stable_weight"],
            ),
            "push_stable_ungated_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_PUSH_STABLE_UNGATED_WEIGHT",
                self.residual_penalty_cfg["push_stable_ungated_weight"],
            ),
            "push_stable_score_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_PUSH_STABLE_SCORE_THRESHOLD",
                self.residual_penalty_cfg["push_stable_score_threshold"],
            ),
            "recovery_activation_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_RECOVERY_ACTIVATION_WEIGHT",
                self.residual_penalty_cfg["recovery_activation_weight"],
            ),
            "recovery_activation_gate_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_RECOVERY_ACTIVATION_GATE_THRESHOLD",
                self.residual_penalty_cfg["recovery_activation_gate_threshold"],
            ),
            "recovery_activation_target_norm": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_RECOVERY_ACTIVATION_TARGET_NORM",
                self.residual_penalty_cfg["recovery_activation_target_norm"],
            ),
            "recovery_activation_max_norm": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_RECOVERY_ACTIVATION_MAX_NORM",
                self.residual_penalty_cfg["recovery_activation_max_norm"],
            ),
            "recovery_activation_overgrowth_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_RECOVERY_ACTIVATION_OVERGROWTH_WEIGHT",
                self.residual_penalty_cfg["recovery_activation_overgrowth_weight"],
            ),
            "recovery_ppo_sample_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_PPO_SAMPLE_WEIGHT",
                self.residual_penalty_cfg["recovery_ppo_sample_weight"],
            ),
            "recovery_ppo_sample_gate_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_PPO_SAMPLE_GATE_THRESHOLD",
                self.residual_penalty_cfg["recovery_ppo_sample_gate_threshold"],
            ),
            "recovery_ppo_sample_power": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_PPO_SAMPLE_POWER",
                self.residual_penalty_cfg["recovery_ppo_sample_power"],
            ),
        }
        self.residual_penalty_cfg = self.algorithm.residual_penalty_cfg
        self.algorithm.recovery_teacher_cfg = {
            **self.algorithm.recovery_teacher_cfg,
            "enabled": _env_bool(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER",
                bool(self.algorithm.recovery_teacher_cfg.get("enabled", False)),
            ),
            "coef": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_COEF",
                float(self.algorithm.recovery_teacher_cfg.get("coef", 0.0)),
            ),
            "gate_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_GATE_THRESHOLD",
                float(self.algorithm.recovery_teacher_cfg.get("gate_threshold", 0.55)),
            ),
            "score_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_SCORE_THRESHOLD",
                float(self.algorithm.recovery_teacher_cfg.get("score_threshold", 0.18)),
            ),
            "velocity_gain": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_VELOCITY_GAIN",
                float(self.algorithm.recovery_teacher_cfg.get("velocity_gain", 0.55)),
            ),
            "tilt_gain": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_TILT_GAIN",
                float(self.algorithm.recovery_teacher_cfg.get("tilt_gain", 0.35)),
            ),
            "ang_vel_gain": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_ANG_VEL_GAIN",
                float(self.algorithm.recovery_teacher_cfg.get("ang_vel_gain", 0.10)),
            ),
            "force_gain": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_FORCE_GAIN",
                float(self.algorithm.recovery_teacher_cfg.get("force_gain", 0.0)),
            ),
            "push_delta_gain": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_PUSH_DELTA_GAIN",
                float(self.algorithm.recovery_teacher_cfg.get("push_delta_gain", 0.0)),
            ),
            "max_delta": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_MAX_DELTA",
                float(self.algorithm.recovery_teacher_cfg.get("max_delta", 0.12)),
            ),
            "non_leg_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_NON_LEG_WEIGHT",
                float(self.algorithm.recovery_teacher_cfg.get("non_leg_weight", 0.02)),
            ),
            "apply_to_privileged": _env_bool(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_APPLY_TO_PRIVILEGED",
                bool(self.algorithm.recovery_teacher_cfg.get("apply_to_privileged", False)),
            ),
            "separate_actor_grad_clip": _env_bool(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_SEPARATE_ACTOR_GRAD_CLIP",
                bool(self.algorithm.recovery_teacher_cfg.get("separate_actor_grad_clip", False)),
            ),
            "actor_max_grad_norm": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_TEACHER_ACTOR_MAX_GRAD_NORM",
                float(self.algorithm.recovery_teacher_cfg.get("actor_max_grad_norm", self.algorithm.max_grad_norm)),
            ),
        }
        self.algorithm.privileged_teacher_cfg = {
            **self.algorithm.privileged_teacher_cfg,
            "enabled": _env_bool(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER",
                bool(self.algorithm.privileged_teacher_cfg.get("enabled", False)),
            ),
            "train": _env_bool(
                "BOOSTER_TRACK_ADAPTER_TRAIN_PRIVILEGED_TEACHER",
                bool(self.algorithm.privileged_teacher_cfg.get("train", False)),
            ),
        }
        self.algorithm.privileged_teacher_distill_cfg = {
            **self.algorithm.privileged_teacher_distill_cfg,
            "enabled": _env_bool(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL",
                bool(self.algorithm.privileged_teacher_distill_cfg.get("enabled", False)),
            ),
            "coef": _env_float(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_COEF",
                float(self.algorithm.privileged_teacher_distill_cfg.get("coef", 0.0)),
            ),
            "gate_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_GATE_THRESHOLD",
                float(self.algorithm.privileged_teacher_distill_cfg.get("gate_threshold", 0.55)),
            ),
            "score_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_SCORE_THRESHOLD",
                float(self.algorithm.privileged_teacher_distill_cfg.get("score_threshold", 0.12)),
            ),
            "replay_enabled": _env_bool(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_REPLAY",
                bool(self.algorithm.privileged_teacher_distill_cfg.get("replay_enabled", False)),
            ),
            "replay_buffer_size": _env_int(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_REPLAY_SIZE",
                int(self.algorithm.privileged_teacher_distill_cfg.get("replay_buffer_size", 262144)),
            ),
            "replay_batch_size": _env_int(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_REPLAY_BATCH_SIZE",
                int(self.algorithm.privileged_teacher_distill_cfg.get("replay_batch_size", 4096)),
            ),
            "replay_updates_per_iteration": _env_int(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_REPLAY_UPDATES",
                int(self.algorithm.privileged_teacher_distill_cfg.get("replay_updates_per_iteration", 0)),
            ),
            "replay_warm_start_updates": _env_int(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_WARM_START_UPDATES",
                int(self.algorithm.privileged_teacher_distill_cfg.get("replay_warm_start_updates", 0)),
            ),
            "replay_min_samples": _env_int(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_REPLAY_MIN_SAMPLES",
                int(self.algorithm.privileged_teacher_distill_cfg.get("replay_min_samples", 4096)),
            ),
            "capture_teacher_replay": _env_bool(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_CAPTURE_TEACHER_REPLAY",
                bool(self.algorithm.privileged_teacher_distill_cfg.get("capture_teacher_replay", False)),
            ),
            "replay_path": os.getenv(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_REPLAY_PATH",
                str(self.algorithm.privileged_teacher_distill_cfg.get("replay_path", "")),
            ),
            "replay_load_path": os.getenv(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_REPLAY_LOAD_PATH",
                str(self.algorithm.privileged_teacher_distill_cfg.get("replay_load_path", "")),
            ),
            "replay_save_interval": _env_int(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_REPLAY_SAVE_INTERVAL",
                int(self.algorithm.privileged_teacher_distill_cfg.get("replay_save_interval", 0)),
            ),
            "replay_save_on_checkpoint": _env_bool(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_REPLAY_SAVE_ON_CHECKPOINT",
                bool(self.algorithm.privileged_teacher_distill_cfg.get("replay_save_on_checkpoint", True)),
            ),
            "replay_save_max_samples": _env_int(
                "BOOSTER_TRACK_ADAPTER_PRIVILEGED_TEACHER_DISTILL_REPLAY_SAVE_MAX_SAMPLES",
                int(self.algorithm.privileged_teacher_distill_cfg.get("replay_save_max_samples", 262144)),
            ),
        }
        self.residual_action_gate = {
            **self.residual_action_gate,
            "enabled": _env_bool("BOOSTER_TRACK_ADAPTER_RESIDUAL_ACTION_GATE", self.residual_action_gate["enabled"]),
            "stable_floor": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_STABLE_FLOOR",
                self.residual_action_gate["stable_floor"],
            ),
            "stable_max_gate": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_STABLE_MAX",
                self.residual_action_gate["stable_max_gate"],
            ),
            "stable_style_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_STABLE_STYLE_THRESHOLD",
                self.residual_action_gate["stable_style_threshold"],
            ),
            "stable_recovery_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_STABLE_RECOVERY_THRESHOLD",
                self.residual_action_gate["stable_recovery_threshold"],
            ),
            "zero_command_gate": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_ZERO_COMMAND",
                self.residual_action_gate["zero_command_gate"],
            ),
            "zero_command_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_ZERO_COMMAND_THRESHOLD",
                self.residual_action_gate["zero_command_threshold"],
            ),
            "recovery_scale": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_RECOVERY_SCALE",
                self.residual_action_gate["recovery_scale"],
            ),
            "recovery_power": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_RECOVERY_POWER",
                self.residual_action_gate["recovery_power"],
            ),
            "push_min_gate": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PUSH_MIN",
                self.residual_action_gate["push_min_gate"],
            ),
            "push_phase_enabled": _env_bool(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PUSH_PHASE_ENABLED",
                self.residual_action_gate["push_phase_enabled"],
            ),
            "push_full_gate_s": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PUSH_FULL_S",
                self.residual_action_gate["push_full_gate_s"],
            ),
            "push_taper_s": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PUSH_TAPER_S",
                self.residual_action_gate["push_taper_s"],
            ),
            "push_late_min_gate": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PUSH_LATE_MIN",
                self.residual_action_gate["push_late_min_gate"],
            ),
            "push_high_score_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PUSH_HIGH_SCORE_THRESHOLD",
                self.residual_action_gate["push_high_score_threshold"],
            ),
            "push_high_score_width": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PUSH_HIGH_SCORE_WIDTH",
                self.residual_action_gate["push_high_score_width"],
            ),
            "push_high_score_min_gate": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PUSH_HIGH_SCORE_MIN",
                self.residual_action_gate["push_high_score_min_gate"],
            ),
            "sudden_stop_min_gate": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_SUDDEN_STOP_MIN",
                self.residual_action_gate["sudden_stop_min_gate"],
            ),
            "manual_stop_speed_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_MANUAL_STOP_SPEED_THRESHOLD",
                self.residual_action_gate["manual_stop_speed_threshold"],
            ),
            "phase_recovery_min_gate": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PHASE_RECOVERY_MIN",
                self.residual_action_gate.get("phase_recovery_min_gate", self.residual_action_gate["push_min_gate"]),
            ),
            "phase_return_start_gate": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PHASE_RETURN_START",
                self.residual_action_gate.get("phase_return_start_gate", self.residual_action_gate["push_min_gate"]),
            ),
            "phase_return_end_gate": _env_float(
                "BOOSTER_TRACK_ADAPTER_RESIDUAL_GATE_PHASE_RETURN_END",
                self.residual_action_gate.get("phase_return_end_gate", self.residual_action_gate["stable_floor"]),
            ),
        }
        self.recovery_controller = {
            **self.recovery_controller,
            "enabled": _env_bool(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER",
                bool(self.recovery_controller["enabled"]),
            ),
            "active_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_ACTIVE_THRESHOLD",
                self.recovery_controller["active_threshold"],
            ),
            "enter_score": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_ENTER_SCORE",
                self.recovery_controller["enter_score"],
            ),
            "enter_recovery_gate": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_ENTER_RECOVERY_GATE",
                self.recovery_controller["enter_recovery_gate"],
            ),
            "exit_score": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_EXIT_SCORE",
                self.recovery_controller["exit_score"],
            ),
            "stable_score": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_STABLE_SCORE",
                self.recovery_controller["stable_score"],
            ),
            "enter_delay_s": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_ENTER_DELAY_S",
                self.recovery_controller["enter_delay_s"],
            ),
            "min_recovery_s": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_MIN_RECOVERY_S",
                self.recovery_controller["min_recovery_s"],
            ),
            "min_stable_s": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_MIN_STABLE_S",
                self.recovery_controller["min_stable_s"],
            ),
            "return_stable_s": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_RETURN_STABLE_S",
                self.recovery_controller["return_stable_s"],
            ),
            "return_decay_s": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_RETURN_DECAY_S",
                self.recovery_controller["return_decay_s"],
            ),
            "ramp_s": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_RAMP_S",
                self.recovery_controller["ramp_s"],
            ),
            "recovery_style_gate_max": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_STYLE_MAX",
                self.recovery_controller["recovery_style_gate_max"],
            ),
            "return_style_floor": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_RETURN_STYLE_FLOOR",
                self.recovery_controller["return_style_floor"],
            ),
            "recovery_gate_min": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_RECOVERY_GATE_MIN",
                self.recovery_controller["recovery_gate_min"],
            ),
            "return_recovery_gate_min": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_RETURN_RECOVERY_GATE_MIN",
                self.recovery_controller["return_recovery_gate_min"],
            ),
            "recovery_command_reward_scale": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_COMMAND_REWARD_SCALE",
                self.recovery_controller["recovery_command_reward_scale"],
            ),
            "return_command_reward_floor": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_CONTROLLER_RETURN_COMMAND_FLOOR",
                self.recovery_controller["return_command_reward_floor"],
            ),
        }
        self.recovery_command_override = {
            **self.recovery_command_override,
            "enabled": _env_bool(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE",
                self.recovery_command_override["enabled"],
            ),
            "zero_command_only": _env_bool(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_ZERO_ONLY",
                self.recovery_command_override["zero_command_only"],
            ),
            "zero_command_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_ZERO_THRESHOLD",
                self.recovery_command_override["zero_command_threshold"],
            ),
            "active_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_ACTIVE_THRESHOLD",
                self.recovery_command_override["active_threshold"],
            ),
            "direction_threshold": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_DIRECTION_THRESHOLD",
                self.recovery_command_override["direction_threshold"],
            ),
            "push_direction_gain": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_PUSH_DIRECTION_GAIN",
                self.recovery_command_override["push_direction_gain"],
            ),
            "velocity_direction_gain": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_VELOCITY_DIRECTION_GAIN",
                self.recovery_command_override["velocity_direction_gain"],
            ),
            "tilt_direction_gain": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_TILT_DIRECTION_GAIN",
                self.recovery_command_override["tilt_direction_gain"],
            ),
            "ang_vel_direction_gain": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_ANG_VEL_DIRECTION_GAIN",
                self.recovery_command_override["ang_vel_direction_gain"],
            ),
            "push_speed_gain": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_PUSH_SPEED_GAIN",
                self.recovery_command_override["push_speed_gain"],
            ),
            "velocity_speed_gain": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_VELOCITY_SPEED_GAIN",
                self.recovery_command_override["velocity_speed_gain"],
            ),
            "signal_speed_gain": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_SIGNAL_SPEED_GAIN",
                self.recovery_command_override["signal_speed_gain"],
            ),
            "min_speed": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_MIN_SPEED",
                self.recovery_command_override["min_speed"],
            ),
            "max_speed": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_MAX_SPEED",
                self.recovery_command_override["max_speed"],
            ),
            "max_abs_x": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_MAX_ABS_X",
                self.recovery_command_override["max_abs_x"],
            ),
            "max_abs_y": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_MAX_ABS_Y",
                self.recovery_command_override["max_abs_y"],
            ),
            "yaw_damping_gain": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_YAW_DAMPING_GAIN",
                self.recovery_command_override["yaw_damping_gain"],
            ),
            "max_yaw": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_MAX_YAW",
                self.recovery_command_override["max_yaw"],
            ),
            "ramp_s": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_RAMP_S",
                self.recovery_command_override["ramp_s"],
            ),
            "blend": _env_float(
                "BOOSTER_TRACK_ADAPTER_RECOVERY_COMMAND_OVERRIDE_BLEND",
                self.recovery_command_override["blend"],
            ),
        }
        self.terminal_penalty = {
            **self.terminal_penalty,
            "enabled": _env_bool("BOOSTER_TRACK_ADAPTER_TERMINAL_PENALTY", self.terminal_penalty["enabled"]),
            "terms": {
                **self.terminal_penalty["terms"],
                "base_contact": _env_float(
                    "BOOSTER_TRACK_ADAPTER_TERMINAL_BASE_CONTACT_PENALTY",
                    self.terminal_penalty["terms"]["base_contact"],
                ),
            },
        }
        self.reward_group_weights = {
            **self.reward_group_weights,
            "task": _env_float("BOOSTER_TRACK_ADAPTER_TASK_WEIGHT", self.reward_group_weights["task"]),
            "style": _env_float("BOOSTER_TRACK_ADAPTER_STYLE_WEIGHT", self.reward_group_weights["style"]),
            "recovery": _env_float("BOOSTER_TRACK_ADAPTER_RECOVERY_WEIGHT", self.reward_group_weights["recovery"]),
            "reg": _env_float("BOOSTER_TRACK_ADAPTER_REG_WEIGHT", self.reward_group_weights["reg"]),
        }
        self.disturbance_gate = {
            **self.disturbance_gate,
            "beta": _env_float("BOOSTER_TRACK_ADAPTER_GATE_BETA", self.disturbance_gate["beta"]),
            "min_style_gate": _env_float(
                "BOOSTER_TRACK_ADAPTER_MIN_STYLE_GATE", self.disturbance_gate["min_style_gate"]
            ),
            "command_min_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_COMMAND_MIN_WEIGHT", self.disturbance_gate["command_min_weight"]
            ),
            "tilt_weight": _env_float("BOOSTER_TRACK_ADAPTER_GATE_TILT_WEIGHT", self.disturbance_gate["tilt_weight"]),
            "ang_vel_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_GATE_ANG_VEL_WEIGHT", self.disturbance_gate["ang_vel_weight"]
            ),
            "velocity_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_GATE_VELOCITY_WEIGHT", self.disturbance_gate["velocity_weight"]
            ),
            "height_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_GATE_HEIGHT_WEIGHT", self.disturbance_gate["height_weight"]
            ),
            "contact_weight": _env_float(
                "BOOSTER_TRACK_ADAPTER_GATE_CONTACT_WEIGHT", self.disturbance_gate["contact_weight"]
            ),
            "push_recovery_gate_min": _env_float(
                "BOOSTER_TRACK_ADAPTER_PUSH_RECOVERY_GATE_MIN",
                self.disturbance_gate["push_recovery_gate_min"],
            ),
            "push_style_gate_max": _env_float(
                "BOOSTER_TRACK_ADAPTER_PUSH_STYLE_GATE_MAX",
                self.disturbance_gate["push_style_gate_max"],
            ),
            "push_command_gate_min": _env_float(
                "BOOSTER_TRACK_ADAPTER_PUSH_COMMAND_GATE_MIN",
                self.disturbance_gate["push_command_gate_min"],
            ),
            "push_command_gate_max": _env_float(
                "BOOSTER_TRACK_ADAPTER_PUSH_COMMAND_GATE_MAX",
                self.disturbance_gate["push_command_gate_max"],
            ),
            "push_command_gate_late_max": _env_float(
                "BOOSTER_TRACK_ADAPTER_PUSH_COMMAND_GATE_LATE_MAX",
                self.disturbance_gate["push_command_gate_late_max"],
            ),
            "push_command_gate_full_s": _env_float(
                "BOOSTER_TRACK_ADAPTER_PUSH_COMMAND_GATE_FULL_S",
                self.disturbance_gate["push_command_gate_full_s"],
            ),
            "push_command_gate_taper_s": _env_float(
                "BOOSTER_TRACK_ADAPTER_PUSH_COMMAND_GATE_TAPER_S",
                self.disturbance_gate["push_command_gate_taper_s"],
            ),
            "reg_recovery_scale": _env_float(
                "BOOSTER_TRACK_ADAPTER_REG_RECOVERY_SCALE", self.disturbance_gate["reg_recovery_scale"]
            ),
        }
        self.safety_gate = {
            **self.safety_gate,
            "min_iteration": _env_int(
                "BOOSTER_TRACK_ADAPTER_SAFETY_MIN_ITERATION", self.safety_gate["min_iteration"]
            ),
            "min_iteration_relative": _env_bool(
                "BOOSTER_TRACK_ADAPTER_SAFETY_MIN_ITERATION_RELATIVE",
                self.safety_gate.get("min_iteration_relative", False),
            ),
            "max_base_contact": _env_float(
                "BOOSTER_TRACK_ADAPTER_SAFETY_MAX_BASE_CONTACT", self.safety_gate["max_base_contact"]
            ),
            "max_scaled_residual_action_l2": _env_float(
                "BOOSTER_TRACK_ADAPTER_SAFETY_MAX_SCALED_RESIDUAL_ACTION_L2",
                self.safety_gate["max_scaled_residual_action_l2"],
            ),
            "patience": _env_int("BOOSTER_TRACK_ADAPTER_SAFETY_PATIENCE", self.safety_gate["patience"]),
        }

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
import math
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp as mdp

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.OmniVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        # Resume from run_amp_y but widen the command range for backward and
        # faster lateral motion. This is intentionally still trained with the
        # 56-col AMP layout below; 62-col/root-velocity AMP is reserved for a
        # later low-weight fine-tune ablation.
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        rel_yaw_only_envs=0.15,
        yaw_only_ang_vel_abs_range=(0.15, 0.3),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.3, 0.3)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=False)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05),clip=(-100.0, 100.0),scale=1.0,)
        projected_gravity = ObsTerm(func=mdp.projected_gravity,noise=GaussianNoise(mean=0.0, std=0.025),clip=(-100.0, 100.0),scale=1.0,)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=1.0,)
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=1.0,)
        actions = ObsTerm(func=mdp.last_action, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=1.0,)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        # robot base measurements
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel,clip=(-100.0, 100.0),scale=1.0,)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel,clip=(-100.0, 100.0),scale=1.0,)
        projected_gravity = ObsTerm(func=mdp.projected_gravity,clip=(-100.0, 100.0),scale=1.0,)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # robot joint measurements
        joint_pos = ObsTerm(func=mdp.joint_pos, clip=(-100.0, 100.0), scale=1.0,)
        joint_vel = ObsTerm(func=mdp.joint_vel, clip=(-100.0, 100.0), scale=1.0,)

        # last action
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0,)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    @configclass
    class AMPObsCfg(ObsGroup):
        # 56-col layout for compatibility with run_amp_y:
        # joint_pos[22] + joint_vel[22] + EE_pos_b[12].
        # Do not add root velocity terms here for this resume run.
        joint_pos = ObsTerm(func=mdp.joint_pos, clip=(-100.0, 100.0), scale=1.0,)
        joint_vel = ObsTerm(func=mdp.joint_vel, clip=(-100.0, 100.0), scale=1.0,)
        left_hand_pos = ObsTerm(func=mdp.get_lefthand_pos, clip=(-100.0, 100.0), scale=1.0,)
        right_hand_pos = ObsTerm(func=mdp.get_righthand_pos, clip=(-100.0, 100.0), scale=1.0,)
        left_foot_pos = ObsTerm(func=mdp.get_leftfoot_pos, clip=(-100.0, 100.0), scale=1.0,)
        right_foot_pos = ObsTerm(func=mdp.get_rightfoot_pos, clip=(-100.0, 100.0), scale=1.0,)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()
    amp_observations :AMPObsCfg = AMPObsCfg()

@configclass
class EventCfg:
    """Configuration for events."""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.3, 0.6),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-0.2, 0.8),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (-2.0, 2.0),
            "torque_range": (-2.0, 2.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # -- task
    # Experiment B: keep gait-shaping reductions from A (feet_air_time 0.5,
    # head_height 3.0) and additionally boost tracking weights so velocity
    # tracking becomes the dominant signal for the policy. track_lin_vel_xy
    # from 2.0 -> 5.0, track_ang_vel_z from 1.2 -> 2.5.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight= 0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "command_name": "base_velocity", "threshold": 0.5 },
    )
    head_height = RewTerm(func=mdp.tracking_head_height, params={"target_head_height": 0.35 ,"threshold": 0.8179, "std": 0.3, "command_name": "base_velocity", "command_threshold": 0.2, "asset_cfg": SceneEntityCfg("robot", body_names=["Head_2"])}, weight=3.0)
    keep_balance = RewTerm(
        func=mdp.stay_alive,
        weight=1.0
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight= -0.5,
        params={"command_name": "base_velocity"},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight= -2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot_.*"]),
        },
    )
    # Experiment C: penalize horizontal foot velocity while in contact.
    # With the boosted tracking weight (Experiment B) the policy can reach
    # commanded speeds by sliding its feet; this term closes that shortcut.
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot_.*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot_.*"]),
        },
    )

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=5.0, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.5, params={"command_name": "base_velocity", "std": math.sqrt(0.5)}
    )
    yaw_only_track_ang_vel_z_exp = RewTerm(
        func=mdp.yaw_only_track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.25, "lin_threshold": 0.15, "yaw_threshold": 0.15},
    )
    yaw_only_feet_air_time = RewTerm(
        func=mdp.yaw_only_feet_air_time,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot_.*"]),
            "command_name": "base_velocity",
            "threshold": 0.12,
            "lin_threshold": 0.15,
            "yaw_threshold": 0.15,
        },
    )
    # # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)
    head_ang_vel_xy_l2 = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Head_2"])},
    )
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_Shank",".*Hip.*", ".*hand.*", ".*Arm.*", "Head_.*", "Trunk"]), "threshold": 1.0},
    )
    # -- optional penalties
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ "Trunk"]), "threshold": 1.0},
    )
    # bad_orientation = DoneTerm(
    #     func=mdp.bad_orientation,
    #     params={"limit_angle": 0.7},
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.origin_type = "world"   # for rotation the view by mouse and keyboard
        self.viewer.eye = (3.0, -4.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)

"""V4 Stage 2 — Trap Skill env config (single-agent, role-conditioned).

Layout:
  * Scene: K1 robot + soccer ball + flat terrain. NO goal posts (the trap
    task does not need them).
  * Actor obs: legacy proprio + perceived ball pos/mask/dt + role one-hot
    + ball history (Step A).
  * Critic obs: proprio + GT ball pos/vel + predicted contact distance
    + role + history.
  * Rewards: trap-skill goal rewards (``receiver_ball_at_feet_skill``,
    ``trap_anticipation``, ``trap_success_skill``) + aux shaping
    (intercept alignment, idle-when-blind, search, last-seen, head-yaw),
    + posture / alive / terminated, + standard regularizers.
  * Term: time_out + fall_height + fall_tilt (no goal / ball_out).

Registered as ``Booster-Soccer-Trap-Skill-v0`` via the sibling ``__init__``.
"""
from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

import booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp as mdp
from booster_rl_tasks.assets.objects import SOCCER_BALL_CFG
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_trap_commands import (
    SoccerTrapCommandCfg,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_perception import (
    VirtualPerceptionCfg,
    soccer_vision_train_cfg,
)


# =========================================================================
# Scene — K1 robot + ball + flat terrain (no goal posts)
# =========================================================================


@configclass
class SoccerTrapSceneCfg(InteractiveSceneCfg):
    """Scene for the trap-skill task: robot + ball + terrain. No goal posts."""

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
    robot: ArticulationCfg = MISSING
    ball: RigidObjectCfg = SOCCER_BALL_CFG  # type: ignore[assignment]
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )


# =========================================================================
# Commands — single-agent trap receiver
# =========================================================================


@configclass
class CommandsCfg:
    """Stage 2 command: single receiver, role="receiver", history_len=10."""

    soccer_trap = SoccerTrapCommandCfg(
        debug_vis=False,
        perception=soccer_vision_train_cfg(),
        role="receiver",
        ball_history_len=10,
    )


# =========================================================================
# Actions
# =========================================================================


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=False
    )


# =========================================================================
# Observations
# =========================================================================


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # proprio (noisy)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
        )
        actions = ObsTerm(
            func=mdp.last_action,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
        )
        # task-conditioning observations
        ball_pos_b = ObsTerm(
            func=mdp.soccer_trap_obs.trap_ball_pos_b,
            params={"command_name": "soccer_trap"},
            clip=(-30.0, 30.0),
        )
        ball_mask = ObsTerm(
            func=mdp.soccer_trap_obs.trap_ball_mask,
            params={"command_name": "soccer_trap"},
        )
        last_seen_dt = ObsTerm(
            func=mdp.soccer_trap_obs.trap_last_seen_dt,
            params={"command_name": "soccer_trap"},
        )
        # V4 role + history
        role_one_hot = ObsTerm(
            func=mdp.soccer_role_obs.role_one_hot,
            params={"command_name": "soccer_trap"},
        )
        ball_history = ObsTerm(
            func=mdp.soccer_role_obs.ball_history_flat,
            params={"command_name": "soccer_trap"},
            clip=(-30.0, 30.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        # proprio (clean)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        joint_pos = ObsTerm(func=mdp.joint_pos, clip=(-100.0, 100.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, clip=(-100.0, 100.0))
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))
        # ground-truth task-relevant state
        ball_pos_b_gt = ObsTerm(
            func=mdp.soccer_trap_obs.trap_ball_pos_b_gt,
            params={"command_name": "soccer_trap"},
            clip=(-30.0, 30.0),
        )
        ball_vel_b_gt = ObsTerm(
            func=mdp.soccer_trap_obs.trap_ball_vel_b_gt,
            params={"command_name": "soccer_trap"},
            clip=(-30.0, 30.0),
        )
        predicted_contact_distance = ObsTerm(
            func=mdp.soccer_trap_obs.trap_predicted_contact_distance,
            params={"command_name": "soccer_trap"},
            clip=(-10.0, 10.0),
        )
        role_one_hot = ObsTerm(
            func=mdp.soccer_role_obs.role_one_hot,
            params={"command_name": "soccer_trap"},
        )
        ball_history = ObsTerm(
            func=mdp.soccer_role_obs.ball_history_flat,
            params={"command_name": "soccer_trap"},
            clip=(-30.0, 30.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AMPObsCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, clip=(-100.0, 100.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, clip=(-100.0, 100.0))
        left_hand_pos = ObsTerm(func=mdp.get_lefthand_pos, clip=(-100.0, 100.0))
        right_hand_pos = ObsTerm(func=mdp.get_righthand_pos, clip=(-100.0, 100.0))
        left_foot_pos = ObsTerm(func=mdp.get_leftfoot_pos, clip=(-100.0, 100.0))
        right_foot_pos = ObsTerm(func=mdp.get_rightfoot_pos, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()
    amp_observations: AMPObsCfg = AMPObsCfg()


# =========================================================================
# Events — physics/material/mass DR. (Robot + ball reset is handled in cmd.)
# =========================================================================


@configclass
class EventCfg:
    physics_material_robot = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )
    physics_material_ball = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball", body_names=".*"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.2, 0.5),
            "num_buckets": 32,
        },
    )
    ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball", body_names=".*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    trunk_mass = EventTerm(
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
            "com_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "z": (-0.01, 0.01)},
        },
    )


# =========================================================================
# Rewards
# =========================================================================


@configclass
class RewardsCfg:
    # ---- Trap "goal" rewards ----
    receiver_ball_at_feet = RewTerm(
        func=mdp.soccer_trap_rewards.receiver_ball_at_feet_skill,
        weight=6.0,
        params={"command_name": "soccer_trap", "sigma": 0.30},
    )
    trap_anticipation = RewTerm(
        func=mdp.soccer_trap_rewards.trap_anticipation,
        weight=2.0,
        params={
            "command_name": "soccer_trap",
            "contact_radius": 0.5,
            "min_approach_speed": 0.5,
        },
    )
    trap_success = RewTerm(
        func=mdp.soccer_trap_rewards.trap_success_skill,
        weight=30.0,
        params={"command_name": "soccer_trap"},
    )

    # ---- Trap "aux" shaping ----
    intercept_alignment = RewTerm(
        func=mdp.soccer_trap_rewards.intercept_alignment_skill,
        weight=-1.0,
        params={"command_name": "soccer_trap"},
    )
    receiver_idle_when_blind = RewTerm(
        func=mdp.soccer_trap_rewards.receiver_idle_when_blind,
        weight=0.3,
        params={"command_name": "soccer_trap", "posture_tolerance": 0.3},
    )
    search_yaw_velocity = RewTerm(
        func=mdp.soccer_rewards.search_yaw_velocity,
        weight=2.0,
        params={"command_name": "soccer_trap", "min_rate": 0.5, "max_rate": 3.0},
    )
    last_seen_dt_penalty = RewTerm(
        func=mdp.soccer_rewards.last_seen_dt_penalty,
        weight=-1.0,
        params={"command_name": "soccer_trap", "max_dt": 5.0},
    )
    head_yaw_search = RewTerm(
        func=mdp.soccer_rewards.head_yaw_search,
        weight=0.5,
        params={"command_name": "soccer_trap", "min_abs_rate": 0.5},
    )

    # ---- Posture / survival ----
    pelvis_orientation = RewTerm(
        func=mdp.soccer_rewards.pelvis_orientation_penalty,
        weight=-1.0,
    )
    feet_proximity = RewTerm(
        func=mdp.soccer_rewards.feet_proximity_penalty,
        weight=-1.0,
    )
    alive = RewTerm(func=mdp.soccer_trap_rewards.trap_alive, weight=0.5)
    terminated = RewTerm(func=mdp.soccer_trap_rewards.trap_terminated, weight=-20.0)

    # ---- Standard regularizers (shared with locomotion baseline) ----
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_Shank", ".*Hip.*", ".*hand.*", ".*Arm.*", "Head_.*", "Trunk"],
            ),
            "threshold": 1.0,
        },
    )


# =========================================================================
# Terminations — time_out + fall (no goal / ball_out)
# =========================================================================


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall_height = DoneTerm(
        func=mdp.soccer_terminations.fall_height,
        params={"min_height": 0.30},
    )
    fall_tilt = DoneTerm(
        func=mdp.soccer_terminations.fall_tilt,
        params={"max_tilt": 1.3},
    )


# =========================================================================
# Env config
# =========================================================================


@configclass
class SoccerTrapSkillEnvCfg(ManagerBasedRLEnvCfg):
    """V4 Stage 2 env cfg — single-agent trap skill."""

    scene: SoccerTrapSceneCfg = SoccerTrapSceneCfg(num_envs=4096, env_spacing=16.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # Simulation timing — 50Hz control, 200Hz physics (matches kick).
        self.decimation = 4
        self.episode_length_s = 4.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15

        # Viewer
        self.viewer.origin_type = "world"
        self.viewer.eye = (6.0, -6.0, 3.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)

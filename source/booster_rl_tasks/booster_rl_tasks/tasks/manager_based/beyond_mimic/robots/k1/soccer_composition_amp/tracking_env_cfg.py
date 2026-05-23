"""V4 Stage 4 — Composition validation env config (pass+trap eval scene).

Layout:
  * Scene = 2 K1 robots (``robot`` = kicker, ``receiver``) + ball + goal +
    flat terrain. Reuses the V3.1 scene/event topology so the receiver's
    pose is reset by ``reset_receiver_to_default`` and the kicker drives
    the ball via :class:`SoccerKickCommand` with pass-only (``shoot_prob=0``).
  * Two command terms run side-by-side:
      - ``soccer_kick`` — :class:`SoccerKickCommand` with ``receiver_name="receiver"``
        so the pass target tracks the receiver xy each step.
      - ``soccer_trap`` — :class:`PassiveTrapCommand` (subclass that skips
        robot/ball respawn but keeps trap detection + history buffer for
        the receiver).
  * Observations: two separate policy groups so each policy sees its native
    Stage 1 / Stage 2 layout (no obs-shape adapter required).
      - ``policy_kicker`` ≡ Stage 1 ``policy`` layout (22 proprio + 8 task
        + 4 role + 50 history = 84 dims).
      - ``policy_receiver`` ≡ Stage 2 ``policy`` layout (22 proprio + 4 task
        + 4 role + 50 history = 80 dims).
      - ``critic`` is a placeholder (we don't train).
  * Actions: 22 (kicker) + 22 (receiver) = 44-dim concatenated.
  * Rewards: a single ``alive`` term — evaluation only.
  * Terminations: time_out + kicker fall + receiver fall.

Composition metrics (pass_landing_rate, trap_success_rate,
combined_success_rate, ...) are computed in
``scripts/rsl_rl/play_multi.py`` from command-term state, not from rewards.
"""
from __future__ import annotations

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
from booster_rl_tasks.assets.objects import (
    SOCCER_BALL_CFG,
    soccer_goal_assets,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_commands import (
    SoccerKickCommandCfg,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_perception import (
    VirtualPerceptionCfg,
    soccer_vision_train_cfg,
)
from dataclasses import MISSING

from .passive_trap_command import PassiveTrapCommandCfg


# =========================================================================
# Scene — 2 K1 robots + ball + goal posts + flat terrain
# =========================================================================


@configclass
class SoccerCompositionSceneCfg(InteractiveSceneCfg):
    """Two-K1 scene: kicker (``robot``) + receiver (``receiver``) + ball + goal."""

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
    receiver: ArticulationCfg = MISSING
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
# Commands — kicker (pass mode) + passive trap (receiver bookkeeping)
# =========================================================================


@configclass
class CommandsCfg:
    soccer_kick = SoccerKickCommandCfg(
        debug_vis=False,
        perception=soccer_vision_train_cfg(),
        kick_ball_speed_thresh=1.0,
        kick_success_speed_thresh=1.5,
        receiver_name="receiver",
        role="kicker",
        ball_history_len=10,
        # Composition validation: always evaluate pass mode.
        shoot_prob=0.0,
    )
    soccer_trap = PassiveTrapCommandCfg(
        debug_vis=False,
        perception=soccer_vision_train_cfg(),
        asset_name="receiver",
        ball_name="ball",
        role="receiver",
        ball_history_len=10,
    )


# =========================================================================
# Actions — 22 (kicker) + 22 (receiver) = 44-dim
# =========================================================================


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=False
    )
    joint_pos_receiver = mdp.JointPositionActionCfg(
        asset_name="receiver", joint_names=[".*"], use_default_offset=False
    )


# =========================================================================
# Observations — separate policy groups per agent
# =========================================================================


@configclass
class ObservationsCfg:
    """Two policy groups (one per agent) + a dummy critic group.

    Each policy group's term order MUST match the Stage 1 / Stage 2
    ``PolicyCfg`` exactly so the saved checkpoint can be evaluated
    without an observation adapter.
    """

    # ----- Kicker policy group: Stage 1 PolicyCfg layout (84 dims) ---------
    @configclass
    class PolicyKickerCfg(ObsGroup):
        # proprio (noisy)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
        )
        actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "joint_pos"},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
        )
        # task-conditioning (kicker)
        ball_pos_b = ObsTerm(
            func=mdp.soccer_observations.ball_pos_b,
            params={"command_name": "soccer_kick"},
            clip=(-30.0, 30.0),
        )
        ball_mask = ObsTerm(
            func=mdp.soccer_observations.ball_mask,
            params={"command_name": "soccer_kick"},
        )
        last_seen_dt = ObsTerm(
            func=mdp.soccer_observations.last_seen_dt,
            params={"command_name": "soccer_kick"},
        )
        target_dir_b = ObsTerm(
            func=mdp.soccer_observations.target_dir_b,
            params={"command_name": "soccer_kick"},
        )
        target_strength = ObsTerm(
            func=mdp.soccer_observations.target_strength_norm,
            params={"command_name": "soccer_kick"},
        )
        is_shoot = ObsTerm(
            func=mdp.soccer_observations.is_shoot_flag,
            params={"command_name": "soccer_kick"},
        )
        # V4 role one-hot + history
        role_one_hot = ObsTerm(
            func=mdp.soccer_role_obs.role_one_hot,
            params={"command_name": "soccer_kick"},
        )
        ball_history = ObsTerm(
            func=mdp.soccer_role_obs.ball_history_flat,
            params={"command_name": "soccer_kick"},
            clip=(-30.0, 30.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # ----- Receiver policy group: Stage 2 PolicyCfg layout (80 dims) -------
    @configclass
    class PolicyReceiverCfg(ObsGroup):
        # proprio (noisy) — receiver
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("receiver")},
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("receiver")},
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("receiver")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("receiver")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
        )
        actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "joint_pos_receiver"},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
        )
        # task-conditioning (trap — driven by PassiveTrapCommand)
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
        # V4 role + history (driven by soccer_trap)
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

    # ----- Critic placeholder (we don't train; the env_cfg requires
    # *some* default policy group named "policy" if the wrapper assumes it,
    # so we mirror PolicyKickerCfg as the default "policy" group too). ----
    @configclass
    class PolicyCfg(PolicyKickerCfg):
        """Default ``policy`` alias for the kicker observation.

        Provided so that single-policy tooling (``play.py``,
        ``train.py``) that hard-codes ``obs["policy"]`` still finds a
        valid tensor — it falls back to the kicker view.
        """

        def __post_init__(self):
            super().__post_init__()

    policy: PolicyCfg = PolicyCfg()
    policy_kicker: PolicyKickerCfg = PolicyKickerCfg()
    policy_receiver: PolicyReceiverCfg = PolicyReceiverCfg()


# =========================================================================
# Events — DR + receiver default-pose reset (V3.1 topology)
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
    # Reset receiver to a position 2.5–4 m in front of the kicker.
    # (Kicker / ball pose come from SoccerKickCommand._resample_command.)
    reset_receiver = EventTerm(
        func=mdp.soccer_trap_events.reset_receiver_to_default,
        mode="reset",
        params={
            "asset_name": "receiver",
            "offset_x_range": (2.5, 4.0),
            "offset_y_range": (-1.0, 1.0),
            "spawn_z": 0.57,
        },
    )


# =========================================================================
# Rewards — minimal (eval only)
# =========================================================================


@configclass
class RewardsCfg:
    """Composition eval — single dummy reward so the manager has a term."""

    alive = RewTerm(func=mdp.is_alive, weight=0.0)


# =========================================================================
# Terminations — time_out + both robots' fall terms
# =========================================================================


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    kicker_fall_height = DoneTerm(
        func=mdp.soccer_terminations.fall_height,
        params={"min_height": 0.30},
    )
    kicker_fall_tilt = DoneTerm(
        func=mdp.soccer_terminations.fall_tilt,
        params={"max_tilt": 1.3},
    )
    receiver_fall_height = DoneTerm(
        func=mdp.soccer_terminations.receiver_fall_height,
        params={"min_height": 0.30},
    )
    receiver_fall_tilt = DoneTerm(
        func=mdp.soccer_terminations.receiver_fall_tilt,
        params={"max_tilt": 1.3},
    )


# =========================================================================
# Env config
# =========================================================================


@configclass
class SoccerCompositionEnvCfg(ManagerBasedRLEnvCfg):
    """V4 Stage 4 composition validation env (pass+trap)."""

    scene: SoccerCompositionSceneCfg = SoccerCompositionSceneCfg(
        num_envs=64, env_spacing=16.0
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # Attach goal posts to the scene as additional AssetBaseCfg fields.
        for name, cfg in soccer_goal_assets().items():
            setattr(self.scene, name, cfg)

        # Simulation timing — same as Stage 1/2 (50Hz control, 200Hz physics).
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Two articulations + ball + goal — bump GPU patch budget.
        self.sim.physx.gpu_max_rigid_patch_count = 32 * 2**15

        # Viewer
        self.viewer.origin_type = "world"
        self.viewer.eye = (6.0, -6.0, 3.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)

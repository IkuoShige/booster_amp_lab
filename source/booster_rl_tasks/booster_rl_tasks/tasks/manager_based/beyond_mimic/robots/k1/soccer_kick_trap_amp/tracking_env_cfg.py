"""V3 soccer kick + trap env config (scaffold stage — kicker + passive receiver).

This is the scene-topology smoke test for the V3 multi-agent design (see
``docs/soccer_amp_v3_design.md``). The scene now holds **two K1 articulations**:

* ``robot``    — the kicker (full action / observation / reward stack from V1)
* ``receiver`` — a second K1 sibling of the kicker, **without** an action term

The receiver's actuators are PD-controlled toward its default joint pose, so
once a reset event (``reset_receiver_to_default``) writes the default joint
state + a randomized base xy / yaw, the receiver remains roughly upright and
stationary for the duration of the episode. This proves that two articulations
can coexist under the same env_origin before we invest in the full two-policy
training architecture.

Everything else (commands, rewards, terminations, perception) is reused
verbatim from the V1 kicker task to keep the scaffolding minimal.
"""
from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp as mdp
from booster_rl_tasks.assets.objects import (
    SOCCER_BALL_CFG,
    soccer_goal_assets,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.robots.k1.soccer_kick_amp.tracking_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)


# =========================================================================
# Scene — kicker + passive receiver + ball + goal posts + terrain
# =========================================================================


@configclass
class SoccerKickTrapSceneCfg(InteractiveSceneCfg):
    """V3 scene: two K1 robots ('robot' = kicker, 'receiver' = passive) + ball + goal."""

    # ground terrain (flat) — same material setup as V1.
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
    # Two robots — kicker (controllable) + receiver (passive). Both populated
    # in the env-cfg subclass's __post_init__ from BOOSTER_K1_CFG.replace(...).
    robot: ArticulationCfg = MISSING
    receiver: ArticulationCfg = MISSING
    # Soccer ball (RigidObject).
    ball: RigidObjectCfg = SOCCER_BALL_CFG  # type: ignore[assignment]
    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    # Contact sensor covers only the kicker bodies — receiver contacts are not
    # referenced in any reward / termination yet (kept symmetric for V3.1+).
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )


# =========================================================================
# Events — V1 set + receiver default-pose reset
# =========================================================================


@configclass
class EventCfg:
    # Domain randomization (same as V1, applied to the kicker + ball only).
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

    # Kicker + ball poses are reset by ``SoccerKickCommand._resample_command``.
    # The receiver's pose is reset here (passive — no action term in V3).
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

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 10.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


# =========================================================================
# Env config
# =========================================================================


@configclass
class SoccerKickTrapEnvCfg(ManagerBasedRLEnvCfg):
    """Base env cfg for the V3 soccer kick+trap scaffold (single-policy kicker)."""

    scene: SoccerKickTrapSceneCfg = SoccerKickTrapSceneCfg(num_envs=4096, env_spacing=16.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # Attach goal posts to the scene as additional AssetBaseCfg fields.
        for name, cfg in soccer_goal_assets().items():
            setattr(self.scene, name, cfg)

        # V3: pass-mode envs aim at the receiver's actual world xy each step.
        # ``receiver_name`` activates the live-tracking path inside
        # ``SoccerKickCommand``; the receiver itself is reset by
        # ``mdp.soccer_trap_events.reset_receiver_to_default``.
        self.commands.soccer_kick.receiver_name = "receiver"
        # Train both shoot and pass modes (V3 explicit 50/50 split).
        self.commands.soccer_kick.shoot_prob = 0.5

        # Privileged-obs hookup: kicker sees the receiver xy in its body-yaw
        # frame so the critic can credit-assign pass-direction errors.
        self.observations.critic.receiver_pos_b = ObsTerm(
            func=mdp.soccer_observations.receiver_pos_b,
            params={"command_name": "soccer_kick"},
            clip=(-30.0, 30.0),
            scale=1.0,
        )

        # Simulation timing — same as V1.
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Two articulations + ball + goal → bump GPU patch budget a little more.
        self.sim.physx.gpu_max_rigid_patch_count = 32 * 2**15

        # Viewer
        self.viewer.origin_type = "world"
        self.viewer.eye = (6.0, -6.0, 3.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)

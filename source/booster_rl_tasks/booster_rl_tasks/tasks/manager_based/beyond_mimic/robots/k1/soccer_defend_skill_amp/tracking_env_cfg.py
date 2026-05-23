"""V4 Stage 3 — Defend Skill env config (single-agent, role-conditioned).

Layout:
  * Scene: K1 robot (defender) + soccer ball + own-goal posts (mirror of the
    shoot goal at x = -GOAL_LINE_X) + flat terrain.
  * Actor obs: proprio + perceived ball xy/mask/last_seen_dt + own-goal
    direction + role one-hot + ball history flat.
  * Critic obs: same proprio + GT ball xy/vel + own-goal direction +
    is_in_lane flag + block / steal latches + role one-hot + ball history.
  * AMP obs: identical to legacy soccer (joint pos/vel + hand/foot pos).
  * Rewards: block_success (+30), steal_success (+25),
    defensive_line_hold (+5/step), ball_pushed_away (+10),
    own_goal_proximity_penalty (-15), plus alive/terminated/posture/regs.
  * Term: timeout (5 s), fall (height < 0.30 or tilt > 1.3), own-goal scored.

Registered as ``Booster-Soccer-Defend-Skill-v0`` via the sibling ``__init__``.
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
from booster_rl_tasks.assets.objects import (
    SOCCER_BALL_CFG,
    soccer_own_goal_assets,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_defend_commands import (
    SoccerDefendCommandCfg,
)
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp.soccer_perception import (
    VirtualPerceptionCfg,
    soccer_vision_train_cfg,
)


# =========================================================================
# Scene
# =========================================================================


@configclass
class SoccerDefendSceneCfg(InteractiveSceneCfg):
    """Scene for the soccer defend task: defender + ball + own-goal + terrain."""

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
# Commands
# =========================================================================


@configclass
class CommandsCfg:
    """Stage 3 command: defender vs incoming ball."""

    soccer_defend = SoccerDefendCommandCfg(
        debug_vis=False,
        perception=soccer_vision_train_cfg(),
        role="defender",
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
        # ---- proprio (noisy) ----
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        # ---- task-conditioning (perception-based) ----
        ball_pos_b = ObsTerm(
            func=mdp.soccer_defend_obs.defend_ball_pos_b,
            params={"command_name": "soccer_defend"},
            clip=(-30.0, 30.0),
            scale=1.0,
        )
        ball_mask = ObsTerm(
            func=mdp.soccer_defend_obs.defend_ball_mask,
            params={"command_name": "soccer_defend"},
        )
        last_seen_dt = ObsTerm(
            func=mdp.soccer_defend_obs.defend_last_seen_dt,
            params={"command_name": "soccer_defend"},
        )
        own_goal_dir_b = ObsTerm(
            func=mdp.soccer_defend_obs.defend_own_goal_dir_b,
            params={"command_name": "soccer_defend"},
        )
        # ---- role one-hot + ball history (V4 Step A) ----
        role_one_hot = ObsTerm(
            func=mdp.soccer_role_obs.role_one_hot,
            params={"command_name": "soccer_defend"},
        )
        ball_history = ObsTerm(
            func=mdp.soccer_role_obs.ball_history_flat,
            params={"command_name": "soccer_defend"},
            clip=(-30.0, 30.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        # ---- proprio (clean) ----
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0
        )
        joint_pos = ObsTerm(func=mdp.joint_pos, clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel, clip=(-100.0, 100.0), scale=1.0)
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        # ---- GT task state ----
        ball_pos_b_gt = ObsTerm(
            func=mdp.soccer_defend_obs.defend_ball_pos_b_gt,
            params={"command_name": "soccer_defend"},
            clip=(-30.0, 30.0),
            scale=1.0,
        )
        ball_vel_b_gt = ObsTerm(
            func=mdp.soccer_defend_obs.defend_ball_vel_b_gt,
            params={"command_name": "soccer_defend"},
            clip=(-30.0, 30.0),
            scale=1.0,
        )
        own_goal_dir_b = ObsTerm(
            func=mdp.soccer_defend_obs.defend_own_goal_dir_b,
            params={"command_name": "soccer_defend"},
        )
        is_in_lane = ObsTerm(
            func=mdp.soccer_defend_obs.defend_is_in_lane,
            params={"command_name": "soccer_defend"},
        )
        block_flags = ObsTerm(
            func=mdp.soccer_defend_obs.defend_block_flags,
            params={"command_name": "soccer_defend"},
        )
        # ---- role + history (mirror policy for symmetric value-fn input) ----
        role_one_hot = ObsTerm(
            func=mdp.soccer_role_obs.role_one_hot,
            params={"command_name": "soccer_defend"},
        )
        ball_history = ObsTerm(
            func=mdp.soccer_role_obs.ball_history_flat,
            params={"command_name": "soccer_defend"},
            clip=(-30.0, 30.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AMPObsCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel, clip=(-100.0, 100.0), scale=1.0)
        left_hand_pos = ObsTerm(func=mdp.get_lefthand_pos, clip=(-100.0, 100.0), scale=1.0)
        right_hand_pos = ObsTerm(func=mdp.get_righthand_pos, clip=(-100.0, 100.0), scale=1.0)
        left_foot_pos = ObsTerm(func=mdp.get_leftfoot_pos, clip=(-100.0, 100.0), scale=1.0)
        right_foot_pos = ObsTerm(func=mdp.get_rightfoot_pos, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()
    amp_observations: AMPObsCfg = AMPObsCfg()


# =========================================================================
# Events
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
    # ---- Goal-group: defend-specific bonuses & penalties (V4.1 rebalance) ----
    #
    # V4.0 ran 4 h with ``defensive_line_hold`` weight 5/step yielding a
    # ~4.2 / episode contribution that dominated ``block_success`` (~0.005 /
    # episode) by 3 orders of magnitude. The defender learned to park in
    # lane (lane_hold 90 %) and never engage (block+steal 5 %).
    #
    # V4.1 changes:
    #   * drop ``defensive_line_hold`` weight 5 → 0.3 (per-episode ~0.27)
    #   * add ``ball_intercept_proximity`` (continuous Gaussian on
    #     defender↔segment distance) — same role as line_hold but smooth,
    #     so the policy learns to *converge to* the line not just be on it
    #   * add ``defender_approach_ball`` — closes the loop on actually
    #     engaging the ball (small per-episode ~0.5)
    #   * boost block_success 30 → 80, steal_success 25 → 60 so a single
    #     interception event is ~5–10× the dense reward total
    #   * own_goal_proximity stays -15 (terminal disincentive)
    block_success = RewTerm(
        func=mdp.soccer_defend_rewards.block_success,
        weight=80.0,
        params={"command_name": "soccer_defend"},
    )
    steal_success = RewTerm(
        func=mdp.soccer_defend_rewards.steal_success,
        weight=60.0,
        params={"command_name": "soccer_defend"},
    )
    defensive_line_hold = RewTerm(
        func=mdp.soccer_defend_rewards.defensive_line_hold,
        weight=0.3,
        params={"command_name": "soccer_defend"},
    )
    ball_intercept_proximity = RewTerm(
        func=mdp.soccer_defend_rewards.ball_intercept_proximity,
        weight=2.0,
        params={"command_name": "soccer_defend", "sigma": 0.8},
    )
    defender_approach_ball = RewTerm(
        func=mdp.soccer_defend_rewards.defender_approach_ball,
        weight=2.0,
        params={"command_name": "soccer_defend", "sigma": 1.5},
    )
    ball_pushed_away = RewTerm(
        func=mdp.soccer_defend_rewards.ball_pushed_away,
        weight=10.0,
        params={"command_name": "soccer_defend"},
    )
    own_goal_proximity = RewTerm(
        func=mdp.soccer_defend_rewards.own_goal_proximity_penalty,
        weight=-15.0,
        params={"command_name": "soccer_defend"},
    )

    # ---- Aux-group: alive / terminated ----
    alive = RewTerm(func=mdp.soccer_rewards.alive_reward, weight=0.5)
    terminated = RewTerm(func=mdp.soccer_rewards.terminated_penalty, weight=-20.0)

    # ---- Aux-group: posture & regularizers ----
    pelvis_orientation = RewTerm(
        func=mdp.soccer_rewards.pelvis_orientation_penalty,
        weight=-1.0,
    )
    feet_proximity = RewTerm(
        func=mdp.soccer_rewards.feet_proximity_penalty,
        weight=-1.0,
    )
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
                body_names=[".*Hip.*", "Head_.*"],
            ),
            "threshold": 1.0,
        },
    )


# =========================================================================
# Terminations
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
    own_goal = DoneTerm(
        func=mdp.soccer_terminations.own_goal_scored,
        params={"command_name": "soccer_defend"},
    )


# =========================================================================
# Env config
# =========================================================================


@configclass
class SoccerDefendEnvCfg(ManagerBasedRLEnvCfg):
    """Base env cfg for the soccer defend skill task."""

    scene: SoccerDefendSceneCfg = SoccerDefendSceneCfg(num_envs=4096, env_spacing=16.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # Attach own-goal posts to the scene as additional AssetBaseCfg fields.
        for name, cfg in soccer_own_goal_assets().items():
            setattr(self.scene, name, cfg)

        # Simulation timing — 50 Hz control, 200 Hz physics.
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15

        # Viewer
        self.viewer.origin_type = "world"
        self.viewer.eye = (-6.0, -6.0, 3.5)
        self.viewer.lookat = (-3.0, 0.0, 0.5)

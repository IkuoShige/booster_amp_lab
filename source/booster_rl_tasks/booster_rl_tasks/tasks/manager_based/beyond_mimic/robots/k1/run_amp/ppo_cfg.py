import os

from isaaclab.utils import configclass
from booster_assets import BOOSTER_ASSETS_DIR
from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BaseAMPAgentCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg, RslRlRndCfg
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp import symmetry


_AMP_ROOT = os.path.join(BOOSTER_ASSETS_DIR, "motions", "K1", "motion_amp_expert", "omni")


@configclass
class PPORunnerCfg(BaseAMPAgentCfg):
    max_iterations = 50000
    experiment_name = "run_amp_omni"

    # AMP corpus (62-col layout):
    #   joint_pos[22] + joint_vel[22] + EE_pos_b[12] + root_lin_vel_b[3] + root_ang_vel_b[3]
    # The two trailing root-velocity channels let the discriminator penalize
    # gait modes that move feet without translating / rotating the COM (the
    # "kick-without-weight-shift" failure observed in lateral). Adding root
    # velocities also makes pivot motions distinguishable from generic foot
    # shuffling: ω_z ≠ 0 with |v_xy| ≈ 0 only matches pivot expert frames.
    amp_reward_coef = 0.2

    # 20-clip omnidirectional corpus.
    # Sampling weights are picked so each category contributes a comparable
    # fraction of expert transitions:
    #   forward (4 × 0.50) = 2.0  → 25%
    #   lateral (8 × 0.30) = 2.4  → 30%
    #   backward (4 × 0.50) = 2.0 → 25%
    #   pivot (4 × 0.40) = 1.6    → 20%
    # Pivot is mildly down-weighted because pivot clips are short and pivot
    # rewards already get explicit task signal from the angular-velocity
    # tracking term — bumping its sampling weight risks making the
    # discriminator score "feet shuffle in place" too high.
    amp_motion_files = [
        # forward
        os.path.join(_AMP_ROOT, "forward", "walk.txt"),
        os.path.join(_AMP_ROOT, "forward", "walk2run.txt"),
        os.path.join(_AMP_ROOT, "forward", "run.txt"),
        os.path.join(_AMP_ROOT, "forward", "run2walk.txt"),
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
        # pivot
        os.path.join(_AMP_ROOT, "pivot", "pivot_left_slow.txt"),
        os.path.join(_AMP_ROOT, "pivot", "pivot_left_fast.txt"),
        os.path.join(_AMP_ROOT, "pivot", "pivot_right_slow.txt"),
        os.path.join(_AMP_ROOT, "pivot", "pivot_right_fast.txt"),
    ]
    amp_num_preload_transitions = 200000
    # NOTE: amp_task_reward_lerp is read but the vendored discriminator just
    # uses r = disc_r + task_r when >0. Kept for signature compatibility.
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 22

    # AMP gating disabled: the corpus now covers the previously OOD regimes
    # (backward, pivot, near-stationary at low velocity is bracketed by the
    # walk_*_slow + pivot_*_slow clips). The earlier gating attempt caused a
    # pivot collapse when AMP shut off, so we let the discriminator stay
    # active across the whole command space instead.
    amp_command_gating = False
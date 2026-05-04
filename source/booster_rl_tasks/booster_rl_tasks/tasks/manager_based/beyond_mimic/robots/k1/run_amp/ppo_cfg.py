import os

from isaaclab.utils import configclass
from booster_assets import BOOSTER_ASSETS_DIR
from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BaseAMPAgentCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg, RslRlRndCfg
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp import symmetry


_AMP_ROOT = os.path.join(BOOSTER_ASSETS_DIR, "motions", "K1", "motion_amp_expert")


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

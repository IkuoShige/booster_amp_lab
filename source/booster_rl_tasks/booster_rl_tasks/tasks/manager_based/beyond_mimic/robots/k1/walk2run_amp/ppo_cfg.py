import os

from booster_assets import BOOSTER_ASSETS_DIR
from isaaclab.utils import configclass
from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BaseAMPAgentCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg, RslRlRndCfg
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp import symmetry


_AMP_ROOT = os.path.join(BOOSTER_ASSETS_DIR, "motions", "K1", "motion_amp_expert")


@configclass
class PPORunnerCfg(BaseAMPAgentCfg):
    max_iterations = 50000
    experiment_name = "walk2run_amp"

    # amp parameter
    amp_reward_coef = 0.3
    # Per-clip speeds (root_pos from motion_visualization):
    #   walk          0.59 m/s
    #   walk2run      transient (~0.6 -> ~2.0 m/s)
    #   run           2.68 m/s
    #   run2walk      transient (~2.5 -> ~0.5 m/s)
    amp_motion_files = [
        os.path.join(_AMP_ROOT, "walk.txt"),
        os.path.join(_AMP_ROOT, "walk2run.txt"),
        os.path.join(_AMP_ROOT, "run.txt"),
        os.path.join(_AMP_ROOT, "run2walk.txt"),
    ]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 22
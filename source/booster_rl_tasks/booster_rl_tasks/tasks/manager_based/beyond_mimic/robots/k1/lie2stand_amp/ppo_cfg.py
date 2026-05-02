import os

from booster_assets import BOOSTER_ASSETS_DIR
from isaaclab.utils import configclass
from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BaseAMPAgentCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg, RslRlRndCfg
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp import symmetry


_AMP_ROOT = os.path.join(BOOSTER_ASSETS_DIR, "motions", "K1", "motion_amp_expert")


@configclass
class LiedownPPORunnerCfg(BaseAMPAgentCfg):
    max_iterations = 50000
    experiment_name = "liedown_amp"

    # amp parameter
    amp_reward_coef = 0.3
    amp_motion_files = [os.path.join(_AMP_ROOT, "liedown.txt")]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 22

@configclass
class StandupPPORunnerCfg(BaseAMPAgentCfg):
    max_iterations = 50000
    experiment_name = "standup_amp"

    # amp parameter
    amp_reward_coef = 0.3
    amp_motion_files = [os.path.join(_AMP_ROOT, "standup.txt")]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 22
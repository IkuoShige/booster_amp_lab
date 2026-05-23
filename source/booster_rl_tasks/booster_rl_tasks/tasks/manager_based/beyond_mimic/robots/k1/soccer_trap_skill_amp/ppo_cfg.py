"""V4 Stage 2 — PPO runner cfg for the Trap Skill task.

Mirrors :mod:`soccer_kick_skill_amp.ppo_cfg` but with its own experiment
name. The AMP corpus is shared across skills (walk / walk2run / run +
omni/kick/walk_kick) — locomotion is a foundation for trap too.
"""
import glob
import os

from booster_assets import BOOSTER_ASSETS_DIR
from isaaclab.utils import configclass

from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import (
    BaseMultiCriticAMPAgentCfg,
)


_AMP_ROOT = os.path.join(BOOSTER_ASSETS_DIR, "motions", "K1", "motion_amp_expert")
_KICK_FILES = sorted(glob.glob(os.path.join(_AMP_ROOT, "omni", "kick", "walk_kick*.txt")))


@configclass
class PPORunnerCfg(BaseMultiCriticAMPAgentCfg):
    experiment_name = "soccer_trap_skill_amp"
    max_iterations = 8000

    amp_reward_coef = 0.3
    amp_motion_files = [
        os.path.join(_AMP_ROOT, "walk.txt"),
        os.path.join(_AMP_ROOT, "walk2run.txt"),
        os.path.join(_AMP_ROOT, "run.txt"),
        *_KICK_FILES,
    ]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 22

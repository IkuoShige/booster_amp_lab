"""PPO runner config for the V3 soccer kick+trap scaffold.

Single-critic AMP PPO targeting the kicker only — the receiver is passive in
V3 and the multi-policy / multi-critic runner is not implemented yet (see
``docs/soccer_amp_v3_design.md``). This config is intentionally a verbatim
copy of the V1 ``soccer_kick_amp.PPORunnerCfg`` except for the experiment
name, so the V3 scaffold trains exactly like V1 while we validate the
two-articulation scene topology.
"""
import glob
import os

from booster_assets import BOOSTER_ASSETS_DIR
from isaaclab.utils import configclass

from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import (
    BaseAMPAgentCfg,
)


_AMP_ROOT = os.path.join(BOOSTER_ASSETS_DIR, "motions", "K1", "motion_amp_expert")
_KICK_FILES = sorted(glob.glob(os.path.join(_AMP_ROOT, "omni", "kick", "walk_kick*.txt")))


@configclass
class PPORunnerCfg(BaseAMPAgentCfg):
    experiment_name = "soccer_kick_trap_amp"
    max_iterations = 50000

    # AMP corpus: locomotion priors (legacy 56-col) + new 56-col walk+kick
    # clips (same as V1). All clips must match the env-side AMP obs width
    # (joint+EE = 56 cols).
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

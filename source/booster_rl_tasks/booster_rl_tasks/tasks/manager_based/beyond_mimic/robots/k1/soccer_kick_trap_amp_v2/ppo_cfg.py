"""PPO runner config for V3.2 — single shared 44-dim policy, multi-critic AMP.

Uses :class:`MultiCriticAmpOnPolicyRunner` with the same two-group
goal/aux partition used by V1.2; the V3.2 ``RewardsCfgV2.CRITIC_GROUPS``
maps receiver-side trap rewards into those groups (``trap_success`` and
``receiver_ball_at_feet`` → goal; ``receiver_alive``, ``receiver_terminated``,
``receiver_face_ball`` default to aux).

Action dim is 44 (22 kicker + 22 receiver), so ``min_normalized_std`` is
extended to 44 entries to match.
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
    experiment_name = "soccer_kick_trap_amp_v2"
    max_iterations = 50000

    # AMP corpus: locomotion priors (legacy 56-col) + new 56-col walk+kick
    # clips (kicker-only discriminator). All clips must match the env-side
    # AMP obs width (joint+EE = 56 cols).
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
    # 22 kicker joints + 22 receiver joints = 44-dim joint action.
    min_normalized_std = [0.05] * 44

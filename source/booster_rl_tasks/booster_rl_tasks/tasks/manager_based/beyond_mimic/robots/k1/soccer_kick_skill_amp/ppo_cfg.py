"""V4 Stage 1 — PPO runner cfg for the Kick Skill task.

Mirrors :mod:`soccer_kick_amp_mc.ppo_cfg` but:
  * different experiment name so checkpoints land in their own log dir;
  * default ``max_iterations`` is 8000 (matches the §7 Stage 1 plan).
The AMP corpus is unchanged — walk/walk2run/run + omni/kick/walk_kick.
"""
import glob
import os

from booster_assets import BOOSTER_ASSETS_DIR
from isaaclab.utils import configclass

from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import (
    BaseEncoderMultiCriticAMPAgentCfg,
    BaseMultiCriticAMPAgentCfg,
    EncoderCfg,
)


_AMP_ROOT = os.path.join(BOOSTER_ASSETS_DIR, "motions", "K1", "motion_amp_expert")
_KICK_FILES = sorted(glob.glob(os.path.join(_AMP_ROOT, "omni", "kick", "walk_kick*.txt")))


@configclass
class PPORunnerCfg(BaseMultiCriticAMPAgentCfg):
    experiment_name = "soccer_kick_skill_amp"
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


@configclass
class EncoderPPORunnerCfg(BaseEncoderMultiCriticAMPAgentCfg):
    """LVDRS-style Kick V5.3 agent: 50-frame ball history -> 64-dim latent."""

    experiment_name = "soccer_kick_skill_amp"
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

    encoder_cfg = EncoderCfg(
        # Actor layout: legacy 80 + role_one_hot 4 + ball_history.
        history_slice=(84, 334),
        # Critic layout: privileged 95 + role_one_hot 4 + ball_history.
        critic_history_slice=(99, 349),
        history_len=50,
        history_dim=5,
        latent_dim=64,
        # Linear 250->64 projection starts as "latest 10-frame history passthrough"
        # inside EncoderActorCritic, allowing V5.2 policy warm-start.
        encoder_hidden_dims=(),
        # Reconstruct GT ball pos_b(3) + vel_b(3) from the history latent.
        decoder_target_dim=6,
        decoder_hidden_dims=(128, 128),
        decoder_target_slice=(75, 81),
        decoder_loss_coef=0.05,
    )

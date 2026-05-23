"""V4 Stage 4 — composition validation eval runner cfg.

This task is **evaluation only**; the play_multi.py runner does not call
into the rsl_rl trainer. A minimal cfg is provided so the gym registry's
``rsl_rl_cfg_entry_point`` resolves and tooling that introspects agent
hyperparameters can still construct the cfg without error.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import (
    BaseMultiCriticAMPAgentCfg,
)


@configclass
class PPORunnerCfg(BaseMultiCriticAMPAgentCfg):
    """Stub PPO cfg — composition task is not trained. Reuses Stage 1 dims
    so a per-agent policy could in principle be re-trained here later."""

    experiment_name = "soccer_composition_amp"
    max_iterations = 1
    amp_motion_files: list[str] = []
    min_normalized_std = [0.05] * 22

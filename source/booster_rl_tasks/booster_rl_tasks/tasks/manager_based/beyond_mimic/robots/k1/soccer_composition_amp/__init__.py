"""V4 Stage 4 — Composition Validation (pass+trap).

Registers ``Booster-Soccer-Composition-v0``. Two K1 robots share a scene:
the *kicker* (``robot``) holds a ball and passes it to the *receiver*
(``receiver``), which must trap it. The env exposes two separate policy
observation groups (``policy_kicker`` matching the Stage 1 layout and
``policy_receiver`` matching the Stage 2 layout) so the corresponding
checkpoints can be evaluated unchanged.

Rewards are intentionally minimal — this task is for *evaluation*, not
training. The composition metrics live in :file:`scripts/rsl_rl/play_multi.py`
and are computed from command-term state (``trap_success_awarded``,
``pass_target_pos_w``, ball xy, etc.).

See ``docs/soccer_amp_skill_library_design.md`` §7 Stage 4 and §8 Deployment.
"""
import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Booster-Soccer-Composition-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatSoccerCompositionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

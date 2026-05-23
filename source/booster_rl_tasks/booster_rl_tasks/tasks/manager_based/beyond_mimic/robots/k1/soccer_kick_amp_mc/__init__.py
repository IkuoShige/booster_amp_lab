"""Soccer kick AMP multi-critic task (LVDRS §3.2).

Registers ``Booster-Soccer-Kick-AMP-MC-v0``: same env as the single-critic
variant in ``soccer_kick_amp``, but with the goal/aux reward split exposed
through a ``CRITIC_GROUPS`` mapping on ``RewardsCfg`` and trained with the
multi-critic AMP-PPO runner.
"""
import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Booster-Soccer-Kick-AMP-MC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatSoccerKickMultiCriticEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

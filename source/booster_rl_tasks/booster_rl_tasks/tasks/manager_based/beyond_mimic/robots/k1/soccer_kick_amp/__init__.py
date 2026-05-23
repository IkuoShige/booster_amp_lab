"""Soccer kick AMP task (V1 single-agent).

Registers ``Booster-Soccer-Kick-AMP-v0`` with the Isaac Lab managers-based env
runner and the existing AMP-PPO runner in ``rsl_rl``.
"""
import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Booster-Soccer-Kick-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatSoccerKickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

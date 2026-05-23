"""Soccer kick+trap AMP V3.2 task — single shared policy controls BOTH robots.

V3.2 turns the V3/V3.1 ``Booster-Soccer-KickTrap-AMP-v0`` "passive receiver"
scaffold into a fully trainable task. The receiver gets its own action term
(``joint_pos_receiver``) and its own observations / rewards / terminations.
A *single* PPO policy with a 44-dim joint action output controls both
robots — the policy network has to learn to act as kicker via the kicker
joints and as trapper via the receiver joints.

Registers ``Booster-Soccer-KickTrap-AMP-V2-v0`` with the multi-critic AMP
PPO runner so the goal vs aux reward partition stays clean across the
much larger reward stack.
"""
import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Booster-Soccer-KickTrap-AMP-V2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatSoccerKickTrapV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

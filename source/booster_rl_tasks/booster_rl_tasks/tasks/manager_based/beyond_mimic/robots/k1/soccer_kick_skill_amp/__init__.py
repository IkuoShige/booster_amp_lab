"""V4 Stage 1 — Kick Skill (single-agent, role-conditioned, multi-critic AMP).

Registers ``Booster-Soccer-Kick-Skill-v0``. Mirrors :mod:`soccer_kick_amp_mc`
but:
  * adds a 4-dim ``role`` one-hot observation (set to "kicker" in this task);
  * adds a 50-dim flattened ball-history observation (Step A);
  * uses mode-conditional rewards (shoot fast-bonus + pass single-shot
    discipline) so the same policy can be conditioned on ``is_shoot``.

See ``docs/soccer_amp_skill_library_design.md`` §4.1 + §7 (Stage 1).
"""
import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Booster-Soccer-Kick-Skill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatSoccerKickSkillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-Soccer-Kick-Skill-V5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatSoccerKickSkillV5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-Soccer-Kick-Skill-V51-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatSoccerKickSkillV51EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-Soccer-Kick-Skill-V52-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatSoccerKickSkillV52EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-Soccer-Kick-Skill-V53-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatSoccerKickSkillV53EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:EncoderPPORunnerCfg",
    },
)

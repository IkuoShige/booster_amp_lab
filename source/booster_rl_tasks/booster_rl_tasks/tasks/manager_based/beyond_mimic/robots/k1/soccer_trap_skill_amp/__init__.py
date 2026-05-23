"""V4 Stage 2 — Trap Skill (single-agent receiver, multi-critic AMP).

Registers ``Booster-Soccer-Trap-Skill-v0``. Scene: single K1 + ball spawned
1.5–3 m away in a forward cone with an initial velocity (2–5 m/s) pointing
roughly at the robot. The policy must time its foot to deaden the ball
(within 0.30 m of a foot and < 0.4 m/s xy speed).

See ``docs/soccer_amp_skill_library_design.md`` §4.2 + §7 (Stage 2).
"""
import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Booster-Soccer-Trap-Skill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatSoccerTrapSkillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

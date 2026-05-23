"""V4 Stage 3 — Defend Skill (single-agent, role-conditioned, multi-critic AMP).

Registers ``Booster-Soccer-Defend-Skill-v0``. Mirrors :mod:`soccer_kick_skill_amp`
but:
  * defender starts close to its own goal (mirror of the shoot goal, at
    ``x = -GOAL_LINE_X``);
  * ball spawns 4-8 m in front of the defender with an initial velocity of
    2-8 m/s pointed at the own goal;
  * rewards reward blocking / stealing / lane-holding / pushing the ball
    away, and penalise the ball crossing the own-goal line (terminal).
  * role = "defender".

See ``docs/soccer_amp_skill_library_design.md`` §4.3 + §7 (Stage 3).
"""
import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Booster-Soccer-Defend-Skill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatSoccerDefendSkillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

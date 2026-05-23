"""Soccer kick + trap AMP task (V3 scaffold).

V3 introduces a second K1 robot (the *receiver*) sharing the scene with the
kicker. For the scaffold stage only the *kicker* has an action term — the
receiver is passive: its PD actuators hold the default joint pose while a
reset event reseats its base at a random xy / yaw in front of the kicker.
This task registers ``Booster-Soccer-KickTrap-AMP-v0``; the rsl-rl entry
point is the same single-policy AMP-PPO runner used by the V1 task, since
no two-agent training architecture has landed yet (see
``docs/soccer_amp_v3_design.md``).
"""
import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Booster-Soccer-KickTrap-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatSoccerKickTrapEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

from isaaclab.utils import configclass
from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BaseAMPAgentCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg, RslRlRndCfg
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp import symmetry


@configclass
class PPORunnerCfg(BaseAMPAgentCfg):
    max_iterations = 50000
    experiment_name = "run_amp"

    # amp parameter
    # Experiment D: reduce AMP influence (0.3 -> 0.2) so the discriminator
    # pulls the policy toward walk/run modes less strongly in the 1.0-1.3
    # m/s mid range where no single-speed expert exists.
    amp_reward_coef = 0.2
    # High-speed specialist: command range is (0.5, 2.0) m/s. Use the full
    # locomotion expert set (walk 0.59, walk2run transit, run 2.68, run2walk
    # transit) so the discriminator has in-distribution references across
    # the command band. Slow-walk synthetic clips are not used here because
    # commands below 0.5 are not exercised.
    amp_motion_files = [
        "/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/walk.txt",
        "/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/walk2run.txt",
        "/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/run.txt",
        "/root/booster_rl_tasks/booster_assets/motions/K1/motion_amp_expert/run2walk.txt",
    ]
    amp_num_preload_transitions = 200000
    # NOTE: amp_task_reward_lerp is read but not applied by the vendored
    # discriminator (it just uses r = disc_r + task_r when >0). Value here
    # is kept only to preserve the existing call signature.
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 22
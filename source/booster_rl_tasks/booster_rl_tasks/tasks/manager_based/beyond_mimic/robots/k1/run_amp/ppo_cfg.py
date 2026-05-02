import os

from isaaclab.utils import configclass
from booster_assets import BOOSTER_ASSETS_DIR
from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BaseAMPAgentCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg, RslRlRndCfg
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp import symmetry


_AMP_ROOT = os.path.join(BOOSTER_ASSETS_DIR, "motions", "K1", "motion_amp_expert")


@configclass
class PPORunnerCfg(BaseAMPAgentCfg):
    max_iterations = 50000
    experiment_name = "run_amp_y"

    # amp parameter
    # Omni-directional run: corpus has 4 forward modes × {own data} + 4 lateral
    # modes × {left, right} mirrored. Discriminator receives lateral as style
    # prior alongside forward. Coef kept at 0.2.
    amp_reward_coef = 0.2
    # Forward (walk 0.8, walk2run 0.5, run 0.8, run2walk 0.5) — total weight 2.6.
    # Lateral 8 clips × 0.3 = 2.4 → forward / lateral sampling ≈ 52 / 48.
    # Both left and right are included because the AMP discriminator does not
    # auto-mirror joint slots — strafe_left and strafe_right have distinct
    # joint signatures (Left_Hip vs Right_Hip slot exchange + roll sign flip)
    # so the discriminator must see both to score either direction as natural.
    amp_motion_files = [
        os.path.join(_AMP_ROOT, "walk.txt"),
        os.path.join(_AMP_ROOT, "walk2run.txt"),
        os.path.join(_AMP_ROOT, "run.txt"),
        os.path.join(_AMP_ROOT, "run2walk.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_walk_left.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_walk_right.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_walk2run_left.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_walk2run_right.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_run_left.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_run_right.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_run2walk_left.txt"),
        os.path.join(_AMP_ROOT, "lateral", "strafe_run2walk_right.txt"),
    ]
    amp_num_preload_transitions = 200000
    # NOTE: amp_task_reward_lerp is read but not applied by the vendored
    # discriminator (it just uses r = disc_r + task_r when >0). Value here
    # is kept only to preserve the existing call signature.
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 22
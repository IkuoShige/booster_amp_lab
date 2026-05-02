import os
from dataclasses import MISSING

from booster_assets import BOOSTER_ASSETS_DIR
from isaaclab.utils import configclass
from booster_rl_tasks.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BaseAMPAgentCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg, RslRlRndCfg
from booster_rl_tasks.tasks.manager_based.beyond_mimic.mdp import symmetry


_AMP_ROOT = os.path.join(BOOSTER_ASSETS_DIR, "motions", "K1", "motion_amp_expert")

@configclass
class depth_predictor_cfg:
    lr: float= 3e-4
    weight_decay: float= 1e-4
    training_interval: int= 10
    training_iters: int= 1000
    batch_size: int= 1024
    loss_scale: int= 100
    #  
    resized: tuple= (64, 64)
    update_interval: int= 5  # 5 works without retraining, 8 worse
    camera_num_envs: int = 1024
    forward_height_dim: int = 525 
    use_camera: bool = False


@configclass
class WMRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    latent_dim: int = MISSING
    wm_latent_dim: int = MISSING
    wm_encoder_hidden_dims: list = MISSING
    encoder_hidden_dims: list = MISSING


@configclass
class PPORunnerCfg(BaseAMPAgentCfg):
    max_iterations = 50000
    experiment_name = "wm_amp"

    # amp parameter
    amp_reward_coef = 0.3
    amp_motion_files = [os.path.join(_AMP_ROOT, "walk_t1.txt")]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512]
    min_normalized_std = [0.05] * 22

    policy = WMRslRlPpoActorCriticCfg(
        class_name="ActorCriticWMP",
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",

        latent_dim = 32 + 3,
        wm_latent_dim = 32,
        wm_encoder_hidden_dims = [64, 64],
        encoder_hidden_dims = [256, 128],

    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="WMAMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    depth_predictor = depth_predictor_cfg()


    prop_dim = 78
    height_dim = 121
    privileged_dim = 189

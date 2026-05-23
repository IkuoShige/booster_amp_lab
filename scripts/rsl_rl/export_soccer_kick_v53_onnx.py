"""Export the V5.3 soccer-kick encoder policy as a deployable ONNX graph.

The generic IsaacLab/RSL-RL exporter cannot export EncoderActorCritic directly
because the public observation is 334-dim, while the internal actor MLP receives
the 250-dim ball-history slice compressed to a 64-dim latent. This exporter
wraps the deploy path explicitly:

    raw 334-dim obs -> empirical normalizer -> encoder(history[84:334])
    -> actor input 148-dim -> 22 actions

The resulting ONNX model has one input:

    obs: float32[batch, 334]

and one output:

    actions: float32[batch, 22]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from rsl_rl.modules import EncoderActorCritic


DEFAULT_RUN = (
    "logs/rsl_rl/soccer_kick_skill_amp/"
    "2026-05-23_13-28-28_kick_skill_v53_lvdrs_critic_split_from_2800_curriculum_restore"
)
DEFAULT_CHECKPOINT = f"{DEFAULT_RUN}/model_5400.pt"


OBS_LAYOUT = [
    {"name": "base_ang_vel", "start": 0, "end": 3},
    {"name": "projected_gravity", "start": 3, "end": 6},
    {"name": "joint_pos", "start": 6, "end": 28},
    {"name": "joint_vel", "start": 28, "end": 50},
    {"name": "previous_actions", "start": 50, "end": 72},
    {"name": "ball_pos_b_xy", "start": 72, "end": 74},
    {"name": "ball_mask", "start": 74, "end": 75},
    {"name": "last_seen_dt", "start": 75, "end": 76},
    {"name": "target_dir_b", "start": 76, "end": 78},
    {"name": "target_strength_norm", "start": 78, "end": 79},
    {"name": "is_shoot", "start": 79, "end": 80},
    {"name": "role_one_hot", "start": 80, "end": 84},
    {"name": "ball_history", "start": 84, "end": 334},
]


class SoccerKickV53DeployWrapper(nn.Module):
    """Fixed deploy graph: normalizer + history encoder + actor MLP."""

    def __init__(
        self,
        policy: EncoderActorCritic,
        *,
        obs_mean: torch.Tensor | None,
        obs_std: torch.Tensor | None,
        normalizer_eps: float = 1e-2,
    ) -> None:
        super().__init__()
        self.encoder = policy.encoder
        self.actor = policy.actor
        self.history_start = int(policy.history_slice[0])
        self.history_end = int(policy.history_slice[1])
        self.normalizer_eps = float(normalizer_eps)
        self.use_normalizer = obs_mean is not None and obs_std is not None
        if self.use_normalizer:
            self.register_buffer("obs_mean", obs_mean.detach().float().reshape(1, -1))
            self.register_buffer("obs_std", obs_std.detach().float().reshape(1, -1))
        else:
            self.register_buffer("obs_mean", torch.zeros(1, policy.expected_actor_obs))
            self.register_buffer("obs_std", torch.ones(1, policy.expected_actor_obs))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.use_normalizer:
            obs = (obs - self.obs_mean) / (self.obs_std + self.normalizer_eps)
        history = obs[:, self.history_start:self.history_end]
        latent = self.encoder(history)
        actor_obs = torch.cat(
            [obs[:, : self.history_start], latent, obs[:, self.history_end:]],
            dim=-1,
        )
        return self.actor(actor_obs)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.unsafe_load(f)


def _infer_dims(state_dict: dict[str, torch.Tensor], encoder_cfg: dict[str, Any]) -> tuple[int, int, int]:
    history_start, history_end = [int(x) for x in encoder_cfg["history_slice"]]
    history_width = history_end - history_start
    latent_dim = int(encoder_cfg["latent_dim"])
    actor_input_dim = int(state_dict["actor.0.weight"].shape[1])
    obs_dim = actor_input_dim + history_width - latent_dim

    critic_history_start, critic_history_end = [int(x) for x in encoder_cfg["critic_history_slice"]]
    critic_history_width = critic_history_end - critic_history_start
    critic_input_dim = int(state_dict["critics.0.0.weight"].shape[1])
    critic_obs_dim = critic_input_dim + critic_history_width - latent_dim

    num_actions = int(state_dict["actor.6.bias"].shape[0])
    return obs_dim, critic_obs_dim, num_actions


def _build_policy(agent_cfg: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> EncoderActorCritic:
    policy_cfg = dict(agent_cfg["policy"])
    policy_cfg.pop("class_name", None)
    # These keys are accepted by runner configs but are not policy constructor
    # inputs in the local EncoderActorCritic path.
    policy_cfg.pop("actor_obs_normalization", None)
    policy_cfg.pop("critic_obs_normalization", None)

    encoder_cfg = dict(agent_cfg["encoder_cfg"])
    for key in (
        "history_slice",
        "critic_history_slice",
        "history_len",
        "history_dim",
        "latent_dim",
        "encoder_hidden_dims",
        "decoder_target_dim",
        "decoder_hidden_dims",
    ):
        policy_cfg[key] = encoder_cfg[key]

    alg_cfg = dict(agent_cfg["algorithm"])
    policy_cfg["num_critic_groups"] = int(alg_cfg.get("num_critic_groups", 2))
    policy_cfg["critic_group_names"] = tuple(alg_cfg.get("critic_group_names", ("goal", "aux")))

    obs_dim, critic_obs_dim, num_actions = _infer_dims(state_dict, encoder_cfg)
    policy = EncoderActorCritic(
        obs_dim,
        critic_obs_dim,
        num_actions,
        **policy_cfg,
    )
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy


def _default_output(checkpoint: Path) -> Path:
    return checkpoint.parent / "exported" / f"{checkpoint.stem}_soccer_kick_v53_deploy.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--agent_cfg",
        type=Path,
        default=None,
        help="Path to params/agent.yaml. Defaults to <checkpoint_dir>/params/agent.yaml.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no_normalizer",
        action="store_true",
        help="Export encoder+actor only. By default the empirical obs normalizer is embedded.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    agent_cfg_path = args.agent_cfg
    if agent_cfg_path is None:
        agent_cfg_path = checkpoint.parent / "params" / "agent.yaml"
    agent_cfg_path = agent_cfg_path.expanduser().resolve()
    output = (args.output or _default_output(checkpoint)).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    agent_cfg = _load_yaml(agent_cfg_path)
    # This is a trusted local training artifact from this workspace.
    loaded = torch.load(checkpoint, map_location=args.device, weights_only=False)
    state_dict = loaded["model_state_dict"]
    policy = _build_policy(agent_cfg, state_dict).to(args.device)

    obs_dim = int(policy.expected_actor_obs)
    obs_mean = obs_std = None
    if not args.no_normalizer:
        obs_norm = loaded.get("obs_norm_state_dict")
        if obs_norm is None:
            raise RuntimeError("Checkpoint is missing obs_norm_state_dict; use --no_normalizer to export raw policy.")
        obs_mean = obs_norm["_mean"].to(args.device)
        obs_std = obs_norm["_std"].to(args.device)

    wrapper = SoccerKickV53DeployWrapper(policy, obs_mean=obs_mean, obs_std=obs_std).to(args.device)
    wrapper.eval()

    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32, device=args.device)
    with torch.no_grad():
        torch_actions = wrapper(dummy_obs).detach().cpu()

    torch.onnx.export(
        wrapper,
        (dummy_obs,),
        str(output),
        input_names=["obs"],
        output_names=["actions"],
        opset_version=args.opset,
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)

    metadata = {
        "checkpoint": str(checkpoint),
        "agent_cfg": str(agent_cfg_path),
        "onnx": str(output),
        "input": {"name": "obs", "shape": ["batch", obs_dim], "dtype": "float32"},
        "output": {"name": "actions", "shape": ["batch", int(policy.actor[-1].out_features)], "dtype": "float32"},
        "normalizer_embedded": not args.no_normalizer,
        "normalizer_eps": 1e-2,
        "history_slice": list(policy.history_slice),
        "history_len": int(policy.history_len),
        "history_dim": int(policy.history_dim),
        "latent_dim": int(policy.latent_dim),
        "actor_input_dim_after_encoding": int(policy.actor[0].in_features),
        "obs_layout": OBS_LAYOUT,
        "ball_history_slot": ["ball_pos_b_x", "ball_pos_b_y", "ball_mask", "last_seen_dt", "ball_speed_b"],
        "torch_zero_obs_action_sample": torch_actions.flatten().tolist(),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[OK] exported ONNX: {output}")
    print(f"[OK] wrote metadata: {metadata_path}")
    print(f"[OK] embedded normalizer: {not args.no_normalizer}")


if __name__ == "__main__":
    main()

# Copyright (c) 2025-2026, The Booster Soccer Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Multi-policy evaluation script for V4 Stage 4 composition validation.

Loads up to three role-specialized policies (kicker / receiver / defender)
into a shared scene and rolls out a configurable number of episodes,
collecting quantitative composition metrics (pass landing rate, trap success,
combined success, kicker / receiver fall rates, episode length, etc.).

When a checkpoint argument is omitted, that role is driven by a *random*
policy. This lets us validate the plumbing before the Stage 1/2 checkpoints
finish training.

Usage:
  python scripts/rsl_rl/play_multi.py \\
    --task Booster-Soccer-Composition-v0 \\
    --checkpoint_kicker logs/rsl_rl/soccer_kick_skill_amp/<run>/model_7999.pt \\
    --checkpoint_receiver logs/rsl_rl/soccer_trap_skill_amp/<run>/model_7999.pt \\
    --num_envs 64 --num_episodes 200 --headless

Outputs:
  * JSON summary printed to stdout (parseable by external tooling).
  * Same summary written to ``--results_path`` if provided.

See ``docs/soccer_amp_skill_library_design.md`` §7 Stage 4 + §8 Deployment.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# argparse — must run before AppLauncher.launch() so --headless etc. resolve.
parser = argparse.ArgumentParser(
    description="V4 Stage 4 multi-policy composition validation runner."
)
parser.add_argument(
    "--task",
    type=str,
    default="Booster-Soccer-Composition-v0",
    help="Gym task id (must expose ``policy_kicker`` + ``policy_receiver`` obs groups).",
)
parser.add_argument(
    "--checkpoint_kicker",
    type=str,
    default=None,
    help="Path to the kicker policy checkpoint (.pt). Random policy if omitted.",
)
parser.add_argument(
    "--checkpoint_receiver",
    type=str,
    default=None,
    help="Path to the receiver policy checkpoint (.pt). Random policy if omitted.",
)
parser.add_argument(
    "--checkpoint_defender",
    type=str,
    default=None,
    help="Path to the defender policy checkpoint (.pt). Currently unused (Stage 4 pass+trap only).",
)
parser.add_argument(
    "--num_envs", type=int, default=64, help="Number of parallel environments."
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=200,
    help="Total number of episodes (across all envs) to evaluate.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=4000,
    help="Hard limit on outer-loop sim steps. Safety bound for the random-policy smoke path.",
)
parser.add_argument(
    "--results_path",
    type=str,
    default=None,
    help="If set, the JSON summary is also written to this path.",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Seed for env + numpy/torch."
)
# Standard AppLauncher args (--headless, --device, ...).
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Clear out hydra leftovers so downstream tooling parses cleanly.
sys.argv = [sys.argv[0]] + hydra_args

# Launch Omniverse first (required before importing any isaac module).
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import time

import gymnasium as gym
import torch
from tensordict import TensorDict

import isaaclab_tasks  # noqa: F401 — registers stock Isaac Lab tasks.
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import booster_rl_tasks.tasks  # noqa: F401 — registers Booster tasks.

# rsl_rl modules are heavy — defer their import until needed.
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.normalizer import EmpiricalNormalization

try:
    from rsl_rl.modules.multi_critic_actor_critic import MultiCriticActorCritic
except ImportError:  # pragma: no cover — vendored rsl_rl must include this.
    MultiCriticActorCritic = None  # type: ignore[assignment]


# =========================================================================
# Policy wrapper — loads a single checkpoint + normalizer for a role.
# =========================================================================


class _PolicyWrapper:
    """Holds one policy + its empirical normalizer; produces actions.

    When ``checkpoint_path`` is None, ``act`` returns zero-mean Gaussian
    actions of shape ``(num_envs, 22)`` (smoke / plumbing-validation mode).
    """

    def __init__(
        self,
        *,
        role_name: str,
        checkpoint_path: str | None,
        num_obs: int,
        num_actions: int,
        device: torch.device,
        policy_cfg_dict: dict | None = None,
    ) -> None:
        self.role_name = role_name
        self.num_obs = int(num_obs)
        self.num_actions = int(num_actions)
        self.device = device
        self.is_random = checkpoint_path is None
        self.checkpoint_path = checkpoint_path
        self._policy_net: torch.nn.Module | None = None
        self._normalizer: torch.nn.Module | None = None

        if self.is_random:
            print(
                f"[play_multi] role={role_name}: no checkpoint provided -> random policy"
            )
            return

        # Build the actor-critic with the configured architecture, then
        # restore weights + the empirical-norm running stats.
        cfg = dict(policy_cfg_dict or {})
        actor_hidden = cfg.pop("actor_hidden_dims", [512, 256, 128])
        critic_hidden = cfg.pop("critic_hidden_dims", [512, 256, 128])
        activation = cfg.pop("activation", "elu")
        init_noise_std = cfg.pop("init_noise_std", 1.0)
        noise_std_type = cfg.pop("noise_std_type", "log")
        num_critic_groups = cfg.pop("num_critic_groups", 2)
        critic_group_names = cfg.pop("critic_group_names", ("goal", "aux"))

        if MultiCriticActorCritic is not None:
            self._policy_net = MultiCriticActorCritic(
                num_actor_obs=self.num_obs,
                num_critic_obs=self.num_obs,
                num_actions=self.num_actions,
                actor_hidden_dims=actor_hidden,
                critic_hidden_dims=critic_hidden,
                activation=activation,
                init_noise_std=init_noise_std,
                noise_std_type=noise_std_type,
                num_critic_groups=num_critic_groups,
                critic_group_names=tuple(critic_group_names),
            ).to(device)
        else:
            self._policy_net = ActorCritic(
                num_actor_obs=self.num_obs,
                num_critic_obs=self.num_obs,
                num_actions=self.num_actions,
                actor_hidden_dims=actor_hidden,
                critic_hidden_dims=critic_hidden,
                activation=activation,
                init_noise_std=init_noise_std,
                noise_std_type=noise_std_type,
            ).to(device)

        # Load the checkpoint weights (training-side state dict). If keys
        # don't match (e.g. the user supplied a single-critic ActorCritic
        # checkpoint), fall back to ``strict=False`` and report the diff.
        # Filter out critic.* keys — for inference we only need the actor,
        # and the training-time critic obs dim (with privileged obs) usually
        # differs from the composition env's critic dim, which would cause
        # a shape mismatch even with strict=False.
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = loaded.get("model_state_dict", loaded)
        sd_actor_only = {
            k: v for k, v in sd.items()
            if not (k.startswith("critic") or k.startswith("critics"))
        }
        # ActorCritic.load_state_dict returns a bool (not the standard tuple),
        # so we call the underlying nn.Module method directly to access the
        # missing/unexpected key lists for debugging.
        n_critic_skipped = len(sd) - len(sd_actor_only)
        try:
            ret = torch.nn.Module.load_state_dict(
                self._policy_net, sd_actor_only, strict=False
            )
            missing = list(getattr(ret, "missing_keys", []))
            unexpected = list(getattr(ret, "unexpected_keys", []))
        except TypeError:
            # Older torch returns (missing, unexpected) tuple directly.
            missing, unexpected = self._policy_net.load_state_dict(
                sd_actor_only, strict=False
            )
        if missing or unexpected or n_critic_skipped:
            print(
                f"[play_multi] role={role_name}: load_state_dict (actor-only, non-strict) "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"critic_keys_skipped={n_critic_skipped}"
            )
        self._policy_net.eval()

        # Empirical normalizer: rebuild and restore running stats.
        self._normalizer = EmpiricalNormalization(shape=[self.num_obs], until=1.0e8).to(
            device
        )
        if "obs_norm_state_dict" in loaded:
            try:
                self._normalizer.load_state_dict(loaded["obs_norm_state_dict"])
            except Exception as exc:  # pragma: no cover — defensive
                print(
                    f"[play_multi] role={role_name}: obs_norm load failed ({exc}); "
                    "falling back to identity."
                )
                self._normalizer = torch.nn.Identity().to(device)
        else:
            print(
                f"[play_multi] role={role_name}: checkpoint has no obs_norm_state_dict; "
                "using identity (raw obs)."
            )
            self._normalizer = torch.nn.Identity().to(device)
        if isinstance(self._normalizer, EmpiricalNormalization):
            self._normalizer.eval()

        print(
            f"[play_multi] role={role_name}: loaded checkpoint {checkpoint_path} "
            f"(num_obs={self.num_obs}, num_actions={self.num_actions})"
        )

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        if self.is_random:
            # Bounded uniform noise — keeps actuators in a reasonable range.
            return (
                torch.rand(obs.shape[0], self.num_actions, device=self.device) * 2.0
                - 1.0
            )
        assert self._policy_net is not None
        normalized = (
            self._normalizer(obs) if self._normalizer is not None else obs
        )
        # Deterministic mean action (no exploration noise at eval time).
        return self._policy_net.act_inference(normalized)


# =========================================================================
# Episode statistics aggregator
# =========================================================================


class _EpisodeStats:
    """Per-env running episode counters; flushed when a done flag fires."""

    def __init__(self, num_envs: int, device: torch.device) -> None:
        self.num_envs = num_envs
        self.device = device
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.kicks = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.kick_success_prev = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.pass_landed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.trap_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.kicker_fall = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.receiver_fall = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Completed-episode accumulators (filled at done events).
        self.completed_episodes: list[dict] = []

    def update_step(
        self,
        *,
        kick_success_awarded: torch.Tensor,
        pass_landed_now: torch.Tensor,
        trap_success_now: torch.Tensor,
        kicker_fall_now: torch.Tensor,
        receiver_fall_now: torch.Tensor,
    ) -> None:
        self.step_count += 1
        # New kicks: rising edge of cmd.kick_success_awarded.
        new_kicks = kick_success_awarded & (~self.kick_success_prev)
        self.kicks += new_kicks.long()
        self.kick_success_prev = kick_success_awarded.clone()
        # Latch episode flags.
        self.pass_landed |= pass_landed_now
        self.trap_success |= trap_success_now
        self.kicker_fall |= kicker_fall_now
        self.receiver_fall |= receiver_fall_now

    def flush_done(self, done_mask: torch.Tensor) -> int:
        if not done_mask.any():
            return 0
        idxs = done_mask.nonzero(as_tuple=False).flatten().tolist()
        for i in idxs:
            self.completed_episodes.append(
                dict(
                    steps=int(self.step_count[i].item()),
                    kicks=int(self.kicks[i].item()),
                    pass_landed=bool(self.pass_landed[i].item()),
                    trap_success=bool(self.trap_success[i].item()),
                    kicker_fall=bool(self.kicker_fall[i].item()),
                    receiver_fall=bool(self.receiver_fall[i].item()),
                )
            )
        # Reset per-env accumulators for envs that just ended.
        self.step_count[done_mask] = 0
        self.kicks[done_mask] = 0
        self.kick_success_prev[done_mask] = False
        self.pass_landed[done_mask] = False
        self.trap_success[done_mask] = False
        self.kicker_fall[done_mask] = False
        self.receiver_fall[done_mask] = False
        return len(idxs)


# =========================================================================
# Metric reducer
# =========================================================================


def _reduce(stats: _EpisodeStats, *, decimation: int, dt: float) -> dict:
    eps = stats.completed_episodes
    n = len(eps)
    if n == 0:
        return {
            "num_episodes": 0,
            "note": "no episodes completed — try a longer --max_steps",
        }
    pass_landing = sum(e["pass_landed"] for e in eps) / n
    trap_success = sum(e["trap_success"] for e in eps) / n
    combined = sum(e["pass_landed"] and e["trap_success"] for e in eps) / n
    kicker_fall = sum(e["kicker_fall"] for e in eps) / n
    receiver_fall = sum(e["receiver_fall"] for e in eps) / n
    avg_kicks = sum(e["kicks"] for e in eps) / n
    avg_steps = sum(e["steps"] for e in eps) / n
    avg_seconds = avg_steps * decimation * dt
    return {
        "num_episodes": n,
        "pass_landing_rate": float(pass_landing),
        "trap_success_rate": float(trap_success),
        "combined_success_rate": float(combined),
        "kicker_fall_rate": float(kicker_fall),
        "receiver_fall_rate": float(receiver_fall),
        "avg_kicks_per_episode": float(avg_kicks),
        "avg_episode_steps": float(avg_steps),
        "avg_episode_seconds": float(avg_seconds),
    }


# =========================================================================
# Main eval loop
# =========================================================================


def main() -> None:
    torch.manual_seed(args_cli.seed)

    # ----- Build env cfg + env -----------------------------------------
    env_cfg: ManagerBasedRLEnvCfg = load_cfg_from_registry(
        args_cli.task, "env_cfg_entry_point"
    )
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Build the policy cfg dict (used to instantiate the actor-critic for
    # the kicker / receiver — both use the same arch as Stage 1/2).
    try:
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        policy_cfg_dict = agent_cfg.policy.to_dict() if hasattr(agent_cfg.policy, "to_dict") else dict(vars(agent_cfg.policy))
        # Drop the class name — we always build MultiCriticActorCritic.
        policy_cfg_dict.pop("class_name", None)
        # Defaults aligned with the Stage 1/2 BaseMultiCriticAMPAgentCfg.
        policy_cfg_dict.setdefault("num_critic_groups", 2)
        policy_cfg_dict.setdefault("critic_group_names", ("goal", "aux"))
    except Exception as exc:  # pragma: no cover
        print(f"[play_multi] WARN: could not load agent cfg ({exc}); using arch defaults")
        policy_cfg_dict = {
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "init_noise_std": 1.0,
            "noise_std_type": "log",
            "num_critic_groups": 2,
            "critic_group_names": ("goal", "aux"),
        }

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    # ----- Determine per-agent obs dims (reset once + introspect) -----
    obs_td, _ = env.reset()
    assert isinstance(obs_td, TensorDict), f"unexpected obs type: {type(obs_td)}"
    if "policy_kicker" not in obs_td.keys() or "policy_receiver" not in obs_td.keys():
        raise RuntimeError(
            "Composition env must expose ``policy_kicker`` + ``policy_receiver`` "
            f"observation groups; got keys={list(obs_td.keys())}"
        )
    kicker_obs_dim = int(obs_td["policy_kicker"].shape[-1])
    receiver_obs_dim = int(obs_td["policy_receiver"].shape[-1])
    print(
        f"[play_multi] obs dims — kicker={kicker_obs_dim}, receiver={receiver_obs_dim}"
    )

    # Each agent emits 22 joint actions; the env concatenates them in the
    # order ``joint_pos`` (kicker), ``joint_pos_receiver`` (receiver).
    num_joints_per_agent = 22
    total_actions = num_envs * 0  # silence unused-var lint
    total_actions = int(env.unwrapped.action_manager.total_action_dim)
    assert total_actions == 2 * num_joints_per_agent, (
        f"Expected 44-dim action (22 kicker + 22 receiver); got {total_actions}"
    )

    # ----- Build policy wrappers --------------------------------------
    kicker_policy = _PolicyWrapper(
        role_name="kicker",
        checkpoint_path=args_cli.checkpoint_kicker,
        num_obs=kicker_obs_dim,
        num_actions=num_joints_per_agent,
        device=device,
        policy_cfg_dict=policy_cfg_dict,
    )
    receiver_policy = _PolicyWrapper(
        role_name="receiver",
        checkpoint_path=args_cli.checkpoint_receiver,
        num_obs=receiver_obs_dim,
        num_actions=num_joints_per_agent,
        device=device,
        policy_cfg_dict=policy_cfg_dict,
    )

    # ----- Command-term handles for metric extraction -----------------
    cm = env.unwrapped.command_manager
    kick_cmd = cm.get_term("soccer_kick")
    trap_cmd = cm.get_term("soccer_trap")

    # ----- Termination handles for fall-rate metrics ------------------
    tm = env.unwrapped.termination_manager
    term_names = set(tm.active_terms)

    def _term_done(name: str) -> torch.Tensor:
        if name in term_names:
            try:
                return tm.get_term(name).bool()
            except Exception:
                pass
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    # ----- Stats accumulator ------------------------------------------
    stats = _EpisodeStats(num_envs=num_envs, device=device)

    dt = float(env_cfg.sim.dt)
    decimation = int(env_cfg.decimation)
    pass_landing_radius = float(kick_cmd.cfg.pass_landing_radius)
    pass_speed_lo, pass_speed_hi = (
        float(kick_cmd.cfg.pass_landing_speed_window[0]),
        float(kick_cmd.cfg.pass_landing_speed_window[1]),
    )
    print(
        f"[play_multi] eval thresholds — pass_radius={pass_landing_radius:.2f} m, "
        f"pass_speed=[{pass_speed_lo},{pass_speed_hi}] m/s, "
        f"trap_radius={float(trap_cmd.cfg.trap_success_radius):.2f} m"
    )

    # ----- Roll out ---------------------------------------------------
    step = 0
    t_start = time.time()
    while (
        len(stats.completed_episodes) < args_cli.num_episodes
        and step < args_cli.max_steps
        and simulation_app.is_running()
    ):
        with torch.inference_mode():
            kicker_obs = obs_td["policy_kicker"]
            receiver_obs = obs_td["policy_receiver"]
            kicker_act = kicker_policy.act(kicker_obs)
            receiver_act = receiver_policy.act(receiver_obs)
            actions = torch.cat([kicker_act, receiver_act], dim=-1)

            obs_td, _rew, dones, _extras = env.step(actions)

            # Pass-landing detection (live xy check while ball is in flight).
            ball_xy = kick_cmd.ball_pos_w[:, :2]
            target_xy = kick_cmd.pass_target_pos_w
            ball_speed_xy = torch.linalg.norm(kick_cmd.ball_vel_w[:, :2], dim=-1)
            dist = torch.linalg.norm(ball_xy - target_xy, dim=-1)
            pass_landed_now = (
                (dist < pass_landing_radius)
                & (ball_speed_xy >= pass_speed_lo)
                & (ball_speed_xy <= pass_speed_hi)
            )
            trap_success_now = trap_cmd.trap_success_awarded

            kicker_fall_now = _term_done("kicker_fall_height") | _term_done(
                "kicker_fall_tilt"
            )
            receiver_fall_now = _term_done("receiver_fall_height") | _term_done(
                "receiver_fall_tilt"
            )

            stats.update_step(
                kick_success_awarded=kick_cmd.kick_success_awarded,
                pass_landed_now=pass_landed_now,
                trap_success_now=trap_success_now,
                kicker_fall_now=kicker_fall_now,
                receiver_fall_now=receiver_fall_now,
            )
            n_flushed = stats.flush_done(dones.bool())
            if n_flushed > 0 and (len(stats.completed_episodes) % 25 == 0 or n_flushed > 5):
                print(
                    f"[play_multi] step={step}: completed_episodes="
                    f"{len(stats.completed_episodes)} / {args_cli.num_episodes}"
                )
            step += 1

    elapsed = time.time() - t_start
    summary = _reduce(stats, decimation=decimation, dt=dt)
    summary["task"] = args_cli.task
    summary["num_envs"] = num_envs
    summary["wallclock_seconds"] = elapsed
    summary["steps_simulated"] = step
    summary["checkpoint_kicker"] = args_cli.checkpoint_kicker
    summary["checkpoint_receiver"] = args_cli.checkpoint_receiver
    summary["checkpoint_defender"] = args_cli.checkpoint_defender

    print("=" * 70)
    print("[play_multi] composition validation summary:")
    print(json.dumps(summary, indent=2))
    print("=" * 70)

    if args_cli.results_path:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.results_path)), exist_ok=True)
        with open(args_cli.results_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[play_multi] wrote results to {args_cli.results_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

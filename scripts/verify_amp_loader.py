"""Smoke-test AMPLoader against the freshly-generated 62-col corpus.

Confirms:
  - all 20 clips load without dim mismatch errors
  - observation_dim == 62
  - has_root_vel is True
  - root_lin_vel slice has plausible magnitudes (forward = +x, backward = -x,
    lateral = ±y, pivot = small |v_xy| with non-zero ω_z)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "rsl_rl"))
from rsl_rl.utils import AMPLoader


_AMP_ROOT = os.path.join(_REPO_ROOT, "booster_assets", "motions", "K1", "motion_amp_expert", "omni")
_GROUPS = {
    "forward":  ["walk", "walk2run", "run", "run2walk"],
    "lateral":  [
        "strafe_walk_left", "strafe_walk_right",
        "strafe_walk2run_left", "strafe_walk2run_right",
        "strafe_run_left", "strafe_run_right",
        "strafe_run2walk_left", "strafe_run2walk_right",
    ],
    "backward": ["backward_walk", "backward_walk2run", "backward_run", "backward_run2walk"],
    "pivot":    ["pivot_left_slow", "pivot_left_fast", "pivot_right_slow", "pivot_right_fast"],
}


def _all_files() -> list[str]:
    files = []
    for cat, names in _GROUPS.items():
        for n in names:
            p = os.path.join(_AMP_ROOT, cat, f"{n}.txt")
            if os.path.isfile(p):
                files.append(p)
            else:
                print(f"  ! missing: {p}")
    return files


def main() -> None:
    files = _all_files()
    print(f"Found {len(files)} clips under {_AMP_ROOT}")

    loader = AMPLoader(
        device=torch.device("cpu"),
        time_between_frames=0.02,
        preload_transitions=True,
        num_preload_transitions=20000,
        motion_files=files,
    )
    print(f"\nobservation_dim = {loader.observation_dim}")
    print(f"has_root_vel    = {loader.has_root_vel}")
    assert loader.observation_dim == AMPLoader.OBS_DIM_WITH_ROOT, "expected 62-col"
    assert loader.has_root_vel

    sample, _ = next(loader.feed_forward_generator(1, 4096))
    lin_b = AMPLoader.get_root_lin_vel_batch(sample).numpy()
    ang_b = AMPLoader.get_root_ang_vel_batch(sample).numpy()
    print(f"\nrandom-sample root_lin_vel_b stats (over 4096):")
    print(f"  vx mean={lin_b[:,0].mean():+.3f}  std={lin_b[:,0].std():.3f}  "
          f"min={lin_b[:,0].min():+.3f}  max={lin_b[:,0].max():+.3f}")
    print(f"  vy mean={lin_b[:,1].mean():+.3f}  std={lin_b[:,1].std():.3f}  "
          f"min={lin_b[:,1].min():+.3f}  max={lin_b[:,1].max():+.3f}")
    print(f"random-sample root_ang_vel_b stats:")
    print(f"  wz mean={ang_b[:,2].mean():+.3f}  std={ang_b[:,2].std():.3f}  "
          f"min={ang_b[:,2].min():+.3f}  max={ang_b[:,2].max():+.3f}")

    # Per-clip stats: load one trajectory at a time so we can see each
    # category's signature isolation.
    print("\nper-clip body-frame velocity signatures:")
    print(f"{'category':<10} {'clip':<32} {'vx_mean':>9} {'vy_mean':>9} {'wz_mean':>9}")
    for cat, names in _GROUPS.items():
        for n in names:
            p = os.path.join(_AMP_ROOT, cat, f"{n}.txt")
            if not os.path.isfile(p):
                continue
            single = AMPLoader(
                device=torch.device("cpu"),
                time_between_frames=0.02,
                preload_transitions=False,
                motion_files=[p],
            )
            traj = single.trajectories_full[0]
            vx = AMPLoader.get_root_lin_vel_batch(traj)[:, 0].mean().item()
            vy = AMPLoader.get_root_lin_vel_batch(traj)[:, 1].mean().item()
            wz = AMPLoader.get_root_ang_vel_batch(traj)[:, 2].mean().item()
            print(f"{cat:<10} {n:<32} {vx:>+9.3f} {vy:>+9.3f} {wz:>+9.3f}")


if __name__ == "__main__":
    main()

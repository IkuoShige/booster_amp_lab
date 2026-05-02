"""Convert retargeted K1 motion pkls into AMP txt corpus in one shot.

Combines what ``gmr_data_conversion.py`` (Stage 1: pkl → intermediate root+dof
text) and ``replay_amp_txt.py`` (Stage 2: replay + FK to extract EE positions)
do, but as a batch driver that launches Isaac Sim once and processes every
``*.pkl`` in an input directory.

Output format (56 columns, matches existing ``motion_amp_expert/*.txt``):
    [22 joint_pos (IsaacLab BFS order)]
    [22 joint_vel (IsaacLab BFS order)]
    [12 EE_pos in body frame: left_hand, right_hand, left_foot, right_foot]

Usage::

    python scripts/build_lateral_amp.py \\
      --input_dir=/workspace/motions_k1_lateral \\
      --output_dir=booster_assets/motions/K1/motion_amp_expert/lateral \\
      --motion_weight=0.5
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batch convert K1 retargeted pkls to AMP txt.")
parser.add_argument("--input_dir", type=str, required=True, help="Directory of retargeted pkls.")
parser.add_argument("--output_dir", type=str, required=True, help="Where to write AMP txt files.")
parser.add_argument("--fps", type=float, default=30.0, help="Default fps if not in pkl.")
parser.add_argument("--motion_weight", type=float, default=0.5, help="MotionWeight written into each AMP txt.")
parser.add_argument("--robot", type=str, default="booster_k1", choices=["booster_k1"], help="Robot type.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul, axis_angle_from_quat

from booster_rl_tasks.assets.robots.booster import BOOSTER_K1_CFG

# pkl dof order (CSV / K1_JOINT_NAMES) → IsaacLab BFS articulation order.
# Mirrors ``replay_amp_txt.reorder`` for booster_k1.
_PKL_TO_LAB_K1 = [
    0,   # AAHead_yaw
    2,   # ALeft_Shoulder_Pitch
    6,   # ARight_Shoulder_Pitch
    10,  # Left_Hip_Pitch
    16,  # Right_Hip_Pitch
    1,   # Head_pitch
    3,   # Left_Shoulder_Roll
    7,   # Right_Shoulder_Roll
    11,  # Left_Hip_Roll
    17,  # Right_Hip_Roll
    4,   # Left_Elbow_Pitch
    8,   # Right_Elbow_Pitch
    12,  # Left_Hip_Yaw
    18,  # Right_Hip_Yaw
    5,   # Left_Elbow_Yaw
    9,   # Right_Elbow_Yaw
    13,  # Left_Knee_Pitch
    19,  # Right_Knee_Pitch
    14,  # Left_Ankle_Pitch
    20,  # Right_Ankle_Pitch
    15,  # Left_Ankle_Roll
    21,  # Right_Ankle_Roll
]


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(1.0, 1.0, 1.0)),
    )
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _finite_diff_root_lin_vel(root_pos: np.ndarray, dt: float) -> np.ndarray:
    return np.diff(root_pos, axis=0) / dt


def _finite_diff_root_ang_vel(root_rot_wxyz: torch.Tensor, dt: float) -> torch.Tensor:
    """Root angular velocity from quaternion finite differences (axis-angle / dt)."""
    q1_conj = quat_conjugate(root_rot_wxyz[:-1])
    dq = quat_mul(q1_conj, root_rot_wxyz[1:])
    axis_angle = axis_angle_from_quat(dq)
    return axis_angle / dt


def _finite_diff_dof_vel(dof_pos: np.ndarray, dt: float) -> np.ndarray:
    return np.diff(dof_pos, axis=0) / dt


def _reorder_pkl_to_lab(arr: np.ndarray) -> np.ndarray:
    return arr[:, _PKL_TO_LAB_K1]


def _write_amp_txt(out_path: str, frames: np.ndarray, fps: float, motion_weight: float) -> None:
    n = frames.shape[0]
    with open(out_path, "w") as f:
        f.write("{\n")
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0 / fps:.4f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write(f'"MotionWeight": {motion_weight:.3f},\n\n')
        f.write('"Frames":\n[\n')
        for i, row in enumerate(frames):
            row_str = ", ".join(f"{v:.6f}" for v in row)
            term = "" if i == n - 1 else ","
            f.write(f"  [{row_str}]{term}\n")
        f.write("]\n}\n")


def _build_one(
    pkl_path: str,
    out_path: str,
    scene: InteractiveScene,
    sim: SimulationContext,
    motion_weight: float,
    default_fps: float,
) -> None:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    fps = float(d.get("fps", default_fps))
    dt = 1.0 / fps

    root_pos = np.asarray(d["root_pos"], dtype=np.float64)             # (N, 3)
    root_rot_xyzw = np.asarray(d["root_rot"], dtype=np.float64)        # (N, 4) xyzw
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]                     # → wxyz
    dof_pos = np.asarray(d["dof_pos"], dtype=np.float64)               # (N, 22) pkl order

    n = root_pos.shape[0]
    if dof_pos.shape[0] != n or root_rot_wxyz.shape[0] != n:
        raise RuntimeError(f"{pkl_path}: row count mismatch among root_pos/root_rot/dof_pos")

    # Drop last frame so finite differences match (yields N-1 frames).
    root_lin_vel_w = _finite_diff_root_lin_vel(root_pos, dt)           # (N-1, 3)
    root_rot_wxyz_t = torch.tensor(root_rot_wxyz, dtype=torch.float32, device=scene.device)
    root_ang_vel_w = _finite_diff_root_ang_vel(root_rot_wxyz_t, dt).cpu().numpy()  # (N-1, 3)
    dof_vel_pkl = _finite_diff_dof_vel(dof_pos, dt)                    # (N-1, 22) pkl order

    # Trim everything to (N-1) frames so velocities align with positions.
    n_use = n - 1
    root_pos = root_pos[:n_use]
    root_rot_wxyz = root_rot_wxyz[:n_use]
    dof_pos_pkl = dof_pos[:n_use]

    # Reorder to IsaacLab BFS order.
    dof_pos_lab = _reorder_pkl_to_lab(dof_pos_pkl)
    dof_vel_lab = _reorder_pkl_to_lab(dof_vel_pkl)

    # Resolve EE body indices.
    robot: Articulation = scene["robot"]
    hand_ids, _ = robot.find_bodies(name_keys=["left_hand_link", "right_hand_link"], preserve_order=True)
    foot_ids, _ = robot.find_bodies(name_keys=["left_foot_link", "right_foot_link"], preserve_order=True)
    left_hand_local = torch.tensor([0.0, 0.2, 0.0], device=scene.device).repeat((scene.num_envs, 1))
    right_hand_local = torch.tensor([0.0, -0.2, 0.0], device=scene.device).repeat((scene.num_envs, 1))

    ee_frames = np.zeros((n_use, 12), dtype=np.float32)

    # Replay each frame in sim, read EE pose in body frame.
    dof_pos_lab_t = torch.tensor(dof_pos_lab, dtype=torch.float32, device=scene.device)
    dof_vel_lab_t = torch.tensor(dof_vel_lab, dtype=torch.float32, device=scene.device)
    root_state = torch.zeros((scene.num_envs, 13), dtype=torch.float32, device=scene.device)

    for i in range(n_use):
        # set joint state
        robot.write_joint_position_to_sim(dof_pos_lab_t[i:i + 1].repeat(scene.num_envs, 1))
        robot.write_joint_velocity_to_sim(dof_vel_lab_t[i:i + 1].repeat(scene.num_envs, 1))

        # set root state (xyz, qwxyz, lin_vel_w, ang_vel_w)
        root_state[:, 0:3] = torch.tensor(root_pos[i], dtype=torch.float32, device=scene.device)
        # nudge upward by 5cm to match replay_amp_txt convention (avoid ground penetration in viz)
        root_state[:, 2] += 0.05
        root_state[:, 3:7] = torch.tensor(root_rot_wxyz[i], dtype=torch.float32, device=scene.device)
        root_state[:, 7:10] = torch.tensor(root_lin_vel_w[i], dtype=torch.float32, device=scene.device)
        root_state[:, 10:13] = torch.tensor(root_ang_vel_w[i], dtype=torch.float32, device=scene.device)
        robot.write_root_state_to_sim(root_state)

        scene.write_data_to_sim()
        sim.render()
        scene.update(sim.get_physics_dt())

        # hand: body pose * local offset, expressed in robot body frame
        left_hand_w = robot.data.body_state_w[:, hand_ids[0], :3] - robot.data.root_state_w[:, 0:3] + \
            quat_apply(robot.data.body_state_w[:, hand_ids[0], 3:7], left_hand_local)
        right_hand_w = robot.data.body_state_w[:, hand_ids[1], :3] - robot.data.root_state_w[:, 0:3] + \
            quat_apply(robot.data.body_state_w[:, hand_ids[1], 3:7], right_hand_local)
        left_hand_b = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), left_hand_w)
        right_hand_b = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), right_hand_w)
        # foot: body pose in robot body frame
        left_foot_w = robot.data.body_state_w[:, foot_ids[0], :3] - robot.data.root_state_w[:, 0:3]
        right_foot_w = robot.data.body_state_w[:, foot_ids[1], :3] - robot.data.root_state_w[:, 0:3]
        left_foot_b = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), left_foot_w)
        right_foot_b = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), right_foot_w)

        ee = torch.cat([left_hand_b[0], right_hand_b[0], left_foot_b[0], right_foot_b[0]]).detach().cpu().numpy()
        ee_frames[i] = ee

    frames = np.concatenate([dof_pos_lab.astype(np.float32), dof_vel_lab.astype(np.float32), ee_frames], axis=1)
    assert frames.shape[1] == 56, f"unexpected width {frames.shape[1]} for {pkl_path}"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _write_amp_txt(out_path, frames, fps=fps, motion_weight=motion_weight)
    print(f"  ✓ {os.path.basename(out_path)}: {frames.shape[0]} frames @ {fps}fps")


def main() -> None:
    pkls = sorted(glob.glob(os.path.join(args_cli.input_dir, "*.pkl")))
    if not pkls:
        print(f"[build_lateral_amp] no pkls found under {args_cli.input_dir}")
        return
    print(f"[build_lateral_amp] {len(pkls)} pkls → {args_cli.output_dir}")

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / args_cli.fps, device=args_cli.device or "cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.5, 2.5, 2.5), target=(0.0, 0.0, 0.5))

    scene_cfg = _SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[build_lateral_amp] sim ready")

    os.makedirs(args_cli.output_dir, exist_ok=True)
    for pkl in pkls:
        name = os.path.splitext(os.path.basename(pkl))[0]
        out = os.path.join(args_cli.output_dir, f"{name}.txt")
        try:
            _build_one(pkl, out, scene, sim, args_cli.motion_weight, args_cli.fps)
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            raise
    print("[build_lateral_amp] done.")


if __name__ == "__main__":
    main()
    simulation_app.close()

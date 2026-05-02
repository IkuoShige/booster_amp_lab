"""Diagnose whether lateral source pkl root_pos actually translates laterally.

For each strafe_*.pkl:
  - read root_pos (T, 3) in world frame
  - read root_rot (T, 4) xyzw
  - compute initial yaw -> world->body rotation
  - project root displacement into body frame
  - report: total displacement (forward, lateral), mean lateral velocity,
    lateral oscillation amplitude (should be small for clean strafe),
    whether lateral motion is monotonic (good) or net-zero (= COM not shifting)
"""
import pickle
import numpy as np
import pathlib

_PKL_DIR = pathlib.Path("/workspace/motions_k1_lateral")


def quat_to_yaw_xyzw(q):
    # q: (..., 4) xyzw
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def diagnose(pkl_path: pathlib.Path):
    with pkl_path.open("rb") as f:
        d = pickle.load(f)
    fps = int(d["fps"])
    root_pos = np.asarray(d["root_pos"])  # (T, 3)
    root_rot = np.asarray(d["root_rot"])  # (T, 4) xyzw
    T = root_pos.shape[0]
    duration = T / fps

    yaw0 = quat_to_yaw_xyzw(root_rot[0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)

    rel = root_pos - root_pos[0]
    body_x = c * rel[:, 0] - s * rel[:, 1]
    body_y = s * rel[:, 0] + c * rel[:, 1]
    body_z = rel[:, 2]

    fwd_total = body_x[-1]
    lat_total = body_y[-1]

    fwd_speed = fwd_total / duration
    lat_speed = lat_total / duration

    body_y_min = body_y.min()
    body_y_max = body_y.max()
    body_x_min = body_x.min()
    body_x_max = body_x.max()
    body_z_min = body_z.min()
    body_z_max = body_z.max()

    abs_lat = np.abs(np.diff(body_y)).sum()
    abs_fwd = np.abs(np.diff(body_x)).sum()
    monotonicity = abs(lat_total) / max(abs_lat, 1e-6)

    return dict(
        name=pkl_path.stem,
        T=T,
        fps=fps,
        dur=duration,
        fwd_total=fwd_total,
        lat_total=lat_total,
        fwd_speed=fwd_speed,
        lat_speed=lat_speed,
        body_y_range=(body_y_min, body_y_max),
        body_x_range=(body_x_min, body_x_max),
        body_z_range=(body_z_min, body_z_max),
        path_len_lat=abs_lat,
        path_len_fwd=abs_fwd,
        lat_monotonicity=monotonicity,
        body_y_traj=body_y,
        body_x_traj=body_x,
    )


def fmt(v):
    return f"{v:+7.3f}"


def main():
    pkls = sorted(_PKL_DIR.glob("strafe_*.pkl"))
    if not pkls:
        print(f"no pkls found in {_PKL_DIR}")
        return

    print(f"\nDiagnose {len(pkls)} lateral pkls (body frame, body_x=forward, body_y=left)\n")
    print(
        f"{'name':<28} {'dur':>5} | {'lat_tot':>8} {'fwd_tot':>8} | "
        f"{'lat_v':>7} {'fwd_v':>7} | {'lat_amp':>7} {'fwd_amp':>7} | {'mono':>5}"
    )
    print("-" * 110)

    all_results = []
    for p in pkls:
        r = diagnose(p)
        all_results.append(r)
        lat_amp = r["body_y_range"][1] - r["body_y_range"][0]
        fwd_amp = r["body_x_range"][1] - r["body_x_range"][0]
        print(
            f"{r['name']:<28} {r['dur']:5.2f} | {fmt(r['lat_total'])} {fmt(r['fwd_total'])} | "
            f"{fmt(r['lat_speed'])} {fmt(r['fwd_speed'])} | "
            f"{lat_amp:7.3f} {fwd_amp:7.3f} | {r['lat_monotonicity']:5.2f}"
        )

    print()
    print("Legend:")
    print("  body_x = forward (initial body +X)   body_y = left (initial body +Y)")
    print("  lat_v  = mean lateral velocity (m/s in body frame)")
    print("  lat_amp = body_y peak-to-peak range (sanity: ~ |lat_total| if no overshoot)")
    print("  mono   = |lat_total| / sum(|d body_y|).  1.0 = perfectly monotonic (no back-and-forth)")
    print("           low = sideways oscillation without net displacement")
    print()

    print("Per-clip body_y trajectory (downsampled, in cm):")
    for r in all_results:
        ts = np.linspace(0, r["dur"], 9)
        idxs = np.linspace(0, r["T"] - 1, 9).astype(int)
        ys = r["body_y_traj"][idxs] * 100.0
        s = " ".join(f"{y:+5.1f}" for y in ys)
        print(f"  {r['name']:<28} y(cm)={s}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Extract the four feature families (experiment plan §4) from a LeRobotDataset, per episode.

Maps the plan's metrics onto the end-effector pose trajectory stored in observation.state:
    speed       : completion time, mean/peak end-effector speed, mean angular speed
    smoothness  : acceleration, JERK (3rd derivative of position), and their SDs,
                  plus log-dimensionless-jerk (a standard smoothness score)
    accuracy    : end-effector position / orientation error vs a reference trajectory
                  (only if --ref_repo_id is given; plan §3 X_ref)
    error_rate  : collisions / failures (only if --labels_csv is given; plan §3.x)

The end-effector pose [x,y,z,yaw,pitch,roll] is read from the state feature by the column
names the converter wrote (ee_x..ee_roll). Derivatives use the dataset's `timestamp` column
(falls back to 1/fps). Output is one CSV row per episode for later t-tests / ANOVA.

USAGE (run from the inner lerobot-main project dir):
    uv run --extra training python experiments/camera_ablation/feature_extraction.py \
        --repo_id local/cond_b_camera --condition B --out outputs/features_B.csv
"""

import argparse
from collections import defaultdict

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

EE_NAMES = ["ee_x", "ee_y", "ee_z", "ee_yaw", "ee_pitch", "ee_roll"]


def _ee_indices(state_names: list[str]) -> list[int]:
    missing = [n for n in EE_NAMES if n not in state_names]
    if missing:
        raise SystemExit(
            f"state is missing end-effector columns {missing}. "
            "Convert with --dh_config so the FK pose is included in observation.state."
        )
    return [state_names.index(n) for n in EE_NAMES]


def _deriv(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Time derivative along axis 0 using non-uniform central differences."""
    return np.gradient(x, t, axis=0, edge_order=2)


def smoothness_speed_features(pose: np.ndarray, t: np.ndarray) -> dict:
    """pose (T,6) = [x,y,z,yaw,pitch,roll]; t (T,) seconds. Returns scalar features."""
    pos = pose[:, :3]
    ori = np.unwrap(pose[:, 3:6], axis=0)  # unwrap euler angles before differentiating

    vel = _deriv(pos, t)                  # (T,3) linear velocity
    speed = np.linalg.norm(vel, axis=1)   # (T,)
    acc = _deriv(vel, t)                   # (T,3)
    jerk = _deriv(acc, t)                  # (T,3) 3rd derivative of position
    ang_vel = _deriv(ori, t)              # (T,3)

    acc_mag = np.linalg.norm(acc, axis=1)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    ang_speed = np.linalg.norm(ang_vel, axis=1)

    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    path_len = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    peak_speed = float(speed.max()) if len(speed) else 0.0

    # Log dimensionless jerk (Balasubramanian et al.): larger (less negative) = smoother.
    ldlj = np.nan
    if duration > 0 and peak_speed > 1e-9:
        _trap = getattr(np, "trapezoid", np.trapz)  # numpy>=2.0 renamed trapz->trapezoid
        jerk_sq_integral = float(_trap(jerk_mag ** 2, t))
        if jerk_sq_integral > 0:
            ldlj = -np.log((duration ** 3 / peak_speed ** 2) * jerk_sq_integral)

    return {
        # speed
        "completion_time_s": duration,
        "path_length_m": path_len,
        "mean_speed": float(speed.mean()),
        "peak_speed": peak_speed,
        "mean_angular_speed": float(ang_speed.mean()),
        # smoothness
        "mean_acc": float(acc_mag.mean()),
        "acc_sd": float(acc_mag.std(ddof=1)) if len(acc_mag) > 1 else 0.0,
        "mean_jerk": float(jerk_mag.mean()),
        "jerk_sd": float(jerk_mag.std(ddof=1)) if len(jerk_mag) > 1 else 0.0,
        "ang_speed_sd": float(ang_speed.std(ddof=1)) if len(ang_speed) > 1 else 0.0,
        "log_dimensionless_jerk": float(ldlj),
    }


def _resample_traj(pose: np.ndarray, n: int) -> np.ndarray:
    """Resample a (T,6) trajectory to (n,6) along normalized time [0,1] (linear interp)."""
    src = np.linspace(0.0, 1.0, len(pose))
    dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, pose[:, d]) for d in range(pose.shape[1])], axis=1)


def _ypr_to_quat(ypr: np.ndarray) -> np.ndarray:
    """ZYX intrinsic Euler [yaw,pitch,roll] -> quaternion (w,x,y,z). Input (...,3)."""
    yaw, pitch, roll = ypr[..., 0], ypr[..., 1], ypr[..., 2]
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    return np.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], axis=-1)


def build_reference(ref_episodes: dict, n: int) -> np.ndarray:
    """Average the expert demonstrations (time-normalized) into one reference (n,6)."""
    resampled = [_resample_traj(pose, n) for pose, _ in ref_episodes.values()]
    return np.mean(np.stack(resampled), axis=0)


def accuracy_features(pose: np.ndarray, ref_pose: np.ndarray, n: int) -> dict:
    """Position/orientation error vs the expert reference, after time-normalized alignment.

    Both trajectories are resampled to n points over normalized time [0,1] so demos of
    different durations are compared point-for-point (plan §3: X_ref / q_ref).
    """
    p = _resample_traj(pose, n)
    r = ref_pose  # already (n,6)
    pos_err = np.linalg.norm(p[:, :3] - r[:, :3], axis=1)
    # Orientation error = 2 * arccos(|q . q_ref|)  (geodesic angle between quaternions)
    qp, qr = _ypr_to_quat(p[:, 3:6]), _ypr_to_quat(r[:, 3:6])
    dot = np.clip(np.abs(np.sum(qp * qr, axis=1)), -1.0, 1.0)
    ori_err = 2.0 * np.arccos(dot)
    return {
        "mean_position_error_m": float(pos_err.mean()),
        "max_position_error_m": float(pos_err.max()),
        "rmse_position_m": float(np.sqrt((pos_err ** 2).mean())),
        "mean_orientation_error_rad": float(ori_err.mean()),
    }


def load_episodes(repo_id: str, root: str | None, video_backend: str) -> tuple[dict, list[str], float]:
    """Return {episode_index: (pose (T,6), t (T,))}, state_names, fps."""
    ds = LeRobotDataset(repo_id, root=root, video_backend=video_backend)
    state_names = ds.meta.features["observation.state"]["names"]
    ee_idx = _ee_indices(state_names)
    fps = ds.meta.fps

    poses: dict[int, list] = defaultdict(list)
    times: dict[int, list] = defaultdict(list)
    for i in range(len(ds)):
        item = ds[i]
        ep = int(item["episode_index"])
        state = np.asarray(item["observation.state"], dtype=float)
        poses[ep].append(state[ee_idx])
        times[ep].append(float(item["timestamp"]))

    episodes = {}
    for ep in poses:
        pose = np.stack(poses[ep])
        t = np.asarray(times[ep])
        if len(t) < 4:
            print(f"  skip episode {ep}: only {len(t)} frames (need >=4 for jerk)")
            continue
        episodes[ep] = (pose, t)
    return episodes, state_names, fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--condition", required=True, help="Label written to each row, e.g. A or B.")
    parser.add_argument("--out", required=True, help="Output features CSV.")
    parser.add_argument("--video_backend", default="pyav")
    parser.add_argument("--ref_repo_id", default=None,
                        help="Expert dataset for accuracy (X_ref): a skilled operator's demo(s).")
    parser.add_argument("--ref_root", default=None)
    parser.add_argument("--labels_csv", default=None,
                        help="CSV 'episode,success[,failure_type]' for success/failure rate.")
    parser.add_argument("--resample_n", type=int, default=100,
                        help="Points for time-normalized accuracy alignment to the expert.")
    args = parser.parse_args()

    import pandas as pd

    episodes, _, fps = load_episodes(args.repo_id, args.root, args.video_backend)
    print(f"{args.repo_id}: {len(episodes)} usable episodes, fps={fps}")

    ref_pose = None
    if args.ref_repo_id:
        ref_episodes, _, _ = load_episodes(args.ref_repo_id, args.ref_root, args.video_backend)
        if not ref_episodes:
            raise SystemExit(f"No usable expert episodes in {args.ref_repo_id}")
        ref_pose = build_reference(ref_episodes, args.resample_n)
        print(f"Built expert reference from {len(ref_episodes)} demo(s) at {args.resample_n} pts")

    rows = []
    for ep, (pose, t) in sorted(episodes.items()):
        feats = {"condition": args.condition, "episode": ep, "n_frames": len(t)}
        feats.update(smoothness_speed_features(pose, t))
        if ref_pose is not None:
            feats.update(accuracy_features(pose, ref_pose, args.resample_n))
        rows.append(feats)

    df = pd.DataFrame(rows)

    # Error rate (plan §3.x): merge success/failure labels, report SR / FR + failure breakdown.
    if args.labels_csv:
        labels = pd.read_csv(args.labels_csv)
        df = df.merge(labels, on="episode", how="left")
        if "success" in df.columns:
            sr = df["success"].mean()
            print(f"\nSuccess rate SR = {sr:.1%}   Failure rate FR = {1 - sr:.1%}   (n={len(df)})")
            if "failure_type" in df.columns:
                fails = df.loc[df["success"] == 0, "failure_type"].dropna().value_counts()
                if len(fails):
                    print("Failure breakdown:")
                    for k, v in fails.items():
                        print(f"  {k}: {int(v)}")

    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(rows)} episode feature rows to {args.out}")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df.select_dtypes(include=[np.number]).describe().round(4))


if __name__ == "__main__":
    main()

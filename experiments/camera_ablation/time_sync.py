#!/usr/bin/env python
"""Time-synchronize the three capture streams (experiment plan §4) into aligned episodes.

The robot, camera and phone each run on their own clock/rate. This resamples all three onto
a common time grid so every output frame has a matching joint state, phone command and camera
image. Output is in the folder format that convert_raw_to_lerobot.py consumes.

INPUT  (one folder per episode, each stream timestamped in SECONDS):
    <raw_ts>/episode_000/
        robot.csv        # timestamp, j1..jN              (robot joint angles)
        phone.csv        # timestamp, dx,dy,dz,dyaw,dpitch,droll   (phone end-effector delta)
        video.mp4        # camera recording (Condition B only)
        video_meta.json  # optional {"start_ts": <s>, "fps": <video_fps>}; default start=0, fps=mp4 fps

OUTPUT (ready for the converter):
    <aligned>/episode_000/
        states.csv   # timestamp,j1..jN   (robot, linearly interpolated to the grid)
        actions.csv  # timestamp,dx..droll (phone, zero-order hold: last command still in effect)
        video.mp4    # one camera frame per grid point (nearest in time)

Resampling: robot state = linear interp (continuous signal); phone delta = zero-order hold
(a command stays in effect until the next one); camera = nearest frame. The common grid spans
[max(start times), min(end times)] so every stream actually covers it.

USAGE:
    uv run --extra training python experiments/camera_ablation/time_sync.py \
        --in_dir raw_ts/cond_b --out_dir raw/cond_b --out_fps 30
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _read_ts_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (timestamps (T,), values (T,D), value_column_names)."""
    import pandas as pd

    df = pd.read_csv(path)
    ts_col = next((c for c in df.columns if str(c).strip().lower() in {"timestamp", "time", "t"}), None)
    if ts_col is None:
        raise SystemExit(f"{path}: needs a 'timestamp' column (seconds).")
    t = df[ts_col].to_numpy(dtype=float)
    val_cols = [c for c in df.columns if c != ts_col]
    vals = df[val_cols].to_numpy(dtype=float)
    return t, vals, val_cols


def _video_timestamps(meta_path: Path, mp4_path: Path) -> tuple[np.ndarray, float]:
    import cv2

    cap = cv2.VideoCapture(str(mp4_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    start = 0.0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        start = float(meta.get("start_ts", 0.0))
        fps = float(meta.get("fps", fps))
    return start + np.arange(n) / fps, fps


def _interp(t_grid: np.ndarray, t: np.ndarray, vals: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_grid, t, vals[:, d]) for d in range(vals.shape[1])], axis=1)


def _zoh(t_grid: np.ndarray, t: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Zero-order hold: each grid point takes the most recent sample at or before it."""
    idx = np.searchsorted(t, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 1)
    return vals[idx]


def sync_episode(ep_in: Path, ep_out: Path, out_fps: int) -> None:
    import cv2
    import pandas as pd

    robot_t, robot_v, robot_cols = _read_ts_csv(ep_in / "robot.csv")
    phone_t, phone_v, phone_cols = _read_ts_csv(ep_in / "phone.csv")

    starts = [robot_t[0], phone_t[0]]
    ends = [robot_t[-1], phone_t[-1]]

    has_video = (ep_in / "video.mp4").exists()
    if has_video:
        vid_t, _ = _video_timestamps(ep_in / "video_meta.json", ep_in / "video.mp4")
        starts.append(vid_t[0])
        ends.append(vid_t[-1])

    t0, t1 = max(starts), min(ends)
    if t1 <= t0:
        raise SystemExit(f"{ep_in.name}: streams do not overlap in time [{t0}, {t1}]")
    t_grid = np.arange(t0, t1, 1.0 / out_fps)

    states = _interp(t_grid, robot_t, robot_v)       # robot: linear
    actions = _zoh(t_grid, phone_t, phone_v)         # phone delta: zero-order hold

    ep_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(np.column_stack([t_grid, states]), columns=["timestamp", *robot_cols]).to_csv(
        ep_out / "states.csv", index=False)
    pd.DataFrame(np.column_stack([t_grid, actions]), columns=["timestamp", *phone_cols]).to_csv(
        ep_out / "actions.csv", index=False)

    if has_video:
        cap = cv2.VideoCapture(str(ep_in / "video.mp4"))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        frames = np.stack(frames)
        nearest = np.clip(np.searchsorted(vid_t, t_grid), 0, len(frames) - 1)
        h, w = frames.shape[1:3]
        writer = cv2.VideoWriter(str(ep_out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 out_fps, (w, h))
        for idx in nearest:
            writer.write(frames[idx])
        writer.release()

    print(f"  {ep_in.name}: {len(t_grid)} aligned frames over {t1 - t0:.2f}s"
          f"{' + video' if has_video else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_dir", required=True, help="Dir of episode_* folders with raw timestamped streams.")
    parser.add_argument("--out_dir", required=True, help="Output dir of aligned episode_* folders.")
    parser.add_argument("--out_fps", type=int, default=30)
    args = parser.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    eps = sorted(p for p in in_dir.iterdir() if p.is_dir() and (p / "robot.csv").exists())
    if not eps:
        raise SystemExit(f"No episode_* folders with robot.csv under {in_dir}")
    print(f"Aligning {len(eps)} episodes at {args.out_fps} Hz")
    for ep in eps:
        sync_episode(ep, out_dir / ep.name, args.out_fps)
    print(f"\nDone. Aligned episodes in {out_dir}. Feed them to convert_raw_to_lerobot.py.")


if __name__ == "__main__":
    main()

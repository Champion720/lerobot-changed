#!/usr/bin/env python
"""Convert raw recordings (from your own phone-app / WebRTC capture system) into a
LeRobot dataset, so you can train ACT on real Condition-A / Condition-B data.

This is the bridge between your Target1.1 capture system and LeRobot. You record each
demonstration with your own setup, dump it to the simple folder format below, and this
script packs it into the standard LeRobotDataset that `lerobot-train` consumes.

Current experiment definition:
  Condition A = camera stream displayed on the phone client; phone controls the end effector.
  Condition B = camera stream displayed on the PC client; phone controls the end effector.
Both conditions should normally include video.mp4. The --no_camera flag is kept only for
legacy/debug datasets.

DATA SEMANTICS (agreed for this project):
  action = [dx, dy, dz, dyaw, dpitch, droll]   end-effector DELTA sent by the phone (6D)
  state  = [joint angles j1..jN] (+ end-effector pose [x,y,z,yaw,pitch,roll] via FK)
           The robot reports joint angles; the end-effector pose is computed from them
           with your DH table (--dh_config). Without --dh_config, state = joint angles only
           (no information lost: joints already determine the pose).

EXPECTED RAW FORMAT  (one folder per recorded episode):

    <raw_dir>/
        episode_000/
            states.csv     # one row per frame: joint angles j1..jN (what the robot reports)
            actions.csv    # one row per frame: the 6D end-effector delta the phone sent
            video.mp4       # camera recording (current A/B experiment: present in both conditions)
        episode_001/ ...

  * states.csv and actions.csv must have the SAME number of rows (= frames). Any
    'timestamp'/'time'/'t'/'frame'/'index' column is dropped automatically.
  * When video.mp4 is present, its frame count should match; frames are matched by index.

USAGE (run from the inner lerobot-main project dir):
    # Condition A (mobile display) with FK-augmented state and camera video
    uv run --extra training python experiments/camera_ablation/convert_raw_to_lerobot.py \
        --raw_dir raw/cond_a_mobile --repo_id local/cond_a_mobile --fps 30 --task "pick and place" \
        --dh_config experiments/camera_ablation/dh_params.json --resize 480x640

    # Condition B (PC display) with FK-augmented state and camera video
    uv run --extra training python experiments/camera_ablation/convert_raw_to_lerobot.py \
        --raw_dir raw/cond_b_pc --repo_id local/cond_b_pc --fps 30 --task "pick and place" \
        --dh_config experiments/camera_ablation/dh_params.json --resize 480x640
"""

import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from forward_kinematics import ForwardKinematics  # noqa: E402  (same-dir import)

DROP_COLS = {"timestamp", "time", "t", "frame", "index", "frame_index"}
CAM_KEY = "observation.images.cam"
ACTION_NAMES = ["d_x", "d_y", "d_z", "d_yaw", "d_pitch", "d_roll"]
EE_NAMES = ["ee_x", "ee_y", "ee_z", "ee_yaw", "ee_pitch", "ee_roll"]


def read_csv(path: Path) -> np.ndarray:
    """Read a numeric CSV into (T, dim), dropping any timestamp-like columns."""
    import pandas as pd

    df = pd.read_csv(path)
    drop = [c for c in df.columns if str(c).strip().lower() in DROP_COLS]
    df = df.drop(columns=drop)
    arr = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{path}: expected a non-empty 2D numeric table, got shape {arr.shape}")
    return arr


def read_video(path: Path, resize_hw: tuple[int, int] | None) -> np.ndarray:
    """Read an mp4 into (T, H, W, 3) uint8 RGB frames."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if resize_hw is not None:
            h, w = resize_hw
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()
    if not frames:
        raise ValueError(f"{path}: no frames decoded")
    return np.stack(frames).astype(np.uint8)


def build_features(n_joints: int, action_dim: int, has_camera: bool,
                   img_hw: tuple[int, int] | None, with_fk: bool):
    state_names = [f"joint_{i + 1}" for i in range(n_joints)]
    if with_fk:
        state_names = state_names + EE_NAMES
    action_names = ACTION_NAMES[:action_dim] if action_dim <= len(ACTION_NAMES) else None

    feats = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": action_names},
    }
    if has_camera:
        h, w = img_hw
        feats[CAM_KEY] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    return feats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", required=True, help="Dir containing episode_* folders.")
    parser.add_argument("--repo_id", required=True, help="Output dataset id, e.g. local/cond_b_camera.")
    parser.add_argument("--fps", type=int, required=True, help="Capture frame rate.")
    parser.add_argument("--task", required=True, help="Natural-language task description.")
    parser.add_argument("--dh_config", default=None,
                        help="DH params JSON. If set, state = [joint angles + FK end-effector pose].")
    parser.add_argument(
        "--no_camera",
        action="store_true",
        help="Legacy/debug mode: episodes have no video.mp4. Current A/B experiment should not use this.",
    )
    parser.add_argument("--resize", default=None, help="Force camera frames to HxW, e.g. 480x640.")
    parser.add_argument("--output_dir", default=None, help="Where to write the dataset.")
    parser.add_argument("--states_name", default="states.csv")
    parser.add_argument("--actions_name", default="actions.csv")
    parser.add_argument("--video_name", default="video.mp4")
    args = parser.parse_args()

    has_camera = not args.no_camera
    fk = ForwardKinematics.from_json(args.dh_config) if args.dh_config else None
    resize_hw = None
    if args.resize:
        h, w = (int(x) for x in args.resize.lower().split("x"))
        resize_hw = (h, w)

    def make_state(joint_rows: np.ndarray) -> np.ndarray:
        """joint_rows (T, n_joints) -> state (T, n_joints[+6])."""
        if fk is None:
            return joint_rows
        poses = np.stack([fk.pose(joint_rows[t]) for t in range(len(joint_rows))]).astype(np.float32)
        return np.concatenate([joint_rows, poses], axis=1)

    raw_dir = Path(args.raw_dir)
    ep_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir() and (p / args.states_name).exists())
    if not ep_dirs:
        raise SystemExit(f"No episode folders with {args.states_name} found under {raw_dir}")
    print(f"Found {len(ep_dirs)} episodes under {raw_dir}")

    # Infer dims from the first episode.
    j0 = read_csv(ep_dirs[0] / args.states_name)
    a0 = read_csv(ep_dirs[0] / args.actions_name)
    n_joints = j0.shape[1]
    if fk is not None and fk.n_joints != n_joints:
        raise SystemExit(f"DH config has {fk.n_joints} joints but states.csv has {n_joints} columns")
    if a0.shape[1] != 6:
        print(f"  WARN: action has {a0.shape[1]} columns; expected 6 (dx,dy,dz,dyaw,dpitch,droll)")

    img_hw = None
    if has_camera:
        v0 = read_video(ep_dirs[0] / args.video_name, resize_hw)
        img_hw = (v0.shape[1], v0.shape[2])
    print(f"joints={n_joints}, state_dim={n_joints + (6 if fk else 0)}, action_dim={a0.shape[1]}, "
          f"camera={'%dx%d' % img_hw if has_camera else 'none'}")

    features = build_features(n_joints, a0.shape[1], has_camera, img_hw, with_fk=fk is not None)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.output_dir,
        use_videos=has_camera,
        video_backend="pyav",
    )

    for ep in ep_dirs:
        joints = read_csv(ep / args.states_name)
        actions = read_csv(ep / args.actions_name)
        states = make_state(joints)
        n = min(len(states), len(actions))
        video = None
        if has_camera:
            video = read_video(ep / args.video_name, resize_hw)
            n = min(n, len(video))
        if n == 0:
            print(f"  WARN: {ep.name} has 0 usable frames, skipping")
            continue

        for t in range(n):
            frame = {
                "observation.state": states[t].astype(np.float32),
                "action": actions[t].astype(np.float32),
                "task": args.task,
            }
            if has_camera:
                frame[CAM_KEY] = video[t]
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"  {ep.name}: {n} frames")

    dataset.finalize()
    print(f"\nDone. Dataset '{args.repo_id}' written to {dataset.root}")
    print(f"Train it with:  --dataset.repo_id={args.repo_id} --dataset.video_backend=pyav")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Convert raw recordings (from your own phone-app / WebRTC capture system) into a
LeRobot dataset, so you can train ACT on real Condition-A / Condition-B data.

This is the bridge between your Target1.1 capture system and LeRobot. You record each
demonstration with your own setup, dump it to the simple folder format below, and this
script packs it into the standard LeRobotDataset that `lerobot-train` consumes.

Current experiment definition:
  A_mobile_colocated = wrist stream displayed on the controlling phone.
  B_desktop_separated = wrist stream displayed on a fixed desktop while the phone controls motion.
Both conditions must include video.mp4 from the same wrist-camera source.

DATA SEMANTICS (agreed for this project):
  action = [dx, dy, dz, dyaw, dpitch, droll]   accepted DELTA applied by the bridge (6D)
  state  = [joint angles j1..jN] (+ end-effector pose [x,y,z,yaw,pitch,roll] via FK)
           The robot reports joint angles; the end-effector pose is computed from them
           with your DH table (--dh_config). Without --dh_config, state = joint angles only
           (no information lost: joints already determine the pose).

EXPECTED RAW FORMAT  (one folder per recorded episode):

    <raw_dir>/
        episode_000/
            states.csv     # one row per frame: joint angles j1..jN (what the robot reports)
            actions.csv    # one row per frame: the synchronized 6D delta actually applied
            video.mp4       # camera recording (current A/B experiment: present in both conditions)
        episode_001/ ...

  * states.csv and actions.csv must have the SAME number of rows (= frames). Any
    'timestamp'/'time'/'t'/'frame'/'index' column is dropped automatically.
  * When video.mp4 is present, its frame count should match; frames are matched by index.

USAGE (run from the inner lerobot-main project dir):
    # A_mobile_colocated with FK-augmented state and wrist-camera video
    uv run --extra training python experiments/wrist_view_presentation/convert_raw_to_lerobot.py \
        --raw_dir raw/A_mobile_colocated --repo_id local/A_mobile_colocated --fps 30 \
        --task "pick and place" \
        --dh_config experiments/wrist_view_presentation/dh_params.json --resize 480x640

    # B_desktop_separated with the identical state/video schema
    uv run --extra training python experiments/wrist_view_presentation/convert_raw_to_lerobot.py \
        --raw_dir raw/B_desktop_separated --repo_id local/B_desktop_separated --fps 30 \
        --task "pick and place" \
        --dh_config experiments/wrist_view_presentation/dh_params.json --resize 480x640
"""

import argparse
import contextlib
import csv
import hashlib
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import numpy as np

if __package__:
    from .forward_kinematics import ForwardKinematics
else:
    from forward_kinematics import ForwardKinematics

DROP_COLS = {"timestamp", "time", "t", "frame", "index", "frame_index"}
CAM_KEY = "observation.images.cam"
ACTION_NAMES = ["dx", "dy", "dz", "dyaw", "dpitch", "droll"]
EE_NAMES = ["ee_x", "ee_y", "ee_z", "ee_yaw", "ee_pitch", "ee_roll"]


def sha256_file(path: Path) -> str:
    """Hash one immutable capture input without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cleanup_owned_snapshot(owner_root: Path) -> None:
    """Remove only the exact input-snapshot directory created by this process."""

    if owner_root.is_symlink() or owner_root.is_file():
        raise RuntimeError(f"owned input snapshot changed type unexpectedly: {owner_root}")
    if owner_root.exists():
        shutil.rmtree(owner_root)


def _snapshot_relative_path(filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"input filename must stay inside each episode directory: {filename!r}")
    return relative


def make_input_snapshots(
    episode_dirs: Sequence[Path],
    filenames: Sequence[str],
    *,
    owner_parent: Path,
    owner_prefix: str,
) -> tuple[Path, list[Path], dict[str, dict[str, dict[str, str]]]]:
    """Copy and verify every capture input before any parser or decoder opens it.

    Each source is hashed before copying. The copied bytes and the source's
    immediately-current bytes must both still match that first digest. Once this
    returns, callers use only the process-owned snapshot paths; later source ABA
    replacements therefore cannot affect conversion.
    """

    owner_parent.mkdir(parents=True, exist_ok=True)
    owner_root = Path(tempfile.mkdtemp(prefix=owner_prefix, dir=owner_parent))
    snapshot_dirs: list[Path] = []
    fingerprints: dict[str, dict[str, dict[str, str]]] = {}
    try:
        for source_episode in episode_dirs:
            snapshot_episode = owner_root / source_episode.name
            snapshot_episode.mkdir(exist_ok=False)
            snapshot_dirs.append(snapshot_episode)
            episode_fingerprints: dict[str, dict[str, str]] = {}
            for filename in filenames:
                relative = _snapshot_relative_path(filename)
                source_path = source_episode / relative
                snapshot_path = snapshot_episode / relative
                before_digest = sha256_file(source_path)
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, snapshot_path)
                snapshot_digest = sha256_file(snapshot_path)
                try:
                    current_digest = sha256_file(source_path) if source_path.is_file() else None
                except OSError:
                    current_digest = None
                if snapshot_digest != before_digest or current_digest != before_digest:
                    raise RuntimeError(
                        f"{source_path}: input changed while creating immutable conversion "
                        "snapshot; stop capture and retry"
                    )
                episode_fingerprints[filename] = {
                    "path": str(source_path.absolute()),
                    "sha256": before_digest,
                }
            fingerprints[source_episode.name] = episode_fingerprints
        return owner_root, snapshot_dirs, fingerprints
    except BaseException:
        cleanup_owned_snapshot(owner_root)
        raise


def resolve_output_root(output_dir: str | None, repo_id: str) -> Path:
    """Match ``LeRobotDataset.create``'s default root without creating it."""

    if output_dir is not None:
        return Path(output_dir).expanduser()

    from lerobot.utils.constants import HF_LEROBOT_HOME

    return HF_LEROBOT_HOME / repo_id


def validate_output_target(target_root: Path) -> bool:
    """Reject unsafe targets and report whether an existing target is empty."""

    if target_root.is_symlink():
        raise ValueError(f"Output dataset root must not be a symbolic link: {target_root}")
    if not target_root.exists():
        return False
    if not target_root.is_dir():
        raise ValueError(f"Output path is not a directory: {target_root}")
    if next(target_root.iterdir(), None) is not None:
        raise FileExistsError(f"Output dataset root already exists and is not empty: {target_root}")
    return True


def make_staging_area(target_root: Path) -> tuple[Path, Path]:
    """Atomically own a sibling staging parent and return its dataset child."""

    target_root.parent.mkdir(parents=True, exist_ok=True)
    owner_root = Path(
        tempfile.mkdtemp(
            prefix=f".{target_root.name}.staging-{uuid4().hex}-",
            dir=target_root.parent,
        )
    )
    return owner_root, owner_root / "dataset"


def publish_staging_root(
    staging_root: Path,
    target_root: Path,
    *,
    target_was_empty: bool,
) -> None:
    """Atomically rename a finalized sibling staging directory into place."""

    removed_empty_target = False
    if target_root.exists() or target_root.is_symlink():
        if not target_was_empty:
            raise FileExistsError(
                f"Output dataset root appeared while converting; refusing to overwrite: {target_root}"
            )
        validate_output_target(target_root)
        target_root.rmdir()
        removed_empty_target = True

    try:
        staging_root.rename(target_root)
    except BaseException:
        # Preserve a pre-existing empty output directory if publication itself fails.
        if removed_empty_target and not target_root.exists():
            with contextlib.suppress(OSError):
                target_root.mkdir()
        raise


def cleanup_staging_root(owner_root: Path) -> None:
    """Remove only the atomically created staging parent owned by this process."""

    if owner_root.is_symlink() or owner_root.is_file():
        raise RuntimeError(f"owned staging parent changed type unexpectedly: {owner_root}")
    if owner_root.exists():
        shutil.rmtree(owner_root)


def finalize_before_discard(dataset) -> None:
    """Best-effort resource release before deleting a failed staging dataset."""

    with contextlib.suppress(Exception):
        dataset.finalize()


def read_csv_table(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read a numeric CSV and retain the exact value-column contract."""
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.reader(stream)
            header = next(rows, None)
            if header is None:
                raise ValueError(f"{path}: CSV is empty")
            raw_columns = tuple(column.strip() for column in header)
            if any(not column for column in raw_columns):
                raise ValueError(f"{path}: CSV header names must not be blank")
            if len(set(raw_columns)) != len(raw_columns):
                raise ValueError(f"{path}: CSV header names must be unique after trimming, got {raw_columns}")
            for line_number, row in enumerate(rows, start=2):
                if len(row) != len(header):
                    raise ValueError(
                        f"{path}: CSV row {line_number} has {len(row)} fields; header has {len(header)}"
                    )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: CSV must be UTF-8 encoded") from exc
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path}: CSV is empty") from exc
    if len(df.columns) != len(raw_columns):
        raise ValueError(
            f"{path}: parsed {len(df.columns)} CSV columns but the raw header has {len(raw_columns)}"
        )
    # Use the already validated raw names so pandas cannot silently mangle a
    # duplicate header into a different value-column schema.
    df.columns = list(raw_columns)
    if len(df) == 0:
        raise ValueError(f"{path}: contains no rows")

    drop = [c for c in df.columns if str(c).strip().lower() in DROP_COLS]
    for column in drop:
        try:
            ordering = pd.to_numeric(df[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: alignment column {column!r} must be numeric") from exc
        if not np.isfinite(ordering).all():
            raise ValueError(f"{path}: alignment column {column!r} contains NaN or infinity")
        if len(ordering) > 1 and np.any(np.diff(ordering) <= 0):
            raise ValueError(
                f"{path}: alignment column {column!r} must be strictly increasing "
                "(duplicates and out-of-order rows are not safe for frame-index conversion)"
            )

    values = df.drop(columns=drop)
    if values.shape[1] == 0:
        raise ValueError(f"{path}: expected at least one value column after dropping {drop}")
    columns = tuple(str(column).strip() for column in values.columns)
    if any(not column for column in columns):
        raise ValueError(f"{path}: value column names must not be blank")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{path}: value column names must be unique, got {columns}")
    try:
        numeric = values.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: all value columns must be numeric") from exc
    arr = numeric.to_numpy(dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{path}: expected a non-empty 2D numeric table, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: value columns contain NaN or infinity")
    return arr, columns


def read_csv(
    path: Path,
    *,
    expected_columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Read values and optionally enforce an exact, ordered column contract."""

    values, columns = read_csv_table(path)
    if expected_columns is not None:
        expected = tuple(expected_columns)
        if columns != expected:
            raise ValueError(
                f"{path}: value columns must be exactly {list(expected)} in this order; got {list(columns)}"
            )
    return values


def parse_joint_names(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = tuple(part.strip() for part in value.split(","))
    if not names or any(not name for name in names):
        raise ValueError("--joint_names must be a comma-separated list of non-empty names")
    if len(set(names)) != len(names):
        raise ValueError("--joint_names entries must be unique")
    return names


def iter_video_frames(path: Path, resize_hw: tuple[int, int] | None):
    """Yield validated RGB frames while keeping only one decoded frame in memory."""
    import cv2

    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    if resize_hw is not None and (
        len(resize_hw) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in resize_hw)
        or any(value <= 0 for value in resize_hw)
    ):
        raise ValueError(f"resize_hw must contain two positive integers, got {resize_hw!r}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"{path}: OpenCV could not open the video")
    reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_count = 0
    expected_shape = None
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
                raise ValueError(f"{path}: decoded an invalid video frame")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if resize_hw is not None:
                h, w = resize_hw
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            if expected_shape is None:
                expected_shape = rgb.shape
            elif rgb.shape != expected_shape:
                raise ValueError(f"{path}: frame dimensions changed from {expected_shape} to {rgb.shape}")
            decoded_count += 1
            yield rgb.astype(np.uint8, copy=False)
    finally:
        cap.release()
    if decoded_count == 0:
        raise ValueError(f"{path}: no frames decoded")
    if reported_count > 0 and reported_count != decoded_count:
        raise ValueError(
            f"{path}: container reports {reported_count} frames but only {decoded_count} decoded"
        )


def probe_video(
    path: Path,
    resize_hw: tuple[int, int] | None,
) -> tuple[int, tuple[int, int]]:
    """Decode once for integrity/count/shape checks without stacking frames."""

    count = 0
    image_hw = None
    for frame in iter_video_frames(path, resize_hw):
        count += 1
        image_hw = (frame.shape[0], frame.shape[1])
    if image_hw is None:
        raise ValueError(f"{path}: no frames decoded")
    return count, image_hw


def read_video(path: Path, resize_hw: tuple[int, int] | None) -> np.ndarray:
    """Compatibility helper for short tests; the converter itself streams frames."""

    return np.stack(list(iter_video_frames(path, resize_hw))).astype(np.uint8)


def resolve_episode_length(
    stream_lengths: dict[str, int],
    max_length_mismatch: int = 0,
    *,
    context: str = "episode",
) -> int:
    """Return an explicitly allowed common length instead of silently truncating.

    The default tolerance is zero. When a positive tolerance is supplied, all streams
    may differ by at most that many frames and are intentionally trimmed to the shortest.
    """
    if isinstance(max_length_mismatch, bool) or not isinstance(max_length_mismatch, (int, np.integer)):
        raise TypeError("max_length_mismatch must be an integer")
    max_length_mismatch = int(max_length_mismatch)
    if max_length_mismatch < 0:
        raise ValueError("max_length_mismatch must be non-negative")
    if not stream_lengths:
        raise ValueError(f"{context}: no stream lengths were provided")
    if any(
        isinstance(length, bool) or not isinstance(length, (int, np.integer))
        for length in stream_lengths.values()
    ):
        raise TypeError(f"{context}: stream lengths must be integers")
    if any(length <= 0 for length in stream_lengths.values()):
        raise ValueError(f"{context}: every stream must contain at least one frame, got {stream_lengths}")

    shortest = min(stream_lengths.values())
    longest = max(stream_lengths.values())
    mismatch = longest - shortest
    if mismatch > max_length_mismatch:
        details = ", ".join(f"{name}={length}" for name, length in stream_lengths.items())
        raise ValueError(
            f"{context}: stream length mismatch ({details}); difference {mismatch} exceeds "
            f"allowed {max_length_mismatch}. Re-run time_sync.py or explicitly set "
            "--max_length_mismatch to permit a small trim."
        )
    return int(shortest)


def parse_resize(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"--resize must use HxW format, got {value!r}")
    try:
        height, width = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"--resize must use integer HxW dimensions, got {value!r}") from exc
    if height <= 0 or width <= 0:
        raise ValueError(f"--resize dimensions must be positive, got {value!r}")
    return height, width


def build_features(
    joint_names: Sequence[str],
    img_hw: tuple[int, int],
    with_fk: bool,
):
    n_joints = len(joint_names)
    if n_joints <= 0:
        raise ValueError("joint_names must contain at least one name")
    if len(set(joint_names)) != n_joints:
        raise ValueError("joint_names must be unique")
    state_names = list(joint_names)
    if with_fk:
        state_names = state_names + EE_NAMES
    feats = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": ACTION_NAMES,
        },
    }
    h, w = img_hw
    feats[CAM_KEY] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    return feats


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", required=True, help="Dir containing episode_* folders.")
    parser.add_argument(
        "--repo_id",
        required=True,
        help="Output dataset id, e.g. local/B_desktop_separated.",
    )
    parser.add_argument("--fps", type=int, required=True, help="Capture frame rate.")
    parser.add_argument("--task", required=True, help="Natural-language task description.")
    parser.add_argument(
        "--dh_config",
        default=None,
        help="DH params JSON. If set, state = [joint angles + FK end-effector pose].",
    )
    parser.add_argument("--resize", default=None, help="Force camera frames to HxW, e.g. 480x640.")
    parser.add_argument("--output_dir", default=None, help="Where to write the dataset.")
    parser.add_argument("--states_name", default="states.csv")
    parser.add_argument("--actions_name", default="actions.csv")
    parser.add_argument("--video_name", default="video.mp4")
    parser.add_argument(
        "--joint_names",
        default=None,
        help=(
            "Exact comma-separated states.csv value-column order, for example "
            "joint_1,joint_2,... . Required with --dh_config; otherwise the first "
            "episode header becomes the contract for all episodes."
        ),
    )
    parser.add_argument(
        "--joint_angle_unit",
        choices=("rad", "deg"),
        default="rad",
        help=(
            "Raw states.csv joint-angle unit when --dh_config is absent. Output states "
            "are always radians. With --dh_config, its angle_unit is authoritative."
        ),
    )
    parser.add_argument(
        "--max_length_mismatch",
        type=int,
        default=0,
        help=(
            "Maximum frame-count difference allowed between state/action/video streams. "
            "Default 0 rejects every mismatch; a positive value explicitly trims to the shortest stream."
        ),
    )
    parser.add_argument(
        "--max_episode_frames",
        type=int,
        default=18_000,
        help="Reject longer aligned episodes before dataset creation (default: 18000).",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        raise SystemExit(f"--fps must be positive, got {args.fps}")
    if args.max_length_mismatch < 0:
        raise SystemExit(f"--max_length_mismatch must be non-negative, got {args.max_length_mismatch}")
    if args.max_episode_frames <= 0:
        raise SystemExit(f"--max_episode_frames must be positive, got {args.max_episode_frames}")
    if not args.repo_id.strip():
        raise SystemExit("--repo_id must not be empty")
    if not args.task.strip():
        raise SystemExit("--task must not be empty")

    fk = ForwardKinematics.from_json(args.dh_config) if args.dh_config else None
    try:
        resize_hw = parse_resize(args.resize)
        configured_joint_names = parse_joint_names(args.joint_names)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if fk is not None:
        if configured_joint_names is None and fk.has_explicit_joint_names:
            configured_joint_names = fk.joint_names
        elif configured_joint_names is None:
            raise SystemExit(
                "provide --joint_names or DH config joint_names so FK joint order cannot be inferred silently"
            )
        elif fk.has_explicit_joint_names and configured_joint_names != fk.joint_names:
            raise SystemExit("--joint_names does not match DH config joint_names")

    def make_state(joint_rows: np.ndarray) -> np.ndarray:
        """Normalize joints to radians and optionally append an FK pose in m/rad."""
        if fk is None:
            return np.deg2rad(joint_rows).astype(np.float32) if args.joint_angle_unit == "deg" else joint_rows
        normalized_joints = fk.normalize_joint_angles(joint_rows).astype(np.float32)
        poses = np.stack([fk.pose(joint_rows[t]) for t in range(len(joint_rows))]).astype(np.float32)
        return np.concatenate([normalized_joints, poses], axis=1)

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        raise SystemExit(f"Raw input directory not found: {raw_dir}")
    target_root = resolve_output_root(args.output_dir, args.repo_id)
    if target_root.resolve() == raw_dir.resolve():
        raise SystemExit("Raw input and dataset output directories must be different")
    try:
        target_was_empty = validate_output_target(target_root)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    all_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    named_episode_dirs = [p for p in all_dirs if p.name.startswith("episode_")]
    ep_dirs = named_episode_dirs or [p for p in all_dirs if (p / args.states_name).exists()]
    if not ep_dirs:
        raise SystemExit(f"No episode folders with {args.states_name} found under {raw_dir}")
    for ep in ep_dirs:
        required = [ep / args.states_name, ep / args.actions_name, ep / args.video_name]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise SystemExit(f"{ep}: missing required file(s): {', '.join(missing)}")
    input_names = [args.states_name, args.actions_name, args.video_name]
    snapshot_owner, snapshot_ep_dirs, episode_fingerprints = make_input_snapshots(
        ep_dirs,
        input_names,
        owner_parent=target_root.parent,
        owner_prefix=f".{target_root.name}.input-snapshot-",
    )
    ep_dirs = snapshot_ep_dirs

    try:
        print(f"Found {len(ep_dirs)} episodes under {raw_dir}")
        # Establish the exact schema from the first immutable episode snapshot.
        j0, first_joint_columns = read_csv_table(ep_dirs[0] / args.states_name)
        joint_names = configured_joint_names or first_joint_columns
        if first_joint_columns != tuple(joint_names):
            raise SystemExit(
                f"{ep_dirs[0] / args.states_name}: value columns must be exactly "
                f"{list(joint_names)} in this order; got {list(first_joint_columns)}"
            )
        a0 = read_csv(
            ep_dirs[0] / args.actions_name,
            expected_columns=ACTION_NAMES,
        )
        n_joints = j0.shape[1]
        if fk is not None and fk.n_joints != n_joints:
            raise SystemExit(f"DH config has {fk.n_joints} joints but states.csv has {n_joints} columns")
        # Fully validate every episode before creating any dataset output. This prevents a
        # bad later episode from leaving an apparently trainable partial dataset.
        episode_lengths: dict[Path, dict[str, int]] = {}
        img_hw = None
        for ep in ep_dirs:
            try:
                joints = (
                    j0 if ep == ep_dirs[0] else read_csv(ep / args.states_name, expected_columns=joint_names)
                )
                actions = (
                    a0
                    if ep == ep_dirs[0]
                    else read_csv(ep / args.actions_name, expected_columns=ACTION_NAMES)
                )
                if joints.shape[1] != n_joints:
                    raise ValueError(
                        f"{ep / args.states_name}: has {joints.shape[1]} joint columns; expected {n_joints}"
                    )
                if actions.shape[1] != len(ACTION_NAMES):
                    raise ValueError(
                        f"{ep / args.actions_name}: has {actions.shape[1]} action columns; "
                        f"expected fixed 6D schema {ACTION_NAMES}"
                    )
                # Exercise joint normalization and FK before output creation.
                make_state(joints)
                lengths = {"states": len(joints), "actions": len(actions)}
                video_count, episode_hw = probe_video(ep / args.video_name, resize_hw)
                lengths["video"] = video_count
                if img_hw is None:
                    img_hw = episode_hw
                elif episode_hw != img_hw:
                    raise ValueError(
                        f"{ep / args.video_name}: frame size {episode_hw}; expected {img_hw}. "
                        "Use --resize HxW to normalize episode resolution."
                    )
                resolve_episode_length(
                    lengths,
                    args.max_length_mismatch,
                    context=ep.name,
                )
                if max(lengths.values()) > args.max_episode_frames:
                    raise ValueError(
                        f"{ep.name}: stream length {max(lengths.values())} exceeds "
                        f"--max_episode_frames={args.max_episode_frames}"
                    )
                episode_lengths[ep] = lengths
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
    except BaseException:
        cleanup_owned_snapshot(snapshot_owner)
        raise

    try:
        if img_hw is None:
            raise RuntimeError("internal error: wrist-camera dimensions were not established")
        camera_summary = f"{img_hw[0]}x{img_hw[1]}"
        print(
            f"joints={n_joints}, state_dim={n_joints + (6 if fk else 0)}, "
            f"action_dim={a0.shape[1]}, camera={camera_summary}"
        )
        features = build_features(
            joint_names,
            img_hw,
            with_fk=fk is not None,
        )
        staging_owner, staging_root = make_staging_area(target_root)
    except BaseException:
        cleanup_owned_snapshot(snapshot_owner)
        raise
    dataset = None
    published = False
    try:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=features,
            root=staging_root,
            use_videos=True,
            video_backend="pyav",
        )

        for ep in ep_dirs:
            joints = j0 if ep == ep_dirs[0] else read_csv(ep / args.states_name, expected_columns=joint_names)
            actions = (
                a0 if ep == ep_dirs[0] else read_csv(ep / args.actions_name, expected_columns=ACTION_NAMES)
            )

            states = make_state(joints)
            stream_lengths = episode_lengths[ep]
            n = resolve_episode_length(
                stream_lengths,
                args.max_length_mismatch,
                context=ep.name,
            )
            if len(set(stream_lengths.values())) > 1:
                print(
                    f"  NOTE: {ep.name} explicitly trims streams {stream_lengths} to {n} frames "
                    f"(allowed by --max_length_mismatch={args.max_length_mismatch})"
                )

            video_frames = iter_video_frames(ep / args.video_name, resize_hw)
            try:
                for t in range(n):
                    frame = {
                        "observation.state": states[t].astype(np.float32),
                        "action": actions[t].astype(np.float32),
                        "task": args.task,
                    }
                    frame[CAM_KEY] = next(video_frames)
                    dataset.add_frame(frame)
            finally:
                video_frames.close()
            dataset.save_episode()
            print(f"  {ep.name}: {n} frames")

        fingerprint_path = staging_root / "meta" / "source_fingerprints.json"
        fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "hash_algorithm": "sha256",
                    "episodes": episode_fingerprints,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        dataset.finalize()
        publish_staging_root(
            staging_root,
            target_root,
            target_was_empty=target_was_empty,
        )
        published = True
        cleanup_staging_root(staging_owner)
    finally:
        try:
            if not published:
                if dataset is not None:
                    finalize_before_discard(dataset)
                cleanup_staging_root(staging_owner)
        finally:
            cleanup_owned_snapshot(snapshot_owner)

    print(f"\nDone. Dataset '{args.repo_id}' written to {target_root}")
    print(f"Train it with:  --dataset.repo_id={args.repo_id} --dataset.video_backend=pyav")


if __name__ == "__main__":
    main()

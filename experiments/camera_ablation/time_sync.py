#!/usr/bin/env python
"""Time-synchronize the three capture streams (experiment plan section 4) into aligned episodes.

The robot, camera and phone each run on their own clock/rate. This resamples all three onto
a common time grid so every output frame has a matching joint state, phone command and camera
image. Output is in the folder format that convert_raw_to_lerobot.py consumes.

INPUT  (one folder per episode, each stream timestamped in SECONDS):
    <raw_ts>/episode_000/
        robot.csv        # timestamp, j1..jN              (robot joint angles)
        phone.csv        # audit stream: every received phone delta, including rejected input
        applied_actions.csv
                         # timestamp,dx,dy,dz,dyaw,dpitch,droll; accepted/applied deltas only
        video.mp4        # camera recording (current A/B experiment: present in both conditions)
        video_meta.json  # formal runs: {"frame_timestamps_s":[absolute timestamp per frame], ...}

OUTPUT (ready for the converter):
    <aligned>/episode_000/
        states.csv   # timestamp,j1..jN   (robot, linearly interpolated to the grid)
        actions.csv  # timestamp,dx..droll (ordered SE(3) composition in each output interval)
        video.mp4    # one camera frame per grid point (nearest in time)

Resampling: robot state = linear interpolation; accepted phone increments are composed in
timestamp order over half-open grid intervals [t_i, t_i+1); camera = nearest frame using
actual per-frame timestamps when supplied.  Ordered SE(3) composition preserves
non-commuting rotations and tool-frame translations when phone and output rates differ.
``start_ts + n/fps`` is only a legacy constant-frame-rate fallback.

USAGE:
    uv run --extra training python experiments/camera_ablation/time_sync.py \
        --in_dir raw_ts/cond_b --out_dir raw/cond_b --out_fps 30 \
        --robot_bridge_config experiments/camera_ablation/robot_bridge_config.json
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np

ACTION_COLUMNS = ["dx", "dy", "dz", "dyaw", "dpitch", "droll"]
FrameName = Literal["base", "tool"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cleanup_owned_input_snapshot(owner_root: Path) -> None:
    """Remove only the exact input-snapshot directory created by this process."""

    if owner_root.is_symlink() or owner_root.is_file():
        raise RuntimeError(f"owned input snapshot changed type unexpectedly: {owner_root}")
    if owner_root.exists():
        shutil.rmtree(owner_root)


def _snapshot_episode_inputs(
    ep_in: Path,
    filenames: list[str],
    *,
    owner_parent: Path,
    owner_prefix: str,
) -> tuple[Path, Path, dict[str, dict[str, str]]]:
    """Create one verified immutable snapshot before any input content is parsed."""

    owner_parent.mkdir(parents=True, exist_ok=True)
    owner_root = Path(tempfile.mkdtemp(prefix=owner_prefix, dir=owner_parent))
    snapshot_episode = owner_root / ep_in.name
    fingerprints: dict[str, dict[str, str]] = {}
    try:
        snapshot_episode.mkdir(exist_ok=False)
        for filename in filenames:
            relative = Path(filename)
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise ValueError(f"input filename must stay inside the episode directory: {filename!r}")
            source_path = ep_in / relative
            snapshot_path = snapshot_episode / relative
            before_digest = _sha256_file(source_path)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, snapshot_path)
            snapshot_digest = _sha256_file(snapshot_path)
            try:
                current_digest = _sha256_file(source_path) if source_path.is_file() else None
            except OSError:
                current_digest = None
            if snapshot_digest != before_digest or current_digest != before_digest:
                raise RuntimeError(
                    f"{source_path}: input changed while creating immutable synchronization "
                    "snapshot; stop capture and retry"
                )
            fingerprints[filename] = {
                "path": str(source_path.absolute()),
                "sha256": before_digest,
            }
        return owner_root, snapshot_episode, fingerprints
    except BaseException:
        _cleanup_owned_input_snapshot(owner_root)
        raise


def _validate_csv_width(path: Path) -> tuple[str, ...]:
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
            expected = len(header)
            for line_number, row in enumerate(rows, start=2):
                if len(row) != expected:
                    raise ValueError(
                        f"{path}: CSV row {line_number} has {len(row)} fields; header has {expected}"
                    )
            return raw_columns
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: CSV must be UTF-8 encoded") from exc


def _read_ts_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return sorted (timestamps (T,), values (T,D), value column names).

    Out-of-order rows are sorted stably. Duplicate timestamps are rejected because
    interpolation/zero-order hold would otherwise depend on arbitrary CSV row order.
    """
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(f"Timestamped CSV not found: {path}")
    raw_columns = _validate_csv_width(path)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path}: CSV is empty") from exc
    if len(df.columns) != len(raw_columns):
        raise ValueError(
            f"{path}: parsed {len(df.columns)} CSV columns but the raw header has {len(raw_columns)}"
        )
    df.columns = list(raw_columns)
    ts_col = next((c for c in df.columns if str(c).strip().lower() in {"timestamp", "time", "t"}), None)
    if ts_col is None:
        raise ValueError(f"{path}: needs a 'timestamp' column (seconds)")
    val_cols = [c for c in df.columns if c != ts_col]
    if not val_cols:
        raise ValueError(f"{path}: needs at least one numeric value column")
    if len(df) == 0:
        raise ValueError(f"{path}: contains no samples")

    try:
        t = pd.to_numeric(df[ts_col], errors="raise").to_numpy(dtype=float)
        vals = df[val_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: timestamp and value columns must be numeric") from exc
    if not np.isfinite(t).all():
        raise ValueError(f"{path}: timestamp column contains NaN or infinity")
    if not np.isfinite(vals).all():
        raise ValueError(f"{path}: value columns contain NaN or infinity")

    order = np.argsort(t, kind="stable")
    t = t[order]
    vals = vals[order]
    duplicate = np.flatnonzero(np.diff(t) == 0)
    if duplicate.size:
        repeated = t[duplicate[0]]
        raise ValueError(f"{path}: duplicate timestamp {repeated!r}; timestamps must be unique")
    return t, vals, val_cols


def _load_video_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    if not meta_path.is_file():
        raise ValueError(f"Video metadata path is not a file: {meta_path}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{meta_path}: invalid JSON ({exc.msg})") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path}: metadata must be a JSON object")
    return meta


def _video_timestamps(
    meta_path: Path,
    mp4_path: Path,
    frame_count: int | None = None,
) -> tuple[np.ndarray, float]:
    import cv2

    if not mp4_path.is_file():
        raise FileNotFoundError(f"Video file not found: {mp4_path}")
    cap = cv2.VideoCapture(str(mp4_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"{mp4_path}: OpenCV could not open the video")
        reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()

    if frame_count is None:
        frame_count = reported_count
    if isinstance(frame_count, bool) or not isinstance(frame_count, (int, np.integer)) or frame_count <= 0:
        raise ValueError(f"{mp4_path}: video has no decodable frames")

    meta = _load_video_meta(meta_path)
    explicit_timestamps = meta.get("frame_timestamps_s")
    if explicit_timestamps is not None:
        if not isinstance(explicit_timestamps, list):
            raise ValueError(f"{meta_path}: frame_timestamps_s must be a JSON array")
        try:
            timestamps = np.asarray(explicit_timestamps, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{meta_path}: frame_timestamps_s must contain only numeric values") from exc
        if timestamps.shape != (int(frame_count),):
            raise ValueError(
                f"{meta_path}: frame_timestamps_s has {timestamps.size} entries but "
                f"the video has {frame_count} decoded frames"
            )
        if not np.isfinite(timestamps).all():
            raise ValueError(f"{meta_path}: frame_timestamps_s contains NaN or infinity")
        if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"{meta_path}: frame_timestamps_s must be strictly increasing")
        if "first_frame_ts" in meta:
            try:
                first_frame_ts = float(meta["first_frame_ts"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{meta_path}: first_frame_ts must be numeric") from exc
            if not np.isfinite(first_frame_ts) or not np.isclose(
                first_frame_ts,
                timestamps[0],
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError(f"{meta_path}: first_frame_ts must equal frame_timestamps_s[0]")
        if "episode_start_ts" in meta:
            try:
                episode_start_ts = float(meta["episode_start_ts"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{meta_path}: episode_start_ts must be numeric") from exc
            if not np.isfinite(episode_start_ts) or episode_start_ts > timestamps[0]:
                raise ValueError(
                    f"{meta_path}: episode_start_ts must be finite and no later than the first video frame"
                )
        return timestamps, reported_fps

    try:
        start = float(meta.get("first_frame_ts", meta.get("start_ts", 0.0)))
        fps = float(meta.get("fps", reported_fps))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{meta_path}: start_ts and fps must be numeric") from exc
    if not np.isfinite(start):
        raise ValueError(f"{meta_path}: start_ts must be finite")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            f"{mp4_path}: video fps must be positive and finite; "
            f"provide a valid 'fps' in {meta_path.name} if the container has none"
        )

    timestamps = start + np.arange(int(frame_count), dtype=float) / fps
    if not np.isfinite(timestamps).all():
        raise ValueError(f"{mp4_path}: generated video timestamps are not finite")
    return timestamps, fps


def _read_video_frames(path: Path) -> np.ndarray:
    """Decode a video into same-sized BGR frames and reject empty/corrupt input."""
    import cv2

    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"{path}: OpenCV could not open the video")
    reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    expected_shape = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"{path}: decoded an invalid video frame")
            if expected_shape is None:
                expected_shape = frame.shape
            elif frame.shape != expected_shape:
                raise ValueError(f"{path}: frame dimensions changed from {expected_shape} to {frame.shape}")
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"{path}: no frames decoded")
    if reported_count > 0 and reported_count != len(frames):
        raise ValueError(f"{path}: container reports {reported_count} frames but only {len(frames)} decoded")
    return np.stack(frames)


def _nearest_indices(reference_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    """Indices of truly nearest reference timestamps; ties choose the earlier frame."""
    reference_t = np.asarray(reference_t, dtype=float)
    query_t = np.asarray(query_t, dtype=float)
    if reference_t.ndim != 1 or reference_t.size == 0:
        raise ValueError("reference_t must be a non-empty one-dimensional array")
    if query_t.ndim != 1:
        raise ValueError("query_t must be one-dimensional")
    if not np.isfinite(reference_t).all() or not np.isfinite(query_t).all():
        raise ValueError("timestamp arrays must contain only finite values")
    if np.any(np.diff(reference_t) <= 0):
        raise ValueError("reference_t must be strictly increasing")

    right = np.searchsorted(reference_t, query_t, side="left")
    left = np.clip(right - 1, 0, len(reference_t) - 1)
    right = np.clip(right, 0, len(reference_t) - 1)
    choose_right = np.abs(reference_t[right] - query_t) < np.abs(query_t - reference_t[left])
    return np.where(choose_right, right, left)


def _count_decodable_frames(path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return 0
    count = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
    finally:
        cap.release()
    return count


def _write_video(path: Path, frames: np.ndarray, indices: np.ndarray, fps: int) -> None:
    """Write selected BGR frames atomically and verify all output frames decode."""
    import cv2

    path = Path(path)
    frames = np.asarray(frames)
    indices = np.asarray(indices)
    if frames.ndim != 4 or frames.shape[0] == 0 or frames.shape[3] != 3:
        raise ValueError(f"{path}: frames must have shape (T, H, W, 3), got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"{path}: frames must have uint8 dtype, got {frames.dtype}")
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{path}: refusing to write an empty video")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{path}: frame indices must be integers")
    if np.any(indices < 0) or np.any(indices >= len(frames)):
        raise ValueError(f"{path}: frame indices are outside [0, {len(frames) - 1}]")
    if isinstance(fps, bool) or not isinstance(fps, (int, np.integer)) or fps <= 0:
        raise ValueError(f"{path}: fps must be a positive integer, got {fps!r}")
    if not path.parent.is_dir():
        raise NotADirectoryError(f"Video output directory not found: {path.parent}")

    h, w = frames.shape[1:3]
    tmp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()

    writer = None
    try:
        writer = cv2.VideoWriter(
            str(tmp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (w, h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"{path}: OpenCV could not open the output video writer")
        for idx in indices:
            writer.write(frames[int(idx)])
        writer.release()
        writer = None
        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise RuntimeError(f"{path}: video writer produced no output")
        decoded_count = _count_decodable_frames(tmp_path)
        if decoded_count != len(indices):
            raise RuntimeError(f"{path}: wrote {len(indices)} frames but only {decoded_count} can be decoded")
        tmp_path.replace(path)
    except BaseException:
        if writer is not None:
            writer.release()
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _write_resampled_video(
    source_path: Path,
    output_path: Path,
    indices: np.ndarray,
    fps: int,
) -> None:
    """Stream selected source frames to an atomic output without stacking a video."""
    import cv2

    source_path = Path(source_path)
    output_path = Path(output_path)
    indices = np.asarray(indices)
    if not source_path.is_file():
        raise FileNotFoundError(f"Video file not found: {source_path}")
    if (
        indices.ndim != 1
        or indices.size == 0
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(np.diff(indices) < 0)
    ):
        raise ValueError("streaming video indices must be non-empty, non-negative, and sorted")
    if isinstance(fps, bool) or not isinstance(fps, (int, np.integer)) or fps <= 0:
        raise ValueError(f"{output_path}: fps must be a positive integer, got {fps!r}")
    if not output_path.parent.is_dir():
        raise NotADirectoryError(f"Video output directory not found: {output_path.parent}")

    tmp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"{source_path}: OpenCV could not open the video")
    writer = None
    selection_index = 0
    source_index = 0
    try:
        while selection_index < len(indices):
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"{source_path}: decoded an invalid video frame")
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(tmp_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {tmp_path}")
            while selection_index < len(indices) and int(indices[selection_index]) == source_index:
                writer.write(frame)
                selection_index += 1
            source_index += 1
        if selection_index != len(indices):
            raise ValueError(
                f"{source_path}: requested source frame {int(indices[selection_index])}, "
                f"but only {source_index} frame(s) decoded"
            )
        if writer is None:
            raise ValueError(f"{source_path}: no frames decoded")
        writer.release()
        writer = None
        capture.release()
        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Video writer produced no data: {tmp_path}")
        decoded_count = _count_decodable_frames(tmp_path)
        if decoded_count != len(indices):
            raise RuntimeError(f"{tmp_path}: wrote {len(indices)} frames but decoded {decoded_count}")
        tmp_path.replace(output_path)
    except BaseException:
        capture.release()
        if writer is not None:
            writer.release()
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _interp(t_grid: np.ndarray, t: np.ndarray, vals: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_grid, t, vals[:, d]) for d in range(vals.shape[1])], axis=1)


def _rotation_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return a ZYX yaw/pitch/roll rotation matrix."""

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _rotation_to_zyx(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to one canonical ZYX representation."""

    pitch = math.atan2(
        -float(rotation[2, 0]),
        math.hypot(float(rotation[0, 0]), float(rotation[1, 0])),
    )
    if abs(math.cos(pitch)) > 1e-8:
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    else:
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
        roll = 0.0
    return np.asarray([yaw, pitch, roll], dtype=float)


def _validate_frame(frame: str, field: str) -> FrameName:
    if frame not in ("base", "tool"):
        raise ValueError(f"{field} must be 'base' or 'tool', got {frame!r}")
    return frame


def _validate_optional_positive_limit(value: float | None, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field} must be a finite positive number or null")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be a finite positive number or null")
    return result


def _validate_action_envelope(
    action: np.ndarray,
    *,
    max_translation_step_m: float | None,
    max_rotation_step_rad: float | None,
    context: str,
) -> None:
    translation_norm = float(np.linalg.norm(action[:3]))
    rotation_norm = float(np.linalg.norm(action[3:]))
    if max_translation_step_m is not None and translation_norm > max_translation_step_m:
        raise ValueError(
            f"{context}: translation step {translation_norm:.6f} m exceeds "
            f"{max_translation_step_m:.6f} m; increase --out_fps instead of "
            "placing an unsafe combined step in one training frame"
        )
    if max_rotation_step_rad is not None and rotation_norm > max_rotation_step_rad:
        raise ValueError(
            f"{context}: rotation step {rotation_norm:.6f} rad exceeds "
            f"{max_rotation_step_rad:.6f} rad; increase --out_fps instead of "
            "placing an unsafe combined step in one training frame"
        )


def _validate_deployment_velocity(
    action: np.ndarray,
    *,
    output_period_s: float | None,
    max_translation_velocity_m_s: float | None,
    max_rotation_velocity_rad_s: float | None,
    context: str,
) -> None:
    if max_translation_velocity_m_s is None and max_rotation_velocity_rad_s is None:
        return
    if output_period_s is None:
        raise ValueError("output_period_s is required when deployment velocity limits are provided")
    translation_velocity = float(np.linalg.norm(action[:3])) / output_period_s
    rotation_velocity = float(np.linalg.norm(action[3:])) / output_period_s
    if max_translation_velocity_m_s is not None and translation_velocity > max_translation_velocity_m_s:
        raise ValueError(
            f"{context}: nominal deployment translation velocity "
            f"{translation_velocity:.6f} m/s exceeds "
            f"{max_translation_velocity_m_s:.6f} m/s; increase --out_fps"
        )
    if max_rotation_velocity_rad_s is not None and rotation_velocity > max_rotation_velocity_rad_s:
        raise ValueError(
            f"{context}: nominal deployment rotation velocity "
            f"{rotation_velocity:.6f} rad/s exceeds "
            f"{max_rotation_velocity_rad_s:.6f} rad/s; increase --out_fps"
        )


def _compose_increment_bin(
    increments: np.ndarray,
    *,
    translation_frame: FrameName,
    rotation_frame: FrameName,
) -> np.ndarray:
    """Compose one timestamp-ordered bin using the bridge's exact delta semantics."""

    pose = np.eye(4, dtype=float)
    saw_prior_rotation = False
    for action in increments:
        # Tool translations followed by base-frame rotations do not have a
        # pose-independent six-vector composition: D @ R0 @ d depends on R0.
        # A single command remains exact, as do translations before any rotation.
        if (
            translation_frame == "tool"
            and rotation_frame == "base"
            and saw_prior_rotation
            and np.linalg.norm(action[:3]) > 0
        ):
            raise ValueError(
                "cannot aggregate a tool-frame translation after a base-frame rotation "
                "without the bin's initial FK pose; increase --out_fps so these commands "
                "land in separate output intervals"
            )

        delta_position = action[:3]
        if translation_frame == "tool":
            delta_position = pose[:3, :3] @ delta_position
        pose[:3, 3] += delta_position

        delta_rotation = _rotation_zyx(*action[3:])
        if rotation_frame == "tool":
            pose[:3, :3] = pose[:3, :3] @ delta_rotation
        else:
            pose[:3, :3] = delta_rotation @ pose[:3, :3]
        saw_prior_rotation = saw_prior_rotation or bool(np.linalg.norm(action[3:]) > 0)

    return np.concatenate([pose[:3, 3], _rotation_to_zyx(pose[:3, :3])])


def _compose_increments(
    t_grid: np.ndarray,
    interval_end_s: float,
    timestamps: np.ndarray,
    increments: np.ndarray,
    *,
    translation_frame: FrameName = "base",
    rotation_frame: FrameName = "tool",
    max_translation_step_m: float | None = None,
    max_rotation_step_rad: float | None = None,
    output_period_s: float | None = None,
    max_translation_velocity_m_s: float | None = None,
    max_rotation_velocity_rad_s: float | None = None,
) -> np.ndarray:
    """Compose discrete six-dimensional increments in each half-open interval.

    Output row ``i`` contains every increment with a timestamp in
    ``[t_grid[i], t_grid[i + 1])``.  The final row covers
    ``[t_grid[-1], interval_end_s)``. Commands retain timestamp order, so rotations
    are not incorrectly treated as commutative Euler-vector addition.
    """

    translation_frame = _validate_frame(translation_frame, "translation_frame")
    rotation_frame = _validate_frame(rotation_frame, "rotation_frame")
    max_translation_step_m = _validate_optional_positive_limit(
        max_translation_step_m,
        "max_translation_step_m",
    )
    max_rotation_step_rad = _validate_optional_positive_limit(
        max_rotation_step_rad,
        "max_rotation_step_rad",
    )
    output_period_s = _validate_optional_positive_limit(
        output_period_s,
        "output_period_s",
    )
    max_translation_velocity_m_s = _validate_optional_positive_limit(
        max_translation_velocity_m_s,
        "max_translation_velocity_m_s",
    )
    max_rotation_velocity_rad_s = _validate_optional_positive_limit(
        max_rotation_velocity_rad_s,
        "max_rotation_velocity_rad_s",
    )
    t_grid = np.asarray(t_grid, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    increments = np.asarray(increments, dtype=float)
    if t_grid.ndim != 1 or t_grid.size == 0 or np.any(np.diff(t_grid) <= 0):
        raise ValueError("t_grid must be a non-empty strictly increasing array")
    if (
        timestamps.ndim != 1
        or increments.ndim != 2
        or increments.shape[1] != 6
        or len(timestamps) != len(increments)
    ):
        raise ValueError("timestamps and increments must have shapes (T,) and (T,6)")
    if not np.isfinite(interval_end_s) or interval_end_s <= t_grid[-1]:
        raise ValueError("interval_end_s must be finite and later than the final grid point")
    if not np.isfinite(timestamps).all() or not np.isfinite(increments).all():
        raise ValueError("increment timestamps and values must be finite")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("increment timestamps must be strictly increasing")

    result = np.zeros((len(t_grid), 6), dtype=float)
    in_range = (timestamps >= t_grid[0]) & (timestamps < interval_end_s)
    selected_timestamps = timestamps[in_range]
    selected_increments = increments[in_range]
    if selected_timestamps.size:
        bins = np.searchsorted(t_grid, selected_timestamps, side="right") - 1
        for source_index, action in enumerate(selected_increments):
            _validate_action_envelope(
                action,
                max_translation_step_m=max_translation_step_m,
                max_rotation_step_rad=max_rotation_step_rad,
                context=f"raw command at timestamp {selected_timestamps[source_index]:.9f}",
            )
        for bin_index in np.unique(bins):
            composed = _compose_increment_bin(
                selected_increments[bins == bin_index],
                translation_frame=translation_frame,
                rotation_frame=rotation_frame,
            )
            # LeRobot stores actions as float32. Validate the exact quantized value
            # that training/deployment will emit, not a more permissive float64 precursor.
            result[bin_index] = composed.astype(np.float32).astype(float)
            _validate_action_envelope(
                result[bin_index],
                max_translation_step_m=max_translation_step_m,
                max_rotation_step_rad=max_rotation_step_rad,
                context=f"aggregated output interval starting at {t_grid[bin_index]:.9f}",
            )
            _validate_deployment_velocity(
                result[bin_index],
                output_period_s=output_period_s,
                max_translation_velocity_m_s=max_translation_velocity_m_s,
                max_rotation_velocity_rad_s=max_rotation_velocity_rad_s,
                context=f"aggregated output interval starting at {t_grid[bin_index]:.9f}",
            )
    return result


def _sum_increments(
    t_grid: np.ndarray,
    interval_end_s: float,
    timestamps: np.ndarray,
    increments: np.ndarray,
) -> np.ndarray:
    """Compatibility alias for the corrected ordered SE(3) aggregation."""

    return _compose_increments(t_grid, interval_end_s, timestamps, increments)


def sync_episode(
    ep_in: Path,
    ep_out: Path,
    out_fps: int,
    *,
    max_episode_duration_s: float = 600.0,
    max_output_frames: int = 18_000,
    translation_frame: FrameName = "base",
    rotation_frame: FrameName = "tool",
    max_translation_step_m: float | None = None,
    max_rotation_step_rad: float | None = None,
    max_translation_velocity_m_s: float | None = None,
    max_rotation_velocity_rad_s: float | None = None,
) -> None:
    ep_in = Path(ep_in)
    ep_out = Path(ep_out)
    if not ep_in.is_dir():
        raise NotADirectoryError(f"Episode input directory not found: {ep_in}")
    if ep_out.exists() or ep_out.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing episode output: {ep_out}")
    if ep_in.resolve() == ep_out.resolve():
        raise ValueError("Episode input and output directories must be different")
    if isinstance(out_fps, bool) or not isinstance(out_fps, (int, np.integer)) or out_fps <= 0:
        raise ValueError(f"out_fps must be a positive integer, got {out_fps!r}")
    if (
        isinstance(max_episode_duration_s, bool)
        or not isinstance(max_episode_duration_s, (int, float))
        or not np.isfinite(max_episode_duration_s)
        or max_episode_duration_s <= 0
    ):
        raise ValueError("max_episode_duration_s must be finite and positive")
    if (
        isinstance(max_output_frames, bool)
        or not isinstance(max_output_frames, (int, np.integer))
        or max_output_frames <= 0
    ):
        raise ValueError("max_output_frames must be a positive integer")
    translation_frame = _validate_frame(translation_frame, "translation_frame")
    rotation_frame = _validate_frame(rotation_frame, "rotation_frame")
    max_translation_step_m = _validate_optional_positive_limit(
        max_translation_step_m,
        "max_translation_step_m",
    )
    max_rotation_step_rad = _validate_optional_positive_limit(
        max_rotation_step_rad,
        "max_rotation_step_rad",
    )
    max_translation_velocity_m_s = _validate_optional_positive_limit(
        max_translation_velocity_m_s,
        "max_translation_velocity_m_s",
    )
    max_rotation_velocity_rad_s = _validate_optional_positive_limit(
        max_rotation_velocity_rad_s,
        "max_rotation_velocity_rad_s",
    )

    has_video = (ep_in / "video.mp4").is_file()
    input_names = ["robot.csv", "phone.csv", "applied_actions.csv"]
    if has_video:
        input_names.append("video.mp4")
        if (ep_in / "video_meta.json").is_file():
            input_names.append("video_meta.json")
    if (ep_in / "frame_timestamps.csv").is_file():
        input_names.append("frame_timestamps.csv")
    source_episode_name = ep_in.name
    snapshot_owner, snapshot_episode, input_fingerprints = _snapshot_episode_inputs(
        ep_in,
        input_names,
        owner_parent=ep_out.parent,
        owner_prefix=f".{ep_out.name}.input-snapshot-",
    )
    try:
        _sync_episode_from_snapshot(
            snapshot_episode,
            ep_out,
            out_fps,
            source_episode_name=source_episode_name,
            input_fingerprints=input_fingerprints,
            max_episode_duration_s=max_episode_duration_s,
            max_output_frames=max_output_frames,
            translation_frame=translation_frame,
            rotation_frame=rotation_frame,
            max_translation_step_m=max_translation_step_m,
            max_rotation_step_rad=max_rotation_step_rad,
            max_translation_velocity_m_s=max_translation_velocity_m_s,
            max_rotation_velocity_rad_s=max_rotation_velocity_rad_s,
        )
    finally:
        _cleanup_owned_input_snapshot(snapshot_owner)


def _sync_episode_from_snapshot(
    ep_in: Path,
    ep_out: Path,
    out_fps: int,
    *,
    source_episode_name: str,
    input_fingerprints: dict[str, dict[str, str]],
    max_episode_duration_s: float,
    max_output_frames: int,
    translation_frame: FrameName,
    rotation_frame: FrameName,
    max_translation_step_m: float | None,
    max_rotation_step_rad: float | None,
    max_translation_velocity_m_s: float | None,
    max_rotation_velocity_rad_s: float | None,
) -> None:
    """Synchronize exclusively from an already verified process-owned snapshot."""

    import pandas as pd

    has_video = (ep_in / "video.mp4").is_file()
    robot_t, robot_v, robot_cols = _read_ts_csv(ep_in / "robot.csv")
    if not (ep_in / "phone.csv").is_file():
        raise FileNotFoundError(f"Audit stream not found in input snapshot: {ep_in / 'phone.csv'}")
    applied_path = ep_in / "applied_actions.csv"
    phone_t, phone_v, phone_cols = _read_ts_csv(applied_path)
    if phone_cols != ACTION_COLUMNS:
        raise ValueError(
            f"{applied_path}: action columns must be exactly {ACTION_COLUMNS} in this order; got {phone_cols}"
        )

    starts = [robot_t[0], phone_t[0]]
    # Phone rows are instantaneous increments rather than a continuously sampled
    # stream. Their last timestamp therefore does not define episode coverage.
    ends = [robot_t[-1]]

    if has_video:
        frame_count = _count_decodable_frames(ep_in / "video.mp4")
        if frame_count <= 0:
            raise ValueError(f"{ep_in / 'video.mp4'}: no frames decoded")
        vid_t, _ = _video_timestamps(
            ep_in / "video_meta.json",
            ep_in / "video.mp4",
            frame_count=frame_count,
        )
        starts.append(vid_t[0])
        ends.append(vid_t[-1])
    t0, t1 = max(starts), min(ends)
    if t1 <= t0:
        raise ValueError(f"{source_episode_name}: streams do not overlap in time [{t0}, {t1}]")
    duration_s = t1 - t0
    if duration_s > max_episode_duration_s:
        raise ValueError(
            f"{source_episode_name}: overlap duration {duration_s:.3f}s exceeds "
            f"max_episode_duration_s={max_episode_duration_s}; check timestamp units"
        )
    estimated_frames = int(np.ceil(duration_s * out_fps))
    if estimated_frames > max_output_frames:
        raise ValueError(
            f"{source_episode_name}: output would contain about {estimated_frames} frames, exceeding "
            f"max_output_frames={max_output_frames}"
        )
    t_grid = np.arange(t0, t1, 1.0 / out_fps, dtype=float)
    if t_grid.size == 0:
        raise ValueError(f"{source_episode_name}: overlap is too short to produce an output frame")

    states = _interp(t_grid, robot_t, robot_v)  # robot: linear
    actions = _compose_increments(
        t_grid,
        t1,
        phone_t,
        phone_v,
        translation_frame=translation_frame,
        rotation_frame=rotation_frame,
        max_translation_step_m=max_translation_step_m,
        max_rotation_step_rad=max_rotation_step_rad,
        output_period_s=1.0 / out_fps,
        max_translation_velocity_m_s=max_translation_velocity_m_s,
        max_rotation_velocity_rad_s=max_rotation_velocity_rad_s,
    )
    nearest = _nearest_indices(vid_t, t_grid) if has_video else None

    ep_out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{ep_out.name}.staging-",
            dir=ep_out.parent,
        )
    )
    try:
        if has_video:
            _write_resampled_video(
                ep_in / "video.mp4",
                staging / "video.mp4",
                nearest,
                out_fps,
            )

        pd.DataFrame(
            np.column_stack([t_grid, states]),
            columns=["timestamp", *robot_cols],
        ).to_csv(staging / "states.csv", index=False)
        pd.DataFrame(
            np.column_stack([t_grid, actions]),
            columns=["timestamp", *phone_cols],
        ).to_csv(staging / "actions.csv", index=False)
        (staging / "source_fingerprints.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "hash_algorithm": "sha256",
                    "source_episode": source_episode_name,
                    "files": input_fingerprints,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        required_outputs = [
            staging / "states.csv",
            staging / "actions.csv",
            staging / "source_fingerprints.json",
        ]
        if has_video:
            required_outputs.append(staging / "video.mp4")
        if any(not path.is_file() or path.stat().st_size == 0 for path in required_outputs):
            raise RuntimeError(f"{source_episode_name}: staging output verification failed")
        if ep_out.exists() or ep_out.is_symlink():
            raise FileExistsError(
                f"episode output appeared while synchronizing; refusing to overwrite: {ep_out}"
            )
        staging.rename(ep_out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        f"  {source_episode_name}: {len(t_grid)} aligned frames over "
        f"{t1 - t0:.2f}s{' + video' if has_video else ''}"
    )


def _load_bridge_action_config(
    path: Path,
) -> tuple[FrameName, FrameName, float, float, float, float]:
    """Read action semantics plus verified step/velocity bounds for resampling."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc.msg})") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("safety"), dict):
        raise ValueError(f"{path}: expected an object with a 'safety' object")
    safety = payload["safety"]
    translation_frame = _validate_frame(
        safety.get("translation_frame", "base"),
        "safety.translation_frame",
    )
    rotation_frame = _validate_frame(
        safety.get("rotation_frame", "tool"),
        "safety.rotation_frame",
    )
    max_translation = _validate_optional_positive_limit(
        safety.get("max_translation_step_m"),
        "safety.max_translation_step_m",
    )
    max_rotation = _validate_optional_positive_limit(
        safety.get("max_rotation_step_rad"),
        "safety.max_rotation_step_rad",
    )
    max_translation_velocity = _validate_optional_positive_limit(
        safety.get("max_translation_velocity_m_s"),
        "safety.max_translation_velocity_m_s",
    )
    max_rotation_velocity = _validate_optional_positive_limit(
        safety.get("max_rotation_velocity_rad_s"),
        "safety.max_rotation_velocity_rad_s",
    )
    if (
        max_translation is None
        or max_rotation is None
        or max_translation_velocity is None
        or max_rotation_velocity is None
    ):
        raise ValueError(f"{path}: Cartesian step and velocity safety limits are required")
    return (
        translation_frame,
        rotation_frame,
        max_translation,
        max_rotation,
        max_translation_velocity,
        max_rotation_velocity,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in_dir", required=True, help="Dir of episode_* folders with raw timestamped streams."
    )
    parser.add_argument("--out_dir", required=True, help="Output dir of aligned episode_* folders.")
    parser.add_argument("--out_fps", type=int, default=30)
    parser.add_argument("--max_episode_duration_s", type=float, default=600.0)
    parser.add_argument("--max_output_frames", type=int, default=18_000)
    parser.add_argument(
        "--robot_bridge_config",
        required=True,
        help=("Verified robot bridge JSON supplying translation/rotation frames and per-step safety limits."),
    )
    args = parser.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"Input directory not found: {in_dir}")
    if out_dir.exists() or out_dir.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output directory: {out_dir}")
    if in_dir.resolve() == out_dir.resolve():
        raise SystemExit("Input and output directories must be different")
    if args.out_fps <= 0:
        raise SystemExit(f"--out_fps must be positive, got {args.out_fps}")
    try:
        (
            translation_frame,
            rotation_frame,
            max_translation_step_m,
            max_rotation_step_rad,
            max_translation_velocity_m_s,
            max_rotation_velocity_rad_s,
        ) = _load_bridge_action_config(Path(args.robot_bridge_config))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Invalid --robot_bridge_config: {exc}") from exc

    all_dirs = sorted(p for p in in_dir.iterdir() if p.is_dir())
    named_episode_dirs = [p for p in all_dirs if p.name.startswith("episode_")]
    eps = named_episode_dirs or [p for p in all_dirs if (p / "robot.csv").exists()]
    if not eps:
        raise SystemExit(f"No episode_* folders with robot.csv under {in_dir}")
    for ep in eps:
        missing = [
            name for name in ("robot.csv", "phone.csv", "applied_actions.csv") if not (ep / name).is_file()
        ]
        if missing:
            raise SystemExit(f"{ep}: missing required file(s): {', '.join(missing)}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{out_dir.name}.staging-",
            dir=out_dir.parent,
        )
    )
    print(f"Aligning {len(eps)} episodes at {args.out_fps} Hz")
    try:
        for ep in eps:
            sync_episode(
                ep,
                staging_root / ep.name,
                args.out_fps,
                max_episode_duration_s=args.max_episode_duration_s,
                max_output_frames=args.max_output_frames,
                translation_frame=translation_frame,
                rotation_frame=rotation_frame,
                max_translation_step_m=max_translation_step_m,
                max_rotation_step_rad=max_rotation_step_rad,
                max_translation_velocity_m_s=max_translation_velocity_m_s,
                max_rotation_velocity_rad_s=max_rotation_velocity_rad_s,
            )
        if out_dir.exists() or out_dir.is_symlink():
            raise FileExistsError(
                f"output directory appeared while synchronizing; refusing to overwrite: {out_dir}"
            )
        staging_root.rename(out_dir)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    print(f"\nDone. Aligned episodes in {out_dir}. Feed them to convert_raw_to_lerobot.py.")


if __name__ == "__main__":
    main()

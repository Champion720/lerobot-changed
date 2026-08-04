#!/usr/bin/env python
"""Extract per-episode trajectory features for the display-condition experiment.

The end-effector pose ``[x, y, z, yaw, pitch, roll]`` is read from
``observation.state``.  The output contains speed, smoothness, optional expert-reference
accuracy, and optional success/failure labels.

``raw_duration_s`` is always retained. ``completion_time_s`` is populated only when a
validated label says ``success=1``; failed, timed-out, and unlabeled trials are NaN so they
cannot silently enter the successful-task completion-time analysis. If labels provide an
explicit ``completion_time_s``, that recorded value is preserved as
``reported_completion_time_s`` and used instead of raw episode duration. Too-short or
invalid trajectories remain in the output with ``quality_status`` so success/failure rates
cannot be biased by silently dropping bad recordings.

Example:
    uv run --extra training python experiments/camera_ablation/feature_extraction.py \
        --repo_id local/cond_b_pc --condition B_pc --labels_csv labels_B.csv \
        --out outputs/features_B.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    from .analysis_reproducibility import (
        fingerprint_inputs,
        fingerprint_path,
        git_state,
        runtime_environment,
        write_json,
    )
except ImportError:  # Direct ``python path/to/feature_extraction.py`` execution.
    from analysis_reproducibility import (  # type: ignore[no-redef]
        fingerprint_inputs,
        fingerprint_path,
        git_state,
        runtime_environment,
        write_json,
    )

EE_NAMES = ["ee_x", "ee_y", "ee_z", "ee_yaw", "ee_pitch", "ee_roll"]
SMOOTHING_WINDOW_S = 0.15
SMOOTHING_POLYORDER = 3
SAMPLING_UNIFORM_REL_TOL = 0.05
SAMPLING_UNIFORM_ABS_TOL_S = 1e-9
MIN_SAVGOL_FRAMES = 5
MISSING_IDENTIFIER_TOKENS = {"", "nan", "none", "<na>", "null"}


def _ee_indices(state_names: list[str]) -> list[int]:
    missing = [name for name in EE_NAMES if name not in state_names]
    if missing:
        raise SystemExit(
            f"state is missing end-effector columns {missing}. "
            "Convert with --dh_config so FK pose is included in observation.state."
        )
    return [state_names.index(name) for name in EE_NAMES]


def _deriv(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Differentiate along axis 0 with non-uniform central differences."""
    return np.gradient(values, timestamps, axis=0, edge_order=2)


def _smooth_position(position: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Apply a fixed-time Savitzky-Golay filter before numerical derivatives."""

    quality_error = _sampling_quality_error(timestamps)
    if quality_error is not None:
        raise ValueError(quality_error)
    intervals = np.diff(timestamps)
    median_dt = float(np.median(intervals))
    requested = max(MIN_SAVGOL_FRAMES, int(round(SMOOTHING_WINDOW_S / median_dt)))
    if requested % 2 == 0:
        requested += 1
    largest_odd = len(position) if len(position) % 2 == 1 else len(position) - 1
    window = min(requested, largest_odd)
    if window < MIN_SAVGOL_FRAMES:
        raise ValueError(f"smoothness metrics require at least {MIN_SAVGOL_FRAMES} frames")

    from scipy.signal import savgol_filter

    polyorder = min(SMOOTHING_POLYORDER, window - 2)
    return savgol_filter(
        position,
        window_length=window,
        polyorder=polyorder,
        axis=0,
        mode="interp",
    )


def _quaternion_angular_speed(
    yaw_pitch_roll: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Return geodesic SO(3) speed between adjacent orientation samples."""

    quaternions = _ypr_to_quat(yaw_pitch_roll)
    dots = np.clip(
        np.abs(np.sum(quaternions[:-1] * quaternions[1:], axis=1)),
        0.0,
        1.0,
    )
    return 2.0 * np.arccos(dots) / np.diff(timestamps)


def smoothness_speed_features(pose: np.ndarray, timestamps: np.ndarray) -> dict[str, float]:
    """Calculate pre-registered scalar speed/smoothness metrics.

    Position is smoothed with a 0.15 s, cubic Savitzky-Golay filter before
    differentiation. Angular speed uses adjacent quaternion geodesic distance,
    not derivatives of Euler components.
    """

    position = pose[:, :3]
    filtered_position = _smooth_position(position, timestamps)

    velocity = _deriv(filtered_position, timestamps)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = _deriv(velocity, timestamps)
    jerk = _deriv(acceleration, timestamps)

    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    angular_speed = _quaternion_angular_speed(pose[:, 3:6], timestamps)
    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    path_length = float(np.sum(np.linalg.norm(np.diff(position, axis=0), axis=1)))
    peak_speed = float(speed.max()) if len(speed) else 0.0

    # Log dimensionless jerk: larger (less negative) values indicate smoother motion.
    log_dimensionless_jerk = np.nan
    if duration > 0 and peak_speed > 1e-9:
        trapezoid = getattr(np, "trapezoid", np.trapz)
        jerk_squared_integral = float(trapezoid(jerk_magnitude**2, timestamps))
        if jerk_squared_integral > 0:
            log_dimensionless_jerk = -np.log((duration**3 / peak_speed**2) * jerk_squared_integral)

    return {
        "raw_duration_s": duration,
        "path_length_m": path_length,
        "mean_speed": float(speed.mean()),
        "peak_speed": peak_speed,
        "mean_angular_speed": float(angular_speed.mean()),
        "mean_acc": float(acceleration_magnitude.mean()),
        "acc_sd": (float(acceleration_magnitude.std(ddof=1)) if len(acceleration_magnitude) > 1 else 0.0),
        "mean_jerk": float(jerk_magnitude.mean()),
        "jerk_sd": (float(jerk_magnitude.std(ddof=1)) if len(jerk_magnitude) > 1 else 0.0),
        "ang_speed_sd": (float(angular_speed.std(ddof=1)) if len(angular_speed) > 1 else 0.0),
        "log_dimensionless_jerk": float(log_dimensionless_jerk),
    }


def _resample_traj(pose: np.ndarray, size: int) -> np.ndarray:
    """Linearly resample a ``(T, 6)`` trajectory on normalized time."""
    pose = np.asarray(pose, dtype=float).copy()
    # Interpolating wrapped Euler angles directly can turn a short crossing at +/-pi into
    # an artificial full rotation.
    pose[:, 3:6] = np.unwrap(pose[:, 3:6], axis=0)
    source = np.linspace(0.0, 1.0, len(pose))
    destination = np.linspace(0.0, 1.0, size)
    return np.stack(
        [np.interp(destination, source, pose[:, dim]) for dim in range(pose.shape[1])],
        axis=1,
    )


def _ypr_to_quat(yaw_pitch_roll: np.ndarray) -> np.ndarray:
    """Convert intrinsic ZYX Euler angles to ``(w, x, y, z)`` quaternions."""
    yaw = yaw_pitch_roll[..., 0]
    pitch = yaw_pitch_roll[..., 1]
    roll = yaw_pitch_roll[..., 2]
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    return np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=-1,
    )


def _quat_to_ypr(quaternions: np.ndarray) -> np.ndarray:
    """Convert normalized ``(w, x, y, z)`` quaternions to intrinsic ZYX Euler."""

    quaternions = np.asarray(quaternions, dtype=float)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("quaternions must be finite and non-zero")
    w, x, y, z = np.moveaxis(quaternions / norms, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return np.stack([yaw, pitch, roll], axis=-1)


def _markley_quaternion_mean(quaternions: np.ndarray) -> np.ndarray:
    """Return the unweighted Markley mean of equivalent unit quaternions."""

    values = np.asarray(quaternions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) == 0:
        raise ValueError("quaternion mean requires a non-empty (N, 4) array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("quaternion mean input must be finite and non-zero")
    values = values / norms
    anchor = values[0]
    aligned = np.where((values @ anchor)[:, None] < 0, -values, values)
    eigenvalues, eigenvectors = np.linalg.eigh(aligned.T @ aligned)
    mean = eigenvectors[:, int(np.argmax(eigenvalues))]
    if float(mean @ anchor) < 0:
        mean = -mean
    return mean / np.linalg.norm(mean)


def build_reference(ref_episodes: dict[int, tuple[np.ndarray, np.ndarray]], size: int) -> np.ndarray:
    """Average positions in R3 and orientations intrinsically on SO(3)."""
    stacked = np.stack([_resample_traj(pose, size) for pose, _ in ref_episodes.values()])
    reference = np.empty((size, 6), dtype=float)
    reference[:, :3] = stacked[:, :, :3].mean(axis=0)
    quaternions = _ypr_to_quat(stacked[:, :, 3:6])
    reference[:, 3:6] = _quat_to_ypr(
        np.stack([_markley_quaternion_mean(quaternions[:, index]) for index in range(size)])
    )
    return reference


def _normalize_identifier(series: pd.Series, field: str, *, source: str = "") -> pd.Series:
    """Normalize identifiers while rejecting null sentinels before string coercion."""

    prefix = f"{source} " if source else ""
    if series.isna().any():
        raise ValueError(f"{prefix}{field} must be populated")
    values = series.astype(str).str.strip()
    if values.str.lower().isin(MISSING_IDENTIFIER_TOKENS).any():
        raise ValueError(f"{prefix}{field} must be populated")
    return values


def validate_task_labels(
    labels: pd.DataFrame,
    episode_ids: list[int],
    *,
    source_name: str,
) -> pd.DataFrame:
    """Validate an exact episode -> task_id mapping."""
    missing = sorted({"episode", "task_id"} - set(labels.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")
    normalized = labels[["episode", "task_id"]].copy()
    episodes = pd.to_numeric(normalized["episode"], errors="coerce")
    if episodes.isna().any() or (episodes < 0).any() or (episodes % 1 != 0).any():
        raise ValueError(f"{source_name} episode must contain non-negative integers")
    normalized["episode"] = episodes.astype(int)
    if normalized["episode"].duplicated().any():
        raise ValueError(f"{source_name} has duplicate episode rows")
    tasks = _normalize_identifier(normalized["task_id"], "task_id", source=source_name)
    normalized["task_id"] = tasks
    expected, provided = set(episode_ids), set(normalized["episode"])
    if expected != provided:
        raise ValueError(
            f"{source_name} must map every dataset episode exactly; "
            f"missing={sorted(expected - provided)}, extra={sorted(provided - expected)}"
        )
    return normalized


def build_references_by_task(
    ref_episodes: dict[int, tuple[np.ndarray, np.ndarray]],
    task_labels: pd.DataFrame,
    size: int,
) -> dict[str, np.ndarray]:
    """Build one expert trajectory per task; never average across tasks."""
    task_by_episode = dict(zip(task_labels["episode"], task_labels["task_id"], strict=True))
    grouped: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    for episode, trajectory in ref_episodes.items():
        grouped[str(task_by_episode[episode])][episode] = trajectory
    return {task_id: build_reference(episodes, size) for task_id, episodes in grouped.items()}


def accuracy_features(
    pose: np.ndarray,
    reference_pose: np.ndarray,
    size: int,
) -> dict[str, float]:
    """Calculate position and orientation error against a normalized-time reference."""
    sampled = _resample_traj(pose, size)
    position_error = np.linalg.norm(sampled[:, :3] - reference_pose[:, :3], axis=1)
    sampled_quaternion = _ypr_to_quat(sampled[:, 3:6])
    reference_quaternion = _ypr_to_quat(reference_pose[:, 3:6])
    dot = np.clip(
        np.abs(np.sum(sampled_quaternion * reference_quaternion, axis=1)),
        -1.0,
        1.0,
    )
    orientation_error = 2.0 * np.arccos(dot)
    return {
        "mean_position_error_m": float(position_error.mean()),
        "max_position_error_m": float(position_error.max()),
        "rmse_position_m": float(np.sqrt((position_error**2).mean())),
        "mean_orientation_error_rad": float(orientation_error.mean()),
    }


def load_episodes(
    repo_id: str,
    root: str | None,
    video_backend: str,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[int, dict[str, object]],
    list[str],
    float,
]:
    """Return valid trajectories plus a quality row for every dataset episode."""
    dataset = LeRobotDataset(repo_id, root=root, video_backend=video_backend)
    state_names = dataset.meta.features["observation.state"]["names"]
    ee_indices = _ee_indices(state_names)
    fps = dataset.meta.fps

    poses: dict[int, list[np.ndarray]] = defaultdict(list)
    timestamps: dict[int, list[float]] = defaultdict(list)
    # Select raw scalar/state columns so feature extraction never decodes camera video.
    trajectory_rows = dataset.select_columns(["episode_index", "observation.state", "timestamp"])
    for item in trajectory_rows:
        episode = int(item["episode_index"])
        state = np.asarray(item["observation.state"], dtype=float)
        poses[episode].append(state[ee_indices])
        timestamps[episode].append(float(item["timestamp"]))

    episodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    quality: dict[int, dict[str, object]] = {}
    for episode in poses:
        pose = np.stack(poses[episode])
        time = np.asarray(timestamps[episode])
        episode_quality = _episode_quality(pose, time)
        quality[episode] = episode_quality
        if not episode_quality["feature_usable"]:
            print(f"  episode {episode} retained as outcome-only row: {episode_quality['quality_status']}")
            continue
        episodes[episode] = (pose, time)
    return episodes, quality, state_names, fps


def _episode_quality(pose: np.ndarray, timestamps: np.ndarray) -> dict[str, object]:
    """Classify whether derivatives are usable without dropping the episode outcome."""
    duration = np.nan
    if (
        len(timestamps) > 1
        and np.isfinite(timestamps[0])
        and np.isfinite(timestamps[-1])
        and timestamps[-1] >= timestamps[0]
    ):
        duration = float(timestamps[-1] - timestamps[0])

    sampling_error = _sampling_quality_error(timestamps)
    if sampling_error is not None:
        status = sampling_error
    elif not np.isfinite(pose).all():
        status = "invalid_pose"
    else:
        status = "ok"
    return {
        "n_frames": len(timestamps),
        "quality_status": status,
        "feature_usable": status == "ok",
        "raw_duration_s": duration,
    }


def _sampling_quality_error(timestamps: np.ndarray) -> str | None:
    """Return the exact status used by both the SG gate and feature extractor."""

    timestamps = np.asarray(timestamps, dtype=float)
    if len(timestamps) < MIN_SAVGOL_FRAMES:
        return "too_short_for_jerk"
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        return "invalid_timestamps"
    intervals = np.diff(timestamps)
    median_dt = float(np.median(intervals))
    maximum_deviation = float(np.max(np.abs(intervals - median_dt)))
    if maximum_deviation > max(
        SAMPLING_UNIFORM_ABS_TOL_S,
        SAMPLING_UNIFORM_REL_TOL * median_dt,
    ):
        return "irregular_timestamps"
    return None


def _normalize_success(value: object) -> int:
    """Normalize common binary encodings while rejecting ambiguous values."""
    if pd.isna(value):
        raise ValueError("success contains a missing value")
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
        return int(value)
    text = str(value).strip().lower()
    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "no": 0,
        "yes": 1,
        "failure": 0,
        "failed": 0,
        "success": 1,
        "successful": 1,
    }
    if text not in mapping:
        raise ValueError(f"success must be binary (0/1 or true/false), got {value!r}")
    return mapping[text]


def validate_labels(labels: pd.DataFrame, episode_ids: list[int]) -> pd.DataFrame:
    """Validate and normalize labels before a one-to-one episode merge."""
    missing_columns = sorted({"episode", "success", "participant_id"} - set(labels.columns))
    if missing_columns:
        raise ValueError(f"labels CSV is missing required columns: {missing_columns}")

    normalized = labels.copy()
    episodes = pd.to_numeric(normalized["episode"], errors="coerce")
    invalid_episode = episodes.isna() | (episodes < 0) | (episodes % 1 != 0)
    if invalid_episode.any():
        bad = normalized.loc[invalid_episode, "episode"].tolist()
        raise ValueError(f"episode must contain non-negative integers; invalid values={bad}")
    normalized["episode"] = episodes.astype(int)
    participants = _normalize_identifier(normalized["participant_id"], "participant_id")
    normalized["participant_id"] = participants
    if "task_id" in normalized.columns:
        normalized["task_id"] = _normalize_identifier(normalized["task_id"], "task_id")
    if "seed" in normalized.columns:
        seeds = _normalize_identifier(normalized["seed"], "seed")
        normalized["seed"] = seeds
    if "split" in normalized.columns:
        splits = _normalize_identifier(normalized["split"], "split")
        normalized["split"] = splits
        splits_per_participant = normalized.groupby("participant_id")["split"].nunique(dropna=False)
        crossed = splits_per_participant[splits_per_participant > 1].index.tolist()
        if crossed:
            raise ValueError(f"participant_id must not cross label splits; violations={crossed}")

    duplicated = normalized["episode"].duplicated(keep=False)
    if duplicated.any():
        bad = sorted(normalized.loc[duplicated, "episode"].unique().tolist())
        raise ValueError(f"labels CSV has duplicate episode rows: {bad}")

    try:
        normalized["success"] = normalized["success"].map(_normalize_success).astype("int8")
    except ValueError as exc:
        raise ValueError(f"invalid success label: {exc}") from exc

    if "raw_duration_s" in normalized.columns:
        raise ValueError(
            "labels CSV must not define raw_duration_s; that field is measured from dataset timestamps"
        )
    if "completion_time_s" in normalized.columns:
        original_completion = normalized["completion_time_s"]
        completion = pd.to_numeric(original_completion, errors="coerce")
        nonnumeric = original_completion.notna() & completion.isna()
        if nonnumeric.any():
            bad = original_completion.loc[nonnumeric].tolist()
            raise ValueError(f"completion_time_s contains non-numeric values: {bad}")
        invalid_provided_time = completion.notna() & (~np.isfinite(completion) | (completion <= 0))
        if invalid_provided_time.any():
            bad_episodes = normalized.loc[invalid_provided_time, "episode"].tolist()
            raise ValueError(
                "provided completion_time_s values must be finite and positive; "
                f"invalid episodes={bad_episodes}"
            )
        invalid_success_time = normalized["success"].eq(1) & (completion.isna())
        if invalid_success_time.any():
            bad_episodes = normalized.loc[invalid_success_time, "episode"].tolist()
            raise ValueError(
                "successful episodes require a finite, positive completion_time_s; "
                f"invalid episodes={bad_episodes}"
            )
        normalized = normalized.rename(columns={"completion_time_s": "reported_completion_time_s"})
        normalized["reported_completion_time_s"] = completion

    expected = set(episode_ids)
    provided = set(normalized["episode"])
    missing_labels = sorted(expected - provided)
    if missing_labels:
        raise ValueError(f"labels CSV has no label for usable episodes: {missing_labels}")
    extra_labels = sorted(provided - expected)
    if extra_labels:
        raise ValueError(
            "labels CSV contains episodes absent from the dataset; refusing to drop their "
            f"outcomes from the denominator: {extra_labels}"
        )
    return normalized.copy()


def attach_labels_and_completion_times(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Merge labels and expose completion time only for successful trials."""
    if "condition" in labels.columns:
        expected_conditions = set(features["condition"].dropna().astype(str))
        label_conditions = set(labels["condition"].dropna().astype(str))
        if label_conditions and label_conditions != expected_conditions:
            raise ValueError(
                "labels condition does not match --condition: "
                f"labels={sorted(label_conditions)}, expected={sorted(expected_conditions)}"
            )
        labels = labels.drop(columns="condition")
    merged = features.merge(labels, on="episode", how="left", validate="one_to_one")
    if "reported_completion_time_s" in merged.columns:
        completion_source = merged["reported_completion_time_s"]
        merged["completion_time_source"] = "labels_csv"
    else:
        completion_source = merged["raw_duration_s"]
        merged["completion_time_source"] = "dataset_timestamps"
    merged["completion_time_s"] = completion_source.where(merged["success"].eq(1))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--condition", required=True, help="Row label, e.g. A_mobile or B_pc.")
    parser.add_argument("--out", required=True, help="Output feature CSV.")
    parser.add_argument("--video_backend", default="pyav")
    parser.add_argument(
        "--ref_repo_id",
        default=None,
        help="Expert dataset used to build the accuracy reference.",
    )
    parser.add_argument("--ref_root", default=None)
    parser.add_argument(
        "--ref_labels_csv",
        default=None,
        help="Required with --ref_repo_id: exact expert episode,task_id mapping.",
    )
    parser.add_argument(
        "--labels_csv",
        default=None,
        help=(
            "CSV with episode,success,participant_id and experiment metadata; task_id "
            "is required when expert-reference accuracy is enabled."
        ),
    )
    parser.add_argument(
        "--resample_n",
        type=int,
        default=100,
        help="Points used for normalized-time expert alignment.",
    )
    args = parser.parse_args()
    if args.resample_n <= 0:
        raise SystemExit("--resample_n must be a positive integer")

    episodes, quality, _, fps = load_episodes(args.repo_id, args.root, args.video_backend)
    print(
        f"{args.repo_id}: {len(episodes)}/{len(quality)} feature-usable episodes, "
        f"all {len(quality)} retained for outcome analysis, fps={fps}"
    )

    labels = None
    if args.labels_csv:
        try:
            labels = validate_labels(pd.read_csv(args.labels_csv), sorted(quality))
        except ValueError as exc:
            raise SystemExit(f"Invalid labels CSV: {exc}") from exc

    references_by_task: dict[str, np.ndarray] | None = None
    task_by_episode: dict[int, str] = {}
    if args.ref_repo_id:
        if labels is None:
            raise SystemExit("--ref_repo_id requires --labels_csv with target task_id mappings")
        if not args.ref_labels_csv:
            raise SystemExit(
                "--ref_repo_id requires --ref_labels_csv; global cross-task averaging is forbidden"
            )
        reference_episodes, reference_quality, _, _ = load_episodes(
            args.ref_repo_id, args.ref_root, args.video_backend
        )
        if not reference_episodes:
            raise SystemExit(f"No usable expert episodes in {args.ref_repo_id}")
        try:
            target_tasks = validate_task_labels(labels, sorted(quality), source_name="target labels CSV")
            expert_tasks = validate_task_labels(
                pd.read_csv(args.ref_labels_csv),
                sorted(reference_quality),
                source_name="expert labels CSV",
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid task mapping: {exc}") from exc
        references_by_task = build_references_by_task(reference_episodes, expert_tasks, args.resample_n)
        task_by_episode = dict(zip(target_tasks["episode"], target_tasks["task_id"], strict=True))
        missing_reference_tasks = sorted(set(task_by_episode.values()) - set(references_by_task))
        if missing_reference_tasks:
            raise SystemExit(
                f"No usable expert reference for target task_id values: {missing_reference_tasks}"
            )
        print(
            f"Built {len(references_by_task)} task-specific expert references from "
            f"{len(reference_episodes)} demos at {args.resample_n} points "
            f"({len(reference_quality) - len(reference_episodes)} unusable demos excluded)"
        )

    rows: list[dict[str, object]] = []
    for episode, quality_row in sorted(quality.items()):
        features: dict[str, object] = {
            "condition": args.condition,
            "episode": episode,
            **quality_row,
        }
        if quality_row["feature_usable"]:
            pose, timestamps = episodes[episode]
            try:
                features.update(smoothness_speed_features(pose, timestamps))
                if references_by_task is not None:
                    task_id = task_by_episode[episode]
                    features.update(accuracy_features(pose, references_by_task[task_id], args.resample_n))
            except ValueError as exc:
                # The quality gate and extractor share one rule, but fail closed if a
                # future numerical edge case escapes the gate.
                features["feature_usable"] = False
                features["quality_status"] = f"feature_unusable:{exc}"
                print(f"  episode {episode} retained as outcome-only row: {exc}")
        rows.append(features)

    dataframe = pd.DataFrame(rows)
    if labels is not None:
        try:
            dataframe = attach_labels_and_completion_times(dataframe, labels)
        except ValueError as exc:
            raise SystemExit(f"Invalid labels CSV: {exc}") from exc

        success_rate = dataframe["success"].mean()
        successful_times = dataframe["completion_time_s"].dropna()
        successful_count = int(dataframe["success"].sum())
        print(f"\nSuccess rate={success_rate:.1%}; failure rate={1 - success_rate:.1%} (n={len(dataframe)})")
        if len(successful_times):
            print(
                "Successful-task completion time "
                f"mean={successful_times.mean():.3f}s "
                f"(n={len(successful_times)}; failures excluded)"
            )
            if len(successful_times) != successful_count:
                print(
                    "  warning: "
                    f"{successful_count - len(successful_times)} successful outcome(s) "
                    "lack a valid completion time and are excluded from time analysis only"
                )
        else:
            if successful_count:
                print(
                    "Successful-task completion time: successful outcomes exist, but none "
                    "has a valid recorded/raw duration"
                )
            else:
                print("Successful-task completion time: no successful labeled episodes")
        if "failure_type" in dataframe.columns:
            failures = dataframe.loc[dataframe["success"] == 0, "failure_type"].dropna().value_counts()
            if len(failures):
                print("Failure breakdown:")
                for failure_type, count in failures.items():
                    print(f"  {failure_type}: {int(count)}")
    else:
        dataframe["completion_time_s"] = np.nan
        dataframe["completion_time_source"] = "unknown_without_labels"
        print(
            "\nNo labels CSV supplied: raw_duration_s is available, but "
            "completion_time_s is intentionally NaN because success is unknown."
        )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)
    input_paths = {
        "dataset_root": args.root,
        "labels_csv": args.labels_csv,
        "expert_dataset_root": args.ref_root,
        "expert_labels_csv": args.ref_labels_csv,
    }
    metadata_path = output.with_name(f"{output.stem}_metadata.json")
    seed_count = int(dataframe["seed"].nunique()) if "seed" in dataframe.columns else 0
    participant_count = (
        int(dataframe["participant_id"].nunique()) if "participant_id" in dataframe.columns else 0
    )
    write_json(
        metadata_path,
        {
            "artifact_type": "camera_ablation_feature_extraction",
            "arguments": vars(args),
            "datasets": {
                "target": {
                    "repo_id": args.repo_id,
                    "root": args.root,
                    "video_backend": args.video_backend,
                },
                "expert": {
                    "repo_id": args.ref_repo_id,
                    "root": args.ref_root,
                    "video_backend": args.video_backend if args.ref_repo_id else None,
                },
            },
            "algorithm": {
                "position_smoothing": "Savitzky-Golay",
                "smoothing_window_s": SMOOTHING_WINDOW_S,
                "smoothing_polyorder": SMOOTHING_POLYORDER,
                "minimum_frames": MIN_SAVGOL_FRAMES,
                "uniform_relative_tolerance": SAMPLING_UNIFORM_REL_TOL,
                "uniform_absolute_tolerance_s": SAMPLING_UNIFORM_ABS_TOL_S,
                "angular_speed": "adjacent quaternion geodesic",
                "reference_position_mean": "arithmetic_R3",
                "reference_orientation_mean": "Markley_quaternion_SO3",
                "reference_resample_points": args.resample_n,
            },
            "input_hashes": fingerprint_inputs(input_paths),
            "output": fingerprint_path(output),
            "counts": {
                "episode_rows": len(dataframe),
                "feature_usable_episodes": int(dataframe["feature_usable"].sum()),
                "participants": participant_count,
                "seeds": seed_count,
            },
            "runtime": runtime_environment(),
            "git": git_state(Path(__file__).parent),
        },
    )
    print(f"\nWrote {len(dataframe)} episode feature rows to {args.out}")
    print(f"Wrote reproducibility metadata to {metadata_path}")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(dataframe.select_dtypes(include=[np.number]).describe().round(4))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Preflight validation for the fixed mobile-vs-PC display experiment.

Run this before a pilot and again before model training. It validates the protocol
configuration, paired episode manifest, and (when supplied) the raw episode directory.
It never connects to or moves a robot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from acquisition_interfaces import (  # noqa: E402
    FORMAL_FRAME_TIMESTAMPS_FILENAME,
    FORMAL_VIDEO_ARTIFACT_TYPE,
    FORMAL_VIDEO_WRITER_CAPABILITY,
)
from make_episode_splits import CONDITIONS, validate_manifest  # noqa: E402
from time_sync import _count_decodable_frames, _read_ts_csv, _video_timestamps  # noqa: E402

VIDEO_ARTIFACT_TYPE = FORMAL_VIDEO_ARTIFACT_TYPE
VIDEO_WRITER_CAPABILITY = FORMAL_VIDEO_WRITER_CAPABILITY
FRAME_TIMESTAMPS_FILENAME = FORMAL_FRAME_TIMESTAMPS_FILENAME
FRAME_TIMESTAMP_COLUMNS = (
    "frame_index",
    "frame_timestamp_s",
    "frame_clock_id",
    "unix_timestamp_s",
)


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str


def _finding(severity: str, code: str, message: str) -> Finding:
    return Finding(severity=severity, code=code, message=message)


def _positive_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    )


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _clock_mapping_error(mapping: object, expected_clock_id: str) -> str | None:
    if not isinstance(mapping, Mapping):
        return "clock mapping must be an object"
    if mapping.get("schema_version") != 1:
        return "clock mapping schema_version must be 1"
    if mapping.get("source_clock_id") != expected_clock_id:
        return "clock mapping source_clock_id does not match its declared clock"
    for field in ("source_anchor_s", "unix_anchor_s"):
        if not _finite_number(mapping.get(field)) or float(mapping[field]) < 0:
            return f"clock mapping {field} must be finite and non-negative"
    if not _positive_finite_number(mapping.get("rate")):
        return "clock mapping rate must be finite and positive"
    if not _finite_number(mapping.get("uncertainty_s")) or float(mapping["uncertainty_s"]) < 0:
        return "clock mapping uncertainty_s must be finite and non-negative"
    return None


def _video_clock_contract_error(
    meta: Mapping[str, object],
    max_clock_uncertainty_s: object,
) -> str | None:
    if meta.get("schema_version") != 1:
        return "schema_version must be 1"
    if meta.get("artifact_type") != VIDEO_ARTIFACT_TYPE:
        return f"artifact_type must be {VIDEO_ARTIFACT_TYPE!r}"
    if meta.get("writer_capability") != VIDEO_WRITER_CAPABILITY:
        return f"writer_capability must be {VIDEO_WRITER_CAPABILITY!r}"
    if meta.get("video_capture_verified_by_this_writer") is not True:
        return "video_capture_verified_by_this_writer must be true"
    episode_id = meta.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id.strip():
        return "episode_id must be populated"
    video_path = meta.get("video_path")
    if not isinstance(video_path, str) or Path(video_path).name != "video.mp4":
        return "video_path must name video.mp4"
    frame_timestamps_csv = meta.get("frame_timestamps_csv")
    if (
        not isinstance(frame_timestamps_csv, str)
        or Path(frame_timestamps_csv).name != FRAME_TIMESTAMPS_FILENAME
    ):
        return f"frame_timestamps_csv must name {FRAME_TIMESTAMPS_FILENAME}"
    frame_count = meta.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        return "frame_count must be a positive integer"
    if not _positive_finite_number(max_clock_uncertainty_s):
        return "configured max_clock_uncertainty_s must be finite and positive"
    if meta.get("timestamps_domain") != "unix_s":
        return "timestamps_domain must be 'unix_s'"
    contract = meta.get("clock_contract")
    if not isinstance(contract, Mapping):
        return "clock_contract must be an object"
    if contract.get("schema_version") != 1:
        return "clock_contract.schema_version must be 1"
    frame_clock_id = contract.get("frame_clock_id")
    episode_clock_id = contract.get("episode_clock_id")
    if not isinstance(frame_clock_id, str) or not frame_clock_id.strip():
        return "clock_contract.frame_clock_id must be populated"
    if not isinstance(episode_clock_id, str) or not episode_clock_id.strip():
        return "clock_contract.episode_clock_id must be populated"
    same_clock = contract.get("same_clock_as_episode")
    if not isinstance(same_clock, bool):
        return "clock_contract.same_clock_as_episode must be boolean"
    if same_clock and frame_clock_id != episode_clock_id:
        return "same_clock_as_episode=true requires identical clock IDs"
    if not same_clock and frame_clock_id == episode_clock_id:
        return "same_clock_as_episode=false requires distinct clock IDs"
    frame_mapping = contract.get("frame_to_unix")
    episode_mapping = contract.get("episode_to_unix")
    error = _clock_mapping_error(frame_mapping, frame_clock_id)
    if error:
        return f"frame_to_unix: {error}"
    error = _clock_mapping_error(episode_mapping, episode_clock_id)
    if error:
        return f"episode_to_unix: {error}"
    assert isinstance(frame_mapping, Mapping)
    assert isinstance(episode_mapping, Mapping)
    maximum = float(max_clock_uncertainty_s)
    for name, mapping in (
        ("frame_to_unix", frame_mapping),
        ("episode_to_unix", episode_mapping),
    ):
        if float(mapping["uncertainty_s"]) > maximum:
            return f"{name}.uncertainty_s exceeds configured max_clock_uncertainty_s={maximum}"
    if same_clock and frame_mapping != episode_mapping:
        return "one declared clock must use one identical Unix mapping"
    return None


def _read_frame_timestamp_csv(
    path: Path,
    meta: Mapping[str, object],
) -> tuple[list[float], list[float]]:
    """Read the concrete recorder's per-frame clock evidence without pandas header mangling."""

    contract = meta["clock_contract"]
    assert isinstance(contract, Mapping)
    mapping = contract["frame_to_unix"]
    assert isinstance(mapping, Mapping)
    expected_clock_id = contract["frame_clock_id"]
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.reader(stream)
            header = next(rows, None)
            if tuple(header or ()) != FRAME_TIMESTAMP_COLUMNS:
                raise ValueError(f"columns must be exactly {list(FRAME_TIMESTAMP_COLUMNS)}")
            source_timestamps: list[float] = []
            unix_timestamps: list[float] = []
            for expected_index, row in enumerate(rows):
                if len(row) != len(FRAME_TIMESTAMP_COLUMNS):
                    raise ValueError(
                        f"row {expected_index + 2} has {len(row)} fields; expected "
                        f"{len(FRAME_TIMESTAMP_COLUMNS)}"
                    )
                try:
                    frame_index = int(row[0])
                    source_timestamp = float(row[1])
                    unix_timestamp = float(row[3])
                except ValueError as exc:
                    raise ValueError(
                        f"row {expected_index + 2} has non-numeric index/timestamp data"
                    ) from exc
                if str(frame_index) != row[0].strip() or frame_index != expected_index:
                    raise ValueError("frame_index must be contiguous integers starting at zero")
                if row[2].strip() != expected_clock_id:
                    raise ValueError(f"frame {frame_index} clock_id does not match frame_clock_id")
                if (
                    not math.isfinite(source_timestamp)
                    or source_timestamp < 0
                    or not math.isfinite(unix_timestamp)
                    or unix_timestamp < 0
                ):
                    raise ValueError("frame timestamps must be finite and non-negative")
                mapped = float(mapping["unix_anchor_s"]) + float(mapping["rate"]) * (
                    source_timestamp - float(mapping["source_anchor_s"])
                )
                if not math.isclose(
                    mapped,
                    unix_timestamp,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise ValueError(f"frame {frame_index} Unix timestamp does not match clock mapping")
                source_timestamps.append(source_timestamp)
                unix_timestamps.append(unix_timestamp)
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must be UTF-8 encoded") from exc
    if not source_timestamps:
        raise ValueError("timestamp CSV must contain at least one frame")
    if any(
        later <= earlier
        for earlier, later in zip(
            source_timestamps,
            source_timestamps[1:],
            strict=False,
        )
    ):
        raise ValueError("source frame timestamps must be strictly increasing")
    if any(
        later <= earlier
        for earlier, later in zip(
            unix_timestamps,
            unix_timestamps[1:],
            strict=False,
        )
    ):
        raise ValueError("mapped Unix frame timestamps must be strictly increasing")
    return source_timestamps, unix_timestamps


def validate_protocol(config: dict) -> list[Finding]:
    findings: list[Finding] = []
    if not isinstance(config, Mapping):
        return [_finding("ERROR", "config", "protocol config root must be a JSON object")]
    study = config.get("study", {})
    if not isinstance(study, Mapping):
        study = {}
    conditions = study.get("conditions", {})
    if not isinstance(conditions, Mapping) or set(conditions) != set(CONDITIONS):
        findings.append(
            _finding(
                "ERROR",
                "conditions",
                "study.conditions must contain exactly A_mobile and B_pc",
            )
        )
    else:
        for condition, display in CONDITIONS.items():
            item = conditions[condition]
            if not isinstance(item, Mapping):
                findings.append(
                    _finding(
                        "ERROR",
                        "condition_config",
                        f"{condition} must be a JSON object",
                    )
                )
                continue
            if item.get("display") != display:
                findings.append(_finding("ERROR", "display", f"{condition}.display must be {display!r}"))
            if item.get("camera_present") is not True:
                findings.append(_finding("ERROR", "camera", f"{condition} must record the camera stream"))
            if item.get("control") != "phone_imu":
                findings.append(_finding("ERROR", "control", f"{condition}.control must be 'phone_imu'"))

    if study.get("paired_within_participant") is not True:
        findings.append(
            _finding(
                "ERROR",
                "paired_design",
                "paired_within_participant must be true for the confirmed study design",
            )
        )
    if study.get("counterbalance_condition_order") is not True:
        findings.append(
            _finding(
                "ERROR",
                "counterbalance",
                "counterbalance_condition_order must be true",
            )
        )

    capture = config.get("capture", {})
    if not isinstance(capture, Mapping):
        capture = {}
    fps = capture.get("target_fps")
    if not _positive_finite_number(fps):
        findings.append(_finding("ERROR", "fps", "capture.target_fps must be positive"))
    if capture.get("position_unit") != "m" or capture.get("angle_unit") != "rad":
        findings.append(
            _finding(
                "ERROR",
                "units",
                "capture units must be position_unit='m' and angle_unit='rad'",
            )
        )
    required_files = set(capture.get("required_files", []))
    expected_files = {
        "robot.csv",
        "phone.csv",
        "applied_actions.csv",
        "video.mp4",
        "video_meta.json",
        FRAME_TIMESTAMPS_FILENAME,
    }
    if required_files != expected_files:
        findings.append(
            _finding(
                "ERROR",
                "required_files",
                f"capture.required_files must be exactly {sorted(expected_files)}",
            )
        )
    if not _positive_finite_number(capture.get("max_clock_uncertainty_s")):
        findings.append(
            _finding(
                "ERROR",
                "max_clock_uncertainty",
                "capture.max_clock_uncertainty_s must be a positive finite measured bound",
            )
        )
    if capture.get("record_video_latency_ms") is not True:
        findings.append(
            _finding(
                "ERROR",
                "record_video_latency",
                "capture.record_video_latency_ms must be true",
            )
        )
    if capture.get("record_dropped_frames") is not True:
        findings.append(
            _finding(
                "ERROR",
                "record_dropped_frames",
                "capture.record_dropped_frames must be true",
            )
        )

    joint_names = capture.get("joint_names")
    if (
        not isinstance(joint_names, list)
        or not joint_names
        or any(not isinstance(name, str) or not name.strip() for name in joint_names)
        or len(set(joint_names)) != len(joint_names)
    ):
        findings.append(
            _finding(
                "ERROR",
                "joint_names",
                "capture.joint_names must be a non-empty ordered list of unique names",
            )
        )

    success = config.get("success_definition", {})
    if not isinstance(success, Mapping):
        success = {}
    placement = success.get("placement_error_threshold_m")
    timeout = success.get("timeout_s")
    if not _positive_finite_number(placement):
        findings.append(
            _finding(
                "ERROR",
                "placement_threshold",
                "set a positive success_definition.placement_error_threshold_m",
            )
        )
    if not _positive_finite_number(timeout):
        findings.append(_finding("ERROR", "timeout", "set a positive success_definition.timeout_s"))

    tasks = config.get("tasks", [])
    if not isinstance(tasks, list) or any(not isinstance(task, Mapping) for task in tasks):
        findings.append(_finding("ERROR", "tasks", "tasks must be a list of JSON objects"))
        tasks = []
    difficulties = {str(task.get("difficulty", "")).strip().lower() for task in tasks}
    if not {"simple", "medium", "hard"}.issubset(difficulties):
        findings.append(_finding("ERROR", "tasks", "define at least one simple, medium, and hard task"))
    if any("replace_with" in str(task.get("task_id", "")).lower() for task in tasks):
        findings.append(_finding("ERROR", "task_placeholder", "replace every placeholder task_id"))
    task_ids = [str(task.get("task_id", "")).strip() for task in tasks]
    if any(not task_id for task_id in task_ids) or len(set(task_ids)) != len(task_ids):
        findings.append(_finding("ERROR", "task_id", "task_id values must be non-empty and unique"))

    training = config.get("training", {})
    if not isinstance(training, Mapping):
        training = {}
    fractions = [
        training.get("train_fraction"),
        training.get("validation_fraction"),
        training.get("test_fraction"),
    ]
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in fractions
        )
        or abs(
            sum(
                value
                for value in fractions
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            )
            - 1.0
        )
        > 1e-9
    ):
        findings.append(
            _finding(
                "ERROR",
                "split_fractions",
                "training train/validation/test fractions must be non-negative and sum to 1",
            )
        )
    seeds = training.get("random_seeds", [])
    valid_seeds = (
        isinstance(seeds, list)
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
        and len(seeds) == len(set(seeds))
        and len(seeds) >= 3
    )
    if not valid_seeds:
        findings.append(
            _finding(
                "ERROR",
                "random_seeds",
                "training.random_seeds must contain at least three distinct integers",
            )
        )

    analysis = config.get("analysis", {})
    if not isinstance(analysis, Mapping):
        analysis = {}
    planned_participants = analysis.get("planned_participants")
    minimum_per_split = analysis.get("minimum_participants_per_split")
    if (
        isinstance(planned_participants, bool)
        or not isinstance(planned_participants, int)
        or planned_participants <= 0
    ):
        findings.append(
            _finding(
                "ERROR",
                "planned_participants",
                "analysis.planned_participants must be a positive integer from the "
                "preregistered sample-size/power plan",
            )
        )
    if isinstance(minimum_per_split, bool) or not isinstance(minimum_per_split, int) or minimum_per_split < 2:
        findings.append(
            _finding(
                "ERROR",
                "minimum_participants",
                "analysis.minimum_participants_per_split must be an integer of at least 2",
            )
        )
    elif isinstance(planned_participants, int) and not isinstance(
        planned_participants,
        bool,
    ):
        undersized = [
            split
            for split, fraction in zip(
                ("train", "validation", "test"),
                fractions,
                strict=True,
            )
            if isinstance(fraction, (int, float))
            and not isinstance(fraction, bool)
            and math.isfinite(float(fraction))
            and fraction > 0
            and math.ceil(planned_participants * fraction) < minimum_per_split
        ]
        if undersized:
            findings.append(
                _finding(
                    "ERROR",
                    "planned_split_size",
                    f"planned_participants and split fractions cannot provide "
                    f"{minimum_per_split} participants in {undersized}",
                )
            )
    return findings


def validate_manifest_metadata(
    df: pd.DataFrame,
    config: dict | None = None,
) -> tuple[pd.DataFrame | None, list[Finding]]:
    findings: list[Finding] = []
    try:
        normalized = validate_manifest(df)
    except ValueError as exc:
        return None, [_finding("ERROR", "manifest", str(exc))]

    missing_tokens = {"", "nan", "none", "<na>", "null"}
    for identifier in ("task_id", "participant_id", "seed", "split"):
        if identifier not in normalized.columns:
            continue
        raw_identifier = normalized[identifier]
        invalid_identifier = raw_identifier.isna() | raw_identifier.astype(str).str.strip().str.lower().isin(
            missing_tokens
        )
        if invalid_identifier.any():
            findings.append(
                _finding(
                    "ERROR",
                    identifier,
                    f"{identifier} must be populated and cannot use a null sentinel",
                )
            )

    required_metadata = {
        "trial_index",
        "condition_order",
        "success",
        "failure_type",
        "placement_error_m",
        "completion_time_s",
        "video_latency_ms",
        "dropped_frames",
    }
    missing = sorted(required_metadata - set(normalized.columns))
    if missing:
        findings.append(_finding("ERROR", "manifest_metadata", f"manifest is missing columns: {missing}"))
        return normalized, findings

    raw_order = normalized["condition_order"]
    numeric_order = pd.to_numeric(raw_order, errors="coerce")
    invalid_order = numeric_order.isna() | (numeric_order % 1 != 0) | ~numeric_order.isin([1, 2])
    if invalid_order.any():
        findings.append(
            _finding(
                "ERROR",
                "condition_order",
                "condition_order must contain only exact integer values 1 or 2",
            )
        )
    for pair_id, pair in normalized.assign(_condition_order=numeric_order).groupby("pair_id"):
        order = set(pair["_condition_order"].dropna().tolist())
        if order != {1, 2}:
            findings.append(
                _finding(
                    "ERROR",
                    "condition_order",
                    f"pair_id {pair_id!r} must contain condition_order 1 and 2",
                )
            )

    participants_per_difficulty = (
        normalized.drop_duplicates(["participant_id", "difficulty"])
        .groupby("difficulty")["participant_id"]
        .nunique()
    )
    sparse = {
        str(difficulty): int(count) for difficulty, count in participants_per_difficulty.items() if count < 3
    }
    if sparse:
        findings.append(
            _finding(
                "ERROR",
                "split_capacity",
                "fewer than three participants cannot populate train/validation/test "
                f"within a difficulty stratum: {sparse}",
            )
        )

    raw_success = normalized["success"]
    success = pd.to_numeric(raw_success, errors="coerce")
    success_blank = raw_success.isna() | raw_success.astype(str).str.strip().eq("")
    invalid_success = ~success_blank & success.isna()
    if invalid_success.any():
        findings.append(
            _finding("ERROR", "success", "success must be 0, 1, or blank; non-numeric labels are invalid")
        )
    present = success.dropna()
    if not present.isin([0, 1]).all():
        findings.append(_finding("ERROR", "success", "success values must be 0, 1, or blank"))
    if success.isna().any():
        findings.append(
            _finding(
                "WARN",
                "outcomes_incomplete",
                "some outcome labels are blank; fill them after collection and before analysis",
            )
        )

    numeric_rules = {
        "placement_error_m": False,
        "completion_time_s": False,
        "video_latency_ms": False,
        "dropped_frames": True,
    }
    for column, require_integer in numeric_rules.items():
        raw = normalized[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        blank = raw.isna() | raw.astype(str).str.strip().eq("")
        finite = pd.Series(np.isfinite(numeric.to_numpy(dtype=float)), index=numeric.index)
        invalid = (~blank & (numeric.isna() | ~finite)) | numeric.lt(0)
        if require_integer:
            invalid |= numeric.notna() & (numeric % 1 != 0)
        if invalid.any():
            rule = "a non-negative integer" if require_integer else "non-negative numeric"
            findings.append(_finding("ERROR", column, f"{column} values must be {rule} or blank"))
        capture = config.get("capture", {}) if isinstance(config, Mapping) else {}
        required_when_recording = (
            column == "video_latency_ms"
            and isinstance(capture, Mapping)
            and capture.get("record_video_latency_ms") is True
        ) or (
            column == "dropped_frames"
            and isinstance(capture, Mapping)
            and capture.get("record_dropped_frames") is True
        )
        if blank.any():
            findings.append(
                _finding(
                    "ERROR" if required_when_recording else "WARN",
                    f"{column}_incomplete",
                    (
                        f"some {column} values are blank, but protocol capture requires this measurement"
                        if required_when_recording
                        else f"some {column} values are blank"
                    ),
                )
            )

    failed = success.eq(0)
    missing_failure = normalized["failure_type"].fillna("").astype(str).str.strip().eq("")
    if (failed & missing_failure).any():
        findings.append(_finding("ERROR", "failure_type", "every failed episode needs a failure_type"))
    succeeded = success.eq(1)
    unexpected_failure = ~missing_failure & succeeded
    if unexpected_failure.any():
        findings.append(
            _finding(
                "ERROR",
                "failure_type",
                "successful episodes must not have a failure_type",
            )
        )

    if config is not None:
        task_rows = config.get("tasks", [])
        if not isinstance(task_rows, list):
            task_rows = []
        task_to_difficulty = {
            str(task.get("task_id", "")).strip(): str(task.get("difficulty", "")).strip().lower()
            for task in task_rows
            if isinstance(task, dict)
        }
        for row in normalized[["task_id", "difficulty"]].drop_duplicates().itertuples(index=False):
            expected_difficulty = task_to_difficulty.get(str(row.task_id))
            if expected_difficulty is None:
                findings.append(
                    _finding(
                        "ERROR",
                        "task_id",
                        f"manifest task_id {row.task_id!r} is absent from protocol tasks",
                    )
                )
            elif str(row.difficulty).strip().lower() != expected_difficulty:
                findings.append(
                    _finding(
                        "ERROR",
                        "task_difficulty",
                        f"task_id {row.task_id!r} difficulty must be {expected_difficulty!r}",
                    )
                )

        success_definition = config.get("success_definition", {})
        if not isinstance(success_definition, Mapping):
            success_definition = {}
        placement_threshold = success_definition.get("placement_error_threshold_m")
        timeout_s = success_definition.get("timeout_s")
        placement_values = pd.to_numeric(
            normalized["placement_error_m"],
            errors="coerce",
        )
        completion_values = pd.to_numeric(
            normalized["completion_time_s"],
            errors="coerce",
        )
        if (succeeded & placement_values.isna()).any():
            findings.append(
                _finding(
                    "ERROR",
                    "success_measurement",
                    "success=1 requires a numeric placement_error_m",
                )
            )
        if (succeeded & completion_values.isna()).any():
            findings.append(
                _finding(
                    "ERROR",
                    "success_measurement",
                    "success=1 requires a numeric completion_time_s",
                )
            )
        if (
            _positive_finite_number(placement_threshold)
            and (succeeded & placement_values.gt(float(placement_threshold))).any()
        ):
            findings.append(
                _finding(
                    "ERROR",
                    "success_threshold",
                    "success=1 requires placement_error_m not to exceed the configured threshold",
                )
            )
        if _positive_finite_number(timeout_s) and (succeeded & completion_values.gt(float(timeout_s))).any():
            findings.append(
                _finding(
                    "ERROR",
                    "success_timeout",
                    "success=1 requires completion_time_s not to exceed timeout_s",
                )
            )
        allowed_failure_types = success_definition.get("allowed_failure_types")
        if allowed_failure_types is not None:
            if not isinstance(allowed_failure_types, list) or any(
                not isinstance(value, str) or not value.strip() for value in allowed_failure_types
            ):
                findings.append(
                    _finding(
                        "ERROR",
                        "allowed_failure_types",
                        "success_definition.allowed_failure_types must be a list of non-empty strings",
                    )
                )
            else:
                allowed = {value.strip() for value in allowed_failure_types}
                actual_failures = set(normalized.loc[failed, "failure_type"].dropna().astype(str).str.strip())
                unknown_failures = sorted(actual_failures - allowed - {""})
                if unknown_failures:
                    findings.append(
                        _finding(
                            "ERROR",
                            "failure_type",
                            f"failure_type contains values outside protocol: {unknown_failures}",
                        )
                    )

        study_config = config.get("study", {})
        if not isinstance(study_config, Mapping):
            study_config = {}
        if study_config.get("counterbalance_condition_order") is True:
            participant_orders = normalized.assign(_condition_order=numeric_order).pivot_table(
                index="participant_id",
                columns="condition",
                values="_condition_order",
                aggfunc=lambda values: tuple(sorted(set(values.dropna()))),
            )
            inconsistent_participants = []
            sequence_counts = {"A_first": 0, "B_first": 0}
            for participant_id, row in participant_orders.iterrows():
                a_orders = row.get("A_mobile")
                b_orders = row.get("B_pc")
                if a_orders == (1,) and b_orders == (2,):
                    sequence_counts["A_first"] += 1
                elif a_orders == (2,) and b_orders == (1,):
                    sequence_counts["B_first"] += 1
                else:
                    inconsistent_participants.append(str(participant_id))
            if inconsistent_participants:
                findings.append(
                    _finding(
                        "ERROR",
                        "participant_order",
                        "each participant must keep one A/B order across all trials; "
                        f"invalid participants: {inconsistent_participants}",
                    )
                )
            elif (
                min(sequence_counts.values()) == 0
                or abs(sequence_counts["A_first"] - sequence_counts["B_first"]) > 1
            ):
                findings.append(
                    _finding(
                        "ERROR",
                        "counterbalance",
                        f"participant A-first/B-first assignments are not counterbalanced: {sequence_counts}",
                    )
                )

    return normalized, findings


def validate_raw_tree(
    raw_root: Path,
    manifest: pd.DataFrame,
    required_files: list[str],
    joint_names: list[str],
    max_clock_uncertainty_s: object,
) -> list[Finding]:
    findings: list[Finding] = []
    condition_dirs = {"A_mobile": "cond_a_mobile", "B_pc": "cond_b_pc"}
    action_columns = ["timestamp", "dx", "dy", "dz", "dyaw", "dpitch", "droll"]
    robot_columns = ["timestamp", *joint_names]

    for row in manifest.itertuples(index=False):
        episode_dir = raw_root / condition_dirs[row.condition] / f"episode_{int(row.episode):03d}"
        if not episode_dir.is_dir():
            findings.append(_finding("ERROR", "episode_dir", f"missing episode directory: {episode_dir}"))
            continue
        missing = [name for name in required_files if not (episode_dir / name).is_file()]
        if missing:
            findings.append(_finding("ERROR", "episode_files", f"{episode_dir} is missing {missing}"))
            continue

        try:
            robot_header = list(pd.read_csv(episode_dir / "robot.csv", nrows=0).columns)
            phone_header = list(pd.read_csv(episode_dir / "phone.csv", nrows=0).columns)
            applied_header = list(pd.read_csv(episode_dir / "applied_actions.csv", nrows=0).columns)
        except Exception as exc:
            findings.append(
                _finding("ERROR", "csv_header", f"cannot read CSV headers in {episode_dir}: {exc}")
            )
            continue
        if robot_header != robot_columns:
            findings.append(
                _finding(
                    "ERROR",
                    "robot_header",
                    f"{episode_dir / 'robot.csv'} columns must be exactly {robot_columns}",
                )
            )
        if phone_header != action_columns:
            findings.append(
                _finding(
                    "ERROR",
                    "phone_header",
                    f"{episode_dir / 'phone.csv'} columns must be exactly {action_columns}",
                )
            )
        if applied_header != action_columns:
            findings.append(
                _finding(
                    "ERROR",
                    "applied_header",
                    f"{episode_dir / 'applied_actions.csv'} columns must be exactly {action_columns}",
                )
            )

        robot_t = applied_t = video_t = None
        try:
            robot_t, _, robot_value_columns = _read_ts_csv(episode_dir / "robot.csv")
            _, _, phone_value_columns = _read_ts_csv(episode_dir / "phone.csv")
            applied_t, _, applied_value_columns = _read_ts_csv(episode_dir / "applied_actions.csv")
            if robot_value_columns != joint_names:
                raise ValueError(f"robot value columns must be exactly {joint_names}")
            expected_actions = action_columns[1:]
            if phone_value_columns != expected_actions:
                raise ValueError(f"phone value columns must be exactly {expected_actions}")
            if applied_value_columns != expected_actions:
                raise ValueError(f"applied action columns must be exactly {expected_actions}")
        except (OSError, TypeError, ValueError) as exc:
            findings.append(
                _finding(
                    "ERROR",
                    "csv_content",
                    f"invalid timestamped CSV content in {episode_dir}: {exc}",
                )
            )

        try:
            meta = json.loads((episode_dir / "video_meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(
                _finding("ERROR", "video_meta", f"invalid video_meta.json in {episode_dir}: {exc}")
            )
            continue
        if not isinstance(meta, dict):
            findings.append(
                _finding(
                    "ERROR",
                    "video_meta",
                    f"video_meta.json in {episode_dir} must contain a JSON object",
                )
            )
            continue
        clock_contract_error = _video_clock_contract_error(
            meta,
            max_clock_uncertainty_s,
        )
        if clock_contract_error:
            findings.append(
                _finding(
                    "ERROR",
                    "video_clock_contract",
                    f"{episode_dir} video_meta clock contract is invalid: {clock_contract_error}",
                )
            )
        if meta.get("episode_id") != episode_dir.name:
            findings.append(
                _finding(
                    "ERROR",
                    "video_episode_id",
                    f"{episode_dir} video_meta.episode_id must be {episode_dir.name!r}",
                )
            )
        for timestamp_field in ("episode_start_ts", "first_frame_ts"):
            value = meta.get(timestamp_field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                findings.append(
                    _finding(
                        "ERROR",
                        "video_time",
                        f"{episode_dir} video_meta.{timestamp_field} must be finite numeric",
                    )
                )
        episode_start = meta.get("episode_start_ts")
        first_frame = meta.get("first_frame_ts")
        if (
            _finite_number(episode_start)
            and _finite_number(first_frame)
            and float(episode_start) > float(first_frame)
        ):
            findings.append(
                _finding(
                    "ERROR",
                    "video_time",
                    f"{episode_dir} episode_start_ts must not be later than first_frame_ts",
                )
            )
        frame_timestamps = meta.get("frame_timestamps_s")
        valid_frame_timestamps = (
            isinstance(frame_timestamps, list)
            and bool(frame_timestamps)
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in frame_timestamps
            )
            and all(
                float(later) > float(earlier)
                for earlier, later in zip(
                    frame_timestamps,
                    frame_timestamps[1:],
                    strict=False,
                )
            )
        )
        if not valid_frame_timestamps:
            findings.append(
                _finding(
                    "ERROR",
                    "video_frame_times",
                    f"{episode_dir} video_meta.frame_timestamps_s must be a non-empty "
                    "strictly increasing finite numeric array",
                )
            )
        else:
            if meta.get("frame_count") != len(frame_timestamps):
                findings.append(
                    _finding(
                        "ERROR",
                        "video_frame_count",
                        f"{episode_dir} frame_count must equal the number of frame_timestamps_s entries",
                    )
                )
            if isinstance(meta.get("first_frame_ts"), (int, float)) and not math.isclose(
                float(meta["first_frame_ts"]),
                float(frame_timestamps[0]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                findings.append(
                    _finding(
                        "ERROR",
                        "video_first_frame",
                        f"{episode_dir} first_frame_ts must equal frame_timestamps_s[0]",
                    )
                )
        if clock_contract_error is None:
            try:
                _, csv_unix_timestamps = _read_frame_timestamp_csv(
                    episode_dir / FRAME_TIMESTAMPS_FILENAME,
                    meta,
                )
                if valid_frame_timestamps and (
                    len(csv_unix_timestamps) != len(frame_timestamps)
                    or any(
                        not math.isclose(
                            csv_value,
                            float(json_value),
                            rel_tol=0.0,
                            abs_tol=1e-6,
                        )
                        for csv_value, json_value in zip(
                            csv_unix_timestamps,
                            frame_timestamps,
                            strict=False,
                        )
                    )
                ):
                    raise ValueError("CSV Unix timestamps must exactly correspond to frame_timestamps_s")
                if meta.get("frame_count") != len(csv_unix_timestamps):
                    raise ValueError("frame_count must equal the timestamp CSV row count")
            except (OSError, TypeError, ValueError) as exc:
                findings.append(
                    _finding(
                        "ERROR",
                        "video_frame_timestamp_csv",
                        f"invalid {FRAME_TIMESTAMPS_FILENAME} in {episode_dir}: {exc}",
                    )
                )
        if meta.get("display") != CONDITIONS[row.condition]:
            findings.append(
                _finding(
                    "ERROR",
                    "video_display",
                    f"{episode_dir} video_meta.display does not match {row.condition}",
                )
            )

        try:
            decoded_frames = _count_decodable_frames(episode_dir / "video.mp4")
            if decoded_frames <= 0:
                raise ValueError("video has no decodable frames")
            if meta.get("frame_count") != decoded_frames:
                raise ValueError(
                    f"video decodes {decoded_frames} frames but metadata declares {meta.get('frame_count')!r}"
                )
            video_t, _ = _video_timestamps(
                episode_dir / "video_meta.json",
                episode_dir / "video.mp4",
                frame_count=decoded_frames,
            )
        except (OSError, TypeError, ValueError) as exc:
            findings.append(
                _finding(
                    "ERROR",
                    "video_content",
                    f"invalid video/timestamps in {episode_dir}: {exc}",
                )
            )

        if robot_t is not None and applied_t is not None and video_t is not None:
            overlap_start = max(robot_t[0], applied_t[0], video_t[0])
            overlap_end = min(robot_t[-1], video_t[-1])
            if overlap_end <= overlap_start:
                findings.append(
                    _finding(
                        "ERROR",
                        "stream_overlap",
                        f"{episode_dir} robot/applied/video streams do not overlap",
                    )
                )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw_root", default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest_df = pd.read_csv(args.manifest)
    findings = validate_protocol(config)
    protocol_config = config if isinstance(config, dict) else None
    normalized, manifest_findings = validate_manifest_metadata(
        manifest_df,
        protocol_config,
    )
    findings.extend(manifest_findings)
    if args.raw_root and normalized is not None and protocol_config is not None:
        findings.extend(
            validate_raw_tree(
                Path(args.raw_root),
                normalized,
                list(protocol_config.get("capture", {}).get("required_files", [])),
                list(protocol_config.get("capture", {}).get("joint_names", [])),
                protocol_config.get("capture", {}).get("max_clock_uncertainty_s"),
            )
        )

    for finding in findings:
        print(f"[{finding.severity}] {finding.code}: {finding.message}")
    errors = sum(finding.severity == "ERROR" for finding in findings)
    warnings = sum(finding.severity == "WARN" for finding in findings)
    if not findings:
        print("[PASS] experiment configuration and available data passed preflight")
    else:
        print(f"\nPreflight summary: {errors} error(s), {warnings} warning(s)")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()

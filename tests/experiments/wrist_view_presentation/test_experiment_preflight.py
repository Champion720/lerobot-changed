from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.wrist_view_presentation import time_sync

SCRIPT = (
    Path(__file__).parents[3] / "experiments" / "wrist_view_presentation" / "validate_experiment_setup.py"
)
SPEC = importlib.util.spec_from_file_location("validate_experiment_setup", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _config() -> dict:
    return {
        "study": {
            "name": "wrist_view_presentation",
            "paired_within_participant": True,
            "counterbalance_condition_order": True,
            "conditions": {
                "A_mobile_colocated": {
                    "camera_present": True,
                    "camera_view": "wrist",
                    "display": "mobile",
                    "control": "phone_imu",
                    "feedback_control_relation": "colocated",
                },
                "B_desktop_separated": {
                    "camera_present": True,
                    "camera_view": "wrist",
                    "display": "desktop",
                    "control": "phone_imu",
                    "feedback_control_relation": "spatially_separated",
                },
            },
        },
        "capture": {
            "target_fps": 30,
            "position_unit": "m",
            "angle_unit": "rad",
            "joint_names": ["joint_1", "joint_2"],
            "required_files": [
                "robot.csv",
                "phone.csv",
                "applied_actions.csv",
                "video.mp4",
                "video_meta.json",
                "frame_timestamps.csv",
            ],
            "max_clock_uncertainty_s": 0.01,
            "record_video_latency_ms": True,
            "record_dropped_frames": True,
        },
        "success_definition": {
            "placement_error_threshold_m": 0.005,
            "timeout_s": 60,
        },
        "tasks": [
            {"task_id": "task_simple", "difficulty": "simple"},
            {"task_id": "task_medium", "difficulty": "medium"},
            {"task_id": "task_hard", "difficulty": "hard"},
        ],
        "training": {
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "random_seeds": [0, 1, 2],
            "condition_runs": {
                "A_mobile_colocated": {
                    "dataset_repo_id": "local/A_mobile_colocated",
                    "video_backend": "pyav",
                    "policy_type": "act",
                    "device": "cuda",
                    "batch_size": 8,
                    "steps": 5000,
                    "eval_freq": 0,
                    "num_workers": 0,
                    "wandb_enabled": False,
                    "push_to_hub": False,
                    "extra_cli_args": [],
                },
                "B_desktop_separated": {
                    "dataset_repo_id": "local/B_desktop_separated",
                    "video_backend": "pyav",
                    "policy_type": "act",
                    "device": "cuda",
                    "batch_size": 8,
                    "steps": 5000,
                    "eval_freq": 0,
                    "num_workers": 0,
                    "wandb_enabled": False,
                    "push_to_hub": False,
                    "extra_cli_args": [],
                },
            },
        },
        "analysis": {
            "planned_participants": 12,
            "minimum_participants_per_split": 2,
        },
    }


def _manifest() -> pd.DataFrame:
    common = {
        "pair_id": "pair_0",
        "participant_id": "P01",
        "task_id": "task_simple",
        "difficulty": "simple",
        "trial_index": 1,
        "success": 1,
        "failure_type": "",
        "placement_error_m": 0.003,
        "completion_time_s": 10.0,
        "video_latency_ms": 80,
        "dropped_frames": 0,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "condition": "A_mobile_colocated",
                "episode": 0,
                "condition_order": 1,
            },
            {
                **common,
                "condition": "B_desktop_separated",
                "episode": 0,
                "condition_order": 2,
            },
        ]
    )


def _valid_video_meta() -> dict:
    mapping = {
        "schema_version": 1,
        "source_clock_id": "capture_unix",
        "source_anchor_s": 10.0,
        "unix_anchor_s": 10.0,
        "rate": 1.0,
        "uncertainty_s": 0.001,
    }
    return {
        "schema_version": 1,
        "artifact_type": "video_episode_capture",
        "writer_capability": "decoded_frame_video_recorder",
        "video_capture_verified_by_this_writer": True,
        "episode_id": "episode_000",
        "video_path": "video.mp4",
        "frame_timestamps_csv": "frame_timestamps.csv",
        "frame_count": 2,
        "episode_start_ts": 9.9,
        "first_frame_ts": 10.0,
        "frame_timestamps_s": [10.0, 10.1],
        "display": "mobile",
        "timestamps_domain": "unix_s",
        "clock_contract": {
            "schema_version": 1,
            "frame_clock_id": "capture_unix",
            "episode_clock_id": "capture_unix",
            "same_clock_as_episode": True,
            "frame_to_unix": mapping,
            "episode_to_unix": dict(mapping),
        },
    }


def test_complete_protocol_passes() -> None:
    assert MODULE.validate_protocol(_config()) == []


@pytest.mark.parametrize(
    ("condition", "field", "value", "code"),
    [
        ("A_mobile_colocated", "camera_present", False, "camera_present"),
        ("B_desktop_separated", "camera_view", "overhead", "camera_view"),
        ("B_desktop_separated", "control", "desktop", "control"),
        (
            "A_mobile_colocated",
            "feedback_control_relation",
            "spatially_separated",
            "feedback_control_relation",
        ),
    ],
)
def test_protocol_rejects_condition_invariant_drift(
    condition: str,
    field: str,
    value: object,
    code: str,
) -> None:
    config = _config()
    config["study"]["conditions"][condition][field] = value

    assert code in {finding.code for finding in MODULE.validate_protocol(config)}


def test_protocol_rejects_training_drift_between_conditions() -> None:
    config = _config()
    config["training"]["condition_runs"]["B_desktop_separated"]["batch_size"] = 16

    assert "training_invariant" in {finding.code for finding in MODULE.validate_protocol(config)}


def test_protocol_requires_video_and_frame_timestamp_artifacts() -> None:
    config = _config()
    config["capture"]["required_files"].remove("video.mp4")

    assert "required_files" in {finding.code for finding in MODULE.validate_protocol(config)}


def test_placeholder_protocol_reports_blockers() -> None:
    config = _config()
    config["success_definition"]["timeout_s"] = None
    config["tasks"][0]["task_id"] = "replace_with_simple_task"
    codes = {finding.code for finding in MODULE.validate_protocol(config)}
    assert {"timeout", "task_placeholder"}.issubset(codes)


def test_complete_manifest_metadata_passes() -> None:
    normalized, findings = MODULE.validate_manifest_metadata(_manifest())
    assert normalized is not None
    assert [finding.code for finding in findings] == ["split_capacity"]


def test_failed_episode_requires_failure_type() -> None:
    manifest = _manifest()
    manifest.loc[0, "success"] = 0
    _, findings = MODULE.validate_manifest_metadata(manifest)
    assert "failure_type" in {finding.code for finding in findings}


def test_non_numeric_success_and_fractional_order_are_errors() -> None:
    manifest = _manifest()
    manifest["success"] = manifest["success"].astype(object)
    manifest["condition_order"] = manifest["condition_order"].astype(float)
    manifest.loc[0, "success"] = "garbage"
    manifest.loc[0, "condition_order"] = 1.9
    _, findings = MODULE.validate_manifest_metadata(manifest)
    codes = {finding.code for finding in findings if finding.severity == "ERROR"}

    assert {"success", "condition_order"}.issubset(codes)


def test_negative_or_fractional_numeric_metadata_is_rejected() -> None:
    manifest = _manifest()
    manifest["dropped_frames"] = manifest["dropped_frames"].astype(float)
    manifest.loc[0, "placement_error_m"] = -0.1
    manifest.loc[0, "completion_time_s"] = -1
    manifest.loc[0, "video_latency_ms"] = -5
    manifest.loc[0, "dropped_frames"] = 1.5
    _, findings = MODULE.validate_manifest_metadata(manifest)
    codes = {finding.code for finding in findings if finding.severity == "ERROR"}

    assert {
        "placement_error_m",
        "completion_time_s",
        "video_latency_ms",
        "dropped_frames",
    }.issubset(codes)


def test_nan_and_infinity_manifest_numbers_are_rejected() -> None:
    manifest = _manifest()
    manifest["placement_error_m"] = [float("inf"), 0.003]
    manifest["completion_time_s"] = [10.0, float("-inf")]
    manifest["video_latency_ms"] = [float("nan"), float("inf")]
    manifest["dropped_frames"] = [0, float("inf")]

    _, findings = MODULE.validate_manifest_metadata(manifest, _config())
    codes = {finding.code for finding in findings if finding.severity == "ERROR"}

    assert {
        "placement_error_m",
        "completion_time_s",
        "video_latency_ms",
        "dropped_frames",
    }.issubset(codes)


def test_required_capture_measurements_cannot_be_blank() -> None:
    manifest = _manifest()
    manifest.loc[0, "video_latency_ms"] = np.nan
    manifest.loc[1, "dropped_frames"] = np.nan

    _, findings = MODULE.validate_manifest_metadata(manifest, _config())
    errors = {finding.code for finding in findings if finding.severity == "ERROR"}

    assert {"video_latency_ms_incomplete", "dropped_frames_incomplete"}.issubset(errors)


@pytest.mark.parametrize(("field", "value"), [("seed", "None"), ("split", "<NA>")])
def test_optional_identifier_columns_reject_null_sentinel_strings(field: str, value: str) -> None:
    manifest = _manifest()
    manifest["seed"] = ["0", "0"]
    manifest["split"] = ["test", "test"]
    manifest.loc[0, field] = value

    _, findings = MODULE.validate_manifest_metadata(manifest, _config())
    errors = [finding for finding in findings if finding.severity == "ERROR"]

    assert errors
    assert field in errors[0].message


def test_bad_random_seed_values_report_finding_instead_of_crashing() -> None:
    config = _config()
    config["training"]["random_seeds"] = [[0], [1], [2]]

    findings = MODULE.validate_protocol(config)

    assert "random_seeds" in {finding.code for finding in findings}


def test_nan_and_boolean_protocol_numbers_are_rejected_cleanly() -> None:
    config = _config()
    config["capture"]["target_fps"] = True
    config["training"]["train_fraction"] = float("nan")

    codes = {finding.code for finding in MODULE.validate_protocol(config)}

    assert {"fps", "split_fractions"}.issubset(codes)


def test_success_labels_must_match_protocol_thresholds_and_tasks() -> None:
    manifest = _manifest()
    manifest.loc[:, "placement_error_m"] = 0.1
    manifest.loc[:, "completion_time_s"] = 100.0
    manifest.loc[:, "task_id"] = "unknown_task"
    _, findings = MODULE.validate_manifest_metadata(manifest, _config())
    codes = {finding.code for finding in findings if finding.severity == "ERROR"}

    assert {"success_threshold", "success_timeout", "task_id"}.issubset(codes)


def test_participant_condition_order_must_be_counterbalanced() -> None:
    first = _manifest()
    second = _manifest()
    second["participant_id"] = "P02"
    second["pair_id"] = "pair_1"
    second["episode"] = 1
    second["condition_order"] = second["condition_order"].map({1: 2, 2: 1})
    balanced = pd.concat([first, second], ignore_index=True)

    _, balanced_findings = MODULE.validate_manifest_metadata(balanced, _config())
    assert "counterbalance" not in {finding.code for finding in balanced_findings}

    second["condition_order"] = second["condition_order"].map({1: 2, 2: 1})
    unbalanced = pd.concat([first, second], ignore_index=True)
    _, unbalanced_findings = MODULE.validate_manifest_metadata(unbalanced, _config())
    assert "counterbalance" in {finding.code for finding in unbalanced_findings}


def test_video_clock_contract_rejects_unmapped_domains() -> None:
    assert "artifact_type" in MODULE._video_clock_contract_error({"schema_version": 1}, 0.01)
    meta = _valid_video_meta()
    del meta["clock_contract"]
    assert "clock_contract" in MODULE._video_clock_contract_error(meta, 0.01)


def test_video_clock_contract_rejects_timing_only_writer_and_excess_uncertainty() -> None:
    meta = _valid_video_meta()
    meta["writer_capability"] = "timing_metadata_only"
    meta["video_capture_verified_by_this_writer"] = False
    assert "writer_capability" in MODULE._video_clock_contract_error(meta, 0.01)

    meta = _valid_video_meta()
    meta["clock_contract"]["frame_to_unix"]["uncertainty_s"] = 1e9
    meta["clock_contract"]["episode_to_unix"]["uncertainty_s"] = 1e9
    assert "uncertainty_s exceeds" in MODULE._video_clock_contract_error(meta, 0.01)


def test_frame_clock_crosscheck_uses_absolute_not_epoch_scaled_tolerance(
    tmp_path: Path,
) -> None:
    epoch = 1_700_000_000.0
    meta = _valid_video_meta()
    mapping = meta["clock_contract"]["frame_to_unix"]
    mapping["source_anchor_s"] = epoch
    mapping["unix_anchor_s"] = epoch
    meta["clock_contract"]["episode_to_unix"] = dict(mapping)
    path = tmp_path / "frame_timestamps.csv"
    path.write_text(
        "frame_index,frame_timestamp_s,frame_clock_id,unix_timestamp_s\n"
        f"0,{epoch},capture_unix,{epoch + 0.1}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match clock mapping"):
        MODULE._read_frame_timestamp_csv(path, meta)


def test_raw_tree_validator_decodes_video_and_validates_stream_contents(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "A_mobile_colocated" / "episode_000"
    episode.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": [10.0, 10.2],
            "joint_1": [0.0, 0.1],
            "joint_2": [0.0, 0.1],
        }
    ).to_csv(episode / "robot.csv", index=False)
    actions = pd.DataFrame(
        {
            "timestamp": [10.0],
            "dx": [0.0],
            "dy": [0.0],
            "dz": [0.0],
            "dyaw": [0.0],
            "dpitch": [0.0],
            "droll": [0.0],
        }
    )
    actions.to_csv(episode / "phone.csv", index=False)
    actions.to_csv(episode / "applied_actions.csv", index=False)
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    time_sync._write_video(
        episode / "video.mp4",
        frames,
        np.array([0, 1]),
        fps=10,
    )
    (episode / "video_meta.json").write_text(
        json.dumps(_valid_video_meta()),
        encoding="utf-8",
    )
    (episode / "frame_timestamps.csv").write_text(
        "frame_index,frame_timestamp_s,frame_clock_id,unix_timestamp_s\n"
        "0,10.0,capture_unix,10.0\n"
        "1,10.1,capture_unix,10.1\n",
        encoding="utf-8",
    )
    manifest = pd.DataFrame([{"condition": "A_mobile_colocated", "episode": 0}])
    required = [
        "robot.csv",
        "phone.csv",
        "applied_actions.csv",
        "video.mp4",
        "video_meta.json",
        "frame_timestamps.csv",
    ]

    assert (
        MODULE.validate_raw_tree(
            tmp_path,
            manifest,
            required,
            ["joint_1", "joint_2"],
            0.01,
        )
        == []
    )

    (episode / "frame_timestamps.csv").write_text(
        "frame_index,frame_timestamp_s,frame_clock_id,unix_timestamp_s\n"
        "0,10.0,capture_unix,10.0\n"
        "1,10.1,capture_unix,10.15\n",
        encoding="utf-8",
    )
    timestamp_findings = MODULE.validate_raw_tree(
        tmp_path,
        manifest,
        required,
        ["joint_1", "joint_2"],
        0.01,
    )
    assert "video_frame_timestamp_csv" in {finding.code for finding in timestamp_findings}
    (episode / "frame_timestamps.csv").write_text(
        "frame_index,frame_timestamp_s,frame_clock_id,unix_timestamp_s\n"
        "0,10.0,capture_unix,10.0\n"
        "1,10.1,capture_unix,10.1\n",
        encoding="utf-8",
    )

    (episode / "applied_actions.csv").write_text(
        "timestamp,dx,dy,dz,dyaw,dpitch,droll\n10,0,0,0,0,0,0\n10,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    findings = MODULE.validate_raw_tree(
        tmp_path,
        manifest,
        required,
        ["joint_1", "joint_2"],
        0.01,
    )
    assert "csv_content" in {finding.code for finding in findings}

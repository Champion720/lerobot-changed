from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[3]
EXPERIMENT_DIR = ROOT / "experiments" / "wrist_view_presentation"


def _load(name: str):
    path = EXPERIMENT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = _load("generate_experiment_manifest")
SELECTOR = _load("select_training_episodes")
ROLLOUT = _load("make_rollout_manifest")
ROLLOUT_ANALYSIS = _load("analyze_rollouts")
CATALOGS = _load("experiment_catalogs")
GRIPPER = _load("gripper_contract")
PREFLIGHT = _load("validate_experiment_setup")
CONVERTER = _load("convert_raw_to_lerobot")


@pytest.fixture
def config() -> dict:
    return json.loads((EXPERIMENT_DIR / "experiment_config.example.json").read_text(encoding="utf-8"))


def test_formal_manifest_has_exact_frozen_quotas(config: dict) -> None:
    manifest = GENERATOR.generate_manifest(config)
    GENERATOR.validate_generated_manifest(manifest)

    assert len(manifest) == 200
    assert manifest["participant_id"].nunique() == 20
    assert manifest.groupby(["participant_id", "condition"]).size().eq(5).all()
    assert manifest.groupby(["condition", "task_id"]).size().to_dict() == {
        ("A_mobile_colocated", "color_sorting"): 50,
        ("A_mobile_colocated", "stacking"): 50,
        ("B_desktop_separated", "color_sorting"): 50,
        ("B_desktop_separated", "stacking"): 50,
    }
    group_counts = manifest[["participant_id", "group_id"]].drop_duplicates()["group_id"].value_counts()
    assert group_counts.to_dict() == {"G1": 5, "G2": 5, "G3": 5, "G4": 5}


def test_formal_manifest_is_deterministic_and_pair_metadata_matches(config: dict) -> None:
    first = GENERATOR.generate_manifest(config)
    second = GENERATOR.generate_manifest(config)
    pd.testing.assert_frame_equal(first, second)

    for _, pair in first.groupby("pair_id"):
        assert len(pair) == 2
        assert set(pair["condition"]) == set(GENERATOR.CONDITIONS)
        for column in ("participant_id", "group_id", "task_id", "layout_id", "pair_index"):
            assert pair[column].nunique() == 1


def test_condition_blocks_avoid_three_identical_tasks_in_a_row(config: dict) -> None:
    manifest = GENERATOR.generate_manifest(config)
    ordered = manifest.sort_values(["participant_id", "condition_order", "within_condition_trial_index"])
    for _, block in ordered.groupby(["participant_id", "condition"]):
        tasks = block["task_id"].tolist()
        assert all(not (tasks[index] == tasks[index + 1] == tasks[index + 2]) for index in range(3))


def _completed_manifest(config: dict) -> pd.DataFrame:
    manifest = GENERATOR.generate_manifest(config)
    manifest["success"] = 1
    manifest["stage_score"] = 1.0
    manifest["manual_intervention"] = 0
    manifest["recording_valid"] = 1
    manifest["sync_valid"] = 1
    manifest["gripper_valid"] = 1
    return manifest


def test_training_selection_equalizes_all_four_cells(config: dict) -> None:
    manifest = _completed_manifest(config)
    invalid = manifest.index[
        manifest["condition"].eq("A_mobile_colocated") & manifest["task_id"].eq("stacking")
    ][:5]
    manifest.loc[invalid, "success"] = 0
    manifest.loc[invalid, "stage_score"] = 0.5

    selected, summary = SELECTOR.select_balanced_training_rows(manifest, selection_seed=41)
    selected_rows = selected.loc[selected["selected_for_training"]]
    assert summary["target_per_condition_task_cell"] == 45
    assert selected_rows.groupby(["condition", "task_id"]).size().eq(45).all()
    assert not selected.loc[invalid, "selected_for_training"].any()
    assert selected.loc[invalid, "eligibility_reason"].str.contains("not_complete_success").all()


def test_training_selection_is_deterministic(config: dict) -> None:
    manifest = _completed_manifest(config)
    first, first_summary = SELECTOR.select_balanced_training_rows(manifest, selection_seed=19)
    second, second_summary = SELECTOR.select_balanced_training_rows(manifest, selection_seed=19)
    pd.testing.assert_series_equal(first["selected_for_training"], second["selected_for_training"])
    assert first_summary == second_summary


def test_training_selection_rejects_missing_eligible_cell(config: dict) -> None:
    manifest = _completed_manifest(config)
    cell = manifest["condition"].eq("B_desktop_separated") & manifest["task_id"].eq("color_sorting")
    manifest.loc[cell, "recording_valid"] = 0
    with pytest.raises(ValueError, match="no eligible demonstrations"):
        SELECTOR.select_balanced_training_rows(manifest, selection_seed=3)


def test_training_selection_enforces_preregistered_minimum(config: dict) -> None:
    manifest = _completed_manifest(config)
    with pytest.raises(ValueError, match="go/no-go failed"):
        SELECTOR.select_balanced_training_rows(manifest, selection_seed=3, minimum_per_cell=51)


def test_rollout_manifest_has_240_matched_trials(config: dict) -> None:
    manifest = ROLLOUT.generate_rollout_manifest(config)
    ROLLOUT.validate_rollout_manifest(manifest, expected_seeds=[0, 1, 2])

    assert len(manifest) == 240
    assert manifest.groupby(["training_condition", "seed", "task_id"]).size().eq(20).all()
    assert manifest.groupby(["training_condition", "seed", "task_id", "layout_regime"]).size().eq(10).all()
    assert sorted(manifest["execution_order"].tolist()) == list(range(1, 241))


def test_rollout_validator_rejects_a_missing_trial(config: dict) -> None:
    manifest = ROLLOUT.generate_rollout_manifest(config).iloc[:-1].copy()
    with pytest.raises(ValueError, match="must contain 240 rows"):
        ROLLOUT.validate_rollout_manifest(manifest, expected_seeds=[0, 1, 2])


def test_rollout_analysis_preserves_matched_a_minus_b_direction(config: dict) -> None:
    manifest = ROLLOUT.generate_rollout_manifest(config)
    manifest["success"] = manifest["training_condition"].eq("A_mobile_colocated").astype(int)
    manifest["stage_score"] = manifest["success"]
    manifest["completion_time_s"] = 20.0
    manifest["grasp_retries"] = 0
    manifest["drop_count"] = 0
    manifest["manual_intervention"] = 0
    manifest["failure_type"] = manifest["success"].map({1: "", 0: "task_failure"})
    manifest["safety_stop"] = 0
    manifest["inference_latency_ms"] = 40.0
    manifest["video_latency_ms"] = 80.0

    completed = ROLLOUT_ANALYSIS.validate_completed_results(manifest, expected_seeds=[0, 1, 2])
    condition_summary, matched_summary = ROLLOUT_ANALYSIS.summarize_results(completed)
    overall = matched_summary.loc[
        matched_summary["seed"].eq("ALL")
        & matched_summary["task_id"].eq("ALL")
        & matched_summary["layout_regime"].eq("ALL")
    ].iloc[0]
    assert len(condition_summary) == 34
    assert set(condition_summary["seed"].astype(str)) == {"0", "1", "2", "ALL"}
    assert overall["n_pairs"] == 120
    assert overall["mean_success_A_minus_B"] == 1.0
    assert overall["ci95_low_success_A_minus_B"] == 1.0
    assert overall["ci95_high_success_A_minus_B"] == 1.0


def test_rollout_analysis_rejects_incomplete_results(config: dict) -> None:
    manifest = ROLLOUT.generate_rollout_manifest(config)
    with pytest.raises(ValueError, match="incomplete"):
        ROLLOUT_ANALYSIS.validate_completed_results(manifest, expected_seeds=[0, 1, 2])


def _freeze_blocked_protocol_values(config: dict) -> dict:
    config = json.loads(json.dumps(config))
    config["capture"]["max_clock_uncertainty_s"] = 0.01
    config["capture"]["gripper"] = {
        "action_schema_id": "test.gripper_target",
        "action_unit": "normalized",
        "action_column": "gripper_target",
        "state_schema_id": "test.gripper_position",
        "state_unit": "m",
        "state_column": "gripper_position",
        "action_open_value": 0.0,
        "action_closed_value": 1.0,
        "state_open_value": 0.0,
        "state_closed_value": 0.04,
    }
    config["layout_catalog_path"] = "formal_layouts.json"
    config["training"]["minimum_eligible_per_condition_task"] = 1
    for task in config["tasks"]:
        task["max_observation_time_s"] = 60.0
    smolvla = config["training"]["policy_families"]["smolvla"]
    smolvla["batch_size"] = 1
    smolvla["steps"] = 10
    smolvla["chunk_size"] = 50
    smolvla["n_action_steps"] = 5
    config["evaluation"]["reset_catalog_path"] = "formal_resets.json"
    config["evaluation"]["deployment"] = {
        "control_frequency_hz": 10.0,
        "watchdog_timeout_ms": 500.0,
        "command_expiry_ms": 200.0,
    }
    return config


def test_v2_protocol_reports_only_intentional_example_blockers(config: dict) -> None:
    codes = {finding.code for finding in PREFLIGHT.validate_protocol(config)}
    assert {
        "max_clock_uncertainty",
        "gripper_schema",
        "gripper_value",
        "max_observation_time",
        "layout_catalog",
        "training_value",
        "minimum_eligible",
        "reset_catalog",
        "deployment_value",
    }.issubset(codes)


def test_frozen_v2_protocol_and_completed_manifest_pass(config: dict) -> None:
    frozen = _freeze_blocked_protocol_values(config)
    assert PREFLIGHT.validate_protocol(frozen) == []

    manifest = _completed_manifest(frozen)
    manifest["completion_time_s"] = 20.0
    manifest["successful_grasps"] = manifest["task_id"].map({"stacking": 1, "color_sorting": 2})
    manifest["grasp_retries"] = 0
    manifest["drop_count"] = 0
    manifest["stable_for_2s"] = 1
    manifest["failure_type"] = ""
    manifest["video_latency_ms"] = 50.0
    manifest["dropped_frames"] = 0
    manifest["clock_uncertainty_s"] = 0.005
    manifest["training_eligible"] = 1
    manifest["exclusion_reason"] = ""
    normalized, findings = PREFLIGHT.validate_manifest_metadata(manifest, frozen)
    assert normalized is not None
    assert findings == []


def test_gripper_action_and_state_ranges_are_independent(config: dict) -> None:
    contract = _freeze_blocked_protocol_values(config)["capture"]["gripper"]
    GRIPPER.validate_values([[0.5]], contract, stream="action", label="action")
    with pytest.raises(ValueError, match="within"):
        GRIPPER.validate_values([[0.5]], contract, stream="state", label="state")


def test_converter_loads_per_episode_policy_prompts(config: dict, tmp_path: Path) -> None:
    manifest = GENERATOR.generate_manifest(config)
    manifest_path = tmp_path / "manifest.csv"
    config_path = tmp_path / "config.json"
    manifest.to_csv(manifest_path, index=False)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    prompts = CONVERTER.load_episode_task_prompts(
        manifest_path,
        config_path,
        "A_mobile_colocated",
    )
    assert len(prompts) == 100
    expected = {task["task_id"]: task["policy_task_en"] for task in config["tasks"]}
    a_rows = manifest.loc[manifest["condition"].eq("A_mobile_colocated")]
    for row in a_rows.itertuples(index=False):
        assert prompts[row.episode] == expected[row.task_id]


def test_catalogs_require_finite_poses_and_exact_protocol_coverage(config: dict, tmp_path: Path) -> None:
    config = _freeze_blocked_protocol_values(config)
    layout_path = tmp_path / "formal_layouts.json"
    reset_path = tmp_path / "formal_resets.json"
    config["layout_catalog_path"] = layout_path.name
    config["evaluation"]["reset_catalog_path"] = reset_path.name
    pose = {"position_xyz": [0.1, 0.2, 0.3], "orientation_rpy": [0.0, 0.0, 0.0]}
    layouts = [
        {"task_id": task["task_id"], "layout_id": layout_id, "objects": {"fixture": pose}}
        for task in config["tasks"]
        for layout_id in task["collection_layout_ids"]
    ]
    resets = [
        {
            "task_id": task_id,
            "layout_regime": regime,
            "reset_id": reset_id,
            "objects": {"fixture": pose},
        }
        for task_id, regimes in config["evaluation"]["reset_ids"].items()
        for regime, reset_ids in regimes.items()
        for reset_id in reset_ids
    ]
    header = {
        "schema_version": 1,
        "coordinate_frame": "robot_base",
        "position_unit": "m",
        "angle_unit": "rad",
    }
    layout_path.write_text(json.dumps({**header, "layouts": layouts}), encoding="utf-8")
    reset_path.write_text(json.dumps({**header, "resets": resets}), encoding="utf-8")
    CATALOGS.validate_catalog_references(config, config_dir=tmp_path)

    layouts[0]["objects"]["fixture"] = {
        "position_xyz": [None, 0.2, 0.3],
        "orientation_rpy": [0.0, 0.0, 0.0],
    }
    layout_path.write_text(json.dumps({**header, "layouts": layouts}), encoding="utf-8")
    with pytest.raises(ValueError, match="three finite numbers"):
        CATALOGS.validate_catalog_references(config, config_dir=tmp_path)

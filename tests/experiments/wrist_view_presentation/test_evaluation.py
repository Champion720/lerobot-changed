from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.wrist_view_presentation import (
    analysis_reproducibility,
    feature_extraction,
    features_compare,
    offline_compare,
)


def test_manifest_selection_and_pair_mapping_do_not_assume_equal_episode_indices(tmp_path):
    manifest = tmp_path / "splits.csv"
    pd.DataFrame(
        [
            {
                "condition": "A_mobile_colocated",
                "episode": 0,
                "pair_id": "p0",
                "participant_id": "P0",
                "split": "train",
            },
            {
                "condition": "B_desktop_separated",
                "episode": 5,
                "pair_id": "p0",
                "participant_id": "P0",
                "split": "train",
            },
            {
                "condition": "A_mobile_colocated",
                "episode": 2,
                "pair_id": "p1",
                "participant_id": "P1",
                "split": "test",
            },
            {
                "condition": "B_desktop_separated",
                "episode": 8,
                "pair_id": "p1",
                "participant_id": "P1",
                "split": "test",
            },
            {
                "condition": "A_mobile_colocated",
                "episode": 4,
                "pair_id": "p2",
                "participant_id": "P2",
                "split": "test",
            },
            {
                "condition": "B_desktop_separated",
                "episode": 9,
                "pair_id": "p2",
                "participant_id": "P2",
                "split": "test",
            },
        ]
    ).to_csv(manifest, index=False)

    selected_a, train_a = offline_compare._load_manifest_selection(
        str(manifest), condition="A_mobile_colocated", split="test"
    )
    pairs = offline_compare._load_manifest_pairs(
        str(manifest),
        condition_a="A_mobile_colocated",
        condition_b="B_desktop_separated",
        split="test",
    )

    assert selected_a == [2, 4]
    assert train_a == [0]
    assert pairs == [("p1", 2, 8), ("p2", 4, 9)]

    errors_a = {2: {"mae": 1.0}, 4: {"mae": 3.0}}
    errors_b = {8: {"mae": 2.0}, 9: {"mae": 5.0}}
    a, b, test = offline_compare._compare_metric(errors_a, errors_b, "mae", paired=True, episode_pairs=pairs)
    np.testing.assert_array_equal(a, [1.0, 3.0])
    np.testing.assert_array_equal(b, [2.0, 5.0])
    assert test == "paired t-test"


def test_paired_without_manifest_requires_identical_episode_sets():
    with pytest.raises(ValueError, match="identical analysis-unit sets"):
        offline_compare._compare_metric(
            {1: {"mse": 1.0}},
            {2: {"mse": 1.0}},
            "mse",
            paired=True,
        )


def test_episode_spec_and_split_audit(tmp_path):
    assert offline_compare._parse_episode_spec("1, 3-5,3") == [1, 3, 4, 5]
    with pytest.raises(ValueError, match="invalid episode range"):
        offline_compare._parse_episode_spec("4-2")

    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    config_path = checkpoint / "train_config.json"
    config_path.write_text(
        json.dumps({"dataset": {"repo_id": "local/A", "episodes": [0, 1, 2]}}),
        encoding="utf-8",
    )
    status, _ = offline_compare._leakage_status(str(checkpoint), "local/A", [2, 3], declared_train=None)
    assert status == "overlap"

    config_path.write_text(
        json.dumps({"dataset": {"repo_id": "local/A", "episodes": [0, 1]}}),
        encoding="utf-8",
    )
    status, message = offline_compare._leakage_status(
        str(checkpoint), "local/A", [2, 3], declared_train=[0, 1]
    )
    assert status == "verified"
    assert "no train/evaluation overlap" in message


def test_manifest_rejects_participant_crossing_splits(tmp_path):
    manifest = tmp_path / "crossed.csv"
    pd.DataFrame(
        [
            {
                "condition": "A_mobile_colocated",
                "episode": 0,
                "participant_id": "P1",
                "split": "train",
            },
            {
                "condition": "B_desktop_separated",
                "episode": 0,
                "participant_id": "P1",
                "split": "test",
            },
        ]
    ).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="must not cross"):
        offline_compare._read_split_manifest(str(manifest))


def test_offline_aggregates_episodes_then_training_seeds_to_participant(tmp_path):
    manifest = tmp_path / "participant.csv"
    pd.DataFrame(
        [
            {
                "condition": "A_mobile_colocated",
                "episode": 0,
                "participant_id": "P1",
                "seed": "collection_seed_that_must_be_ignored",
                "split": "test",
            },
            {
                "condition": "A_mobile_colocated",
                "episode": 1,
                "participant_id": "P1",
                "seed": "collection_seed_that_must_be_ignored",
                "split": "test",
            },
            {
                "condition": "A_mobile_colocated",
                "episode": 2,
                "participant_id": "P2",
                "seed": "collection_seed_that_must_be_ignored",
                "split": "test",
            },
            {
                "condition": "A_mobile_colocated",
                "episode": 3,
                "participant_id": "P2",
                "seed": "collection_seed_that_must_be_ignored",
                "split": "test",
            },
        ]
    ).to_csv(manifest, index=False)
    participant, provenance, seed_level = offline_compare._aggregate_training_seed_runs_by_participant(
        {
            "0": {
                0: {"translation_rmse_m": 1.0},
                1: {"translation_rmse_m": 3.0},
                2: {"translation_rmse_m": 2.0},
                3: {"translation_rmse_m": 2.0},
            },
            "1": {
                0: {"translation_rmse_m": 5.0},
                1: {"translation_rmse_m": 7.0},
                2: {"translation_rmse_m": 2.0},
                3: {"translation_rmse_m": 2.0},
            },
        },
        str(manifest),
        condition="A_mobile_colocated",
        split="test",
    )
    # P1: seed 0 episode mean=2, seed 1 episode mean=6, cross-seed mean=4.
    assert participant["P1"]["translation_rmse_m"] == pytest.approx(4.0)
    assert participant["P2"]["translation_rmse_m"] == pytest.approx(2.0)
    p1 = next(row for row in provenance if row["participant_id"] == "P1")
    assert p1["n_training_seeds"] == 2
    assert p1["training_seeds"] == ["0", "1"]
    assert seed_level["0"]["P1"]["translation_rmse_m"] == pytest.approx(2.0)

    # The legacy one-checkpoint helper ignores the manifest's collection seed.
    one_checkpoint, one_provenance = offline_compare._aggregate_metrics_by_participant(
        {
            0: {"translation_rmse_m": 1.0},
            1: {"translation_rmse_m": 3.0},
            2: {"translation_rmse_m": 2.0},
            3: {"translation_rmse_m": 2.0},
        },
        str(manifest),
        condition="A_mobile_colocated",
        split="test",
    )
    assert one_checkpoint["P1"]["translation_rmse_m"] == pytest.approx(2.0)
    assert one_provenance[0]["n_training_seeds"] == 1


def test_checkpoint_seed_mappings_must_exactly_match_protocol(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps({"training": {"random_seeds": [0, 1, 2]}}),
        encoding="utf-8",
    )
    mapping = tmp_path / "checkpoints.json"
    mapping.write_text(
        json.dumps({"0": "a/0", "1": "a/1", "2": "a/2"}),
        encoding="utf-8",
    )

    seeds = offline_compare._load_preregistered_training_seeds(str(protocol))
    checkpoints = offline_compare._load_checkpoint_mapping(str(mapping), field="A")
    offline_compare._validate_checkpoint_seed_design(checkpoints, dict(checkpoints), seeds)

    with pytest.raises(ValueError, match="A/B checkpoint seed sets differ"):
        offline_compare._validate_checkpoint_seed_design(
            checkpoints,
            {"0": "b/0", "1": "b/1"},
            seeds,
        )
    with pytest.raises(ValueError, match="exactly match preregistered"):
        offline_compare._validate_checkpoint_seed_design(
            {"0": "a/0", "1": "a/1"},
            {"0": "b/0", "1": "b/1"},
            seeds,
        )
    with pytest.raises(ValueError, match="same checkpoint path"):
        offline_compare._validate_checkpoint_seed_design(
            {"0": "same", "1": "same", "2": "same"},
            {"0": "b/0", "1": "b/1", "2": "b/2"},
            seeds,
        )


def test_formal_checkpoint_artifacts_prove_seed_and_independent_training(tmp_path):
    def checkpoint(name: str, seed: int, artifact: bytes) -> Path:
        root = tmp_path / name
        root.mkdir()
        (root / "train_config.json").write_text(
            json.dumps({"seed": seed}),
            encoding="utf-8",
        )
        (root / "model.safetensors").write_bytes(artifact)
        return root

    a0 = checkpoint("a0", 0, b"a-model-0")
    a1 = checkpoint("a1", 1, b"a-model-1")
    evidence_a = offline_compare._validate_local_checkpoint_artifacts(
        {"0": str(a0), "1": str(a1)},
        condition="A",
    )
    assert evidence_a["0"]["train_config_seed"] == 0
    assert evidence_a["0"]["artifact_sha256"] != evidence_a["1"]["artifact_sha256"]

    wrong_seed = checkpoint("wrong", 7, b"different")
    with pytest.raises(ValueError, match="does not match"):
        offline_compare._validate_local_checkpoint_artifacts(
            {"2": str(wrong_seed)},
            condition="A",
        )

    duplicate = checkpoint("duplicate", 1, b"a-model-0")
    with pytest.raises(ValueError, match="identical checkpoint artifacts"):
        offline_compare._validate_local_checkpoint_artifacts(
            {"0": str(a0), "1": str(duplicate)},
            condition="A",
        )

    b0 = checkpoint("b0", 0, b"a-model-0")
    evidence_b = offline_compare._validate_local_checkpoint_artifacts(
        {"0": str(b0)},
        condition="B",
    )
    with pytest.raises(ValueError, match="A and B reuse identical"):
        offline_compare._reject_cross_condition_checkpoint_reuse(
            {"0": evidence_a["0"]},
            evidence_b,
        )


def test_training_seeds_must_evaluate_the_identical_held_out_episodes():
    audits = {
        "0": {"selected_episodes": [1, 2]},
        "1": {"selected_episodes": [1, 3]},
    }
    with pytest.raises(ValueError, match="identical held-out episodes"):
        offline_compare._require_identical_episode_selection(audits, condition="A_mobile_colocated")


def test_reproducibility_helpers_hash_inputs_and_write_strict_json(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    fingerprint = analysis_reproducibility.fingerprint_path(source)
    assert fingerprint is not None
    assert fingerprint["kind"] == "file"
    assert len(fingerprint["sha256"]) == 64

    sidecar = tmp_path / "metadata.json"
    analysis_reproducibility.write_json(sidecar, {"input": fingerprint})
    assert sidecar.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(ValueError, match="Out of range float"):
        analysis_reproducibility.write_json(sidecar, {"invalid": float("nan")})


def test_masked_action_errors_report_mae_mse_and_rmse():
    predicted = torch.zeros((1, 2, 6))
    target = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
                [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            ]
        ]
    )
    is_pad = torch.tensor([[False, True]])
    absolute, squared, count = offline_compare._masked_error_sums(predicted, target, is_pad)

    np.testing.assert_allclose(absolute.numpy(), [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])
    np.testing.assert_allclose(squared.numpy(), [[1.0, 4.0, 9.0, 0.01, 0.04, 0.09]], rtol=1e-6)
    np.testing.assert_array_equal(count.numpy(), [[1, 1, 1, 1, 1, 1]])
    metrics = offline_compare._finalize_error_metrics(
        {0: absolute.numpy()[0]},
        {0: squared.numpy()[0]},
        {0: count.numpy()[0]},
        ["dx", "dy", "dz", "dyaw", "dpitch", "droll"],
        ["m", "m", "m", "rad", "rad", "rad"],
    )
    assert metrics[0]["dx_mae_m"] == pytest.approx(1.0)
    assert metrics[0]["translation_mse_m2"] == pytest.approx(14 / 3)
    assert metrics[0]["translation_rmse_m"] == pytest.approx(np.sqrt(14 / 3))
    assert metrics[0]["rotation_mse_rad2"] == pytest.approx(0.14 / 3)


def test_offline_inference_uses_public_chunk_api_and_physical_postprocessor():
    class FakePolicy:
        called = False

        def predict_action_chunk(self, batch):
            self.called = True
            assert "observation.images.cam" in batch
            return torch.ones((2, 3, 6))

        @property
        def model(self):
            raise AssertionError("direct model access is forbidden")

    policy = FakePolicy()
    physical = offline_compare._predict_physical_action_chunk(
        policy,
        lambda action: action * 0.01,
        {"observation.images.cam": torch.zeros((2, 3, 8, 8))},
    )
    assert policy.called
    assert physical.shape == (2, 3, 6)
    assert torch.allclose(physical, torch.full_like(physical, 0.01))


def test_offline_requires_scale_metadata_and_a_common_physical_schema():
    cfg = SimpleNamespace(normalization_mapping={"ACTION": "MEAN_STD"})
    with pytest.raises(RuntimeError, match="no action scale statistics"):
        offline_compare._require_action_scale_metadata(cfg, SimpleNamespace(steps=[]))

    names = ["dx", "dy", "dz", "dyaw", "dpitch", "droll"]
    meta_a = SimpleNamespace(features={"action": {"names": names}})
    meta_b = SimpleNamespace(features={"action": {"names": list(names)}})
    action_names, units = offline_compare._validate_common_action_schema(meta_a, meta_b, None)
    assert action_names == names
    assert units == ["m", "m", "m", "rad", "rad", "rad"]

    meta_b.features["action"]["names"][-1] = "gripper"
    with pytest.raises(ValueError, match="schemas differ"):
        offline_compare._validate_common_action_schema(meta_a, meta_b, None)


def test_physical_roundtrip_fails_instead_of_using_normalized_scale():
    target = torch.tensor([[[0.1, 0.2]]])
    normalized = torch.tensor([[[1.0, 2.0]]])
    offline_compare._verify_physical_roundtrip(target, normalized, lambda value: value / 10)
    with pytest.raises(RuntimeError, match="does not round-trip"):
        offline_compare._verify_physical_roundtrip(target, normalized, lambda value: value)


def test_labels_are_one_to_one_and_completion_time_excludes_failures():
    features = pd.DataFrame(
        {
            "condition": ["A_mobile_colocated", "A_mobile_colocated"],
            "episode": [0, 1],
            "raw_duration_s": [10.0, 60.0],
        }
    )
    labels = feature_extraction.validate_labels(
        pd.DataFrame(
            {
                "episode": [0, 1],
                "success": ["yes", "failed"],
                "participant_id": ["P1", "P1"],
                "failure_type": [None, "timeout"],
            }
        ),
        [0, 1],
    )
    merged = feature_extraction.attach_labels_and_completion_times(features, labels)
    assert merged.loc[merged["episode"] == 0, "completion_time_s"].item() == 10.0
    assert np.isnan(merged.loc[merged["episode"] == 1, "completion_time_s"].item())
    assert merged.loc[merged["episode"] == 1, "raw_duration_s"].item() == 60.0

    recorded_labels = feature_extraction.validate_labels(
        pd.DataFrame(
            {
                "episode": [0, 1],
                "success": [1, 0],
                "participant_id": ["P1", "P1"],
                "completion_time_s": [7.5, 60.0],
            }
        ),
        [0, 1],
    )
    recorded = feature_extraction.attach_labels_and_completion_times(features, recorded_labels)
    assert recorded.loc[recorded["episode"] == 0, "completion_time_s"].item() == 7.5
    assert recorded.loc[recorded["episode"] == 0, "reported_completion_time_s"].item() == 7.5
    assert np.isnan(recorded.loc[recorded["episode"] == 1, "completion_time_s"].item())
    assert recorded.loc[recorded["episode"] == 1, "raw_duration_s"].item() == 60.0

    with pytest.raises(ValueError, match="duplicate episode"):
        feature_extraction.validate_labels(
            pd.DataFrame(
                {
                    "episode": [0, 0],
                    "success": [1, 0],
                    "participant_id": ["P1", "P1"],
                }
            ),
            [0],
        )
    with pytest.raises(ValueError, match="no label"):
        feature_extraction.validate_labels(
            pd.DataFrame({"episode": [0], "success": [1], "participant_id": ["P1"]}),
            [0, 1],
        )
    with pytest.raises(ValueError, match="absent from the dataset"):
        feature_extraction.validate_labels(
            pd.DataFrame(
                {
                    "episode": [0, 1],
                    "success": [1, 0],
                    "participant_id": ["P1", "P1"],
                }
            ),
            [0],
        )
    with pytest.raises(ValueError, match="binary"):
        feature_extraction.validate_labels(
            pd.DataFrame(
                {
                    "episode": [0],
                    "success": ["maybe"],
                    "participant_id": ["P1"],
                }
            ),
            [0],
        )
    with pytest.raises(ValueError, match="finite, positive"):
        feature_extraction.validate_labels(
            pd.DataFrame(
                {"episode": [0], "success": [1], "completion_time_s": [np.nan]} | {"participant_id": ["P1"]}
            ),
            [0],
        )


def test_bad_episode_quality_is_an_outcome_row_not_a_silent_drop():
    quality = feature_extraction._episode_quality(np.zeros((3, 6)), np.asarray([0.0, 0.1, 0.2]))
    assert quality == {
        "n_frames": 3,
        "quality_status": "too_short_for_jerk",
        "feature_usable": False,
        "raw_duration_s": pytest.approx(0.2),
    }


def test_savgol_quality_gate_rejects_four_frames_and_irregular_sampling():
    four = feature_extraction._episode_quality(
        np.zeros((4, 6)),
        np.asarray([0.0, 0.1, 0.2, 0.3]),
    )
    assert four["quality_status"] == "too_short_for_jerk"
    assert four["feature_usable"] is False

    timestamps = np.asarray([0.0, 0.1, 0.2, 0.31, 0.41])
    irregular = feature_extraction._episode_quality(np.zeros((5, 6)), timestamps)
    assert irregular["quality_status"] == "irregular_timestamps"
    assert irregular["feature_usable"] is False
    with pytest.raises(ValueError, match="irregular_timestamps"):
        feature_extraction.smoothness_speed_features(np.zeros((5, 6)), timestamps)


def test_angular_speed_uses_quaternion_geodesic_distance():
    orientations = np.array(
        [
            [0.0, 0.0, 0.0],
            [np.pi / 2, np.pi / 2, 0.0],
        ]
    )
    timestamps = np.array([0.0, 0.5])

    angular_speed = feature_extraction._quaternion_angular_speed(
        orientations,
        timestamps,
    )
    quaternions = feature_extraction._ypr_to_quat(orientations)
    expected_angle = 2.0 * np.arccos(abs(np.dot(quaternions[0], quaternions[1])))

    assert angular_speed == pytest.approx([expected_angle / 0.5])
    assert angular_speed[0] != pytest.approx(np.linalg.norm(orientations[1] - orientations[0]) / 0.5)


def test_reference_uses_circular_mean_across_pi_boundary():
    positive = np.zeros((4, 6))
    negative = np.zeros((4, 6))
    positive[:, 3] = np.pi - 0.02
    negative[:, 3] = -np.pi + 0.02
    reference = feature_extraction.build_reference(
        {0: (positive, np.arange(4)), 1: (negative, np.arange(4))},
        size=4,
    )
    # The circular mean is pi-equivalent, not the incorrect arithmetic mean near zero.
    assert np.all(np.abs(reference[:, 3]) > 3.0)


def test_reference_so3_mean_handles_equivalent_euler_representations():
    identity = np.zeros((5, 6))
    equivalent_identity = np.zeros((5, 6))
    equivalent_identity[:, 3:6] = np.pi

    reference = feature_extraction.build_reference(
        {
            0: (identity, np.arange(5, dtype=float)),
            1: (equivalent_identity, np.arange(5, dtype=float)),
        },
        size=5,
    )
    errors = feature_extraction.accuracy_features(identity, reference, size=5)

    np.testing.assert_allclose(reference[:, 3:6], 0.0, atol=1e-7)
    assert errors["mean_orientation_error_rad"] == pytest.approx(0.0, abs=1e-7)


def test_expert_references_are_grouped_by_task():
    task_a = np.zeros((4, 6))
    task_b = np.full((4, 6), 10.0)
    labels = feature_extraction.validate_task_labels(
        pd.DataFrame({"episode": [0, 1], "task_id": ["pick", "place"]}),
        [0, 1],
        source_name="expert",
    )
    references = feature_extraction.build_references_by_task(
        {
            0: (task_a, np.arange(4)),
            1: (task_b, np.arange(4)),
        },
        labels,
        4,
    )
    assert set(references) == {"pick", "place"}
    np.testing.assert_allclose(references["pick"][:, :3], 0)
    np.testing.assert_allclose(references["place"][:, :3], 10)


def test_continuous_comparison_defaults_to_welch_and_supports_pairs():
    independent_a = pd.DataFrame({"metric": [1.0, 2.0, 3.0]})
    independent_b = pd.DataFrame({"metric": [2.0, 4.0, 6.0]})
    independent = features_compare.continuous_comparison(independent_a, independent_b, ["metric"])
    assert independent.loc[0, "test"] == "welch_t"

    paired_a = pd.DataFrame({"pair_id": ["p1", "p2", "p3", "p4"], "metric": [1.0, 4.0, 2.0, 8.0]})
    paired_b = pd.DataFrame({"pair_id": ["p4", "p2", "p1", "p3"], "metric": [7.0, 3.0, 0.0, 1.0]})
    paired = features_compare.continuous_comparison(
        paired_a, paired_b, ["metric"], paired=True, pair_keys=["pair_id"]
    )
    assert paired.loc[0, "test"] == "paired_t"
    assert paired.loc[0, "n_pairs"] == 4
    assert paired.loc[0, "diff_A_minus_B"] == pytest.approx(1.0)
    assert paired.loc[0, "diff_ci95_low"] <= 1.0 <= paired.loc[0, "diff_ci95_high"]
    assert paired.loc[0, "effect_size_type"] == "cohens_dz"


def test_paired_continuous_analysis_rejects_unmatched_or_incomplete_pairs():
    a = pd.DataFrame({"pair_id": ["p1", "p2"], "metric": [1.0, 2.0], "success": [1, 0]})
    unmatched = pd.DataFrame({"pair_id": ["p1", "p3"], "metric": [1.0, 3.0], "success": [1, 1]})
    with pytest.raises(ValueError, match="Refusing an inner join"):
        features_compare.continuous_comparison(a, unmatched, ["metric"], paired=True, pair_keys=["pair_id"])
    one_complete = a.copy()
    one_complete.loc[1, "metric"] = np.nan
    with pytest.raises(ValueError, match="only 1 complete participant pair"):
        features_compare.continuous_comparison(
            one_complete,
            a,
            ["metric"],
            paired=True,
            pair_keys=["pair_id"],
        )

def test_participant_level_features_and_success_are_paired_exactly():
    rows_a = pd.DataFrame(
        {
            "condition": ["A_mobile_colocated"] * 4,
            "participant_id": ["P1", "P1", "P2", "P2"],
            "metric": [1.0, 3.0, 2.0, 4.0],
            "success": [1, 0, 1, 1],
            "task_id": ["stacking", "color_sorting", "stacking", "color_sorting"],
            "condition_order": [1, 1, 2, 2],
        }
    )
    rows_b = pd.DataFrame(
        {
            "condition": ["B_desktop_separated"] * 4,
            "participant_id": ["P1", "P1", "P2", "P2"],
            "metric": [2.0, 4.0, 1.0, 3.0],
            "success": [0, 0, 1, 0],
            "task_id": ["stacking", "color_sorting", "stacking", "color_sorting"],
            "condition_order": [2, 2, 1, 1],
        }
    )
    features_compare.validate_condition_table(rows_a, "A_mobile_colocated")
    features_compare.validate_condition_table(rows_b, "B_desktop_separated")
    participant_a = features_compare.aggregate_participant_features(rows_a, ["metric"])
    assert participant_a.set_index("participant_id").loc["P1", "metric"] == 2.0
    success = features_compare.participant_success_comparison(rows_a, rows_b)
    assert success["test"] == "participant_exact_sign"
    assert success["n_pairs"] == 2
    assert success["SR_A_minus_B"] == pytest.approx(0.5)
    assert success["diff_ci95_low"] <= success["SR_A_minus_B"] <= success["diff_ci95_high"]


def test_condition_is_not_overwritten_and_raw_duration_is_not_auto_tested():
    wrong = pd.DataFrame(
        {
            "condition": ["B_desktop_separated"],
            "participant_id": ["P1"],
            "success": [1],
            "task_id": ["stacking"],
            "condition_order": [1],
        }
    )
    with pytest.raises(ValueError, match="A_mobile_colocated"):
        features_compare.validate_condition_table(wrong, "A_mobile_colocated")
    assert "raw_duration_s" in features_compare.NON_FEATURE


@pytest.mark.parametrize("missing", [None, pd.NA, "None", "<NA>", "null"])
def test_identifier_null_sentinels_are_rejected_before_string_coercion(missing):
    labels = pd.DataFrame(
        {
            "episode": [0],
            "success": [1],
            "participant_id": [missing],
            "task_id": ["task"],
            "seed": ["0"],
            "split": ["test"],
        }
    )
    with pytest.raises(ValueError, match="participant_id must be populated"):
        feature_extraction.validate_labels(labels, [0])

    task_labels = pd.DataFrame({"episode": [0], "task_id": [missing]})
    with pytest.raises(ValueError, match="task_id must be populated"):
        feature_extraction.validate_task_labels(
            task_labels,
            [0],
            source_name="test",
        )


@pytest.mark.parametrize("field", ["task_id", "seed", "split"])
@pytest.mark.parametrize("missing", [None, pd.NA, "None", "<NA>", "null"])
def test_optional_label_identifiers_reject_null_sentinels(field, missing):
    labels = pd.DataFrame(
        {
            "episode": [0],
            "success": [1],
            "participant_id": ["P1"],
            "task_id": ["task"],
            "seed": ["0"],
            "split": ["test"],
        }
    )
    labels.loc[0, field] = missing
    with pytest.raises(ValueError, match=rf"{field} must be populated"):
        feature_extraction.validate_labels(labels, [0])


def test_zero_variance_effect_size_is_not_reported_as_zero():
    assert np.isnan(features_compare.cohens_d(np.ones(3), np.ones(3)))
    assert np.isnan(features_compare.cohens_dz(np.ones(3), np.ones(3)))


def test_protocol_v2_condition_table_and_task_factorial_model():
    rows = []
    for participant_index in range(1, 9):
        for condition in ("A_mobile_colocated", "B_desktop_separated"):
            for task_id in ("stacking", "color_sorting"):
                rows.append(
                    {
                        "participant_id": f"P{participant_index:02d}",
                        "condition": condition,
                        "task_id": task_id,
                        "condition_order": (
                            1
                            if (participant_index % 2 == 1)
                            == (condition == "A_mobile_colocated")
                            else 2
                        ),
                        "success": 1,
                        "smoothness": (
                            participant_index
                            + (2.0 if condition == "B_desktop_separated" else 0.0)
                            + (1.0 if task_id == "color_sorting" else 0.0)
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    features_compare.validate_condition_table(
        frame.loc[frame["condition"] == "A_mobile_colocated"],
        "A_mobile_colocated",
    )
    result = features_compare.factorial_condition_task(
        frame,
        ["smoothness"],
        subject_key="participant_id",
    )
    assert {"condition", "task_id", "condition:task_id"}.issubset(set(result["effect"]))
    assert set(result["task_id_levels"]) == {"color_sorting/stacking"}


def test_protocol_v2_participant_aggregation_weights_tasks_equally():
    frame = pd.DataFrame(
        {
            "participant_id": ["P01", "P01", "P01", "P01", "P02", "P02"],
            "task_id": [
                "stacking",
                "stacking",
                "stacking",
                "color_sorting",
                "stacking",
                "color_sorting",
            ],
            "metric": [0.0, 0.0, 0.0, 10.0, 2.0, 4.0],
        }
    )
    aggregated = features_compare.aggregate_participant_features(frame, ["metric"])
    values = aggregated.set_index("participant_id")["metric"]
    assert values.loc["P01"] == 5.0
    assert values.loc["P02"] == 3.0

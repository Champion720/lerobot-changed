from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[3] / "experiments" / "wrist_view_presentation" / "make_episode_splits.py"
SPEC = importlib.util.spec_from_file_location("make_episode_splits", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _manifest(n_participants: int = 12) -> pd.DataFrame:
    rows = []
    difficulties = ("simple", "medium", "hard")
    episode = 0
    for participant in range(n_participants):
        for task_index, difficulty in enumerate(difficulties):
            for condition in ("A_mobile_colocated", "B_desktop_separated"):
                rows.append(
                    {
                        "condition": condition,
                        "episode": episode,
                        "pair_id": f"P{participant}_pair_{task_index}",
                        "participant_id": f"P{participant}",
                        "task_id": f"task_{task_index}",
                        "difficulty": difficulty,
                        "trial_index": task_index + 1,
                    }
                )
            episode += 1
    return pd.DataFrame(rows)


def test_pair_members_never_cross_splits() -> None:
    result = MODULE.assign_splits(
        _manifest(),
        seed=7,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
    )
    assert (result.groupby("pair_id")["split"].nunique() == 1).all()
    assert (result.groupby("participant_id")["split"].nunique() == 1).all()
    assert set(result["split"]) == {"train", "validation", "test"}


def test_split_is_deterministic() -> None:
    kwargs = {
        "seed": 42,
        "train_fraction": 0.7,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
    }
    first = MODULE.assign_splits(_manifest(), **kwargs)
    second = MODULE.assign_splits(_manifest(), **kwargs)
    pd.testing.assert_frame_equal(first, second)


def test_rejects_missing_condition_in_pair() -> None:
    manifest = _manifest(2)
    manifest = manifest.drop(manifest.index[-1])
    with pytest.raises(ValueError, match="exactly one A_mobile_colocated and one B_desktop_separated"):
        MODULE.validate_manifest(manifest)


def test_rejects_inconsistent_pair_metadata() -> None:
    manifest = _manifest(2)
    manifest.loc[
        (manifest["pair_id"] == "P0_pair_0") & (manifest["condition"] == "B_desktop_separated"),
        "difficulty",
    ] = "hard"
    with pytest.raises(ValueError, match="inconsistent difficulty"):
        MODULE.validate_manifest(manifest)


def test_episode_lists_are_condition_specific() -> None:
    result = MODULE.assign_splits(
        _manifest(),
        seed=3,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
    )
    lists = MODULE.episode_lists(result)
    for split in MODULE.SPLIT_NAMES:
        assert lists["A_mobile_colocated"][split] == lists["B_desktop_separated"][split]


def test_rejects_empty_or_null_manifest_metadata() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MODULE.validate_manifest(_manifest(0))

    manifest = _manifest(3)
    manifest.loc[0, "participant_id"] = pd.NA
    with pytest.raises(ValueError, match="participant_id"):
        MODULE.validate_manifest(manifest)


@pytest.mark.parametrize("value", ["null", "None", "<NA>", "nan"])
def test_rejects_identifier_sentinel_strings(value: str) -> None:
    manifest = _manifest(3)
    manifest.loc[0, "task_id"] = value
    with pytest.raises(ValueError, match="task_id"):
        MODULE.validate_manifest(manifest)


@pytest.mark.parametrize("trial_index", [0, -1, 1.5, "bad"])
def test_rejects_invalid_trial_index(trial_index) -> None:
    manifest = _manifest(3)
    manifest["trial_index"] = manifest["trial_index"].astype(object)
    manifest.loc[0, "trial_index"] = trial_index

    with pytest.raises(ValueError, match="trial_index"):
        MODULE.validate_manifest(manifest)


def test_rejects_inconsistent_trial_index_within_pair() -> None:
    manifest = _manifest(3)
    mask = (manifest["pair_id"] == "P0_pair_0") & (manifest["condition"] == "B_desktop_separated")
    manifest.loc[mask, "trial_index"] = 2

    with pytest.raises(ValueError, match="inconsistent trial_index"):
        MODULE.validate_manifest(manifest)


def test_too_few_participants_is_a_hard_error() -> None:
    with pytest.raises(ValueError, match="cannot populate requested splits"):
        MODULE.assign_splits(
            _manifest(2),
            seed=1,
            train_fraction=0.7,
            validation_fraction=0.15,
            test_fraction=0.15,
        )

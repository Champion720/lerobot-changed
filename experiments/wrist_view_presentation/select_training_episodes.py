#!/usr/bin/env python
"""Freeze equal, quality-qualified successful demonstrations for policy training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

CONDITIONS = ("A_mobile_colocated", "B_desktop_separated")
TASKS = ("stacking", "color_sorting")
CELL_COLUMNS = ("condition", "task_id")
REQUIRED_COLUMNS = {
    "condition",
    "episode",
    "pair_id",
    "participant_id",
    "task_id",
    "layout_id",
    "success",
    "stage_score",
    "manual_intervention",
    "recording_valid",
    "sync_valid",
    "gripper_valid",
}


def _binary(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError(f"{label} must contain only 0 or 1 before training selection")
    return values.astype(int)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qualify_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with derived eligibility and exclusion reasons."""

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"manifest is missing training-selection columns: {missing}")
    if frame.empty:
        raise ValueError("manifest must not be empty")
    result = frame.copy()
    unknown_conditions = sorted(set(result["condition"].astype(str)) - set(CONDITIONS))
    unknown_tasks = sorted(set(result["task_id"].astype(str)) - set(TASKS))
    if unknown_conditions:
        raise ValueError(f"unknown conditions: {unknown_conditions}")
    if unknown_tasks:
        raise ValueError(f"unknown tasks: {unknown_tasks}")

    episode = pd.to_numeric(result["episode"], errors="coerce")
    if episode.isna().any() or (episode < 0).any() or (episode % 1 != 0).any():
        raise ValueError("episode must contain non-negative integers")
    result["episode"] = episode.astype(int)
    if result.duplicated(["condition", "episode"]).any():
        raise ValueError("condition/episode identifiers must be unique")

    success = _binary(result["success"], "success")
    intervention = _binary(result["manual_intervention"], "manual_intervention")
    recording = _binary(result["recording_valid"], "recording_valid")
    sync = _binary(result["sync_valid"], "sync_valid")
    gripper = _binary(result["gripper_valid"], "gripper_valid")
    stage_score = pd.to_numeric(result["stage_score"], errors="coerce")
    if stage_score.isna().any() or ~stage_score.between(0, 1).all():
        raise ValueError("stage_score must be numeric in [0, 1]")

    if "is_practice" in result.columns:
        practice = _binary(result["is_practice"], "is_practice")
    else:
        practice = pd.Series(0, index=result.index, dtype=int)

    reasons: list[str] = []
    eligible: list[bool] = []
    for index in result.index:
        row_reasons: list[str] = []
        if practice.loc[index] == 1:
            row_reasons.append("practice")
        if success.loc[index] != 1:
            row_reasons.append("not_complete_success")
        if abs(float(stage_score.loc[index]) - 1.0) > 1e-9:
            row_reasons.append("stage_score_not_one")
        if intervention.loc[index] != 0:
            row_reasons.append("manual_intervention")
        if recording.loc[index] != 1:
            row_reasons.append("recording_invalid")
        if sync.loc[index] != 1:
            row_reasons.append("sync_invalid")
        if gripper.loc[index] != 1:
            row_reasons.append("gripper_invalid")
        reasons.append(";".join(row_reasons))
        eligible.append(not row_reasons)
    result["training_eligible"] = pd.Series(eligible, index=result.index, dtype=bool)
    result["eligibility_reason"] = reasons
    return result


def _balanced_sample(group: pd.DataFrame, n: int, seed: str) -> list[int]:
    """Select rows while spreading choices across participants and layouts."""

    if n > len(group):
        raise ValueError("cannot sample more rows than the candidate pool")
    rng = random.Random(seed)
    random_keys = {int(index): rng.random() for index in group.index}
    remaining = {int(index) for index in group.index}
    selected: list[int] = []
    participant_counts: Counter[str] = Counter()
    layout_counts: Counter[str] = Counter()
    while len(selected) < n:
        best = min(
            remaining,
            key=lambda index: (
                participant_counts[str(group.at[index, "participant_id"])],
                layout_counts[str(group.at[index, "layout_id"])],
                random_keys[index],
            ),
        )
        remaining.remove(best)
        selected.append(best)
        participant_counts[str(group.at[best, "participant_id"])] += 1
        layout_counts[str(group.at[best, "layout_id"])] += 1
    return selected


def select_balanced_training_rows(
    frame: pd.DataFrame,
    *,
    selection_seed: int,
    paired_complete: bool = False,
    minimum_per_cell: int = 1,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return an auditable manifest with equal selected counts in all four cells."""

    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise ValueError("selection_seed must be an integer")
    if isinstance(minimum_per_cell, bool) or not isinstance(minimum_per_cell, int) or minimum_per_cell < 1:
        raise ValueError("minimum_per_cell must be an integer >= 1")
    qualified = qualify_rows(frame)
    candidates = qualified.loc[qualified["training_eligible"]].copy()
    if paired_complete:
        pair_counts = candidates.groupby("pair_id")["condition"].nunique()
        complete_pair_ids = set(pair_counts[pair_counts.eq(2)].index.astype(str))
        candidates = candidates.loc[candidates["pair_id"].astype(str).isin(complete_pair_ids)].copy()

    counts = candidates.groupby(list(CELL_COLUMNS)).size()
    expected_cells = {(condition, task) for condition in CONDITIONS for task in TASKS}
    actual_cells = set(counts.index.tolist())
    missing_cells = sorted(expected_cells - actual_cells)
    if missing_cells:
        raise ValueError(f"no eligible demonstrations for cells: {missing_cells}")
    insufficient = {
        f"{condition}/{task}": int(counts.get((condition, task), 0))
        for condition, task in sorted(expected_cells)
        if int(counts.get((condition, task), 0)) < minimum_per_cell
    }
    if insufficient:
        raise ValueError(
            f"go/no-go failed: eligible demonstrations below minimum_per_cell={minimum_per_cell}: "
            f"{insufficient}"
        )

    if paired_complete:
        # Select matched pair_ids per task, then include both conditions.
        pair_task = candidates[["pair_id", "task_id", "participant_id", "layout_id"]].drop_duplicates(
            "pair_id"
        )
        task_counts = pair_task.groupby("task_id").size()
        target_pairs = int(task_counts.min())
        selected_pair_ids: set[str] = set()
        for task_id in TASKS:
            group = pair_task.loc[pair_task["task_id"] == task_id]
            indices = _balanced_sample(group, target_pairs, f"{selection_seed}:paired:{task_id}")
            selected_pair_ids.update(group.loc[indices, "pair_id"].astype(str))
        selected_indices = set(
            candidates.index[candidates["pair_id"].astype(str).isin(selected_pair_ids)].astype(int)
        )
        target_per_cell = target_pairs
    else:
        target_per_cell = int(counts.min())
        selected_indices: set[int] = set()
        for condition in CONDITIONS:
            for task_id in TASKS:
                group = candidates.loc[
                    candidates["condition"].eq(condition) & candidates["task_id"].eq(task_id)
                ]
                selected_indices.update(
                    _balanced_sample(group, target_per_cell, f"{selection_seed}:{condition}:{task_id}")
                )

    result = qualified.copy()
    result["selected_for_training"] = result.index.to_series().astype(int).isin(selected_indices)
    selection_reason: list[str] = []
    for _index, row in result.iterrows():
        if bool(row["selected_for_training"]):
            selection_reason.append("selected")
        elif not bool(row["training_eligible"]):
            selection_reason.append(str(row["eligibility_reason"]))
        elif paired_complete and str(row["pair_id"]) not in set(
            result.loc[result["selected_for_training"], "pair_id"].astype(str)
        ):
            selection_reason.append("not_in_selected_complete_pair")
        else:
            selection_reason.append("eligible_not_sampled_for_balance")
    result["selection_reason"] = selection_reason

    selected = result.loc[result["selected_for_training"]]
    selected_counts = selected.groupby(list(CELL_COLUMNS)).size()
    if set(selected_counts.index.tolist()) != expected_cells or not selected_counts.eq(target_per_cell).all():
        raise RuntimeError(f"internal error: selected cells are not equal: {selected_counts.to_dict()}")

    summary: dict[str, object] = {
        "schema_version": 1,
        "selection_seed": selection_seed,
        "paired_complete": paired_complete,
        "minimum_per_condition_task_cell": minimum_per_cell,
        "target_per_condition_task_cell": target_per_cell,
        "candidate_counts": {
            f"{condition}/{task}": int(counts.get((condition, task), 0))
            for condition in CONDITIONS
            for task in TASKS
        },
        "selected_counts": {
            f"{condition}/{task}": int(selected_counts.get((condition, task), 0))
            for condition in CONDITIONS
            for task in TASKS
        },
    }
    return result, summary


def selected_episode_lists(frame: pd.DataFrame) -> dict[str, list[int]]:
    if "selected_for_training" not in frame.columns:
        raise ValueError("selection manifest lacks selected_for_training")
    return {
        condition: sorted(
            frame.loc[
                frame["condition"].eq(condition) & frame["selected_for_training"].astype(bool),
                "episode",
            ]
            .astype(int)
            .tolist()
        )
        for condition in CONDITIONS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--selection_seed", type=int, required=True)
    parser.add_argument("--minimum_per_cell", type=int, required=True)
    parser.add_argument("--paired_complete", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.manifest)
    manifest = pd.read_csv(input_path)
    selected, summary = select_balanced_training_rows(
        manifest,
        selection_seed=args.selection_seed,
        paired_complete=args.paired_complete,
        minimum_per_cell=args.minimum_per_cell,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    summary["input_manifest"] = str(input_path.resolve())
    summary["input_manifest_sha256"] = _sha256(input_path)
    summary["selected_episode_lists"] = selected_episode_lists(selected)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote auditable training selection to {output}")


if __name__ == "__main__":
    main()

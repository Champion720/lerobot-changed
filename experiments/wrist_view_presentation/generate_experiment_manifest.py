#!/usr/bin/env python
"""Generate the frozen 20-participant, 200-attempt formal-study manifest.

The generator reads group quotas and task metadata from the protocol config. It
assigns participants to four five-person groups, creates matched A/B pairs, and
freezes task, layout, and trial order before data collection.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

CONDITIONS = ("A_mobile_colocated", "B_desktop_separated")
FORMAL_GROUPS = ("G1", "G2", "G3", "G4")
OUTCOME_COLUMNS = (
    "success",
    "stage_score",
    "completion_time_s",
    "successful_grasps",
    "grasp_retries",
    "drop_count",
    "manual_intervention",
    "stable_for_2s",
    "failure_type",
    "video_latency_ms",
    "dropped_frames",
    "clock_uncertainty_s",
    "recording_valid",
    "sync_valid",
    "gripper_valid",
    "training_eligible",
    "exclusion_reason",
)


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _task_specs(config: Mapping[str, object]) -> dict[str, dict[str, object]]:
    rows = config.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("tasks must be a non-empty list")
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("every task must be a JSON object")
        task_id = str(row.get("task_id", "")).strip()
        if not task_id or task_id in result:
            raise ValueError("task_id values must be non-empty and unique")
        layout_ids = row.get("collection_layout_ids")
        if (
            not isinstance(layout_ids, list)
            or not layout_ids
            or any(not isinstance(value, str) or not value.strip() for value in layout_ids)
        ):
            raise ValueError(f"task {task_id!r} needs non-empty collection_layout_ids")
        result[task_id] = dict(row)
    if set(result) != {"stacking", "color_sorting"}:
        raise ValueError("formal tasks must be exactly stacking and color_sorting")
    return result


def _group_specs(config: Mapping[str, object], task_ids: set[str]) -> dict[str, dict[str, object]]:
    study = _require_mapping(config.get("study"), "study")
    groups = _require_mapping(study.get("participant_groups"), "study.participant_groups")
    if set(groups) != set(FORMAL_GROUPS):
        raise ValueError(f"participant_groups must contain exactly {list(FORMAL_GROUPS)}")

    result: dict[str, dict[str, object]] = {}
    for group_id in FORMAL_GROUPS:
        row = _require_mapping(groups[group_id], f"participant_groups.{group_id}")
        count = row.get("participant_count")
        if isinstance(count, bool) or not isinstance(count, int) or count != 5:
            raise ValueError(f"{group_id}.participant_count must be exactly 5")
        sequence = row.get("condition_sequence")
        if not isinstance(sequence, list) or tuple(sequence) not in (CONDITIONS, CONDITIONS[::-1]):
            raise ValueError(f"{group_id}.condition_sequence must contain A and B once")
        quotas = _require_mapping(row.get("task_counts_per_condition"), f"{group_id}.task_counts")
        if set(quotas) != task_ids:
            raise ValueError(f"{group_id} task quotas must cover exactly {sorted(task_ids)}")
        normalized_quotas: dict[str, int] = {}
        for task_id, value in quotas.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{group_id} quota for {task_id} must be a positive integer")
            normalized_quotas[str(task_id)] = value
        if sum(normalized_quotas.values()) != 5:
            raise ValueError(f"{group_id} task quotas must sum to 5 per condition")
        result[group_id] = {
            "participant_count": count,
            "condition_sequence": list(sequence),
            "task_counts_per_condition": normalized_quotas,
        }
    return result


def _participant_ids(config: Mapping[str, object], supplied: Sequence[str] | None) -> list[str]:
    analysis = _require_mapping(config.get("analysis"), "analysis")
    planned = analysis.get("planned_participants")
    if isinstance(planned, bool) or not isinstance(planned, int) or planned != 20:
        raise ValueError("analysis.planned_participants must be exactly 20")
    if supplied is None:
        return [f"P{index:02d}" for index in range(1, planned + 1)]
    result = [str(value).strip() for value in supplied]
    if len(result) != planned or any(not value for value in result) or len(set(result)) != planned:
        raise ValueError("participant list must contain exactly 20 unique non-empty identifiers")
    return result


def _balanced_task_order(tasks: list[str], rng: random.Random) -> list[str]:
    """Shuffle a five-trial block while avoiding runs longer than two when possible."""

    candidates: list[list[str]] = []
    for _ in range(200):
        candidate = tasks.copy()
        rng.shuffle(candidate)
        longest = 1
        run = 1
        for previous, current in zip(candidate, candidate[1:], strict=False):
            run = run + 1 if current == previous else 1
            longest = max(longest, run)
        if longest <= 2:
            return candidate
        candidates.append(candidate)
    return candidates[0]


def generate_manifest(
    config: Mapping[str, object],
    *,
    participant_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the deterministic formal manifest described by ``config``."""

    tasks = _task_specs(config)
    groups = _group_specs(config, set(tasks))
    participants = _participant_ids(config, participant_ids)
    study = _require_mapping(config.get("study"), "study")
    assignment_seed = study.get("assignment_seed")
    schedule_seed = study.get("schedule_seed")
    for value, label in ((assignment_seed, "assignment_seed"), (schedule_seed, "schedule_seed")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"study.{label} must be an integer")

    shuffled = participants.copy()
    random.Random(int(assignment_seed)).shuffle(shuffled)
    participant_group: dict[str, str] = {}
    cursor = 0
    for group_id in FORMAL_GROUPS:
        count = int(groups[group_id]["participant_count"])
        for participant_id in shuffled[cursor : cursor + count]:
            participant_group[participant_id] = group_id
        cursor += count
    if cursor != len(shuffled):
        raise RuntimeError("participant group allocation did not consume every participant")

    episode_counters = dict.fromkeys(CONDITIONS, 0)
    rows: list[dict[str, object]] = []
    for participant_id in participants:
        group_id = participant_group[participant_id]
        group = groups[group_id]
        rng = random.Random(f"{schedule_seed}:{participant_id}")
        pair_tasks = [
            task_id
            for task_id, count in group["task_counts_per_condition"].items()
            for _ in range(int(count))
        ]
        pair_tasks = _balanced_task_order(pair_tasks, rng)
        task_pair_number: Counter[str] = Counter()
        pair_rows: list[dict[str, object]] = []
        for pair_index, task_id in enumerate(pair_tasks, start=1):
            task_pair_number[task_id] += 1
            layout_ids = list(tasks[task_id]["collection_layout_ids"])
            layout_offset = rng.randrange(len(layout_ids))
            layout_id = layout_ids[(task_pair_number[task_id] - 1 + layout_offset) % len(layout_ids)]
            pair_rows.append(
                {
                    "pair_index": pair_index,
                    "pair_id": f"{participant_id}_{task_id}_{task_pair_number[task_id]:02d}",
                    "task_id": task_id,
                    "layout_id": layout_id,
                }
            )

        condition_sequence = list(group["condition_sequence"])
        global_trial_index = 0
        for condition_order, condition in enumerate(condition_sequence, start=1):
            ordered_pairs = pair_rows.copy()
            rng.shuffle(ordered_pairs)
            # Preserve the no-long-run constraint after the condition-specific shuffle.
            ordered_task_ids = _balanced_task_order([row["task_id"] for row in ordered_pairs], rng)
            remaining = ordered_pairs.copy()
            ordered_pairs = []
            for task_id in ordered_task_ids:
                match_index = next(index for index, row in enumerate(remaining) if row["task_id"] == task_id)
                ordered_pairs.append(remaining.pop(match_index))

            for within_condition_trial_index, pair in enumerate(ordered_pairs, start=1):
                global_trial_index += 1
                row: dict[str, object] = {
                    "condition": condition,
                    "episode": episode_counters[condition],
                    "pair_id": pair["pair_id"],
                    "participant_id": participant_id,
                    "group_id": group_id,
                    "task_id": pair["task_id"],
                    "layout_id": pair["layout_id"],
                    "layout_regime": "train_layout",
                    "pair_index": pair["pair_index"],
                    "within_condition_trial_index": within_condition_trial_index,
                    "global_trial_index": global_trial_index,
                    "condition_order": condition_order,
                    "is_practice": 0,
                }
                row.update(dict.fromkeys(OUTCOME_COLUMNS, ""))
                rows.append(row)
                episode_counters[condition] += 1

    result = pd.DataFrame(rows)
    expected_columns = [
        "condition",
        "episode",
        "pair_id",
        "participant_id",
        "group_id",
        "task_id",
        "layout_id",
        "layout_regime",
        "pair_index",
        "within_condition_trial_index",
        "global_trial_index",
        "condition_order",
        "is_practice",
        *OUTCOME_COLUMNS,
    ]
    return result.loc[:, expected_columns]


def validate_generated_manifest(frame: pd.DataFrame) -> None:
    """Raise when a generated manifest violates the frozen 20×10 design."""

    if len(frame) != 200:
        raise ValueError(f"formal manifest must contain 200 rows, got {len(frame)}")
    if frame["participant_id"].nunique() != 20:
        raise ValueError("formal manifest must contain exactly 20 participants")
    if set(frame["condition"]) != set(CONDITIONS):
        raise ValueError("formal manifest must contain both formal conditions")
    if set(frame["task_id"]) != {"stacking", "color_sorting"}:
        raise ValueError("formal manifest must contain exactly the two formal tasks")
    group_counts = frame[["participant_id", "group_id"]].drop_duplicates()["group_id"].value_counts()
    if group_counts.to_dict() != dict.fromkeys(FORMAL_GROUPS, 5):
        raise ValueError(f"each formal group must contain five participants, got {group_counts.to_dict()}")
    participant_condition = frame.groupby(["participant_id", "condition"]).size()
    if not participant_condition.eq(5).all():
        raise ValueError("every participant must have exactly five rows in each condition")
    cells = frame.groupby(["condition", "task_id"]).size()
    if not cells.eq(50).all() or len(cells) != 4:
        raise ValueError(f"every condition/task cell must contain 50 rows, got {cells.to_dict()}")
    pair_sizes = frame.groupby("pair_id").size()
    pair_conditions = frame.groupby("pair_id")["condition"].nunique()
    if not pair_sizes.eq(2).all() or not pair_conditions.eq(2).all():
        raise ValueError("every pair_id must contain exactly one row from each condition")
    invariant_columns = ["participant_id", "group_id", "task_id", "layout_id", "pair_index"]
    for column in invariant_columns:
        if not frame.groupby("pair_id")[column].nunique(dropna=False).eq(1).all():
            raise ValueError(f"pair_id rows must have identical {column}")
    if frame.duplicated(["condition", "episode"]).any():
        raise ValueError("condition/episode identifiers must be unique")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--participants_csv",
        help="Optional CSV containing one unique participant_id per row.",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    participant_ids = None
    if args.participants_csv:
        participants = pd.read_csv(args.participants_csv)
        if list(participants.columns) != ["participant_id"]:
            raise SystemExit("participants CSV must contain exactly one column named participant_id")
        participant_ids = participants["participant_id"].tolist()
    manifest = generate_manifest(config, participant_ids=participant_ids)
    validate_generated_manifest(manifest)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(f"Wrote {len(manifest)} frozen formal-study rows to {output}")


if __name__ == "__main__":
    main()

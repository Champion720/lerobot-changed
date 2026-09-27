#!/usr/bin/env python
"""Generate and validate the frozen 240-trial SmolVLA rollout schedule."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

CONDITIONS = ("A_mobile_colocated", "B_desktop_separated")
TASKS = ("stacking", "color_sorting")
LAYOUT_REGIMES = ("train_layout", "novel_position")
RESULT_COLUMNS = (
    "success",
    "stage_score",
    "completion_time_s",
    "grasp_retries",
    "drop_count",
    "manual_intervention",
    "failure_type",
    "safety_stop",
    "inference_latency_ms",
    "video_latency_ms",
)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _integer_list(value: object, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) < 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{label} must contain at least three distinct integer seeds")
    return list(value)


def generate_rollout_manifest(config: Mapping[str, object], *, policy_family: str = "smolvla") -> pd.DataFrame:
    training = _mapping(config.get("training"), "training")
    seeds = _integer_list(training.get("random_seeds"), "training.random_seeds")
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    order_seed = evaluation.get("order_seed")
    if isinstance(order_seed, bool) or not isinstance(order_seed, int):
        raise ValueError("evaluation.order_seed must be an integer")
    resets = _mapping(evaluation.get("reset_ids"), "evaluation.reset_ids")

    reset_rows: dict[str, dict[str, list[str]]] = {}
    for task_id in TASKS:
        task_resets = _mapping(resets.get(task_id), f"evaluation.reset_ids.{task_id}")
        reset_rows[task_id] = {}
        for regime in LAYOUT_REGIMES:
            values = task_resets.get(regime)
            if (
                not isinstance(values, list)
                or len(values) != 10
                or any(not isinstance(item, str) or not item.strip() for item in values)
                or len(set(values)) != 10
            ):
                raise ValueError(f"{task_id}/{regime} must contain exactly 10 unique reset_ids")
            reset_rows[task_id][regime] = [str(item).strip() for item in values]

    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for seed in seeds:
            for task_id in TASKS:
                for regime in LAYOUT_REGIMES:
                    for reset_id in reset_rows[task_id][regime]:
                        row: dict[str, object] = {
                            "rollout_id": (
                                f"{policy_family}_{condition}_seed{seed}_{task_id}_{regime}_{reset_id}"
                            ),
                            "policy_family": policy_family,
                            "training_condition": condition,
                            "seed": seed,
                            "task_id": task_id,
                            "layout_regime": regime,
                            "reset_id": reset_id,
                        }
                        row.update(dict.fromkeys(RESULT_COLUMNS, ""))
                        rows.append(row)

    rng = random.Random(order_seed)
    rng.shuffle(rows)
    for execution_order, row in enumerate(rows, start=1):
        row["execution_order"] = execution_order
    columns = [
        "rollout_id",
        "policy_family",
        "training_condition",
        "seed",
        "task_id",
        "layout_regime",
        "reset_id",
        "execution_order",
        *RESULT_COLUMNS,
    ]
    result = pd.DataFrame(rows).loc[:, columns]
    validate_rollout_manifest(result, expected_seeds=seeds, policy_family=policy_family)
    return result


def validate_rollout_manifest(
    frame: pd.DataFrame,
    *,
    expected_seeds: list[int] | None = None,
    policy_family: str = "smolvla",
) -> None:
    required = {
        "rollout_id",
        "policy_family",
        "training_condition",
        "seed",
        "task_id",
        "layout_regime",
        "reset_id",
        "execution_order",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"rollout manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("rollout manifest must not be empty")
    if frame["rollout_id"].isna().any() or frame["rollout_id"].astype(str).str.strip().eq("").any():
        raise ValueError("rollout_id must be populated")
    if frame["rollout_id"].duplicated().any():
        raise ValueError("rollout_id values must be unique")
    if set(frame["policy_family"].astype(str)) != {policy_family}:
        raise ValueError(f"policy_family must be exactly {policy_family!r}")
    if set(frame["training_condition"].astype(str)) != set(CONDITIONS):
        raise ValueError("rollout manifest must contain both formal training conditions")
    if set(frame["task_id"].astype(str)) != set(TASKS):
        raise ValueError("rollout manifest must contain both formal tasks")
    if set(frame["layout_regime"].astype(str)) != set(LAYOUT_REGIMES):
        raise ValueError("rollout manifest must contain train_layout and novel_position")

    seed_values = pd.to_numeric(frame["seed"], errors="coerce")
    if seed_values.isna().any() or (seed_values % 1 != 0).any():
        raise ValueError("seed must contain integers")
    seeds = sorted(seed_values.astype(int).unique().tolist())
    if expected_seeds is not None and set(seeds) != set(expected_seeds):
        raise ValueError(f"rollout seeds {seeds} do not match expected {sorted(expected_seeds)}")
    expected_total = len(CONDITIONS) * len(seeds) * len(TASKS) * 20
    if len(frame) != expected_total:
        raise ValueError(f"rollout manifest must contain {expected_total} rows, got {len(frame)}")

    cells = frame.groupby(["training_condition", "seed", "task_id"]).size()
    if len(cells) != len(CONDITIONS) * len(seeds) * len(TASKS) or not cells.eq(20).all():
        raise ValueError(f"every condition/seed/task cell must contain 20 rows: {cells.to_dict()}")
    regime_cells = frame.groupby(["training_condition", "seed", "task_id", "layout_regime"]).size()
    if not regime_cells.eq(10).all():
        raise ValueError("every condition/seed/task/layout_regime cell must contain 10 rows")

    reset_sets = frame.groupby(["seed", "task_id", "layout_regime", "training_condition"])[
        "reset_id"
    ].apply(lambda values: frozenset(values.astype(str)))
    for key, group in reset_sets.groupby(level=[0, 1, 2]):
        if group.nunique() != 1:
            raise ValueError(f"A/B reset_id sets differ for {key}")

    order = pd.to_numeric(frame["execution_order"], errors="coerce")
    if order.isna().any() or set(order.astype(int)) != set(range(1, len(frame) + 1)):
        raise ValueError("execution_order must be a permutation of 1..N")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy_family", default="smolvla")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = generate_rollout_manifest(config, policy_family=args.policy_family)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(f"Wrote {len(manifest)} frozen rollout rows to {output}")


if __name__ == "__main__":
    main()

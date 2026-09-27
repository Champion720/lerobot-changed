#!/usr/bin/env python
"""Validate and summarize the frozen matched real-robot rollout results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from .make_rollout_manifest import CONDITIONS, validate_rollout_manifest
except ImportError:  # Direct script execution.
    from make_rollout_manifest import CONDITIONS, validate_rollout_manifest  # type: ignore[no-redef]


PAIR_KEYS = ["policy_family", "seed", "task_id", "layout_regime", "reset_id"]
NUMERIC_RESULTS = [
    "success",
    "stage_score",
    "completion_time_s",
    "grasp_retries",
    "drop_count",
    "manual_intervention",
    "safety_stop",
    "inference_latency_ms",
    "video_latency_ms",
]


def _mean_ci95(values: np.ndarray) -> tuple[float, float]:
    """Return a descriptive two-sided 95% t interval for a matched difference."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    if np.isclose(standard_error, 0.0):
        return mean, mean
    margin = float(stats.t.ppf(0.975, len(values) - 1) * standard_error)
    return mean - margin, mean + margin


def validate_completed_results(frame: pd.DataFrame, *, expected_seeds: list[int] | None = None) -> pd.DataFrame:
    """Return a typed copy and reject incomplete or non-matched formal results."""
    validate_rollout_manifest(frame, expected_seeds=expected_seeds)
    missing = sorted(set(NUMERIC_RESULTS + ["failure_type"]) - set(frame.columns))
    if missing:
        raise ValueError(f"rollout results are missing columns: {missing}")

    result = frame.copy()
    for column in NUMERIC_RESULTS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    complete_required = [
        "success",
        "stage_score",
        "grasp_retries",
        "drop_count",
        "manual_intervention",
        "safety_stop",
        "inference_latency_ms",
        "video_latency_ms",
    ]
    incomplete = result[complete_required].isna().any(axis=1)
    if incomplete.any():
        ids = result.loc[incomplete, "rollout_id"].astype(str).head(10).tolist()
        raise ValueError(f"formal rollout results are incomplete; examples={ids}")
    for column in ("success", "manual_intervention", "safety_stop"):
        if not result[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0/1")
    if not result["stage_score"].between(0, 1).all():
        raise ValueError("stage_score must be within [0, 1]")
    for column in ("grasp_retries", "drop_count", "inference_latency_ms", "video_latency_ms"):
        if (result[column] < 0).any():
            raise ValueError(f"{column} must be non-negative")
    if (result["success"].eq(1) & result["completion_time_s"].isna()).any():
        raise ValueError("successful rollouts require completion_time_s")
    if (result["completion_time_s"].dropna() < 0).any():
        raise ValueError("completion_time_s must be non-negative")

    failures = result["success"].eq(0)
    blank_failure = result["failure_type"].fillna("").astype(str).str.strip().eq("")
    if (failures & blank_failure).any():
        raise ValueError("failed rollouts require failure_type")
    return result


def matched_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one row per matched A/B reset with signed A-minus-B differences."""
    a_name, b_name = CONDITIONS
    a = frame.loc[frame["training_condition"].eq(a_name)].copy()
    b = frame.loc[frame["training_condition"].eq(b_name)].copy()
    columns = PAIR_KEYS + NUMERIC_RESULTS
    pairs = a[columns].merge(
        b[columns],
        on=PAIR_KEYS,
        how="outer",
        suffixes=("_A", "_B"),
        indicator=True,
        validate="one_to_one",
    )
    if not pairs["_merge"].eq("both").all():
        raise ValueError("A/B rollout rows are not exactly matched by seed/task/layout/reset")
    pairs = pairs.drop(columns="_merge")
    for metric in NUMERIC_RESULTS:
        pairs[f"{metric}_A_minus_B"] = pairs[f"{metric}_A"] - pairs[f"{metric}_B"]
    return pairs


def summarize_results(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return condition summaries and matched A-minus-B summaries."""
    metric_aggregations = {
        metric: (metric, "mean")
        for metric in (
            "success",
            "stage_score",
            "completion_time_s",
            "grasp_retries",
            "drop_count",
            "manual_intervention",
            "safety_stop",
            "inference_latency_ms",
            "video_latency_ms",
        )
    }
    cells = (
        frame.groupby(["training_condition", "seed", "task_id", "layout_regime"], observed=True)
        .agg(n=("rollout_id", "size"), **metric_aggregations)
        .reset_index()
    )
    task_layout = (
        frame.groupby(["training_condition", "task_id", "layout_regime"], observed=True)
        .agg(n=("rollout_id", "size"), **metric_aggregations)
        .reset_index()
    )
    task_layout["seed"] = "ALL"
    overall = frame.groupby("training_condition", observed=True).agg(
        n=("rollout_id", "size"), **metric_aggregations
    ).reset_index()
    overall["seed"] = "ALL"
    overall["task_id"] = "ALL"
    overall["layout_regime"] = "ALL"
    condition_summary = pd.concat([cells, task_layout, overall], ignore_index=True)

    pairs = matched_pairs(frame)
    difference_columns = [column for column in pairs.columns if column.endswith("_A_minus_B")]
    rows: list[dict[str, object]] = []
    group_specs: list[tuple[object, str, str, pd.DataFrame]] = [("ALL", "ALL", "ALL", pairs)]
    group_specs.extend(
        ("ALL", str(task_id), str(regime), group)
        for (task_id, regime), group in pairs.groupby(["task_id", "layout_regime"], observed=True)
    )
    group_specs.extend(
        (seed, str(task_id), str(regime), group)
        for (seed, task_id, regime), group in pairs.groupby(
            ["seed", "task_id", "layout_regime"], observed=True
        )
    )
    for seed, task_id, regime, group in group_specs:
        row: dict[str, object] = {
            "seed": seed,
            "task_id": task_id,
            "layout_regime": regime,
            "n_pairs": len(group),
        }
        for column in difference_columns:
            values = group[column].dropna().to_numpy(dtype=float)
            row[f"mean_{column}"] = float(values.mean()) if len(values) else np.nan
            row[f"std_{column}"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            ci_low, ci_high = _mean_ci95(values)
            row[f"ci95_low_{column}"] = ci_low
            row[f"ci95_high_{column}"] = ci_high
        rows.append(row)
    return condition_summary, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--expected_seeds", default="0,1,2")
    args = parser.parse_args()

    expected_seeds = [int(value.strip()) for value in args.expected_seeds.split(",") if value.strip()]
    frame = validate_completed_results(pd.read_csv(args.results), expected_seeds=expected_seeds)
    condition_summary, matched_summary = summarize_results(frame)
    pairs = matched_pairs(frame)

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    condition_summary.to_csv(output / "condition_summary.csv", index=False)
    matched_summary.to_csv(output / "matched_differences_summary.csv", index=False)
    pairs.to_csv(output / "matched_pairs.csv", index=False)
    metadata = {
        "artifact_type": "formal_smolvla_rollout_analysis",
        "n_rollouts": len(frame),
        "n_matched_pairs": len(pairs),
        "difference_direction": "A_mobile_colocated minus B_desktop_separated",
        "primary_metric": "success",
        "tasks_weighted_equally_by_design": True,
        "note": (
            "Per-seed/task/layout matched summaries include descriptive 95% t intervals. "
            "The confirmatory model and multiplicity plan remain preregistration items."
        ),
    }
    (output / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote matched rollout analysis for {len(frame)} trials to {output}")


if __name__ == "__main__":
    main()

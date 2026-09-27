#!/usr/bin/env python
"""Load and cross-check versioned collection-layout and rollout-reset catalogs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

TASKS = {"stacking", "color_sorting"}
LAYOUT_REGIMES = {"train_layout", "novel_position"}


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _pose(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    position = value.get("position_xyz")
    orientation = value.get("orientation_rpy")
    for field, vector in (("position_xyz", position), ("orientation_rpy", orientation)):
        if (
            not isinstance(vector, list)
            or len(vector) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in vector
            )
        ):
            raise ValueError(f"{label}.{field} must contain three finite numbers")


def _catalog_header(payload: object, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{label}.schema_version must be 1")
    _nonempty(payload.get("coordinate_frame"), f"{label}.coordinate_frame")
    if payload.get("position_unit") != "m" or payload.get("angle_unit") != "rad":
        raise ValueError(f"{label} units must be position_unit='m' and angle_unit='rad'")
    return payload


def _objects(value: object, label: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty object map")
    for name, pose in value.items():
        _nonempty(name, f"{label} object name")
        _pose(pose, f"{label}.{name}")


def _read(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc


def validate_layout_catalog(path: Path) -> set[tuple[str, str]]:
    payload = _catalog_header(_read(path, "layout catalog"), "layout_catalog")
    rows = payload.get("layouts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("layout_catalog.layouts must be a non-empty list")
    keys: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"layout_catalog.layouts[{index}] must be an object")
        task_id = _nonempty(row.get("task_id"), f"layouts[{index}].task_id")
        layout_id = _nonempty(row.get("layout_id"), f"layouts[{index}].layout_id")
        if task_id not in TASKS:
            raise ValueError(f"layouts[{index}].task_id is not a formal task")
        key = (task_id, layout_id)
        if key in keys:
            raise ValueError(f"duplicate collection layout {key}")
        keys.add(key)
        _objects(row.get("objects"), f"layouts[{index}].objects")
    return keys


def validate_reset_catalog(path: Path) -> set[tuple[str, str, str]]:
    payload = _catalog_header(_read(path, "reset catalog"), "reset_catalog")
    rows = payload.get("resets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("reset_catalog.resets must be a non-empty list")
    keys: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"reset_catalog.resets[{index}] must be an object")
        task_id = _nonempty(row.get("task_id"), f"resets[{index}].task_id")
        regime = _nonempty(row.get("layout_regime"), f"resets[{index}].layout_regime")
        reset_id = _nonempty(row.get("reset_id"), f"resets[{index}].reset_id")
        if task_id not in TASKS or regime not in LAYOUT_REGIMES:
            raise ValueError(f"resets[{index}] has an invalid task or layout regime")
        key = (task_id, regime, reset_id)
        if key in keys:
            raise ValueError(f"duplicate rollout reset {key}")
        keys.add(key)
        _objects(row.get("objects"), f"resets[{index}].objects")
    return keys


def resolve_catalog_path(value: object, *, config_dir: Path, label: str) -> Path:
    text = _nonempty(value, label)
    path = Path(text)
    return path if path.is_absolute() else config_dir / path


def validate_catalog_references(config: Mapping[str, object], *, config_dir: Path) -> None:
    """Require catalog rows to exactly cover every ID frozen in the protocol."""
    layout_path = resolve_catalog_path(
        config.get("layout_catalog_path"), config_dir=config_dir, label="layout_catalog_path"
    )
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation must be an object")
    reset_path = resolve_catalog_path(
        evaluation.get("reset_catalog_path"), config_dir=config_dir, label="evaluation.reset_catalog_path"
    )
    actual_layouts = validate_layout_catalog(layout_path)
    actual_resets = validate_reset_catalog(reset_path)

    expected_layouts: set[tuple[str, str]] = set()
    tasks = config.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a list")
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ValueError("every task must be an object")
        task_id = str(task.get("task_id", ""))
        layouts = task.get("collection_layout_ids")
        if not isinstance(layouts, list):
            raise ValueError(f"{task_id}.collection_layout_ids must be a list")
        expected_layouts.update((task_id, str(layout_id)) for layout_id in layouts)
    if actual_layouts != expected_layouts:
        raise ValueError(
            "layout catalog does not exactly match collection_layout_ids; "
            f"missing={sorted(expected_layouts - actual_layouts)}, extra={sorted(actual_layouts - expected_layouts)}"
        )

    expected_resets: set[tuple[str, str, str]] = set()
    reset_ids = evaluation.get("reset_ids")
    if not isinstance(reset_ids, Mapping):
        raise ValueError("evaluation.reset_ids must be an object")
    for task_id, task_resets in reset_ids.items():
        if not isinstance(task_resets, Mapping):
            raise ValueError(f"evaluation.reset_ids.{task_id} must be an object")
        for regime, identifiers in task_resets.items():
            if not isinstance(identifiers, list):
                raise ValueError(f"evaluation.reset_ids.{task_id}.{regime} must be a list")
            expected_resets.update((str(task_id), str(regime), str(reset_id)) for reset_id in identifiers)
    if actual_resets != expected_resets:
        raise ValueError(
            "reset catalog does not exactly match evaluation.reset_ids; "
            f"missing={sorted(expected_resets - actual_resets)}, extra={sorted(actual_resets - expected_resets)}"
        )

#!/usr/bin/env python
"""Shared, hardware-agnostic contract for timestamped gripper actions and states."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

REQUIRED_CONFIG_FIELDS = (
    "action_schema_id",
    "action_unit",
    "action_column",
    "state_schema_id",
    "state_unit",
    "state_column",
    "action_open_value",
    "action_closed_value",
    "state_open_value",
    "state_closed_value",
)


def normalize_gripper_contract(value: object) -> dict[str, object]:
    """Validate the protocol fragment and return a canonical scalar gripper contract."""
    if not isinstance(value, Mapping):
        raise ValueError("capture.gripper must be an object")
    result: dict[str, object] = {}
    for field in REQUIRED_CONFIG_FIELDS[:6]:
        item = value.get(field)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"capture.gripper.{field} must be a non-empty string")
        result[field] = item.strip()
    if result["action_column"] == "timestamp" or result["state_column"] == "timestamp":
        raise ValueError("gripper value columns cannot be named timestamp")
    for field in REQUIRED_CONFIG_FIELDS[6:]:
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"capture.gripper.{field} must be finite")
        result[field] = float(item)
    for stream in ("action", "state"):
        open_value = float(result[f"{stream}_open_value"])
        closed_value = float(result[f"{stream}_closed_value"])
        if math.isclose(open_value, closed_value):
            raise ValueError(f"gripper {stream} open and closed values must differ")
        result[f"{stream}_minimum_value"] = min(open_value, closed_value)
        result[f"{stream}_maximum_value"] = max(open_value, closed_value)
    return result


def expected_schema(contract: Mapping[str, object]) -> dict[str, object]:
    """Build the exact per-episode schema document expected by the protocol."""
    normalized = normalize_gripper_contract(contract)
    return {
        "schema_version": 1,
        "action": {
            "schema_id": normalized["action_schema_id"],
            "unit": normalized["action_unit"],
            "column": normalized["action_column"],
            "open_value": normalized["action_open_value"],
            "closed_value": normalized["action_closed_value"],
        },
        "state": {
            "schema_id": normalized["state_schema_id"],
            "unit": normalized["state_unit"],
            "column": normalized["state_column"],
            "open_value": normalized["state_open_value"],
            "closed_value": normalized["state_closed_value"],
        },
    }


def load_and_validate_schema(path: Path, contract: Mapping[str, object]) -> dict[str, object]:
    """Load a per-episode schema and require an exact match with the frozen protocol."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read gripper schema: {exc}") from exc
    expected = expected_schema(contract)
    if payload != expected:
        raise ValueError(f"{path}: gripper schema does not exactly match the frozen protocol")
    return payload


def validate_values(
    values: object,
    contract: Mapping[str, object],
    *,
    stream: str,
    label: str,
) -> None:
    """Require finite scalar values within the frozen open/closed envelope."""
    import numpy as np

    if stream not in {"action", "state"}:
        raise ValueError("stream must be 'action' or 'state'")
    normalized = normalize_gripper_contract(contract)
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 1 or array.shape[0] == 0:
        raise ValueError(f"{label} must contain exactly one non-empty value column")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    minimum = float(normalized[f"{stream}_minimum_value"])
    maximum = float(normalized[f"{stream}_maximum_value"])
    if (array < minimum).any() or (array > maximum).any():
        raise ValueError(f"{label} values must stay within [{minimum}, {maximum}]")

#!/usr/bin/env python
"""Leakage-safe offline ACT evaluation for the wrist-view presentation experiment.

Formal mode evaluates every preregistered training seed on a frozen participant-safe
split, restores actions to physical units, aggregates episodes to participants, and
writes reproducibility metadata. A deliberately explicit single-checkpoint diagnostic
mode remains available for the legacy public-dataset smoke test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION

try:
    from .analysis_reproducibility import fingerprint_inputs, git_state, runtime_environment, write_json
except ImportError:
    from analysis_reproducibility import (  # type: ignore[no-redef]
        fingerprint_inputs,
        git_state,
        runtime_environment,
        write_json,
    )


DEFAULT_ACTION_NAMES = ["dx", "dy", "dz", "dyaw", "dpitch", "droll"]
DEFAULT_ACTION_UNITS = ["m", "m", "m", "rad", "rad", "rad"]
MODEL_FILE_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}


def _parse_episode_spec(value: str | Sequence[int] | None) -> list[int] | None:
    """Parse ``1,3-5`` episode syntax into sorted unique non-negative indices."""

    if value is None:
        return None
    if not isinstance(value, str):
        episodes = [int(item) for item in value]
    else:
        episodes: list[int] = []
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                parts = token.split("-")
                if len(parts) != 2:
                    raise ValueError(f"invalid episode range: {token!r}")
                start, end = (int(part.strip()) for part in parts)
                if start > end:
                    raise ValueError(f"invalid episode range: {token!r}")
                episodes.extend(range(start, end + 1))
            else:
                episodes.append(int(token))
    if any(episode < 0 for episode in episodes):
        raise ValueError("episode indices must be non-negative")
    return sorted(set(episodes))


def _read_split_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"condition", "episode", "participant_id", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"split manifest is missing columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("split manifest is empty")

    manifest = manifest.copy()
    manifest["episode"] = pd.to_numeric(manifest["episode"], errors="raise").astype(int)
    if (manifest["episode"] < 0).any():
        raise ValueError("split manifest episode indices must be non-negative")
    for column in ("condition", "participant_id", "split"):
        if manifest[column].isna().any() or manifest[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"split manifest column {column!r} contains empty values")
        manifest[column] = manifest[column].astype(str).str.strip()

    duplicate = manifest.duplicated(["condition", "episode"], keep=False)
    if duplicate.any():
        rows = manifest.loc[duplicate, ["condition", "episode"]].to_dict("records")
        raise ValueError(f"condition/episode rows must be unique: {rows}")

    participant_splits = manifest.groupby("participant_id", sort=False)["split"].nunique()
    crossed = participant_splits[participant_splits > 1].index.tolist()
    if crossed:
        raise ValueError(f"participants must not cross train/validation/test splits: {crossed}")
    return manifest


def _load_manifest_selection(
    manifest_path: str | Path,
    *,
    condition: str,
    split: str,
) -> tuple[list[int], list[int]]:
    manifest = _read_split_manifest(manifest_path)
    selected = sorted(
        manifest.loc[(manifest["condition"] == condition) & (manifest["split"] == split), "episode"].tolist()
    )
    train = sorted(
        manifest.loc[
            (manifest["condition"] == condition) & (manifest["split"] == "train"), "episode"
        ].tolist()
    )
    if not selected:
        raise ValueError(f"manifest contains no {condition!r} episodes in split {split!r}")
    return selected, train


def _load_manifest_pairs(
    manifest_path: str | Path,
    *,
    condition_a: str,
    condition_b: str,
    split: str,
) -> list[tuple[str, int, int]]:
    manifest = _read_split_manifest(manifest_path)
    if "pair_id" not in manifest.columns:
        raise ValueError("paired analysis requires a pair_id column in the split manifest")
    selected = manifest.loc[
        manifest["condition"].isin([condition_a, condition_b]) & (manifest["split"] == split)
    ].copy()
    if selected["pair_id"].isna().any() or selected["pair_id"].astype(str).str.strip().eq("").any():
        raise ValueError("paired split rows require non-empty pair_id values")
    selected["pair_id"] = selected["pair_id"].astype(str).str.strip()

    result: list[tuple[str, int, int]] = []
    for pair_id, group in selected.groupby("pair_id", sort=True):
        rows_a = group.loc[group["condition"] == condition_a]
        rows_b = group.loc[group["condition"] == condition_b]
        if len(rows_a) != 1 or len(rows_b) != 1 or len(group) != 2:
            raise ValueError(
                f"pair_id {pair_id!r} must contain exactly one {condition_a} and one {condition_b} row"
            )
        if rows_a.iloc[0]["participant_id"] != rows_b.iloc[0]["participant_id"]:
            raise ValueError(f"pair_id {pair_id!r} crosses participants")
        result.append((str(pair_id), int(rows_a.iloc[0]["episode"]), int(rows_b.iloc[0]["episode"])))
    if not result:
        raise ValueError(f"manifest contains no A/B pairs in split {split!r}")
    return result


def _compare_metric(
    errors_a: Mapping[Any, Mapping[str, float]],
    errors_b: Mapping[Any, Mapping[str, float]],
    metric: str,
    *,
    paired: bool,
    episode_pairs: Sequence[tuple[str, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if paired and episode_pairs is not None:
        missing = [
            pair_id
            for pair_id, episode_a, episode_b in episode_pairs
            if episode_a not in errors_a or episode_b not in errors_b
        ]
        if missing:
            raise ValueError(f"paired metrics are missing manifest pairs: {missing}")
        a = np.asarray([errors_a[episode_a][metric] for _, episode_a, _ in episode_pairs], dtype=float)
        b = np.asarray([errors_b[episode_b][metric] for _, _, episode_b in episode_pairs], dtype=float)
        return a, b, "paired t-test"

    if paired:
        units_a = set(errors_a)
        units_b = set(errors_b)
        if units_a != units_b:
            raise ValueError("paired comparison requires identical analysis-unit sets")
        ordered = sorted(units_a, key=str)
        a = np.asarray([errors_a[item][metric] for item in ordered], dtype=float)
        b = np.asarray([errors_b[item][metric] for item in ordered], dtype=float)
        return a, b, "paired t-test"

    a = np.asarray([errors_a[item][metric] for item in sorted(errors_a, key=str)], dtype=float)
    b = np.asarray([errors_b[item][metric] for item in sorted(errors_b, key=str)], dtype=float)
    if not len(a) or not len(b):
        raise ValueError("independent comparison requires non-empty samples")
    return a, b, "Welch t-test"


def _load_train_config(checkpoint: str | Path) -> dict[str, Any] | None:
    path = Path(checkpoint).expanduser() / "train_config.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _leakage_status(
    checkpoint: str,
    repo_id: str,
    evaluation_episodes: Sequence[int],
    *,
    declared_train: Sequence[int] | None,
) -> tuple[str, str]:
    config = _load_train_config(checkpoint)
    if config is None:
        return "unverified", "checkpoint has no train_config.json"
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        return "unverified", "train_config.json does not record dataset configuration"
    trained_repo = dataset.get("repo_id")
    if trained_repo != repo_id:
        return "repo_mismatch", f"checkpoint was trained on {trained_repo!r}, not {repo_id!r}"
    trained = _parse_episode_spec(dataset.get("episodes"))
    if trained is None:
        return "unverified", "training episode list is absent; leakage cannot be audited"
    overlap = sorted(set(trained) & {int(item) for item in evaluation_episodes})
    if overlap:
        return "overlap", f"training/evaluation episode overlap: {overlap}"
    if declared_train is not None and set(trained) != {int(item) for item in declared_train}:
        return "train_mismatch", "checkpoint training episodes do not match the frozen train split"
    return "verified", "verified no train/evaluation overlap"


def _mean_metric_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty metric collection")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows[1:]):
        raise ValueError("metric schemas differ during aggregation")
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in sorted(keys)}


def _aggregate_metrics_by_participant(
    episode_metrics: Mapping[int, Mapping[str, float]],
    manifest_path: str | Path,
    *,
    condition: str,
    split: str,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    manifest = _read_split_manifest(manifest_path)
    selected = manifest.loc[(manifest["condition"] == condition) & (manifest["split"] == split)].copy()
    expected = {int(value) for value in selected["episode"]}
    actual = {int(value) for value in episode_metrics}
    if expected != actual:
        raise ValueError(
            f"{condition} episode metrics do not exactly match the frozen {split} split; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    participants: dict[str, dict[str, float]] = {}
    provenance: list[dict[str, Any]] = []
    for participant_id, group in selected.groupby("participant_id", sort=True):
        episodes = sorted(int(value) for value in group["episode"])
        participants[str(participant_id)] = _mean_metric_rows([episode_metrics[item] for item in episodes])
        provenance.append(
            {
                "participant_id": str(participant_id),
                "condition": condition,
                "split": split,
                "episodes": episodes,
                "n_episodes": len(episodes),
                "training_seeds": ["diagnostic"],
                "n_training_seeds": 1,
            }
        )
    return participants, provenance


def _aggregate_training_seed_runs_by_participant(
    metrics_by_seed: Mapping[str, Mapping[int, Mapping[str, float]]],
    manifest_path: str | Path,
    *,
    condition: str,
    split: str,
) -> tuple[
    dict[str, dict[str, float]],
    list[dict[str, Any]],
    dict[str, dict[str, dict[str, float]]],
]:
    if not metrics_by_seed:
        raise ValueError("at least one training seed is required")
    seed_level: dict[str, dict[str, dict[str, float]]] = {}
    seed_provenance: dict[str, list[dict[str, Any]]] = {}
    for seed in sorted(metrics_by_seed, key=lambda value: int(value)):
        aggregated, provenance = _aggregate_metrics_by_participant(
            metrics_by_seed[seed], manifest_path, condition=condition, split=split
        )
        seed_level[str(seed)] = aggregated
        seed_provenance[str(seed)] = provenance

    participant_sets = [set(values) for values in seed_level.values()]
    if any(values != participant_sets[0] for values in participant_sets[1:]):
        raise ValueError("training seeds produced different participant sets")

    seeds = sorted(seed_level, key=lambda value: int(value))
    participants: dict[str, dict[str, float]] = {}
    provenance: list[dict[str, Any]] = []
    for participant_id in sorted(participant_sets[0]):
        participants[participant_id] = _mean_metric_rows([seed_level[seed][participant_id] for seed in seeds])
        base = next(row for row in seed_provenance[seeds[0]] if row["participant_id"] == participant_id)
        provenance.append(
            {
                **base,
                "training_seeds": seeds,
                "n_training_seeds": len(seeds),
            }
        )
    return participants, provenance, seed_level


def _canonical_seed(value: Any) -> str:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"training seed {value!r} is not an integer") from exc
    if str(value).strip() not in {str(integer), f"+{integer}"}:
        raise ValueError(f"training seed {value!r} is not in canonical integer form")
    return str(integer)


def _load_preregistered_training_seeds(protocol_path: str | Path) -> list[str]:
    payload = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    try:
        raw = payload["training"]["random_seeds"]
    except (KeyError, TypeError) as exc:
        raise ValueError("protocol must define training.random_seeds") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("training.random_seeds must be a non-empty list")
    seeds = [_canonical_seed(value) for value in raw]
    if len(seeds) != len(set(seeds)):
        raise ValueError("training.random_seeds contains duplicates")
    return seeds


def _load_checkpoint_mapping(path: str | Path, *, field: str) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{field} checkpoint mapping must be a non-empty JSON object")
    result: dict[str, str] = {}
    for seed, checkpoint in payload.items():
        canonical = _canonical_seed(seed)
        if canonical in result:
            raise ValueError(f"{field} checkpoint mapping contains duplicate seed {canonical}")
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError(f"{field} checkpoint for seed {canonical} must be a non-empty path")
        result[canonical] = checkpoint.strip()
    return result


def _resolved_checkpoint_paths(mapping: Mapping[str, str]) -> list[str]:
    return [str(Path(path).expanduser().resolve()) for path in mapping.values()]


def _validate_checkpoint_seed_design(
    checkpoints_a: Mapping[str, str],
    checkpoints_b: Mapping[str, str],
    preregistered_seeds: Sequence[str | int],
) -> None:
    seeds_a = set(checkpoints_a)
    seeds_b = set(checkpoints_b)
    if seeds_a != seeds_b:
        raise ValueError("A/B checkpoint seed sets differ")
    expected = {_canonical_seed(seed) for seed in preregistered_seeds}
    if seeds_a != expected:
        raise ValueError("checkpoint seed sets must exactly match preregistered training.random_seeds")
    for condition, mapping in (("A", checkpoints_a), ("B", checkpoints_b)):
        resolved = _resolved_checkpoint_paths(mapping)
        if len(resolved) != len(set(resolved)):
            raise ValueError(f"{condition} maps multiple seeds to the same checkpoint path")


def _hash_checkpoint_artifacts(root: Path) -> tuple[str, list[str]]:
    candidates = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and (path.suffix.lower() in MODEL_FILE_SUFFIXES or "safetensors" in path.name.lower())
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not candidates:
        raise ValueError(f"{root} contains no recognizable model artifacts")
    digest = hashlib.sha256()
    relative_names: list[str] = []
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        relative_names.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\n")
    return digest.hexdigest(), relative_names


def _config_seed(config: Mapping[str, Any]) -> int | None:
    value = config.get("seed")
    if value is None and isinstance(config.get("training"), dict):
        value = config["training"].get("seed")
    return None if value is None else int(value)


def _validate_local_checkpoint_artifacts(
    mapping: Mapping[str, str],
    *,
    condition: str,
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    seen_hashes: dict[str, str] = {}
    for seed in sorted(mapping, key=lambda value: int(value)):
        root = Path(mapping[seed]).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"{condition} seed {seed} checkpoint is not a local directory: {root}")
        config = _load_train_config(root)
        if config is None:
            raise ValueError(f"{condition} seed {seed} checkpoint has no train_config.json")
        recorded_seed = _config_seed(config)
        if recorded_seed != int(seed):
            raise ValueError(
                f"{condition} checkpoint train_config seed {recorded_seed!r} does not match mapping seed {seed}"
            )
        artifact_hash, artifact_files = _hash_checkpoint_artifacts(root)
        if artifact_hash in seen_hashes:
            raise ValueError(
                f"{condition} seeds {seen_hashes[artifact_hash]} and {seed} contain identical checkpoint artifacts"
            )
        seen_hashes[artifact_hash] = seed
        evidence[seed] = {
            "path": mapping[seed],
            "resolved_path": str(root),
            "train_config_seed": recorded_seed,
            "artifact_sha256": artifact_hash,
            "artifact_files": artifact_files,
        }
    return evidence


def _reject_cross_condition_checkpoint_reuse(
    evidence_a: Mapping[str, Mapping[str, Any]],
    evidence_b: Mapping[str, Mapping[str, Any]],
) -> None:
    paths_a = {str(row["resolved_path"]) for row in evidence_a.values()}
    hashes_a = {str(row["artifact_sha256"]) for row in evidence_a.values()}
    for row in evidence_b.values():
        if str(row["resolved_path"]) in paths_a or str(row["artifact_sha256"]) in hashes_a:
            raise ValueError("A and B reuse identical checkpoint paths or model artifacts")


def _require_identical_episode_selection(
    audits: Mapping[str, Mapping[str, Any]],
    *,
    condition: str,
) -> None:
    selections = {
        tuple(int(item) for item in audit.get("selected_episodes", [])) for audit in audits.values()
    }
    if len(selections) != 1:
        raise ValueError(f"all {condition} training seeds must evaluate identical held-out episodes")


def _masked_error_sums(
    predicted: torch.Tensor,
    target: torch.Tensor,
    is_pad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError(
            f"predicted and target action chunks must share (batch,time,dim); got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    if is_pad.ndim == 3 and is_pad.shape[-1] == 1:
        is_pad = is_pad.squeeze(-1)
    if is_pad.shape != predicted.shape[:2]:
        raise ValueError("padding mask shape does not match action chunks")
    valid = (~is_pad.to(dtype=torch.bool, device=predicted.device)).unsqueeze(-1)
    difference = predicted - target.to(predicted.device)
    absolute = (difference.abs() * valid).sum(dim=1)
    squared = (difference.square() * valid).sum(dim=1)
    count = valid.sum(dim=1).expand(-1, predicted.shape[-1])
    return absolute.cpu(), squared.cpu(), count.cpu()


def _unit_squared(unit: str) -> str:
    return f"{unit}2"


def _finalize_error_metrics(
    absolute_sums: Mapping[int, np.ndarray],
    squared_sums: Mapping[int, np.ndarray],
    counts: Mapping[int, np.ndarray],
    action_names: Sequence[str],
    action_units: Sequence[str],
) -> dict[int, dict[str, float]]:
    if not (len(action_names) == len(action_units)):
        raise ValueError("action name/unit lengths differ")
    if set(absolute_sums) != set(squared_sums) or set(absolute_sums) != set(counts):
        raise ValueError("error accumulators contain different episode sets")
    result: dict[int, dict[str, float]] = {}
    for episode in sorted(absolute_sums):
        absolute = np.asarray(absolute_sums[episode], dtype=float)
        squared = np.asarray(squared_sums[episode], dtype=float)
        count = np.asarray(counts[episode], dtype=float)
        if (
            absolute.shape != (len(action_names),)
            or squared.shape != absolute.shape
            or count.shape != absolute.shape
        ):
            raise ValueError(f"episode {episode} error accumulator shape is invalid")
        if np.any(count <= 0):
            raise ValueError(f"episode {episode} has no valid action targets")
        row: dict[str, float] = {}
        for index, (name, unit) in enumerate(zip(action_names, action_units, strict=True)):
            mae = absolute[index] / count[index]
            mse = squared[index] / count[index]
            row[f"{name}_mae_{unit}"] = float(mae)
            row[f"{name}_mse_{_unit_squared(unit)}"] = float(mse)
            row[f"{name}_rmse_{unit}"] = float(math.sqrt(mse))
        for prefix, unit in (("translation", "m"), ("rotation", "rad")):
            indices = [index for index, item in enumerate(action_units) if item == unit]
            if not indices:
                continue
            total_count = float(count[indices].sum())
            mae = float(absolute[indices].sum() / total_count)
            mse = float(squared[indices].sum() / total_count)
            row[f"{prefix}_mae_{unit}"] = mae
            row[f"{prefix}_mse_{_unit_squared(unit)}"] = mse
            row[f"{prefix}_rmse_{unit}"] = math.sqrt(mse)
        result[int(episode)] = row
    return result


def _predict_physical_action_chunk(
    policy: Any,
    postprocessor: Callable[[torch.Tensor], torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    predicted_normalized = policy.predict_action_chunk(batch)
    predicted_physical = postprocessor(predicted_normalized)
    if not isinstance(predicted_physical, torch.Tensor):
        raise TypeError("policy postprocessor must return a torch.Tensor")
    return predicted_physical


def _action_normalization_mode(config: Any) -> str:
    mapping = getattr(config, "normalization_mapping", {}) or {}
    for key, value in mapping.items():
        if "action" in str(getattr(key, "value", key)).lower():
            return str(getattr(value, "value", value)).lower()
    return "identity"


def _require_action_scale_metadata(config: Any, postprocessor: Any) -> None:
    mode = _action_normalization_mode(config)
    if "identity" in mode:
        return
    for step in getattr(postprocessor, "steps", []):
        for attribute in ("stats", "_tensor_stats"):
            stats = getattr(step, attribute, None)
            if not isinstance(stats, Mapping):
                continue
            for key, values in stats.items():
                if (
                    "action" in str(getattr(key, "value", key)).lower()
                    and isinstance(values, Mapping)
                    and values
                ):
                    return
    raise RuntimeError(
        "checkpoint uses action normalization but its postprocessor has no action scale statistics"
    )


def _feature_action_names(meta: Any) -> list[str]:
    features = getattr(meta, "features", {})
    action_feature = features.get(ACTION) if isinstance(features, Mapping) else None
    if not isinstance(action_feature, Mapping):
        raise ValueError("dataset metadata has no action feature")
    names = action_feature.get("names")
    if names is None:
        shape = action_feature.get("shape")
        if shape and int(shape[0]) == len(DEFAULT_ACTION_NAMES):
            return list(DEFAULT_ACTION_NAMES)
        raise ValueError("dataset action feature has no named physical schema")
    return [str(name) for name in names]


def _validate_common_action_schema(
    meta_a: Any,
    meta_b: Any,
    schema_path: str | Path | None,
) -> tuple[list[str], list[str]]:
    names_a = _feature_action_names(meta_a)
    names_b = _feature_action_names(meta_b)
    if names_a != names_b:
        raise ValueError(f"A/B action schemas differ: {names_a} != {names_b}")

    if schema_path is not None:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        if not isinstance(schema, dict):
            raise ValueError("action schema JSON must be an object")
        names = schema.get("names", schema.get("action_names"))
        units = schema.get("units", schema.get("action_units"))
        if names != names_a or not isinstance(units, list) or len(units) != len(names_a):
            raise ValueError("action schema file does not match dataset action names")
        return names_a, [str(unit) for unit in units]

    if names_a != DEFAULT_ACTION_NAMES:
        raise ValueError(
            "non-standard action names require --action_schema_json with explicit physical units"
        )
    return names_a, list(DEFAULT_ACTION_UNITS)


def _verify_physical_roundtrip(
    physical_target: torch.Tensor,
    normalized_target: torch.Tensor,
    postprocessor: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    recovered = postprocessor(normalized_target)
    if not isinstance(recovered, torch.Tensor):
        raise RuntimeError("action postprocessor did not return a tensor")
    expected = physical_target.detach().to(device="cpu", dtype=recovered.dtype)
    actual = recovered.detach().to(device="cpu")
    if expected.shape != actual.shape or not torch.allclose(expected, actual, rtol=1e-4, atol=1e-6):
        raise RuntimeError(
            "action normalization/postprocessing does not round-trip to physical dataset values"
        )


def _build_dataset(
    repo_id: str,
    root: str | None,
    episodes: list[int],
    action_delta_indices: Sequence[int],
    video_backend: str,
) -> LeRobotDataset:
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = {ACTION: [int(index) / metadata.fps for index in action_delta_indices]}
    return LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )


def _per_episode_errors(
    checkpoint: str,
    repo_id: str,
    root: str | None,
    episodes: list[int],
    *,
    device: str,
    video_backend: str,
    batch_size: int,
    action_names: Sequence[str],
    action_units: Sequence[str],
) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = checkpoint
    config.device = device
    action_delta_indices = list(config.action_delta_indices)
    dataset = _build_dataset(repo_id, root, episodes, action_delta_indices, video_backend)
    policy = make_policy(cfg=config, ds_meta=dataset.meta)
    policy.eval()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    _require_action_scale_metadata(config, postprocessor)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pad_key = f"{ACTION}_is_pad"
    absolute_by_episode: dict[int, np.ndarray] = {}
    squared_by_episode: dict[int, np.ndarray] = {}
    count_by_episode: dict[int, np.ndarray] = {}
    roundtrip_checked = False

    with torch.no_grad():
        for raw_batch in loader:
            target_physical = raw_batch[ACTION].detach().clone()
            is_pad = raw_batch.get(pad_key)
            if is_pad is None:
                is_pad = torch.zeros(target_physical.shape[:2], dtype=torch.bool)
            episode_indices = [int(item) for item in raw_batch["episode_index"].tolist()]
            batch = preprocessor(raw_batch)
            if not roundtrip_checked:
                _verify_physical_roundtrip(target_physical, batch[ACTION], postprocessor)
                roundtrip_checked = True
            predicted_physical = _predict_physical_action_chunk(policy, postprocessor, batch)
            absolute, squared, count = _masked_error_sums(
                predicted_physical,
                target_physical.to(predicted_physical.device),
                is_pad.to(predicted_physical.device),
            )
            for row, episode in enumerate(episode_indices):
                if episode not in absolute_by_episode:
                    absolute_by_episode[episode] = np.zeros(len(action_names), dtype=float)
                    squared_by_episode[episode] = np.zeros(len(action_names), dtype=float)
                    count_by_episode[episode] = np.zeros(len(action_names), dtype=float)
                absolute_by_episode[episode] += absolute[row].numpy()
                squared_by_episode[episode] += squared[row].numpy()
                count_by_episode[episode] += count[row].numpy()

    metrics = _finalize_error_metrics(
        absolute_by_episode,
        squared_by_episode,
        count_by_episode,
        action_names,
        action_units,
    )
    audit = {
        "checkpoint": checkpoint,
        "repo_id": repo_id,
        "selected_episodes": sorted(metrics),
        "n_frames": len(dataset),
    }
    return metrics, audit


def _metric_names(rows: Mapping[Any, Mapping[str, float]]) -> list[str]:
    if not rows:
        return []
    values = list(rows.values())
    keys = set(values[0])
    if any(set(row) != keys for row in values[1:]):
        raise ValueError("metric rows have inconsistent schemas")
    return sorted(keys)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _test_summary(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    paired: bool,
) -> dict[str, Any]:
    from scipy import stats

    if paired:
        statistic, p_value = (
            stats.ttest_rel(values_a, values_b) if len(values_a) >= 2 else (math.nan, math.nan)
        )
        differences = values_a - values_b
        spread = float(np.std(differences, ddof=1)) if len(differences) >= 2 else math.nan
        effect = float(np.mean(differences) / spread) if math.isfinite(spread) and spread > 0 else math.nan
        name = "paired t-test"
    else:
        statistic, p_value = (
            stats.ttest_ind(values_a, values_b, equal_var=False)
            if len(values_a) >= 2 and len(values_b) >= 2
            else (math.nan, math.nan)
        )
        pooled = (
            math.sqrt((float(np.var(values_a, ddof=1)) + float(np.var(values_b, ddof=1))) / 2)
            if len(values_a) >= 2 and len(values_b) >= 2
            else math.nan
        )
        effect = (
            float((np.mean(values_a) - np.mean(values_b)) / pooled)
            if math.isfinite(pooled) and pooled > 0
            else math.nan
        )
        name = "Welch t-test"
    return {
        "test": name,
        "n_a": len(values_a),
        "n_b": len(values_b),
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
        "difference_a_minus_b": float(np.mean(values_a) - np.mean(values_b)),
        "statistic": _finite_or_none(float(statistic)),
        "p_value": _finite_or_none(float(p_value)),
        "effect_size": _finite_or_none(effect),
    }


def _comparison_rows(
    metrics_a: Mapping[Any, Mapping[str, float]],
    metrics_b: Mapping[Any, Mapping[str, float]],
    *,
    paired: bool,
    episode_pairs: Sequence[tuple[str, int, int]] | None = None,
) -> list[dict[str, Any]]:
    if set(_metric_names(metrics_a)) != set(_metric_names(metrics_b)):
        raise ValueError("A/B metric schemas differ")
    rows: list[dict[str, Any]] = []
    for metric in _metric_names(metrics_a):
        values_a, values_b, _ = _compare_metric(
            metrics_a,
            metrics_b,
            metric,
            paired=paired,
            episode_pairs=episode_pairs,
        )
        rows.append({"metric": metric, **_test_summary(values_a, values_b, paired=paired)})
    return rows


def _episode_rows(
    metrics_by_seed: Mapping[str, Mapping[int, Mapping[str, float]]],
    *,
    condition: str,
) -> list[dict[str, Any]]:
    return [
        {
            "condition": condition,
            "training_seed": seed,
            "episode": episode,
            **metrics,
        }
        for seed in sorted(metrics_by_seed, key=lambda value: int(value))
        for episode, metrics in sorted(metrics_by_seed[seed].items())
    ]


def _participant_rows(
    seed_level: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    condition: str,
) -> list[dict[str, Any]]:
    return [
        {
            "condition": condition,
            "training_seed": seed,
            "participant_id": participant,
            **metrics,
        }
        for seed in sorted(seed_level, key=lambda value: int(value))
        for participant, metrics in sorted(seed_level[seed].items())
    ]


def _write_tables(
    output: str | Path,
    summary_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    participant_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episode_path = output_path.with_name(f"{output_path.stem}_episodes.csv")
    participant_path = output_path.with_name(f"{output_path.stem}_participants.csv")
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)
    pd.DataFrame(episode_rows).to_csv(episode_path, index=False)
    pd.DataFrame(participant_rows).to_csv(participant_path, index=False)
    return {
        "summary": str(output_path),
        "episodes": str(episode_path),
        "participants": str(participant_path),
    }


def _formal(args: argparse.Namespace) -> None:
    required = {
        "--checkpoints_a_json": args.checkpoints_a_json,
        "--checkpoints_b_json": args.checkpoints_b_json,
        "--protocol_config": args.protocol_config,
        "--split_manifest": args.split_manifest,
    }
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        raise ValueError(f"formal evaluation is missing required arguments: {', '.join(missing)}")

    seeds = _load_preregistered_training_seeds(args.protocol_config)
    checkpoints_a = _load_checkpoint_mapping(args.checkpoints_a_json, field="A")
    checkpoints_b = _load_checkpoint_mapping(args.checkpoints_b_json, field="B")
    _validate_checkpoint_seed_design(checkpoints_a, checkpoints_b, seeds)
    evidence_a = _validate_local_checkpoint_artifacts(checkpoints_a, condition="A")
    evidence_b = _validate_local_checkpoint_artifacts(checkpoints_b, condition="B")
    _reject_cross_condition_checkpoint_reuse(evidence_a, evidence_b)

    selected_a, train_a = _load_manifest_selection(
        args.split_manifest, condition=args.condition_a, split=args.split
    )
    selected_b, train_b = _load_manifest_selection(
        args.split_manifest, condition=args.condition_b, split=args.split
    )
    _load_manifest_pairs(
        args.split_manifest,
        condition_a=args.condition_a,
        condition_b=args.condition_b,
        split=args.split,
    )

    meta_a = LeRobotDatasetMetadata(args.repo_a, root=args.root_a)
    meta_b = LeRobotDatasetMetadata(args.repo_b, root=args.root_b)
    action_names, action_units = _validate_common_action_schema(meta_a, meta_b, args.action_schema_json)

    metrics_a: dict[str, dict[int, dict[str, float]]] = {}
    metrics_b: dict[str, dict[int, dict[str, float]]] = {}
    audits_a: dict[str, dict[str, Any]] = {}
    audits_b: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        print(f"Evaluating training seed {seed}: A_mobile_colocated")
        metrics_a[seed], audits_a[seed] = _per_episode_errors(
            checkpoints_a[seed],
            args.repo_a,
            args.root_a,
            selected_a,
            device=args.device,
            video_backend=args.video_backend,
            batch_size=args.batch_size,
            action_names=action_names,
            action_units=action_units,
        )
        status_a, message_a = _leakage_status(
            checkpoints_a[seed], args.repo_a, selected_a, declared_train=train_a
        )
        audits_a[seed].update({"leakage_status": status_a, "leakage_message": message_a})
        if status_a != "verified":
            raise ValueError(f"A seed {seed} leakage audit failed: {message_a}")

        print(f"Evaluating training seed {seed}: B_desktop_separated")
        metrics_b[seed], audits_b[seed] = _per_episode_errors(
            checkpoints_b[seed],
            args.repo_b,
            args.root_b,
            selected_b,
            device=args.device,
            video_backend=args.video_backend,
            batch_size=args.batch_size,
            action_names=action_names,
            action_units=action_units,
        )
        status_b, message_b = _leakage_status(
            checkpoints_b[seed], args.repo_b, selected_b, declared_train=train_b
        )
        audits_b[seed].update({"leakage_status": status_b, "leakage_message": message_b})
        if status_b != "verified":
            raise ValueError(f"B seed {seed} leakage audit failed: {message_b}")

    _require_identical_episode_selection(audits_a, condition=args.condition_a)
    _require_identical_episode_selection(audits_b, condition=args.condition_b)
    participants_a, provenance_a, seed_level_a = _aggregate_training_seed_runs_by_participant(
        metrics_a,
        args.split_manifest,
        condition=args.condition_a,
        split=args.split,
    )
    participants_b, provenance_b, seed_level_b = _aggregate_training_seed_runs_by_participant(
        metrics_b,
        args.split_manifest,
        condition=args.condition_b,
        split=args.split,
    )
    if set(participants_a) != set(participants_b):
        raise ValueError("formal paired analysis requires identical participant sets in A and B")
    summary = _comparison_rows(participants_a, participants_b, paired=True)
    tables = _write_tables(
        args.out,
        summary,
        _episode_rows(metrics_a, condition=args.condition_a)
        + _episode_rows(metrics_b, condition=args.condition_b),
        _participant_rows(seed_level_a, condition=args.condition_a)
        + _participant_rows(seed_level_b, condition=args.condition_b),
    )
    metadata_path = str(Path(args.out).with_suffix(".metadata.json"))
    write_json(
        metadata_path,
        {
            "analysis_mode": "formal_preregistered_multiseed_participant_paired",
            "conditions": [args.condition_a, args.condition_b],
            "split": args.split,
            "training_seeds": seeds,
            "action_names": action_names,
            "action_units": action_units,
            "tables": tables,
            "checkpoint_evidence": {"A": evidence_a, "B": evidence_b},
            "inference_audits": {"A": audits_a, "B": audits_b},
            "participant_provenance": {"A": provenance_a, "B": provenance_b},
            "inputs": fingerprint_inputs(
                {
                    "protocol_config": args.protocol_config,
                    "split_manifest": args.split_manifest,
                    "checkpoints_a_json": args.checkpoints_a_json,
                    "checkpoints_b_json": args.checkpoints_b_json,
                    "action_schema_json": args.action_schema_json,
                }
            ),
            "runtime": runtime_environment(),
            "git": git_state(),
        },
    )
    print(f"Wrote formal comparison: {tables['summary']}")
    print(f"Wrote reproducibility metadata: {metadata_path}")


def _diagnostic(args: argparse.Namespace) -> None:
    if not args.ckpt_a or not args.ckpt_b:
        raise ValueError("diagnostic mode requires --ckpt_a and --ckpt_b")
    meta_a = LeRobotDatasetMetadata(args.repo_a, root=args.root_a)
    meta_b = LeRobotDatasetMetadata(args.repo_b, root=args.root_b)
    action_names, action_units = _validate_common_action_schema(meta_a, meta_b, args.action_schema_json)
    count_a = max(1, int(round(meta_a.total_episodes * args.test_frac)))
    count_b = max(1, int(round(meta_b.total_episodes * args.test_frac)))
    episodes_a = list(range(meta_a.total_episodes - count_a, meta_a.total_episodes))
    episodes_b = list(range(meta_b.total_episodes - count_b, meta_b.total_episodes))
    metrics_a, _ = _per_episode_errors(
        args.ckpt_a,
        args.repo_a,
        args.root_a,
        episodes_a,
        device=args.device,
        video_backend=args.video_backend,
        batch_size=args.batch_size,
        action_names=action_names,
        action_units=action_units,
    )
    metrics_b, _ = _per_episode_errors(
        args.ckpt_b,
        args.repo_b,
        args.root_b,
        episodes_b,
        device=args.device,
        video_backend=args.video_backend,
        batch_size=args.batch_size,
        action_names=action_names,
        action_units=action_units,
    )
    summary = _comparison_rows(metrics_a, metrics_b, paired=args.paired)
    rows_a = [{"condition": "A_diagnostic", "episode": ep, **row} for ep, row in metrics_a.items()]
    rows_b = [{"condition": "B_diagnostic", "episode": ep, **row} for ep, row in metrics_b.items()]
    tables = _write_tables(args.out, summary, rows_a + rows_b, [])
    metadata_path = str(Path(args.out).with_suffix(".metadata.json"))
    write_json(
        metadata_path,
        {
            "analysis_mode": "diagnostic_single_checkpoint_episode_level",
            "not_for_formal_inference": True,
            "paired": bool(args.paired),
            "test_fraction": args.test_frac,
            "tables": tables,
            "runtime": runtime_environment(),
            "git": git_state(),
        },
    )
    print(f"Wrote diagnostic comparison: {tables['summary']}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo_a",
        required=True,
        help="A_mobile_colocated LeRobot dataset repo id",
    )
    parser.add_argument(
        "--root_a",
        default=None,
        help="Optional local root for A_mobile_colocated",
    )
    parser.add_argument(
        "--repo_b",
        required=True,
        help="B_desktop_separated LeRobot dataset repo id",
    )
    parser.add_argument(
        "--root_b",
        default=None,
        help="Optional local root for B_desktop_separated",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video_backend", default="pyav")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out", default="outputs/offline_action_errors.csv")
    parser.add_argument("--action_schema_json", default=None)

    parser.add_argument("--checkpoints_a_json", default=None)
    parser.add_argument("--checkpoints_b_json", default=None)
    parser.add_argument("--protocol_config", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--condition_a", default="A_mobile_colocated")
    parser.add_argument("--condition_b", default="B_desktop_separated")

    parser.add_argument(
        "--diagnostic_single_checkpoint",
        action="store_true",
        help="Explicitly run the legacy one-checkpoint, episode-level diagnostic path",
    )
    parser.add_argument("--ckpt_a", default=None)
    parser.add_argument("--ckpt_b", default=None)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Diagnostic only: pair identical episode indices from derived datasets",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if not 0 < args.test_frac < 1:
        raise ValueError("--test_frac must be between 0 and 1")
    if args.diagnostic_single_checkpoint:
        _diagnostic(args)
    else:
        if args.ckpt_a or args.ckpt_b or args.paired:
            raise ValueError(
                "single-checkpoint flags require the explicit --diagnostic_single_checkpoint gate"
            )
        _formal(args)


if __name__ == "__main__":
    main()

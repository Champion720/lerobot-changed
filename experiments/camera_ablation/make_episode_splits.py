#!/usr/bin/env python
"""Validate the paired A/B experiment manifest and assign participant-safe splits.

The split is generated *before* model training. All rows from the same
``participant_id`` receive one split, so neither matched A/B trials nor repeated trials
from one person can leak across train and test. By default, participants are stratified
by their complete difficulty profile.

Example:
    uv run --extra training python experiments/camera_ablation/make_episode_splits.py \
        --manifest experiment_manifest.csv --out outputs/experiment_splits.csv \
        --seed 20260728

The printed episode lists can be passed to ``lerobot-train`` as
``--dataset.episodes='[0, 1, ...]'``. Use only the ``train`` episodes for fitting; reserve
``validation`` for model selection and ``test`` for the final offline comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import pandas as pd

CONDITIONS = {"A_mobile": "mobile", "B_pc": "pc"}
REQUIRED_COLUMNS = {
    "condition",
    "episode",
    "pair_id",
    "participant_id",
    "task_id",
    "difficulty",
}
PAIR_INVARIANTS = ("participant_id", "task_id", "difficulty")
SPLIT_NAMES = ("train", "validation", "test")
MISSING_IDENTIFIER_TOKENS = {"", "nan", "none", "<na>", "null"}


def validate_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized manifest or raise ``ValueError`` with actionable details."""
    if df.empty:
        raise ValueError("manifest must contain at least one matched A/B pair")
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")

    out = df.copy()
    if out["condition"].isna().any():
        raise ValueError("condition must be populated for every row")
    out["condition"] = out["condition"].astype(str).str.strip()
    if out["condition"].str.lower().isin(MISSING_IDENTIFIER_TOKENS).any():
        raise ValueError("condition must be populated for every row")
    unknown = sorted(set(out["condition"]) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"unknown condition values {unknown}; expected exactly {sorted(CONDITIONS)}")

    episodes = pd.to_numeric(out["episode"], errors="coerce")
    if episodes.isna().any() or (episodes < 0).any() or (episodes % 1 != 0).any():
        raise ValueError("episode must contain non-negative integer indices")
    out["episode"] = episodes.astype(int)

    for column in ("pair_id", *PAIR_INVARIANTS):
        if out[column].isna().any():
            raise ValueError(f"{column} must be populated for every row")
        values = out[column].astype(str).str.strip()
        if values.str.lower().isin(MISSING_IDENTIFIER_TOKENS).any():
            raise ValueError(f"{column} must be populated for every row")
        out[column] = values

    for column in ("seed", "split"):
        if column not in out.columns:
            continue
        if out[column].isna().any():
            raise ValueError(f"{column} must be populated for every row when present")
        values = out[column].astype(str).str.strip()
        if values.str.lower().isin(MISSING_IDENTIFIER_TOKENS).any():
            raise ValueError(f"{column} must be populated for every row when present")
        out[column] = values

    if "trial_index" in out.columns:
        trial_index = pd.to_numeric(out["trial_index"], errors="coerce")
        if trial_index.isna().any() or (trial_index <= 0).any() or (trial_index % 1 != 0).any():
            raise ValueError("trial_index must contain positive integer indices")
        out["trial_index"] = trial_index.astype(int)

    duplicate = out.duplicated(["condition", "episode"], keep=False)
    if duplicate.any():
        bad = out.loc[duplicate, ["condition", "episode"]].to_dict("records")
        raise ValueError(f"duplicate condition/episode rows: {bad}")

    for pair_id, pair in out.groupby("pair_id", sort=False):
        conditions = set(pair["condition"])
        if conditions != set(CONDITIONS) or len(pair) != 2:
            raise ValueError(
                f"pair_id {pair_id!r} must contain exactly one A_mobile and one B_pc row; "
                f"found {sorted(conditions)} across {len(pair)} row(s)"
            )
        for column in PAIR_INVARIANTS:
            if pair[column].nunique(dropna=False) != 1:
                raise ValueError(f"pair_id {pair_id!r} has inconsistent {column}")
        if "trial_index" in pair.columns and pair["trial_index"].nunique(dropna=False) != 1:
            raise ValueError(f"pair_id {pair_id!r} has inconsistent trial_index")

    if "display" in out.columns:
        expected = out["condition"].map(CONDITIONS)
        actual = out["display"].astype(str).str.strip().str.lower()
        mismatch = actual.ne(expected)
        if mismatch.any():
            raise ValueError("display must be mobile for A_mobile and pc for B_pc")

    return out


def _allocate_counts(n: int, fractions: tuple[float, float, float]) -> list[int]:
    raw = [n * fraction for fraction in fractions]
    counts = [int(value) for value in raw]
    for index in sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)[: n - sum(counts)]:
        counts[index] += 1

    positive = [i for i, fraction in enumerate(fractions) if fraction > 0]
    if n >= len(positive):
        for empty in [i for i in positive if counts[i] == 0]:
            donor = max(positive, key=lambda i: counts[i])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[empty] += 1
    return counts


def assign_splits(
    df: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    stratify_by: str | None = "difficulty_profile",
    min_participants_per_split: int = 2,
) -> pd.DataFrame:
    """Assign one split per participant and reject incomplete requested splits."""
    fractions = (train_fraction, validation_fraction, test_fraction)
    if (
        any(
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or fraction < 0
            for fraction in fractions
        )
        or abs(sum(fractions) - 1.0) > 1e-9
    ):
        raise ValueError("train/validation/test fractions must be non-negative and sum to 1")

    normalized = validate_manifest(df)
    participants = normalized[["participant_id"]].drop_duplicates().copy()
    if (
        isinstance(min_participants_per_split, bool)
        or not isinstance(min_participants_per_split, int)
        or min_participants_per_split <= 0
    ):
        raise ValueError("min_participants_per_split must be a positive integer")
    requested_splits = [split for split, fraction in zip(SPLIT_NAMES, fractions, strict=True) if fraction > 0]
    if len(participants) < len(requested_splits):
        raise ValueError(
            f"{len(participants)} participant(s) cannot populate requested splits "
            f"{requested_splits}; collect at least {len(requested_splits)} participants"
        )

    if stratify_by is not None:
        if stratify_by == "difficulty_profile":
            profiles = (
                normalized.drop_duplicates("pair_id")
                .groupby("participant_id")["difficulty"]
                .apply(lambda values: "|".join(sorted(set(values.astype(str).tolist()))))
            )
            participants["difficulty_profile"] = participants["participant_id"].map(profiles)
        elif stratify_by not in normalized.columns:
            raise ValueError(f"stratify column {stratify_by!r} is absent from the manifest")
        else:
            inconsistent = normalized.groupby("participant_id")[stratify_by].nunique(dropna=False)
            if (inconsistent != 1).any():
                raise ValueError(
                    f"{stratify_by} must be constant within each participant_id; "
                    "use difficulty_profile for repeated multi-difficulty trials"
                )
            values = normalized.groupby("participant_id")[stratify_by].first()
            participants[stratify_by] = participants["participant_id"].map(values)
        strata = participants.groupby(stratify_by, dropna=False, sort=True)
    else:
        strata = [("all", participants)]

    participant_to_split: dict[str, str] = {}
    for stratum, group in strata:
        participant_ids = group["participant_id"].tolist()
        random.Random(f"{seed}:{stratum}").shuffle(participant_ids)
        counts = _allocate_counts(len(participant_ids), fractions)
        start = 0
        for split, count in zip(SPLIT_NAMES, counts, strict=True):
            for participant_id in participant_ids[start : start + count]:
                participant_to_split[participant_id] = split
            start += count

    result = normalized.copy()
    result["split"] = result["participant_id"].map(participant_to_split)
    if result["split"].isna().any():
        raise RuntimeError("internal error: one or more participants were not assigned a split")

    participant_counts = (
        result[["participant_id", "split"]].drop_duplicates().groupby("split")["participant_id"].nunique()
    )
    undersized_splits = {
        split: int(participant_counts.get(split, 0))
        for split in requested_splits
        if participant_counts.get(split, 0) < min_participants_per_split
    }
    if undersized_splits:
        raise ValueError(
            f"participant-safe split has fewer than {min_participants_per_split} "
            f"participant(s) in requested split(s): {undersized_splits}; collect more "
            "participants or change preregistered fractions"
        )
    difficulties = sorted(result["difficulty"].unique())
    empty_cells = [
        f"{split}/{difficulty}"
        for split in requested_splits
        for difficulty in difficulties
        if not (result["split"].eq(split) & result["difficulty"].eq(difficulty)).any()
    ]
    if empty_cells:
        raise ValueError(
            "participant-safe allocation cannot populate every split/difficulty cell: "
            f"{empty_cells}. Collect more participants with every difficulty."
        )
    return result.sort_values(["split", "condition", "episode"]).reset_index(drop=True)


def episode_lists(df: pd.DataFrame) -> dict[str, dict[str, list[int]]]:
    """Build ``{condition: {split: [episode indices]}}`` for CLI use."""
    return {
        condition: {
            split: sorted(
                df.loc[
                    (df["condition"] == condition) & (df["split"] == split),
                    "episode",
                ]
                .astype(int)
                .tolist()
            )
            for split in SPLIT_NAMES
        }
        for condition in CONDITIONS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--min_participants_per_split", type=int, default=2)
    parser.add_argument(
        "--stratify_by",
        default="difficulty_profile",
        help=(
            "Participant-invariant column used for stratification. The default "
            "'difficulty_profile' supports participants who perform several difficulties; "
            "pass 'none' to disable."
        ),
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    result = assign_splits(
        manifest,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        stratify_by=None if args.stratify_by.lower() == "none" else args.stratify_by,
        min_participants_per_split=args.min_participants_per_split,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    lists = episode_lists(result)
    print(json.dumps(lists, ensure_ascii=False, indent=2))
    print(f"\nWrote validated split manifest to {out}")
    print("Train with only each condition's train list; do not train on validation/test episodes.")


if __name__ == "__main__":
    main()

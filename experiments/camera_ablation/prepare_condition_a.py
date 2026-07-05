#!/usr/bin/env python
"""Legacy helper: build a no-camera dataset copy for smoke testing only.

The current real experiment no longer compares WITH vs WITHOUT camera. It compares two
display conditions:
  A = camera stream displayed on the phone client
  B = camera stream displayed on the PC client
Both real datasets should include camera video.

This script remains useful only for validating the training pipeline on a public dataset:
it strips video features to create a small no-camera copy, then run_pipeline.ps1 can train
both the full and stripped copies as a quick smoke test. Do not use this as the real
Condition-A preprocessing step.

This script loads a source LeRobotDataset, auto-detects its video/image columns, and
writes a copy with those columns removed via `modify_features`. Because no video columns
remain, the copy is small and fast to create (parquet only, no video re-encode).

Usage (run from the inner lerobot-main project dir):
    uv run --extra training python experiments/camera_ablation/prepare_condition_a.py \
        --repo_id lerobot/svla_so101_pickplace \
        --out_repo_id svla_so101_pickplace_nocam
"""

import argparse
import shutil
from pathlib import Path

from lerobot.datasets.dataset_tools import modify_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", required=True, help="Source dataset repo_id (Condition B / full).")
    parser.add_argument(
        "--out_repo_id",
        default=None,
        help="Repo id for the no-camera copy. Defaults to <repo_id last part>_nocam.",
    )
    parser.add_argument("--root", default=None, help="Optional local root of the source dataset.")
    parser.add_argument("--output_dir", default=None, help="Where to write the no-camera dataset.")
    parser.add_argument(
        "--video_backend",
        default="pyav",
        help="Video backend (use pyav on machines without system FFmpeg).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If the output dataset already exists, delete and rebuild it.",
    )
    args = parser.parse_args()

    out_repo_id = args.out_repo_id or f"{args.repo_id.split('/')[-1]}_nocam"
    out_root = Path(args.output_dir) if args.output_dir else HF_LEROBOT_HOME / out_repo_id
    if out_root.exists():
        if args.overwrite:
            print(f"--overwrite: removing existing {out_root}")
            shutil.rmtree(out_root)
        else:
            print(f"Condition-A dataset already exists, skipping: {out_root}")
            print("  (reusing it; pass --overwrite to rebuild)")
            return

    print(f"Loading source dataset: {args.repo_id}")
    ds = LeRobotDataset(args.repo_id, root=args.root, video_backend=args.video_backend)

    # Camera features = the video keys. Removing them yields the legacy no-camera smoke-test dataset.
    camera_keys = list(ds.meta.video_keys)
    if not camera_keys:
        raise SystemExit(
            f"No video/camera features found in {args.repo_id}; nothing to strip. "
            "Condition A needs a dataset that actually has camera columns to remove."
        )

    print(f"Camera (video) features to remove: {camera_keys}")
    print(f"Keeping state/action features:     "
          f"{[k for k in ds.meta.features if k not in camera_keys]}")

    new_ds = modify_features(
        dataset=ds,
        remove_features=camera_keys,
        output_dir=args.output_dir,
        repo_id=out_repo_id,
    )

    print("\nDone.")
    print(f"  Legacy no-camera smoke-test dataset: {out_repo_id}")
    print(f"  Root: {new_ds.root}")
    print(f"  Remaining features: {list(new_ds.meta.features)}")
    print(f"  Episodes: {new_ds.meta.total_episodes}, Frames: {new_ds.meta.total_frames}")


if __name__ == "__main__":
    main()

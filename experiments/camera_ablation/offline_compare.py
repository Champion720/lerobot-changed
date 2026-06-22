#!/usr/bin/env python
"""Offline evaluation for the camera-ablation experiment (Condition A vs Condition B).

This implements the *offline* arm of the analysis plan (no physical robot / no simulator
needed): for each held-out frame, run the trained ACT policy and measure the L1 distance
between its predicted action chunk and the expert (ground-truth) action chunk -- i.e. how
close the learned policy stays to the demonstrations. This is the "imitation-learning MSE"
metric from the experiment plan, here as masked L1 in the policy's normalized action space.

We compute a per-episode mean error for each condition, treat episodes as paired samples
(Condition A and B share the same episode indices), and run a paired t-test
(scipy.stats.ttest_rel) so the comparison comes with a p-value.

  Condition A (no camera): policy trained on the *_nocam dataset, state-only ACT.
  Condition B (camera):    policy trained on the full dataset, state+image ACT.

Usage (run from the inner lerobot-main project dir, after both models are trained):
    uv run --extra training python experiments/camera_ablation/offline_compare.py \
        --ckpt_a outputs/train/cond_a/checkpoints/last/pretrained_model \
        --repo_a svla_so101_pickplace_nocam \
        --ckpt_b outputs/train/cond_b/checkpoints/last/pretrained_model \
        --repo_b lerobot/svla_so101_pickplace \
        --test_frac 0.2 --device cuda --video_backend pyav

NOTE: this script could not be executed in the authoring environment (needs the trained
checkpoints + GPU). It follows LeRobot's documented APIs; if a key/import differs on your
version, the error messages point at the exact line to adjust.
"""

import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION


def _build_dataset(repo_id: str, root: str | None, action_delta_indices: list[int], video_backend: str):
    """Load a dataset with the action chunk delta_timestamps the policy expects."""
    meta = LeRobotDataset(repo_id, root=root, video_backend=video_backend).meta
    fps = meta.fps
    delta_timestamps = {ACTION: [i / fps for i in action_delta_indices]}
    return LeRobotDataset(
        repo_id, root=root, delta_timestamps=delta_timestamps, video_backend=video_backend
    )


def _per_episode_errors(ckpt: str, repo_id: str, root: str | None, test_frac: float,
                        device: str, video_backend: str, batch_size: int) -> dict[int, float]:
    """Return {episode_index: mean masked-L1 action error} over the test episodes."""
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = device

    ds = _build_dataset(repo_id, root, list(cfg.action_delta_indices), video_backend)

    policy = make_policy(cfg=cfg, ds_meta=ds.meta)
    policy.eval()
    policy.to(device)

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # Hold out the last `test_frac` of episodes as the test split.
    n_eps = ds.meta.total_episodes
    n_test = max(1, int(round(n_eps * test_frac)))
    test_eps = set(range(n_eps - n_test, n_eps))
    print(f"  [{repo_id}] episodes={n_eps}, test episodes={sorted(test_eps)}")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    pad_key = f"{ACTION}_is_pad"

    err_sum: dict[int, float] = defaultdict(float)
    err_cnt: dict[int, int] = defaultdict(int)

    with torch.no_grad():
        for batch in loader:
            ep_idx = batch["episode_index"]
            keep = torch.tensor([int(e) in test_eps for e in ep_idx])
            if not keep.any():
                continue

            batch = preprocessor(batch)
            actions_hat, _ = policy.model(batch)  # (B, chunk, action_dim), normalized space
            gt = batch[ACTION]                    # (B, chunk, action_dim), normalized space
            mask = (~batch[pad_key]).float()      # (B, chunk)

            abs_err = (gt - actions_hat).abs().mean(dim=-1)              # (B, chunk)
            per_sample = (abs_err * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)  # (B,)

            per_sample = per_sample.cpu()
            for e, v, k in zip(ep_idx.cpu().tolist(), per_sample.tolist(), keep.tolist()):
                if k:
                    err_sum[int(e)] += v
                    err_cnt[int(e)] += 1

    return {ep: err_sum[ep] / err_cnt[ep] for ep in err_sum}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_a", required=True, help="Condition A (no camera) pretrained_model dir.")
    parser.add_argument("--repo_a", required=True, help="Condition A dataset (the *_nocam copy).")
    parser.add_argument("--root_a", default=None)
    parser.add_argument("--ckpt_b", required=True, help="Condition B (camera) pretrained_model dir.")
    parser.add_argument("--repo_b", required=True, help="Condition B dataset (the full dataset).")
    parser.add_argument("--root_b", default=None)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video_backend", default="pyav")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Use a PAIRED t-test on shared episode indices. Only valid when A and B come from the "
        "SAME underlying episodes (e.g. the delete-column smoke test). For two independently captured "
        "datasets (the real experiment), leave this off to use an independent-samples t-test.",
    )
    args = parser.parse_args()

    print("Condition A (no camera):")
    err_a = _per_episode_errors(args.ckpt_a, args.repo_a, args.root_a, args.test_frac,
                                args.device, args.video_backend, args.batch_size)
    print("Condition B (camera):")
    err_b = _per_episode_errors(args.ckpt_b, args.repo_b, args.root_b, args.test_frac,
                                args.device, args.video_backend, args.batch_size)

    if args.paired:
        # Same underlying episodes in both conditions (smoke test with the delete-column dataset).
        shared = sorted(set(err_a) & set(err_b))
        if not shared:
            raise SystemExit("No shared test episodes; cannot run a paired t-test. Drop --paired.")
        a = np.array([err_a[e] for e in shared])
        b = np.array([err_b[e] for e in shared])
        test_name = "Paired t-test (ttest_rel)"
    else:
        # Two independently captured datasets (the real experiment): episodes don't correspond.
        a = np.array(list(err_a.values()))
        b = np.array(list(err_b.values()))
        test_name = "Independent-samples t-test (ttest_ind)"

    print("\n================ Offline action-error comparison ================")
    print(f"  Test episodes: A={len(a)}, B={len(b)}")

    print("\n---- Summary (lower error = closer to expert demonstrations) ----")
    print(f"  Condition A (no camera) mean L1: {a.mean():.5f}  (std {a.std(ddof=1):.5f})")
    print(f"  Condition B (camera)    mean L1: {b.mean():.5f}  (std {b.std(ddof=1):.5f})")
    print(f"  Difference (A - B): {a.mean() - b.mean():.5f}  "
          f"({'B better' if b.mean() < a.mean() else 'A better'})")

    try:
        from scipy import stats

        if args.paired:
            t_stat, p_val = stats.ttest_rel(a, b)
        else:
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        print(f"\n  {test_name}: t = {t_stat:.3f}, p = {p_val:.4g}")
        print(f"  {'Significant (p < 0.05)' if p_val < 0.05 else 'Not significant (p >= 0.05)'}: "
              "the camera condition "
              f"{'measurably changes' if p_val < 0.05 else 'does not measurably change'} "
              "how close the policy stays to the expert.")
    except ImportError:
        print("\n  scipy not installed -- skipping t-test. Install with: uv pip install scipy")


if __name__ == "__main__":
    main()

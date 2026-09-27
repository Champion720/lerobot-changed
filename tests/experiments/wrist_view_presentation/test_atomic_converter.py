from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from experiments.wrist_view_presentation import convert_raw_to_lerobot as converter, time_sync
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _write_raw_episode(raw_dir: Path) -> None:
    episode = raw_dir / "episode_000"
    episode.mkdir(parents=True)
    frame_count = 10
    states = "joint_1\n" + "".join(f"{index / 100}\n" for index in range(frame_count))
    actions = "dx,dy,dz,dyaw,dpitch,droll\n" + "0,0,0,0,0,0\n" * frame_count
    (episode / "states.csv").write_text(states, encoding="utf-8")
    (episode / "actions.csv").write_text(
        actions,
        encoding="utf-8",
    )
    time_sync._write_video(
        episode / "video.mp4",
        np.zeros((frame_count, 64, 64, 3), dtype=np.uint8),
        np.arange(frame_count),
        fps=30,
    )


def _run_main(monkeypatch: pytest.MonkeyPatch, raw_dir: Path, target_root: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_raw_to_lerobot.py",
            "--raw_dir",
            str(raw_dir),
            "--repo_id",
            "local/atomic-test",
            "--fps",
            "30",
            "--task",
            "atomic conversion test",
            "--output_dir",
            str(target_root),
        ],
    )
    converter.main()


class _FakeDataset:
    def __init__(self, root: Path, *, fail_finalize: bool = False) -> None:
        self.root = root
        self.fail_finalize = fail_finalize
        self.finalize_calls = 0
        self.frames = []

    def add_frame(self, frame) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        (self.root / "episode.saved").write_text(
            str(len(self.frames)),
            encoding="utf-8",
        )

    def finalize(self) -> None:
        self.finalize_calls += 1
        if self.fail_finalize:
            raise RuntimeError("injected finalize failure")
        (self.root / "FINALIZED").write_text("yes", encoding="utf-8")


def test_real_dataset_remains_readable_after_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    target_root = tmp_path / "dataset"
    _write_raw_episode(raw_dir)

    _run_main(monkeypatch, raw_dir, target_root)

    info = json.loads((target_root / "meta" / "info.json").read_text(encoding="utf-8"))
    data_files = list((target_root / "data").rglob("*.parquet"))
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 10
    assert len(data_files) == 1
    assert pq.read_table(data_files[0]).num_rows == 10
    assert list(tmp_path.glob(f".{target_root.name}.staging-*")) == []


@pytest.mark.parametrize("preexisting_empty_target", [False, True])
def test_converter_publishes_only_finalized_sibling_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting_empty_target: bool,
) -> None:
    raw_dir = tmp_path / "raw"
    target_root = tmp_path / "dataset"
    _write_raw_episode(raw_dir)
    if preexisting_empty_target:
        target_root.mkdir()

    created: list[_FakeDataset] = []

    def fake_create(**kwargs):
        staging_root = Path(kwargs["root"])
        assert staging_root.parent.parent == target_root.parent
        assert staging_root.parent.name.startswith(f".{target_root.name}.staging-")
        assert staging_root.name == "dataset"
        assert staging_root != target_root
        staging_root.mkdir(exist_ok=False)
        dataset = _FakeDataset(staging_root)
        created.append(dataset)
        return dataset

    monkeypatch.setattr(LeRobotDataset, "create", staticmethod(fake_create))

    _run_main(monkeypatch, raw_dir, target_root)

    assert len(created) == 1
    assert created[0].finalize_calls == 1
    assert (target_root / "FINALIZED").read_text(encoding="utf-8") == "yes"
    assert (target_root / "episode.saved").read_text(encoding="utf-8") == "10"
    assert list(tmp_path.glob(f".{target_root.name}.staging-*")) == []


def test_converter_rejects_nonempty_target_without_touching_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    target_root = tmp_path / "dataset"
    _write_raw_episode(raw_dir)
    target_root.mkdir()
    sentinel = target_root / "keep.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")

    def unexpected_create(**kwargs):
        raise AssertionError(f"create must not be called: {kwargs}")

    monkeypatch.setattr(LeRobotDataset, "create", staticmethod(unexpected_create))

    with pytest.raises(SystemExit, match="already exists and is not empty"):
        _run_main(monkeypatch, raw_dir, target_root)

    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"
    assert list(tmp_path.glob(f".{target_root.name}.staging-*")) == []


def test_converter_cleans_staging_when_finalize_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    target_root = tmp_path / "dataset"
    _write_raw_episode(raw_dir)
    created: list[_FakeDataset] = []

    def fake_create(**kwargs):
        staging_root = Path(kwargs["root"])
        staging_root.mkdir(exist_ok=False)
        dataset = _FakeDataset(staging_root, fail_finalize=True)
        created.append(dataset)
        return dataset

    monkeypatch.setattr(LeRobotDataset, "create", staticmethod(fake_create))

    with pytest.raises(RuntimeError, match="injected finalize failure"):
        _run_main(monkeypatch, raw_dir, target_root)

    assert len(created) == 1
    assert created[0].finalize_calls == 2
    assert not target_root.exists()
    assert list(tmp_path.glob(f".{target_root.name}.staging-*")) == []
    assert list(tmp_path.glob(f".{target_root.name}.input-snapshot-*")) == []


def test_converter_refuses_target_created_during_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    target_root = tmp_path / "dataset"
    _write_raw_episode(raw_dir)

    class TargetRacingDataset(_FakeDataset):
        def finalize(self) -> None:
            super().finalize()
            target_root.mkdir()
            (target_root / "other-process.txt").write_text("keep", encoding="utf-8")

    def fake_create(**kwargs):
        staging_root = Path(kwargs["root"])
        staging_root.mkdir(exist_ok=False)
        return TargetRacingDataset(staging_root)

    monkeypatch.setattr(LeRobotDataset, "create", staticmethod(fake_create))

    with pytest.raises(FileExistsError, match="appeared while converting"):
        _run_main(monkeypatch, raw_dir, target_root)

    assert (target_root / "other-process.txt").read_text(encoding="utf-8") == "keep"
    assert list(tmp_path.glob(f".{target_root.name}.staging-*")) == []


def test_converter_csv_aba_on_original_cannot_affect_snapshot_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    target_root = tmp_path / "dataset"
    _write_raw_episode(raw_dir)
    source_actions = raw_dir / "episode_000" / "actions.csv"
    original_bytes = source_actions.read_bytes()
    original_digest = converter.sha256_file(source_actions)
    malicious_bytes = b"dx,dy,dz,dyaw,dpitch,droll\n9,9,9,9,9,9\n"
    original_read_csv_table = converter.read_csv_table
    injected_paths: list[Path] = []

    def aba_read_csv_table(path: Path):
        path = Path(path)
        if path.name == "actions.csv":
            injected_paths.append(path)
            source_actions.write_bytes(malicious_bytes)
            try:
                return original_read_csv_table(path)
            finally:
                source_actions.write_bytes(original_bytes)
        return original_read_csv_table(path)

    created: list[_FakeDataset] = []

    def fake_create(**kwargs):
        staging_root = Path(kwargs["root"])
        staging_root.mkdir(exist_ok=False)
        dataset = _FakeDataset(staging_root)
        created.append(dataset)
        return dataset

    monkeypatch.setattr(converter, "read_csv_table", aba_read_csv_table)
    monkeypatch.setattr(LeRobotDataset, "create", staticmethod(fake_create))

    _run_main(monkeypatch, raw_dir, target_root)

    assert injected_paths
    assert all(path != source_actions for path in injected_paths)
    assert source_actions.read_bytes() == original_bytes
    assert len(created) == 1
    assert len(created[0].frames) == 10
    assert created[0].frames[0]["action"].tolist() == [0.0] * 6
    provenance = json.loads((target_root / "meta" / "source_fingerprints.json").read_text(encoding="utf-8"))
    assert provenance["schema_version"] == 2
    action_fingerprint = provenance["episodes"]["episode_000"]["actions.csv"]
    assert action_fingerprint == {
        "path": str(source_actions.absolute()),
        "sha256": original_digest,
    }
    assert list(tmp_path.glob(f".{target_root.name}.staging-*")) == []
    assert list(tmp_path.glob(f".{target_root.name}.input-snapshot-*")) == []


def test_converter_cleanup_never_removes_unowned_staging_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    target_root = tmp_path / "dataset"
    _write_raw_episode(raw_dir)
    unowned = tmp_path / ".dataset.staging-other-process"
    unowned.mkdir()
    sentinel = unowned / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    unowned_snapshot = tmp_path / ".dataset.input-snapshot-other-process"
    unowned_snapshot.mkdir()
    snapshot_sentinel = unowned_snapshot / "keep.txt"
    snapshot_sentinel.write_text("keep", encoding="utf-8")

    def failing_create(**kwargs):
        raise RuntimeError(f"injected create failure at {kwargs['root']}")

    monkeypatch.setattr(LeRobotDataset, "create", staticmethod(failing_create))

    with pytest.raises(RuntimeError, match="injected create failure"):
        _run_main(monkeypatch, raw_dir, target_root)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert snapshot_sentinel.read_text(encoding="utf-8") == "keep"

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WRIST_VIEW_PRESENTATION_DIR = Path(__file__).parents[3] / "experiments" / "wrist_view_presentation"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, WRIST_VIEW_PRESENTATION_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


forward_kinematics = _load_script("forward_kinematics")
time_sync = _load_script("time_sync")
convert_raw = _load_script("convert_raw_to_lerobot")


def test_forward_kinematics_converts_degree_angles_once() -> None:
    fk = forward_kinematics.ForwardKinematics(
        {
            "angle_unit": "deg",
            "joints": [
                {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta_offset": 10.0},
            ],
        }
    )

    pose = fk.pose([80.0])

    np.testing.assert_allclose(pose[:3], [0.0, 1.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(pose[3:], [np.pi / 2, 0.0, 0.0], atol=1e-10)


def test_forward_kinematics_converts_degree_alpha() -> None:
    fk = forward_kinematics.ForwardKinematics(
        {
            "angle_unit": "deg",
            "joints": [
                {"a": 0.0, "alpha": 90.0, "d": 0.0},
            ],
        }
    )

    np.testing.assert_allclose(fk.pose([0.0])[3:], [0.0, 0.0, np.pi / 2], atol=1e-10)


def test_forward_kinematics_rejects_bad_shape_and_non_finite_input() -> None:
    fk = forward_kinematics.ForwardKinematics({"joints": [{"a": 1.0, "alpha": 0.0, "d": 0.0}]})

    with pytest.raises(ValueError, match="shape"):
        fk.pose([[0.0]])
    with pytest.raises(ValueError, match="non-finite"):
        fk.pose([np.nan])


def test_forward_kinematics_normalizes_mm_degrees_and_joint_direction() -> None:
    fk = forward_kinematics.ForwardKinematics(
        {
            "length_unit": "mm",
            "angle_unit": "deg",
            "joint_names": ["shoulder"],
            "joints": [
                {
                    "a": 1000.0,
                    "alpha": 0.0,
                    "d": 0.0,
                    "theta_offset": 0.0,
                    "direction": -1,
                }
            ],
        }
    )

    np.testing.assert_allclose(fk.normalize_joint_angles([[90.0]]), [[-np.pi / 2]])
    np.testing.assert_allclose(
        fk.pose([90.0]),
        [0.0, -1.0, 0.0, -np.pi / 2, 0.0, 0.0],
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "transform",
    [
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]],
        [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    ],
)
def test_forward_kinematics_rejects_non_rigid_base_transform(transform) -> None:
    with pytest.raises(ValueError, match="bottom row|orthonormal"):
        forward_kinematics.ForwardKinematics(
            {
                "joints": [{"a": 1.0, "alpha": 0.0, "d": 0.0}],
                "base": transform,
            }
        )


def test_timestamp_csv_is_stably_sorted(tmp_path: Path) -> None:
    path = tmp_path / "robot.csv"
    pd.DataFrame(
        {
            "timestamp": [2.0, 0.0, 1.0],
            "joint": [20.0, 0.0, 10.0],
        }
    ).to_csv(path, index=False)

    timestamps, values, columns = time_sync._read_ts_csv(path)

    np.testing.assert_array_equal(timestamps, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(values[:, 0], [0.0, 10.0, 20.0])
    assert columns == ["joint"]


@pytest.mark.parametrize(
    ("timestamps", "match"),
    [
        ([0.0, 1.0, 1.0], "duplicate timestamp"),
        ([0.0, np.inf, 2.0], "NaN or infinity"),
    ],
)
def test_timestamp_csv_rejects_ambiguous_or_non_finite_time(
    tmp_path: Path,
    timestamps: list[float],
    match: str,
) -> None:
    path = tmp_path / "robot.csv"
    pd.DataFrame({"timestamp": timestamps, "joint": [0.0, 1.0, 2.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match=match):
        time_sync._read_ts_csv(path)


def test_timestamp_csv_rejects_rows_wider_than_header(tmp_path: Path) -> None:
    path = tmp_path / "robot.csv"
    path.write_text(
        "timestamp,j1,j2\n1.0,0.1,0.2,0.3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="4 fields"):
        time_sync._read_ts_csv(path)


def test_timestamp_csv_rejects_duplicate_raw_header_names(tmp_path: Path) -> None:
    path = tmp_path / "robot.csv"
    path.write_text("timestamp,j1,j1\n0.0,0.1,0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="header names must be unique"):
        time_sync._read_ts_csv(path)


def test_nearest_video_frame_compares_both_neighbors() -> None:
    indices = time_sync._nearest_indices(
        np.array([0.0, 1.0, 2.0]),
        np.array([-0.2, 0.1, 0.5, 0.9, 2.2]),
    )

    np.testing.assert_array_equal(indices, [0, 0, 0, 1, 2])


def test_video_metadata_rejects_invalid_fps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    meta = tmp_path / "video_meta.json"
    meta.write_text('{"start_ts": 1.0, "fps": 0}', encoding="utf-8")

    class FakeCapture:
        def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
            return True

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 3.0
            return 30.0

        def release(self) -> None:
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: FakeCapture())

    with pytest.raises(ValueError, match="fps must be positive"):
        time_sync._video_timestamps(meta, video)


def test_video_metadata_uses_irregular_per_frame_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    meta = tmp_path / "video_meta.json"
    meta.write_text(
        '{"episode_start_ts":9.5,"first_frame_ts":10.03,"frame_timestamps_s":[10.03,10.071,10.14]}',
        encoding="utf-8",
    )

    class FakeCapture:
        def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
            return True

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 3.0
            return 30.0

        def release(self) -> None:
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: FakeCapture())

    timestamps, _ = time_sync._video_timestamps(meta, video, frame_count=3)

    np.testing.assert_allclose(timestamps, [10.03, 10.071, 10.14])


@pytest.mark.parametrize(
    "timestamps",
    [
        [1.0, 1.1],
        [1.0, 1.0, 1.2],
        [1.0, np.nan, 1.2],
    ],
)
def test_video_metadata_rejects_bad_per_frame_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timestamps: list[float],
) -> None:
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    meta = tmp_path / "video_meta.json"
    meta.write_text(
        json.dumps({"frame_timestamps_s": timestamps}),
        encoding="utf-8",
    )

    class FakeCapture:
        def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
            return True

        def get(self, prop: int) -> float:
            return 3.0 if prop == cv2.CAP_PROP_FRAME_COUNT else 30.0

        def release(self) -> None:
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: FakeCapture())

    with pytest.raises(ValueError, match="entries|strictly increasing|NaN or infinity"):
        time_sync._video_timestamps(meta, video, frame_count=3)


def test_video_writer_writes_and_verifies_all_frames(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    frames[1, :, :, 1] = 127
    frames[2, :, :, 2] = 255
    output = tmp_path / "aligned.mp4"

    time_sync._write_video(output, frames, np.array([0, 1, 2]), fps=10)

    assert output.stat().st_size > 0
    assert time_sync._count_decodable_frames(output) == 3


def test_video_writer_rejects_invalid_indices_before_writing(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="outside"):
        time_sync._write_video(tmp_path / "bad.mp4", frames, np.array([0, 2]), fps=10)
    assert not (tmp_path / "bad.mp4").exists()


def test_video_writer_cleans_temp_on_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    released: list[bool] = []

    class InterruptingWriter:
        def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoWriter
            return True

        def write(self, _frame) -> None:
            raise KeyboardInterrupt

        def release(self) -> None:
            released.append(True)

    def make_interrupting_writer(path: str, *_args, **_kwargs):
        Path(path).write_bytes(b"partial")
        return InterruptingWriter()

    monkeypatch.setattr(cv2, "VideoWriter", make_interrupting_writer)
    output = tmp_path / "interrupted.mp4"
    frames = np.zeros((1, 8, 8, 3), dtype=np.uint8)

    with pytest.raises(KeyboardInterrupt):
        time_sync._write_video(output, frames, np.array([0]), fps=10)

    assert released
    assert not output.exists()
    assert not (tmp_path / ".interrupted.tmp.mp4").exists()


def test_streaming_video_resampler_does_not_require_stacking(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    frames[1, :, :, 1] = 127
    frames[2, :, :, 2] = 255
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    time_sync._write_video(source, frames, np.array([0, 1, 2]), fps=10)

    time_sync._write_resampled_video(
        source,
        output,
        np.array([0, 0, 2]),
        fps=10,
    )

    assert time_sync._count_decodable_frames(output) == 3


def test_streaming_resampler_cleans_temp_on_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cv2")
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    time_sync._write_video(source, frames, np.array([0, 1]), fps=10)

    def interrupt_decode_check(_path: Path) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(time_sync, "_count_decodable_frames", interrupt_decode_check)
    with pytest.raises(KeyboardInterrupt):
        time_sync._write_resampled_video(
            source,
            output,
            np.array([0, 1]),
            fps=10,
        )

    assert not output.exists()
    assert not (tmp_path / ".output.tmp.mp4").exists()


def test_sync_episode_without_video_sorts_and_resamples(tmp_path: Path) -> None:
    episode_in = tmp_path / "raw" / "episode_000"
    episode_in.mkdir(parents=True)
    pd.DataFrame({"timestamp": [2.0, 0.0, 1.0], "j1": [20.0, 0.0, 10.0]}).to_csv(
        episode_in / "robot.csv", index=False
    )
    action_rows = pd.DataFrame(
        {
            "timestamp": [1.0, 2.0, 0.0],
            "dx": [1.0, 2.0, 0.0],
            "dy": [0.0, 0.0, 0.0],
            "dz": [0.0, 0.0, 0.0],
            "dyaw": [0.0, 0.0, 0.0],
            "dpitch": [0.0, 0.0, 0.0],
            "droll": [0.0, 0.0, 0.0],
        }
    )
    action_rows.to_csv(episode_in / "phone.csv", index=False)
    action_rows.to_csv(episode_in / "applied_actions.csv", index=False)
    episode_out = tmp_path / "aligned" / "episode_000"

    time_sync.sync_episode(episode_in, episode_out, out_fps=2)

    states = pd.read_csv(episode_out / "states.csv")
    actions = pd.read_csv(episode_out / "actions.csv")
    np.testing.assert_allclose(states["timestamp"], [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(states["j1"], [0.0, 5.0, 10.0, 15.0])
    np.testing.assert_allclose(actions["dx"], [0.0, 0.0, 1.0, 0.0])


def test_sync_rejects_implausible_episode_duration_before_grid_allocation(
    tmp_path: Path,
) -> None:
    episode_in = tmp_path / "raw" / "episode_000"
    episode_in.mkdir(parents=True)
    pd.DataFrame({"timestamp": [0.0, 1000.0], "j1": [0.0, 1.0]}).to_csv(episode_in / "robot.csv", index=False)
    actions = pd.DataFrame(
        {
            "timestamp": [0.0],
            "dx": [0.0],
            "dy": [0.0],
            "dz": [0.0],
            "dyaw": [0.0],
            "dpitch": [0.0],
            "droll": [0.0],
        }
    )
    actions.to_csv(episode_in / "phone.csv", index=False)
    actions.to_csv(episode_in / "applied_actions.csv", index=False)

    with pytest.raises(ValueError, match="check timestamp units"):
        time_sync.sync_episode(
            episode_in,
            tmp_path / "aligned" / "episode_000",
            out_fps=30,
            max_episode_duration_s=60,
        )


def test_increment_resampling_preserves_total_motion_across_rates() -> None:
    timestamps = np.arange(0.0, 0.33, 0.033)
    increments = np.column_stack(
        [
            np.full(len(timestamps), 0.001),
            np.zeros((len(timestamps), 5)),
        ]
    )
    grid = np.arange(0.0, 0.36, 1.0 / 30.0)

    resampled = time_sync._sum_increments(
        grid,
        interval_end_s=0.36,
        timestamps=timestamps,
        increments=increments,
    )

    np.testing.assert_allclose(resampled.sum(axis=0), increments.sum(axis=0))


def test_increment_resampling_composes_noncommuting_rotations_in_order() -> None:
    timestamps = np.array([0.01, 0.02])
    increments = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
        ]
    )

    resampled = time_sync._compose_increments(
        np.array([0.0]),
        interval_end_s=0.1,
        timestamps=timestamps,
        increments=increments,
        translation_frame="base",
        rotation_frame="tool",
    )

    expected_rotation = time_sync._rotation_zyx(0.0, 0.1, 0.0) @ time_sync._rotation_zyx(0.1, 0.0, 0.0)
    actual_rotation = time_sync._rotation_zyx(*resampled[0, 3:])
    np.testing.assert_allclose(actual_rotation, expected_rotation, atol=1e-7)
    assert not np.allclose(resampled[0, 3:], increments[:, 3:].sum(axis=0))


def test_increment_resampling_rotates_later_tool_translation() -> None:
    resampled = time_sync._compose_increments(
        np.array([0.0]),
        interval_end_s=0.1,
        timestamps=np.array([0.01, 0.02]),
        increments=np.array(
            [
                [0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        translation_frame="tool",
        rotation_frame="tool",
    )

    np.testing.assert_allclose(
        resampled[0, :3],
        [0.01 * np.cos(0.1), 0.01 * np.sin(0.1), 0.0],
        atol=1e-7,
    )


def test_increment_resampling_rejects_unsafe_aggregated_step() -> None:
    with pytest.raises(ValueError, match="unsafe combined step"):
        time_sync._compose_increments(
            np.array([0.0]),
            interval_end_s=0.1,
            timestamps=np.array([0.01, 0.02]),
            increments=np.array(
                [
                    [0.015, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.015, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            max_translation_step_m=0.02,
            max_rotation_step_rad=0.2,
        )


def test_increment_resampling_rejects_deployment_velocity_above_bridge_limit() -> None:
    with pytest.raises(ValueError, match="deployment translation velocity"):
        time_sync._compose_increments(
            np.array([0.0]),
            interval_end_s=1.0 / 30.0,
            timestamps=np.arange(7, dtype=float) * 0.005,
            increments=np.tile(
                np.array([0.0005, 0.0, 0.0, 0.0, 0.0, 0.0]),
                (7, 1),
            ),
            max_translation_step_m=0.02,
            max_rotation_step_rad=0.2,
            output_period_s=1.0 / 30.0,
            max_translation_velocity_m_s=0.1,
            max_rotation_velocity_rad_s=1.0,
        )


def test_increment_resampling_validates_exact_float32_training_value() -> None:
    with pytest.raises(ValueError, match="rotation step"):
        time_sync._compose_increments(
            np.array([0.0]),
            interval_end_s=0.1,
            timestamps=np.array([0.01]),
            increments=np.array([[0.0, 0.0, 0.0, 0.1, 0.0, 0.0]]),
            max_translation_step_m=0.02,
            max_rotation_step_rad=0.1,
        )


def test_increment_resampling_rejects_pose_dependent_mixed_frames() -> None:
    with pytest.raises(ValueError, match="initial FK pose"):
        time_sync._compose_increments(
            np.array([0.0]),
            interval_end_s=0.1,
            timestamps=np.array([0.01, 0.02]),
            increments=np.array(
                [
                    [0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
                    [0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            translation_frame="tool",
            rotation_frame="base",
        )


def test_sync_episode_failure_leaves_no_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode_in = tmp_path / "raw" / "episode_000"
    episode_in.mkdir(parents=True)
    pd.DataFrame({"timestamp": [0.0, 1.0], "j1": [0.0, 1.0]}).to_csv(
        episode_in / "robot.csv",
        index=False,
    )
    actions = pd.DataFrame(
        {
            "timestamp": [0.0],
            "dx": [0.0],
            "dy": [0.0],
            "dz": [0.0],
            "dyaw": [0.0],
            "dpitch": [0.0],
            "droll": [0.0],
        }
    )
    actions.to_csv(episode_in / "phone.csv", index=False)
    actions.to_csv(episode_in / "applied_actions.csv", index=False)
    episode_out = tmp_path / "aligned" / "episode_000"
    original_to_csv = pd.DataFrame.to_csv
    call_count = 0

    def fail_on_actions(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("simulated actions.csv write failure")
        return original_to_csv(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_on_actions)
    with pytest.raises(OSError, match="simulated"):
        time_sync.sync_episode(episode_in, episode_out, out_fps=2)

    assert not episode_out.exists()
    assert not list(episode_out.parent.glob(".episode_000.staging-*"))
    assert not list(episode_out.parent.glob(".episode_000.input-snapshot-*"))


def test_sync_main_cleans_staging_on_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode_in = tmp_path / "raw" / "episode_000"
    episode_in.mkdir(parents=True)
    for filename in ("robot.csv", "phone.csv", "applied_actions.csv"):
        (episode_in / filename).write_text("placeholder\n", encoding="utf-8")
    output = tmp_path / "aligned"

    monkeypatch.setattr(
        time_sync,
        "_load_bridge_action_config",
        lambda _path: ("base", "tool", 0.1, 0.1, 1.0, 1.0),
    )

    def interrupt_sync(*_args, **_kwargs) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(time_sync, "sync_episode", interrupt_sync)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "time_sync.py",
            "--in_dir",
            str(episode_in.parent),
            "--out_dir",
            str(output),
            "--robot_bridge_config",
            str(tmp_path / "bridge.json"),
        ],
    )

    with pytest.raises(KeyboardInterrupt):
        time_sync.main()

    assert not output.exists()
    assert not list(tmp_path.glob(".aligned.staging-*"))


def test_sync_episode_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    episode_in = tmp_path / "raw" / "episode_000"
    episode_in.mkdir(parents=True)
    episode_out = tmp_path / "aligned" / "episode_000"
    episode_out.mkdir(parents=True)
    sentinel = episode_out / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        time_sync.sync_episode(episode_in, episode_out, out_fps=2)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_sync_episode_video_aba_on_original_cannot_affect_snapshot_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    episode_in = tmp_path / "raw" / "episode_000"
    episode_in.mkdir(parents=True)
    pd.DataFrame({"timestamp": [10.0, 10.3], "j1": [0.0, 0.1]}).to_csv(episode_in / "robot.csv", index=False)
    actions = pd.DataFrame(
        {
            "timestamp": [10.0],
            "dx": [0.0],
            "dy": [0.0],
            "dz": [0.0],
            "dyaw": [0.0],
            "dpitch": [0.0],
            "droll": [0.0],
        }
    )
    actions.to_csv(episode_in / "phone.csv", index=False)
    actions.to_csv(episode_in / "applied_actions.csv", index=False)
    original_frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    time_sync._write_video(
        episode_in / "video.mp4",
        original_frames,
        np.arange(3),
        fps=10,
    )
    source_video = episode_in / "video.mp4"
    original_video_bytes = source_video.read_bytes()
    original_video_digest = time_sync._sha256_file(source_video)
    replacement = episode_in / "replacement.mp4"
    time_sync._write_video(
        replacement,
        np.full((3, 16, 16, 3), 255, dtype=np.uint8),
        np.arange(3),
        fps=10,
    )
    replacement_bytes = replacement.read_bytes()
    replacement.unlink()
    (episode_in / "video_meta.json").write_text(
        json.dumps({"frame_timestamps_s": [10.0, 10.1, 10.2]}),
        encoding="utf-8",
    )
    (episode_in / "frame_timestamps.csv").write_text(
        "frame_index,frame_timestamp_s\n0,10.0\n1,10.1\n2,10.2\n",
        encoding="utf-8",
    )
    original_resampler = time_sync._write_resampled_video
    observed_sources: list[Path] = []

    def replace_restore_then_resample(source_path, output_path, indices, fps):
        source_path = Path(source_path)
        observed_sources.append(source_path)
        source_video.write_bytes(replacement_bytes)
        try:
            return original_resampler(source_path, output_path, indices, fps)
        finally:
            source_video.write_bytes(original_video_bytes)

    monkeypatch.setattr(
        time_sync,
        "_write_resampled_video",
        replace_restore_then_resample,
    )
    episode_out = tmp_path / "aligned" / "episode_000"
    time_sync.sync_episode(episode_in, episode_out, out_fps=10)

    assert observed_sources
    assert all(path != source_video for path in observed_sources)
    assert source_video.read_bytes() == original_video_bytes
    capture = cv2.VideoCapture(str(episode_out / "video.mp4"))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok
    assert frame is not None
    assert float(frame.mean()) < 10.0
    provenance = json.loads((episode_out / "source_fingerprints.json").read_text(encoding="utf-8"))
    assert provenance["schema_version"] == 2
    assert provenance["files"]["video.mp4"] == {
        "path": str(source_video.absolute()),
        "sha256": original_video_digest,
    }
    assert "frame_timestamps.csv" in provenance["files"]
    assert not list(episode_out.parent.glob(".episode_000.staging-*"))
    assert not list(episode_out.parent.glob(".episode_000.input-snapshot-*"))


def test_converter_rejects_reordered_protocol_columns(tmp_path: Path) -> None:
    path = tmp_path / "actions.csv"
    pd.DataFrame(
        {
            "dy": [0.0],
            "dx": [0.0],
            "dz": [0.0],
            "dyaw": [0.0],
            "dpitch": [0.0],
            "droll": [0.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="in this order"):
        convert_raw.read_csv(path, expected_columns=convert_raw.ACTION_NAMES)


def test_converter_supports_normal_package_import() -> None:
    module = importlib.import_module("experiments.wrist_view_presentation.convert_raw_to_lerobot")

    assert module.ForwardKinematics.__name__ == "ForwardKinematics"


def test_converter_csv_rejects_out_of_order_alignment_column(tmp_path: Path) -> None:
    path = tmp_path / "states.csv"
    pd.DataFrame({"timestamp": [0.0, 2.0, 1.0], "j1": [0.0, 2.0, 1.0]}).to_csv(
        path,
        index=False,
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        convert_raw.read_csv(path)


def test_converter_csv_rejects_non_numeric_or_non_finite_values(tmp_path: Path) -> None:
    non_numeric = tmp_path / "non_numeric.csv"
    pd.DataFrame({"j1": [0.0, "bad"]}).to_csv(non_numeric, index=False)
    with pytest.raises(ValueError, match="must be numeric"):
        convert_raw.read_csv(non_numeric)

    non_finite = tmp_path / "non_finite.csv"
    pd.DataFrame({"j1": [0.0, np.inf]}).to_csv(non_finite, index=False)
    with pytest.raises(ValueError, match="NaN or infinity"):
        convert_raw.read_csv(non_finite)


def test_converter_csv_rejects_duplicate_raw_header_names(tmp_path: Path) -> None:
    path = tmp_path / "states.csv"
    path.write_text("j1,j1\n0.1,0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="header names must be unique"):
        convert_raw.read_csv(path)


def test_converter_rejects_length_mismatch_by_default() -> None:
    with pytest.raises(ValueError, match="stream length mismatch"):
        convert_raw.resolve_episode_length(
            {"states": 100, "actions": 99, "video": 100},
            context="episode_000",
        )


def test_converter_only_trims_with_explicit_tolerance() -> None:
    assert (
        convert_raw.resolve_episode_length(
            {"states": 100, "actions": 99, "video": 100},
            max_length_mismatch=1,
            context="episode_000",
        )
        == 99
    )
    with pytest.raises(ValueError, match="exceeds allowed 1"):
        convert_raw.resolve_episode_length(
            {"states": 100, "actions": 98, "video": 100},
            max_length_mismatch=1,
        )


@pytest.mark.parametrize("value", ["480", "480x", "0x640", "axb"])
def test_parse_resize_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="--resize"):
        convert_raw.parse_resize(value)


def test_empty_video_is_rejected(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    path = tmp_path / "video.mp4"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="could not open|no frames"):
        convert_raw.read_video(path, None)

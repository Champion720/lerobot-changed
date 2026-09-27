import json
from pathlib import Path

import pytest

from experiments.wrist_view_presentation.acquisition_interfaces import (
    CARTESIAN_DELTA_ACTION_SCHEMA,
    GRIPPER_EXTENSION_VERSION,
    ClockToUnixMapping,
    FrameTimestamp,
    GripperCommand,
    GripperContractError,
    SignalField,
    VectorSchema,
    VideoClockContract,
    VideoClockContractError,
    VideoEpisodeTiming,
    apply_validated_gripper_command,
    parse_optional_gripper_command,
    read_validated_gripper_state,
    validate_cartesian_delta_action,
    validate_frame_timestamps,
    write_video_timing_metadata,
)


class FakeGripperAdapter:
    extension_version = GRIPPER_EXTENSION_VERSION
    action_schema = VectorSchema(
        "test.parallel_gripper.action",
        2,
        (SignalField("opening", "m"), SignalField("force_limit", "N")),
    )
    state_schema = VectorSchema(
        "test.parallel_gripper.state",
        3,
        (SignalField("measured_opening", "m"), SignalField("measured_force", "N")),
    )

    def __init__(self) -> None:
        self.applied: list[GripperCommand] = []
        self.state: tuple[float, ...] = (0.02, 3.5)
        self.stopped = False

    def apply(self, command: GripperCommand) -> None:
        self.applied.append(command)

    def read_state(self) -> tuple[float, ...]:
        return self.state

    def stop(self) -> None:
        self.stopped = True


def make_same_clock_contract() -> VideoClockContract:
    mapping = ClockToUnixMapping(
        source_clock_id="capture_monotonic",
        source_anchor_s=100.0,
        unix_anchor_s=1_800_000_000.0,
        uncertainty_s=0.002,
    )
    return VideoClockContract(
        frame_clock_id="capture_monotonic",
        episode_clock_id="capture_monotonic",
        same_clock_as_episode=True,
        frame_to_unix=mapping,
        episode_to_unix=mapping,
    )


def test_cartesian_action_stays_exactly_six_dimensional() -> None:
    assert [field.name for field in CARTESIAN_DELTA_ACTION_SCHEMA.fields] == [
        "dx",
        "dy",
        "dz",
        "dyaw",
        "dpitch",
        "droll",
    ]
    assert validate_cartesian_delta_action([0, 0, 0, 0, 0, 0]) == (0.0,) * 6
    with pytest.raises(GripperContractError, match="requires 6"):
        validate_cartesian_delta_action([0, 0, 0, 0, 0, 0, 0.01])


def test_optional_gripper_requires_explicit_adapter_and_exact_schemas() -> None:
    adapter = FakeGripperAdapter()
    payload = {
        "extension_version": 1,
        "schema_id": adapter.action_schema.schema_id,
        "schema_version": adapter.action_schema.version,
        "values": [0.025, 4.0],
    }

    assert parse_optional_gripper_command(None, adapter=None) is None
    with pytest.raises(GripperContractError, match="no gripper adapter"):
        parse_optional_gripper_command(payload, adapter=None)

    command = parse_optional_gripper_command(payload, adapter=adapter)
    assert command is not None
    assert command.values == (0.025, 4.0)
    assert adapter.action_schema.schema_id != CARTESIAN_DELTA_ACTION_SCHEMA.schema_id
    assert adapter.state_schema.schema_id != adapter.action_schema.schema_id

    apply_validated_gripper_command(adapter, command)
    assert adapter.applied == [command]
    assert read_validated_gripper_state(adapter) == (0.02, 3.5)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"extension_version": 99}, "unsupported gripper extension"),
        ({"schema_id": "test.wrong"}, "does not exactly match"),
        ({"schema_version": 999}, "does not exactly match"),
        ({"values": [0.1]}, "requires 2"),
        ({"values": [float("nan"), 1.0]}, "must be finite"),
        ({"extra": True}, "keys do not match"),
    ],
)
def test_gripper_envelope_fails_closed(changes: dict[str, object], message: str) -> None:
    adapter = FakeGripperAdapter()
    payload: dict[str, object] = {
        "extension_version": 1,
        "schema_id": adapter.action_schema.schema_id,
        "schema_version": adapter.action_schema.version,
        "values": [0.025, 4.0],
    }
    payload.update(changes)

    with pytest.raises(GripperContractError, match=message):
        parse_optional_gripper_command(payload, adapter=adapter)
    assert adapter.applied == []


def test_gripper_state_is_checked_before_use() -> None:
    adapter = FakeGripperAdapter()
    adapter.state = (float("inf"), 1.0)

    with pytest.raises(GripperContractError, match="must be finite"):
        read_validated_gripper_state(adapter)


def test_gripper_adapter_rejects_boolean_version_and_missing_methods() -> None:
    adapter = FakeGripperAdapter()
    adapter.extension_version = True
    with pytest.raises(GripperContractError, match="positive integer"):
        parse_optional_gripper_command(
            {
                "extension_version": 1,
                "schema_id": adapter.action_schema.schema_id,
                "schema_version": adapter.action_schema.version,
                "values": [0.02, 1.0],
            },
            adapter=adapter,
        )

    adapter.extension_version = GRIPPER_EXTENSION_VERSION
    adapter.stop = None
    with pytest.raises(GripperContractError, match="callable stop"):
        read_validated_gripper_state(adapter)


def test_clock_contract_maps_frames_to_unix_and_requires_explicit_same_clock_claim() -> None:
    contract = make_same_clock_contract()
    assert contract.frame_to_unix.to_unix_s(100.25) == pytest.approx(1_800_000_000.25)

    event_mapping = ClockToUnixMapping(
        source_clock_id="robot_monotonic",
        source_anchor_s=50.0,
        unix_anchor_s=1_800_000_000.0,
    )
    different_contract = VideoClockContract(
        frame_clock_id="capture_monotonic",
        episode_clock_id="robot_monotonic",
        same_clock_as_episode=False,
        frame_to_unix=contract.frame_to_unix,
        episode_to_unix=event_mapping,
    )
    assert not different_contract.same_clock_as_episode

    with pytest.raises(VideoClockContractError, match="identical clock IDs"):
        VideoClockContract(
            frame_clock_id="capture_monotonic",
            episode_clock_id="robot_monotonic",
            same_clock_as_episode=True,
            frame_to_unix=contract.frame_to_unix,
            episode_to_unix=event_mapping,
        )
    with pytest.raises(VideoClockContractError, match="distinct clock IDs"):
        VideoClockContract(
            frame_clock_id="capture_monotonic",
            episode_clock_id="capture_monotonic",
            same_clock_as_episode=False,
            frame_to_unix=contract.frame_to_unix,
            episode_to_unix=contract.episode_to_unix,
        )


def test_frame_timestamps_require_finite_strict_monotonic_same_clock_data() -> None:
    contract = make_same_clock_contract()
    valid = [
        FrameTimestamp(0, 100.0, "capture_monotonic"),
        FrameTimestamp(1, 100.033, "capture_monotonic"),
    ]
    assert validate_frame_timestamps(valid, contract) == tuple(valid)

    with pytest.raises(VideoClockContractError, match="strictly increasing"):
        validate_frame_timestamps(
            [
                FrameTimestamp(0, 100.0, "capture_monotonic"),
                FrameTimestamp(1, 100.0, "capture_monotonic"),
            ],
            contract,
        )
    with pytest.raises(VideoClockContractError, match="contiguous"):
        validate_frame_timestamps([FrameTimestamp(1, 100.0, "capture_monotonic")], contract)
    with pytest.raises(VideoClockContractError, match="undeclared clock"):
        validate_frame_timestamps([FrameTimestamp(0, 100.0, "browser_clock")], contract)
    with pytest.raises(VideoClockContractError, match="must be finite"):
        FrameTimestamp(0, float("nan"), "capture_monotonic")


def test_reference_writer_only_writes_timing_metadata(tmp_path: Path) -> None:
    contract = make_same_clock_contract()
    video_path = tmp_path / "external_recorder_output.mp4"
    timing = VideoEpisodeTiming(
        episode_id="P001_A_001",
        video_path=str(video_path),
        episode_start_timestamp_s=99.9,
        display="mobile",
        clock_contract=contract,
        frame_timestamps=(
            FrameTimestamp(0, 100.0, "capture_monotonic"),
            FrameTimestamp(1, 100.04, "capture_monotonic"),
        ),
    )
    metadata_path = tmp_path / "timing.json"
    timestamps_path = tmp_path / "frame_timestamps.csv"

    written_metadata, written_timestamps = write_video_timing_metadata(
        timing,
        metadata_path=metadata_path,
        timestamps_csv_path=timestamps_path,
    )

    assert written_metadata == metadata_path
    assert written_timestamps == timestamps_path
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["writer_capability"] == "timing_metadata_only"
    assert metadata["video_capture_verified_by_this_writer"] is False
    assert metadata["frame_count"] == 2
    assert metadata["display"] == "mobile"
    assert metadata["episode_start_ts"] == pytest.approx(1_799_999_999.9)
    assert metadata["first_frame_ts"] == pytest.approx(metadata["frame_timestamps_s"][0])
    assert len(metadata["frame_timestamps_s"]) == 2
    assert "same_clock_as_episode" in metadata["clock_contract"]
    assert "unix_timestamp_s" in timestamps_path.read_text(encoding="utf-8")
    assert not video_path.exists()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_video_timing_metadata(
            timing,
            metadata_path=metadata_path,
            timestamps_csv_path=timestamps_path,
        )


def test_video_timing_uses_formal_display_names() -> None:
    with pytest.raises(VideoClockContractError, match="mobile.*desktop"):
        VideoEpisodeTiming(
            episode_id="episode_000",
            video_path="video.mp4",
            episode_start_timestamp_s=99.9,
            display="pc",
            clock_contract=make_same_clock_contract(),
            frame_timestamps=(FrameTimestamp(0, 100.0, "capture_monotonic"),),
        )

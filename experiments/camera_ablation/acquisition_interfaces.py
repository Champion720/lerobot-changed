"""Strict extension contracts for experiment data acquisition.

This module deliberately contains interfaces and validation only. It does not
discover hardware, actuate a gripper, receive WebRTC frames, or encode video.

The existing robot action remains the six-component Cartesian delta declared by
``CARTESIAN_DELTA_ACTION_SCHEMA``. A gripper command is a separate, optional,
versioned extension. Callers must explicitly persist its action and state schemas;
this module never appends gripper values to the six-dimensional action.

EXPERIMENTER TODO:
    1. Implement ``GripperAdapter`` for the selected end effector and persist its
       exact action/state schemas with every dataset.
    2. Implement ``VideoEpisodeRecorder`` in the process that actually receives
       decoded WebRTC frames and use the capture timestamp of each committed frame.
    3. Measure clock mappings on the deployed devices. Do not copy placeholder
       offsets or infer frame times from nominal FPS.
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

GRIPPER_EXTENSION_VERSION = 1
VIDEO_TIMING_SCHEMA_VERSION = 1
CLOCK_MAPPING_SCHEMA_VERSION = 1
FORMAL_VIDEO_ARTIFACT_TYPE = "video_episode_capture"
FORMAL_VIDEO_WRITER_CAPABILITY = "decoded_frame_video_recorder"
FORMAL_FRAME_TIMESTAMPS_FILENAME = "frame_timestamps.csv"


class AcquisitionContractError(ValueError):
    """Base error for invalid acquisition data or declarations."""


class GripperContractError(AcquisitionContractError):
    """A gripper extension is missing or violates its declared schema."""


class VideoClockContractError(AcquisitionContractError):
    """Video timestamps or clock mappings violate the timing contract."""


def _identifier(value: object, field: str, error_type: type[AcquisitionContractError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{field} must be a non-empty string")
    result = value.strip()
    if any(character.isspace() for character in result):
        raise error_type(f"{field} must not contain whitespace")
    return result


def _version(value: object, field: str, error_type: type[AcquisitionContractError]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise error_type(f"{field} must be a positive integer")
    return value


def _finite_number(
    value: object,
    field: str,
    error_type: type[AcquisitionContractError],
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise error_type(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise error_type(f"{field} must be finite")
    if positive and result <= 0:
        raise error_type(f"{field} must be greater than zero")
    if nonnegative and result < 0:
        raise error_type(f"{field} must be non-negative")
    return result


@dataclass(frozen=True)
class SignalField:
    """One scalar channel in an explicitly versioned vector schema."""

    name: str
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "field name", GripperContractError))
        object.__setattr__(self, "unit", _identifier(self.unit, "field unit", GripperContractError))


@dataclass(frozen=True)
class VectorSchema:
    """Ordered numeric-vector schema used by one hardware adapter."""

    schema_id: str
    version: int
    fields: tuple[SignalField, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_id",
            _identifier(self.schema_id, "schema_id", GripperContractError),
        )
        object.__setattr__(
            self,
            "version",
            _version(self.version, "schema version", GripperContractError),
        )
        fields = tuple(self.fields)
        if not fields:
            raise GripperContractError("a vector schema must declare at least one field")
        if not all(isinstance(field, SignalField) for field in fields):
            raise GripperContractError("schema fields must be SignalField instances")
        names = [field.name for field in fields]
        if len(names) != len(set(names)):
            raise GripperContractError("schema field names must be unique")
        object.__setattr__(self, "fields", fields)

    def validate_vector(self, values: Sequence[object], *, label: str) -> tuple[float, ...]:
        """Return a finite tuple after exact shape validation."""

        if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
            raise GripperContractError(f"{label} must be a numeric sequence")
        if len(values) != len(self.fields):
            raise GripperContractError(
                f"{label} has {len(values)} values; schema {self.schema_id!r} requires {len(self.fields)}"
            )
        return tuple(
            _finite_number(value, f"{label}[{index}]", GripperContractError)
            for index, value in enumerate(values)
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable schema declaration."""

        return {
            "schema_id": self.schema_id,
            "version": self.version,
            "fields": [{"name": field.name, "unit": field.unit} for field in self.fields],
        }


CARTESIAN_DELTA_ACTION_SCHEMA = VectorSchema(
    schema_id="camera_ablation.cartesian_delta_6d",
    version=1,
    fields=(
        SignalField("dx", "m"),
        SignalField("dy", "m"),
        SignalField("dz", "m"),
        SignalField("dyaw", "rad"),
        SignalField("dpitch", "rad"),
        SignalField("droll", "rad"),
    ),
)


def validate_cartesian_delta_action(values: Sequence[object]) -> tuple[float, ...]:
    """Validate the fixed six-dimensional action without any optional extension."""

    return CARTESIAN_DELTA_ACTION_SCHEMA.validate_vector(values, label="Cartesian delta action")


@dataclass(frozen=True)
class GripperCommand:
    """Validated but still separate optional gripper command."""

    extension_version: int
    schema_id: str
    schema_version: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "extension_version",
            _version(self.extension_version, "extension_version", GripperContractError),
        )
        object.__setattr__(
            self,
            "schema_id",
            _identifier(self.schema_id, "schema_id", GripperContractError),
        )
        object.__setattr__(
            self,
            "schema_version",
            _version(self.schema_version, "schema_version", GripperContractError),
        )
        values = tuple(
            _finite_number(value, f"gripper values[{index}]", GripperContractError)
            for index, value in enumerate(self.values)
        )
        if not values:
            raise GripperContractError("gripper values must not be empty")
        object.__setattr__(self, "values", values)


@runtime_checkable
class GripperAdapter(Protocol):
    """Vendor-neutral gripper interface with mandatory action/state schemas.

    Implementations must not reinterpret a command from a different schema
    version. ``stop`` must be fail-safe and bounded; the adapter owner is
    responsible for invoking it when robot control is disarmed or aborted.
    """

    @property
    def extension_version(self) -> int: ...

    @property
    def action_schema(self) -> VectorSchema: ...

    @property
    def state_schema(self) -> VectorSchema: ...

    def apply(self, command: GripperCommand) -> None:
        """Apply one already validated gripper command."""

    def read_state(self) -> Sequence[float]:
        """Return measured state in the exact declared state-schema order."""

    def stop(self) -> None:
        """Place the gripper in its verified fail-safe state."""


def validate_gripper_adapter(adapter: GripperAdapter) -> tuple[VectorSchema, VectorSchema]:
    """Fail closed unless an adapter declares compatible, unambiguous schemas."""

    extension_version = _version(
        getattr(adapter, "extension_version", None),
        "adapter extension_version",
        GripperContractError,
    )
    if extension_version != GRIPPER_EXTENSION_VERSION:
        raise GripperContractError(
            f"unsupported gripper extension version {extension_version!r}; "
            f"expected {GRIPPER_EXTENSION_VERSION}"
        )

    action_schema = getattr(adapter, "action_schema", None)
    state_schema = getattr(adapter, "state_schema", None)
    if not isinstance(action_schema, VectorSchema):
        raise GripperContractError("gripper adapter must declare a VectorSchema action_schema")
    if not isinstance(state_schema, VectorSchema):
        raise GripperContractError("gripper adapter must declare a VectorSchema state_schema")
    if action_schema.schema_id == CARTESIAN_DELTA_ACTION_SCHEMA.schema_id:
        raise GripperContractError("gripper action schema must be distinct from the 6D action schema")
    if action_schema.schema_id == state_schema.schema_id:
        raise GripperContractError("gripper action and state schemas must use distinct schema_id values")
    for method_name in ("apply", "read_state", "stop"):
        if not callable(getattr(adapter, method_name, None)):
            raise GripperContractError(f"gripper adapter must provide callable {method_name}()")
    return action_schema, state_schema


def _decode_json_object(payload: str | bytes | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GripperContractError("gripper payload must be UTF-8") from exc
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise GripperContractError(f"invalid gripper JSON: {exc.msg}") from exc
    elif isinstance(payload, Mapping):
        decoded = dict(payload)
    else:
        raise GripperContractError("gripper payload must be JSON text, UTF-8 bytes, or a mapping")
    if not isinstance(decoded, dict):
        raise GripperContractError("gripper payload root must be a JSON object")
    return decoded


def parse_optional_gripper_command(
    payload: str | bytes | Mapping[str, Any] | None,
    *,
    adapter: GripperAdapter | None,
) -> GripperCommand | None:
    """Parse a separate optional extension; never modify the 6D base action.

    ``None`` means that no gripper command was sent. Any non-null command is
    rejected unless a concrete adapter and its exact action schema are present.
    """

    if payload is None:
        return None
    if adapter is None:
        raise GripperContractError("gripper command received but no gripper adapter is configured")
    action_schema, _ = validate_gripper_adapter(adapter)
    decoded = _decode_json_object(payload)

    expected_keys = {"extension_version", "schema_id", "schema_version", "values"}
    if set(decoded) != expected_keys:
        missing = sorted(expected_keys - set(decoded))
        extra = sorted(set(decoded) - expected_keys)
        raise GripperContractError(
            f"gripper payload keys do not match the versioned envelope; missing={missing}, extra={extra}"
        )

    extension_version = _version(decoded["extension_version"], "extension_version", GripperContractError)
    if extension_version != GRIPPER_EXTENSION_VERSION:
        raise GripperContractError(
            f"unsupported gripper extension version {extension_version}; expected {GRIPPER_EXTENSION_VERSION}"
        )
    schema_id = _identifier(decoded["schema_id"], "schema_id", GripperContractError)
    schema_version = _version(decoded["schema_version"], "schema_version", GripperContractError)
    if (schema_id, schema_version) != (action_schema.schema_id, action_schema.version):
        raise GripperContractError(
            "gripper command schema does not exactly match the configured action schema"
        )

    raw_values = decoded["values"]
    if isinstance(raw_values, (str, bytes, bytearray)) or not isinstance(raw_values, Sequence):
        raise GripperContractError("gripper values must be a numeric sequence")
    values = action_schema.validate_vector(raw_values, label="gripper command")
    return GripperCommand(
        extension_version=extension_version,
        schema_id=schema_id,
        schema_version=schema_version,
        values=values,
    )


def apply_validated_gripper_command(adapter: GripperAdapter, command: GripperCommand) -> None:
    """Validate schema identity again immediately before hardware actuation."""

    action_schema, _ = validate_gripper_adapter(adapter)
    if command.extension_version != adapter.extension_version:
        raise GripperContractError("gripper command extension version changed before actuation")
    if (command.schema_id, command.schema_version) != (
        action_schema.schema_id,
        action_schema.version,
    ):
        raise GripperContractError("gripper command schema changed before actuation")
    action_schema.validate_vector(command.values, label="gripper command")
    adapter.apply(command)


def read_validated_gripper_state(adapter: GripperAdapter) -> tuple[float, ...]:
    """Read measured gripper state and fail if it violates the state schema."""

    _, state_schema = validate_gripper_adapter(adapter)
    return state_schema.validate_vector(adapter.read_state(), label="gripper state")


@dataclass(frozen=True)
class ClockToUnixMapping:
    """Affine mapping from one declared clock domain to Unix seconds.

    ``unix_s = unix_anchor_s + rate * (source_s - source_anchor_s)``.
    ``uncertainty_s`` must include timestamp acquisition and mapping error.
    """

    source_clock_id: str
    source_anchor_s: float
    unix_anchor_s: float
    rate: float = 1.0
    uncertainty_s: float = 0.0
    schema_version: int = CLOCK_MAPPING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_clock_id",
            _identifier(self.source_clock_id, "source_clock_id", VideoClockContractError),
        )
        object.__setattr__(
            self,
            "source_anchor_s",
            _finite_number(
                self.source_anchor_s,
                "source_anchor_s",
                VideoClockContractError,
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "unix_anchor_s",
            _finite_number(
                self.unix_anchor_s,
                "unix_anchor_s",
                VideoClockContractError,
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "rate",
            _finite_number(self.rate, "clock rate", VideoClockContractError, positive=True),
        )
        object.__setattr__(
            self,
            "uncertainty_s",
            _finite_number(
                self.uncertainty_s,
                "uncertainty_s",
                VideoClockContractError,
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "schema_version",
            _version(self.schema_version, "clock mapping schema version", VideoClockContractError),
        )
        if self.schema_version != CLOCK_MAPPING_SCHEMA_VERSION:
            raise VideoClockContractError(f"unsupported clock mapping schema version {self.schema_version}")

    def to_unix_s(self, source_timestamp_s: object) -> float:
        """Map one finite source timestamp to Unix seconds."""

        source = _finite_number(
            source_timestamp_s,
            "source timestamp",
            VideoClockContractError,
            nonnegative=True,
        )
        mapped = self.unix_anchor_s + self.rate * (source - self.source_anchor_s)
        if not math.isfinite(mapped) or mapped < 0:
            raise VideoClockContractError("clock mapping produced an invalid Unix timestamp")
        return mapped

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable mapping metadata."""

        return {
            "schema_version": self.schema_version,
            "source_clock_id": self.source_clock_id,
            "source_anchor_s": self.source_anchor_s,
            "unix_anchor_s": self.unix_anchor_s,
            "rate": self.rate,
            "uncertainty_s": self.uncertainty_s,
        }


@dataclass(frozen=True)
class VideoClockContract:
    """Explicit relation between frame time, episode-event time, and Unix time."""

    frame_clock_id: str
    episode_clock_id: str
    same_clock_as_episode: bool
    frame_to_unix: ClockToUnixMapping
    episode_to_unix: ClockToUnixMapping
    schema_version: int = VIDEO_TIMING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_clock_id",
            _identifier(self.frame_clock_id, "frame_clock_id", VideoClockContractError),
        )
        object.__setattr__(
            self,
            "episode_clock_id",
            _identifier(self.episode_clock_id, "episode_clock_id", VideoClockContractError),
        )
        if not isinstance(self.same_clock_as_episode, bool):
            raise VideoClockContractError("same_clock_as_episode must be a boolean declaration")
        if not isinstance(self.frame_to_unix, ClockToUnixMapping):
            raise VideoClockContractError("frame_to_unix must be a ClockToUnixMapping")
        if not isinstance(self.episode_to_unix, ClockToUnixMapping):
            raise VideoClockContractError("episode_to_unix must be a ClockToUnixMapping")
        if self.frame_to_unix.source_clock_id != self.frame_clock_id:
            raise VideoClockContractError("frame_to_unix source does not match frame_clock_id")
        if self.episode_to_unix.source_clock_id != self.episode_clock_id:
            raise VideoClockContractError("episode_to_unix source does not match episode_clock_id")

        if self.same_clock_as_episode:
            if self.frame_clock_id != self.episode_clock_id:
                raise VideoClockContractError("same_clock_as_episode=True requires identical clock IDs")
            if self.frame_to_unix != self.episode_to_unix:
                raise VideoClockContractError("one declared clock must use one identical Unix mapping")
        elif self.frame_clock_id == self.episode_clock_id:
            raise VideoClockContractError("same_clock_as_episode=False requires distinct clock IDs")

        object.__setattr__(
            self,
            "schema_version",
            _version(self.schema_version, "video timing schema version", VideoClockContractError),
        )
        if self.schema_version != VIDEO_TIMING_SCHEMA_VERSION:
            raise VideoClockContractError(f"unsupported video timing schema version {self.schema_version}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable clock-domain metadata."""

        return {
            "schema_version": self.schema_version,
            "frame_clock_id": self.frame_clock_id,
            "episode_clock_id": self.episode_clock_id,
            "same_clock_as_episode": self.same_clock_as_episode,
            "frame_to_unix": self.frame_to_unix.to_dict(),
            "episode_to_unix": self.episode_to_unix.to_dict(),
        }


@dataclass(frozen=True)
class FrameTimestamp:
    """Capture timestamp for one frame actually accepted by the recorder."""

    frame_index: int
    timestamp_s: float
    clock_id: str

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise VideoClockContractError("frame_index must be an integer")
        if self.frame_index < 0:
            raise VideoClockContractError("frame_index must be non-negative")
        object.__setattr__(
            self,
            "timestamp_s",
            _finite_number(
                self.timestamp_s,
                "frame timestamp",
                VideoClockContractError,
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "clock_id",
            _identifier(self.clock_id, "frame clock_id", VideoClockContractError),
        )


def validate_frame_timestamps(
    timestamps: Sequence[FrameTimestamp],
    clock_contract: VideoClockContract,
) -> tuple[FrameTimestamp, ...]:
    """Require contiguous indices and strictly increasing finite capture times."""

    if isinstance(timestamps, (str, bytes, bytearray)) or not isinstance(timestamps, Sequence):
        raise VideoClockContractError("frame timestamps must be a sequence")
    if not isinstance(clock_contract, VideoClockContract):
        raise VideoClockContractError("clock_contract must be a VideoClockContract")

    validated = tuple(timestamps)
    if not validated:
        raise VideoClockContractError("a completed video episode must contain at least one frame")

    previous_source = -math.inf
    previous_unix = -math.inf
    for expected_index, frame in enumerate(validated):
        if not isinstance(frame, FrameTimestamp):
            raise VideoClockContractError("every frame timestamp must be a FrameTimestamp")
        if frame.frame_index != expected_index:
            raise VideoClockContractError("frame indices must be contiguous, ordered, and start at zero")
        if frame.clock_id != clock_contract.frame_clock_id:
            raise VideoClockContractError(
                f"frame {frame.frame_index} uses undeclared clock {frame.clock_id!r}"
            )
        if frame.timestamp_s <= previous_source:
            raise VideoClockContractError("frame timestamps must be strictly increasing")
        unix_timestamp = clock_contract.frame_to_unix.to_unix_s(frame.timestamp_s)
        if unix_timestamp <= previous_unix:
            raise VideoClockContractError("frame timestamps mapped to Unix time must be strictly increasing")
        previous_source = frame.timestamp_s
        previous_unix = unix_timestamp
    return validated


@dataclass(frozen=True)
class VideoEpisodeTiming:
    """Validated timing result returned by a concrete external video recorder.

    This result names a video artifact but intentionally does not prove that the
    file exists or contains WebRTC frames. Content/decode checks belong in the
    experiment preflight after the concrete recorder finishes.
    """

    episode_id: str
    video_path: str
    episode_start_timestamp_s: float
    display: str
    clock_contract: VideoClockContract
    frame_timestamps: tuple[FrameTimestamp, ...]
    schema_version: int = VIDEO_TIMING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "episode_id",
            _identifier(self.episode_id, "episode_id", VideoClockContractError),
        )
        if not isinstance(self.video_path, str) or not self.video_path.strip():
            raise VideoClockContractError("video_path must be a non-empty string")
        object.__setattr__(self, "video_path", self.video_path.strip())
        object.__setattr__(
            self,
            "episode_start_timestamp_s",
            _finite_number(
                self.episode_start_timestamp_s,
                "episode_start_timestamp_s",
                VideoClockContractError,
                nonnegative=True,
            ),
        )
        display = _identifier(self.display, "display", VideoClockContractError)
        if display not in {"mobile", "pc"}:
            raise VideoClockContractError("display must be 'mobile' or 'pc'")
        object.__setattr__(self, "display", display)
        if not isinstance(self.clock_contract, VideoClockContract):
            raise VideoClockContractError("clock_contract must be a VideoClockContract")
        object.__setattr__(
            self,
            "frame_timestamps",
            validate_frame_timestamps(tuple(self.frame_timestamps), self.clock_contract),
        )
        episode_start_unix_s = self.clock_contract.episode_to_unix.to_unix_s(self.episode_start_timestamp_s)
        first_frame_unix_s = self.clock_contract.frame_to_unix.to_unix_s(self.frame_timestamps[0].timestamp_s)
        if episode_start_unix_s > first_frame_unix_s:
            raise VideoClockContractError(
                "episode start must map to a time no later than the first committed frame"
            )
        object.__setattr__(
            self,
            "schema_version",
            _version(self.schema_version, "video timing schema version", VideoClockContractError),
        )
        if self.schema_version != VIDEO_TIMING_SCHEMA_VERSION:
            raise VideoClockContractError(f"unsupported video timing schema version {self.schema_version}")


@runtime_checkable
class VideoEpisodeRecorder(Protocol):
    """Contract for a future recorder connected to the real decoded-frame path.

    The implementation must call ``record_frame`` with a capture timestamp from
    the declared frame clock only after that frame is accepted for recording. It
    must not synthesize timestamps from nominal FPS. ``finish_episode`` must flush
    the encoder before returning its timing result. The concrete acquisition
    process must then atomically publish ``video.mp4``,
    ``FORMAL_FRAME_TIMESTAMPS_FILENAME`` and ``video_meta.json``. Only that
    concrete process may mark the metadata with ``FORMAL_VIDEO_ARTIFACT_TYPE``,
    ``FORMAL_VIDEO_WRITER_CAPABILITY`` and
    ``video_capture_verified_by_this_writer=true``; the metadata-only reference
    writer below intentionally cannot do so.
    """

    @property
    def clock_contract(self) -> VideoClockContract: ...

    def start_episode(self, episode_id: str, output_video_path: Path) -> None:
        """Start one externally implemented video recording."""

    def record_frame(self, frame: object, timestamp: FrameTimestamp) -> None:
        """Commit one implementation-specific decoded frame and its timestamp."""

    def finish_episode(self) -> VideoEpisodeTiming:
        """Flush video and return validated frame timing metadata."""

    def abort_episode(self, reason: str) -> None:
        """Abort recording and leave no episode eligible for conversion."""


def _write_csv(path: Path, timing: VideoEpisodeTiming) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame_index", "frame_timestamp_s", "frame_clock_id", "unix_timestamp_s"])
        for frame in timing.frame_timestamps:
            writer.writerow(
                [
                    frame.frame_index,
                    format(frame.timestamp_s, ".17g"),
                    frame.clock_id,
                    format(timing.clock_contract.frame_to_unix.to_unix_s(frame.timestamp_s), ".17g"),
                ]
            )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_video_timing_metadata(
    timing: VideoEpisodeTiming,
    *,
    metadata_path: str | Path,
    timestamps_csv_path: str | Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write validated timing metadata without recording or verifying any video.

    The JSON explicitly declares this writer's capability as
    ``timing_metadata_only``. A concrete ``VideoEpisodeRecorder`` and later video
    decode validation are still mandatory.
    """

    if not isinstance(timing, VideoEpisodeTiming):
        raise VideoClockContractError("timing must be a VideoEpisodeTiming")
    metadata = Path(metadata_path)
    timestamps = Path(timestamps_csv_path)
    if metadata.resolve() == timestamps.resolve():
        raise VideoClockContractError("metadata JSON and timestamp CSV paths must differ")
    if not isinstance(overwrite, bool):
        raise VideoClockContractError("overwrite must be a boolean")
    existing = [path for path in (metadata, timestamps) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing timing output: {existing[0]}")

    metadata.parent.mkdir(parents=True, exist_ok=True)
    timestamps.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": timing.schema_version,
        "artifact_type": "video_episode_timing",
        "writer_capability": "timing_metadata_only",
        "video_capture_verified_by_this_writer": False,
        "episode_id": timing.episode_id,
        "video_path": timing.video_path,
        "display": timing.display,
        "timestamps_domain": "unix_s",
        "frame_count": len(timing.frame_timestamps),
        "frame_timestamps_csv": str(timestamps),
        "episode_start_ts": timing.clock_contract.episode_to_unix.to_unix_s(timing.episode_start_timestamp_s),
        "first_frame_ts": timing.clock_contract.frame_to_unix.to_unix_s(
            timing.frame_timestamps[0].timestamp_s
        ),
        "frame_timestamps_s": [
            timing.clock_contract.frame_to_unix.to_unix_s(frame.timestamp_s)
            for frame in timing.frame_timestamps
        ],
        "clock_contract": timing.clock_contract.to_dict(),
    }

    temporary_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{timestamps.name}.",
            suffix=".tmp",
            dir=timestamps.parent,
            delete=False,
        ) as temporary:
            temporary_csv = Path(temporary.name)
        temporary_paths.append(temporary_csv)
        _write_csv(temporary_csv, timing)

        with tempfile.NamedTemporaryFile(
            prefix=f".{metadata.name}.",
            suffix=".tmp",
            dir=metadata.parent,
            delete=False,
        ) as temporary:
            temporary_json = Path(temporary.name)
        temporary_paths.append(temporary_json)
        _write_json(temporary_json, payload)

        temporary_csv.replace(timestamps)
        temporary_paths.remove(temporary_csv)
        temporary_json.replace(metadata)
        temporary_paths.remove(temporary_json)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    return metadata, timestamps

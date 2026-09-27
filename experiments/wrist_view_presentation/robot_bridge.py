#!/usr/bin/env python
"""Hardware-neutral robot bridge for the camera-ablation experiment.

The wire protocol and every public bridge API use SI units:

* Cartesian position: metres
* Cartesian orientation delta: radians, ordered yaw/pitch/roll (ZYX)
* Joint position: radians
* Protocol timestamp: Unix time in milliseconds

Nothing in this module discovers or opens real hardware automatically.  A real robot
can only be used by explicitly providing a non-dry-run ``RobotDriver`` *and* setting
``allow_hardware=True`` when constructing ``RobotBridge``.

EXPERIMENTER TODO:
    1. Implement ``RobotDriver`` for the selected robot controller, including a real,
       independently tested emergency stop.
    2. Implement ``KinematicsProvider`` or wrap LeRobot's placo-based
       ``RobotKinematics`` with ``LeRobotKinematicsAdapter``.
    3. Replace every limit in the example JSON with limits verified for the real arm.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

Delta6 = tuple[float, float, float, float, float, float]
FrameName = Literal["base", "tool"]


def _synchronized(method):
    """Serialize bridge state transitions and driver operations."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._state_lock:
            return method(self, *args, **kwargs)

    return wrapped


class ProtocolMessageError(ValueError):
    """The DataChannel message does not conform to the experiment protocol."""


class SafetyViolation(RuntimeError):  # noqa: N818 - public protocol exception
    """A command violates a configured safety or timing boundary."""


class HardwareExecutionDisabled(RuntimeError):  # noqa: N818 - public safety exception
    """A real driver was supplied without the explicit hardware opt-in."""


class BridgeStateError(RuntimeError):
    """The bridge or driver is not in a state that permits the requested operation."""


class KinematicsError(RuntimeError):
    """A kinematics provider returned an invalid pose or joint vector."""


def _config_finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _config_positive_number(value: object, field: str) -> float:
    result = _config_finite_number(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


@dataclass(frozen=True)
class ControlCommand:
    """Validated phone-to-robot control message."""

    timestamp_ms: int
    delta: Delta6

    @property
    def translation_m(self) -> np.ndarray:
        return np.asarray(self.delta[:3], dtype=float)

    @property
    def yaw_pitch_roll_rad(self) -> np.ndarray:
        return np.asarray(self.delta[3:], dtype=float)


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProtocolMessageError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolMessageError(f"{field} must be finite")
    return result


def parse_control_message(message: str | bytes | Mapping[str, Any]) -> ControlCommand:
    """Parse ``{"t": <ms>, "d": [dx,dy,dz,dyaw,dpitch,droll]}`` strictly."""

    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolMessageError("control message must be UTF-8") from exc

    if isinstance(message, str):
        try:
            payload = json.loads(message)
        except json.JSONDecodeError as exc:
            raise ProtocolMessageError(f"invalid JSON: {exc.msg}") from exc
    elif isinstance(message, Mapping):
        payload = dict(message)
    else:
        raise ProtocolMessageError("control message must be JSON text, UTF-8 bytes, or a mapping")

    if not isinstance(payload, dict):
        raise ProtocolMessageError("control message root must be a JSON object")

    expected_keys = {"t", "d"}
    if set(payload) != expected_keys:
        missing = sorted(expected_keys - set(payload))
        extra = sorted(set(payload) - expected_keys)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ProtocolMessageError(
            "control message must contain only 't' and 'd' (" + ", ".join(details) + ")"
        )

    timestamp = payload["t"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise ProtocolMessageError("t must be a non-negative integer timestamp in milliseconds")

    raw_delta = payload["d"]
    if not isinstance(raw_delta, (list, tuple)) or len(raw_delta) != 6:
        raise ProtocolMessageError("d must contain exactly 6 values")
    delta = tuple(_finite_number(value, f"d[{index}]") for index, value in enumerate(raw_delta))
    return ControlCommand(timestamp_ms=timestamp, delta=delta)  # type: ignore[arg-type]


def encode_joint_feedback(joint_positions_rad: Sequence[float]) -> str:
    """Encode robot-to-phone feedback as compact ``{"j":[j1,...,jN]}`` JSON."""

    joints = _as_finite_vector(joint_positions_rad, "joint feedback")
    if joints.size == 0:
        raise ProtocolMessageError("joint feedback must contain at least one joint")
    return json.dumps({"j": joints.tolist()}, separators=(",", ":"), allow_nan=False)


@runtime_checkable
class KinematicsProvider(Protocol):
    """Replaceable FK/IK interface; all joint values are radians.

    EXPERIMENTER TODO: implement these three members using the chosen robot's
    URDF, DH model, vendor SDK, or existing solver.
    """

    @property
    def dof(self) -> int: ...

    def forward(self, joint_positions_rad: Sequence[float]) -> np.ndarray:
        """Return a 4x4 base-to-tool transform."""

    def inverse(self, current_joint_positions_rad: Sequence[float], target_pose: np.ndarray) -> np.ndarray:
        """Return a joint target in radians, using current joints as the IK seed."""


@dataclass(frozen=True)
class JointMotionConstraints:
    """Controller-side limits that every real driver must enforce atomically."""

    max_velocity_rad_s: float
    max_acceleration_rad_s2: float
    deadline_monotonic_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_velocity_rad_s",
            _config_positive_number(self.max_velocity_rad_s, "max_velocity_rad_s"),
        )
        object.__setattr__(
            self,
            "max_acceleration_rad_s2",
            _config_positive_number(
                self.max_acceleration_rad_s2,
                "max_acceleration_rad_s2",
            ),
        )
        if (
            isinstance(self.deadline_monotonic_ns, bool)
            or not isinstance(self.deadline_monotonic_ns, int)
            or self.deadline_monotonic_ns <= 0
        ):
            raise ValueError("deadline_monotonic_ns must be a positive integer")


@runtime_checkable
class RobotDriver(Protocol):
    """Minimal robot-controller interface required by ``RobotBridge``.

    EXPERIMENTER TODO: implement this protocol for the selected robot.  ``stop``
    should perform a controlled hold/stop; ``emergency_stop`` must invoke the
    controller's documented emergency-stop mechanism, not merely disconnect a
    socket. ``send_joint_positions`` must configure the controller's physical
    velocity/acceleration limits before accepting the target and obey its monotonic
    deadline. ``emergency_stop`` must be thread-safe, use an independent control
    path, return promptly, and cancel or invalidate any in-flight send. The bridge
    intentionally calls it concurrently when a vendor send is stuck.
    """

    @property
    def dof(self) -> int: ...

    @property
    def is_connected(self) -> bool: ...

    @property
    def is_dry_run(self) -> bool: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def read_joint_positions(self) -> np.ndarray:
        """Read measured joint positions in radians."""

    def send_joint_positions(
        self,
        target_joint_positions_rad: Sequence[float],
        constraints: JointMotionConstraints,
    ) -> None:
        """Send one target while enforcing every supplied controller-side constraint."""

    def stop(self) -> None:
        """Perform a controlled hold/stop."""

    def emergency_stop(self, reason: str) -> None:
        """Trigger the controller's real emergency stop."""


@runtime_checkable
class RecordingCallback(Protocol):
    """Event sink used to connect the bridge to episode recording.

    ``on_control`` is the audit stream and includes rejected input.
    ``on_command_applied`` is the only source permitted for training actions.
    """

    def on_control(self, command: ControlCommand, received_timestamp_s: float) -> None: ...

    def on_joint_state(self, timestamp_s: float, joint_positions_rad: np.ndarray) -> None: ...

    def on_command_applied(
        self,
        timestamp_s: float,
        command: ControlCommand,
        target_joint_positions_rad: np.ndarray,
    ) -> None: ...

    def on_event(self, timestamp_s: float, event: str, detail: str) -> None: ...


class NullRecordingCallback:
    """Default event sink; intentionally performs no file I/O."""

    def on_control(self, command: ControlCommand, received_timestamp_s: float) -> None:
        del command, received_timestamp_s

    def on_joint_state(self, timestamp_s: float, joint_positions_rad: np.ndarray) -> None:
        del timestamp_s, joint_positions_rad

    def on_command_applied(
        self,
        timestamp_s: float,
        command: ControlCommand,
        target_joint_positions_rad: np.ndarray,
    ) -> None:
        del timestamp_s, command, target_joint_positions_rad

    def on_event(self, timestamp_s: float, event: str, detail: str) -> None:
        del timestamp_s, event, detail


class InMemoryRecordingCallback(NullRecordingCallback):
    """Test/demo recorder. It never writes an experiment episode to disk."""

    def __init__(self) -> None:
        self.controls: list[tuple[float, ControlCommand]] = []
        self.joint_states: list[tuple[float, np.ndarray]] = []
        self.applied_commands: list[tuple[float, ControlCommand, np.ndarray]] = []
        self.events: list[tuple[float, str, str]] = []

    def on_control(self, command: ControlCommand, received_timestamp_s: float) -> None:
        self.controls.append((received_timestamp_s, command))

    def on_joint_state(self, timestamp_s: float, joint_positions_rad: np.ndarray) -> None:
        self.joint_states.append((timestamp_s, np.asarray(joint_positions_rad, dtype=float).copy()))

    def on_command_applied(
        self,
        timestamp_s: float,
        command: ControlCommand,
        target_joint_positions_rad: np.ndarray,
    ) -> None:
        self.applied_commands.append(
            (timestamp_s, command, np.asarray(target_joint_positions_rad, dtype=float).copy())
        )

    def on_event(self, timestamp_s: float, event: str, detail: str) -> None:
        self.events.append((timestamp_s, event, detail))


class CsvEpisodeRecorder(NullRecordingCallback):
    """Fail-fast CSV sink implementing the formal raw-episode file contract.

    It writes every received command to ``phone.csv`` for audit, accepted commands
    only to ``applied_actions.csv`` for training, one measured state row per callback
    to ``robot.csv``, and bridge diagnostics to ``events.csv``. Existing files are
    never appended to or overwritten.
    """

    _ACTION_COLUMNS = ("dx", "dy", "dz", "dyaw", "dpitch", "droll")

    def __init__(self, episode_dir: str | Path, joint_names: Sequence[str]) -> None:
        names = tuple(str(name).strip() for name in joint_names)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("joint_names must be a non-empty sequence of unique names")
        self.episode_dir = Path(episode_dir)
        self.joint_names = names
        self.dof = len(names)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False
        self._last_applied_timestamp_s: float | None = None
        self._last_joint_timestamp_s: float | None = None
        self._streams: dict[str, Any] = {}
        self._writers: dict[str, Any] = {}
        schemas = {
            "phone": ("timestamp", *self._ACTION_COLUMNS),
            "applied_actions": ("timestamp", *self._ACTION_COLUMNS),
            "robot": ("timestamp", *names),
            "events": ("timestamp", "event", "detail"),
            "command_audit": (
                "received_timestamp_s",
                "source_timestamp_ms",
                "status",
            ),
        }
        created: list[Path] = []
        try:
            for key, header in schemas.items():
                path = self.episode_dir / f"{key}.csv"
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite existing recorder file: {path}")
                stream = path.open("x", encoding="utf-8", newline="")
                created.append(path)
                writer = csv.writer(stream)
                writer.writerow(header)
                stream.flush()
                self._streams[key] = stream
                self._writers[key] = writer
        except Exception:
            for stream in self._streams.values():
                stream.close()
            for path in created:
                path.unlink(missing_ok=True)
            raise

    def __enter__(self) -> CsvEpisodeRecorder:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for stream in self._streams.values():
                stream.flush()
                stream.close()
            self._closed = True

    def on_control(self, command: ControlCommand, received_timestamp_s: float) -> None:
        timestamp = self._finite_timestamp(received_timestamp_s)
        with self._lock:
            self._write("phone", [timestamp, *command.delta])
            self._write(
                "command_audit",
                [timestamp, command.timestamp_ms, "received"],
            )

    def on_joint_state(
        self,
        timestamp_s: float,
        joint_positions_rad: np.ndarray,
    ) -> None:
        timestamp = self._finite_timestamp(timestamp_s)
        joints = _as_finite_vector(
            joint_positions_rad,
            "recorded joints",
            self.dof,
        )
        with self._lock:
            self._require_increasing(
                timestamp,
                self._last_joint_timestamp_s,
                "robot.csv",
            )
            self._write("robot", [timestamp, *joints.tolist()])
            self._last_joint_timestamp_s = timestamp

    def on_command_applied(
        self,
        timestamp_s: float,
        command: ControlCommand,
        target_joint_positions_rad: np.ndarray,
    ) -> None:
        del target_joint_positions_rad
        timestamp = self._finite_timestamp(timestamp_s)
        with self._lock:
            self._require_increasing(
                timestamp,
                self._last_applied_timestamp_s,
                "applied_actions.csv",
            )
            self._write("applied_actions", [timestamp, *command.delta])
            self._write(
                "command_audit",
                [timestamp, command.timestamp_ms, "applied"],
            )
            self._last_applied_timestamp_s = timestamp

    def on_event(self, timestamp_s: float, event: str, detail: str) -> None:
        timestamp = self._finite_timestamp(timestamp_s)
        with self._lock:
            self._write("events", [timestamp, str(event), str(detail)])

    def _write(self, key: str, row: Sequence[Any]) -> None:
        if self._closed:
            raise RuntimeError("CSV episode recorder is closed")
        self._writers[key].writerow(row)
        self._streams[key].flush()

    @staticmethod
    def _finite_timestamp(timestamp_s: float) -> float:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("recording timestamp must be finite")
        return timestamp

    @staticmethod
    def _require_increasing(
        timestamp_s: float,
        previous_timestamp_s: float | None,
        filename: str,
    ) -> None:
        if previous_timestamp_s is not None and timestamp_s <= previous_timestamp_s:
            raise ValueError(
                f"{filename} timestamp {timestamp_s} is not strictly later than {previous_timestamp_s}"
            )


def _as_finite_vector(values: Sequence[float], name: str, expected_size: int | None = None) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise KinematicsError(f"{name} must be numeric") from exc
    if vector.ndim != 1:
        raise KinematicsError(f"{name} must be a one-dimensional vector")
    if expected_size is not None and vector.size != expected_size:
        raise KinematicsError(f"{name} must contain {expected_size} values, got {vector.size}")
    if not np.all(np.isfinite(vector)):
        raise KinematicsError(f"{name} must contain only finite values")
    return vector


def _as_pose(transform: np.ndarray, name: str) -> np.ndarray:
    pose = np.asarray(transform, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise KinematicsError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise KinematicsError(f"{name} has an invalid homogeneous bottom row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise KinematicsError(f"{name} rotation must be orthonormal with determinant +1")
    return pose


def _pose_residuals(target_pose: np.ndarray, achieved_pose: np.ndarray) -> tuple[float, float]:
    target = _as_pose(target_pose, "IK target")
    achieved = _as_pose(achieved_pose, "IK FK-backcheck result")
    position_error_m = float(np.linalg.norm(target[:3, 3] - achieved[:3, 3]))
    relative_rotation = target[:3, :3].T @ achieved[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    orientation_error_rad = math.acos(cosine)
    return position_error_m, orientation_error_rad


def _require_pose_residual(
    target_pose: np.ndarray,
    achieved_pose: np.ndarray,
    *,
    max_position_error_m: float,
    max_orientation_error_rad: float,
    exception_type: type[Exception],
) -> None:
    position_error_m, orientation_error_rad = _pose_residuals(target_pose, achieved_pose)
    if position_error_m > max_position_error_m or orientation_error_rad > max_orientation_error_rad:
        raise exception_type(
            "IK FK-backcheck residual exceeds tolerance: "
            f"position={position_error_m:.9f} m (max {max_position_error_m:.9f}), "
            f"orientation={orientation_error_rad:.9f} rad "
            f"(max {max_orientation_error_rad:.9f})"
        )


def _rotation_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _rotation_to_zyx(rotation: np.ndarray) -> np.ndarray:
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) > 1e-8:
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
    else:
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
        roll = 0.0
    return np.array([yaw, pitch, roll], dtype=float)


def compose_delta_pose(
    current_pose: np.ndarray,
    command: ControlCommand,
    *,
    translation_frame: FrameName = "base",
    rotation_frame: FrameName = "tool",
) -> np.ndarray:
    """Compose the six-dimensional protocol delta with a measured tool pose."""

    pose = _as_pose(current_pose, "current pose").copy()
    delta_position = command.translation_m
    if translation_frame == "tool":
        delta_position = pose[:3, :3] @ delta_position
    elif translation_frame != "base":
        raise ValueError(f"unknown translation frame: {translation_frame}")
    pose[:3, 3] += delta_position

    yaw, pitch, roll = command.yaw_pitch_roll_rad
    delta_rotation = _rotation_zyx(float(yaw), float(pitch), float(roll))
    if rotation_frame == "tool":
        pose[:3, :3] = pose[:3, :3] @ delta_rotation
    elif rotation_frame == "base":
        pose[:3, :3] = delta_rotation @ pose[:3, :3]
    else:
        raise ValueError(f"unknown rotation frame: {rotation_frame}")
    return pose


@dataclass(frozen=True)
class SafetyConfig:
    """Safety envelope that must be replaced with verified robot-specific values."""

    workspace_min_m: tuple[float, float, float]
    workspace_max_m: tuple[float, float, float]
    joint_min_rad: tuple[float, ...]
    joint_max_rad: tuple[float, ...]
    max_translation_step_m: float
    max_rotation_step_rad: float
    max_joint_step_rad: float
    max_translation_velocity_m_s: float
    max_rotation_velocity_rad_s: float
    max_joint_velocity_rad_s: float
    max_joint_acceleration_rad_s2: float
    max_ik_position_error_m: float
    max_ik_orientation_error_rad: float
    driver_command_timeout_ms: int
    translation_frame: FrameName = "base"
    rotation_frame: FrameName = "tool"
    require_strictly_increasing_t: bool = True
    max_command_age_ms: int | None = None
    max_future_skew_ms: int = 50
    max_receive_interval_ms: int = 250

    def __post_init__(self) -> None:
        workspace_min = tuple(
            _config_finite_number(value, f"workspace_min_m[{index}]")
            for index, value in enumerate(self.workspace_min_m)
        )
        workspace_max = tuple(
            _config_finite_number(value, f"workspace_max_m[{index}]")
            for index, value in enumerate(self.workspace_max_m)
        )
        joint_min = tuple(
            _config_finite_number(value, f"joint_min_rad[{index}]")
            for index, value in enumerate(self.joint_min_rad)
        )
        joint_max = tuple(
            _config_finite_number(value, f"joint_max_rad[{index}]")
            for index, value in enumerate(self.joint_max_rad)
        )
        object.__setattr__(self, "workspace_min_m", workspace_min)
        object.__setattr__(self, "workspace_max_m", workspace_max)
        object.__setattr__(self, "joint_min_rad", joint_min)
        object.__setattr__(self, "joint_max_rad", joint_max)

        if len(workspace_min) != 3 or len(workspace_max) != 3:
            raise ValueError("workspace bounds must each contain 3 values")
        if not joint_min or len(joint_min) != len(joint_max):
            raise ValueError("joint_min_rad and joint_max_rad must have the same non-zero length")
        for name, values in (
            ("workspace_min_m", workspace_min),
            ("workspace_max_m", workspace_max),
            ("joint_min_rad", joint_min),
            ("joint_max_rad", joint_max),
        ):
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain only finite values")
        if any(low >= high for low, high in zip(workspace_min, workspace_max, strict=True)):
            raise ValueError("every workspace minimum must be smaller than its maximum")
        if any(low >= high for low, high in zip(joint_min, joint_max, strict=True)):
            raise ValueError("every joint minimum must be smaller than its maximum")
        for name in (
            "max_translation_step_m",
            "max_rotation_step_rad",
            "max_joint_step_rad",
            "max_translation_velocity_m_s",
            "max_rotation_velocity_rad_s",
            "max_joint_velocity_rad_s",
            "max_joint_acceleration_rad_s2",
            "max_ik_position_error_m",
            "max_ik_orientation_error_rad",
        ):
            value = _config_positive_number(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.translation_frame not in ("base", "tool"):
            raise ValueError("translation_frame must be 'base' or 'tool'")
        if self.rotation_frame not in ("base", "tool"):
            raise ValueError("rotation_frame must be 'base' or 'tool'")
        if not isinstance(self.require_strictly_increasing_t, bool):
            raise ValueError("require_strictly_increasing_t must be a boolean")
        for name in (
            "max_future_skew_ms",
            "max_receive_interval_ms",
            "driver_command_timeout_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.driver_command_timeout_ms > self.max_receive_interval_ms:
            raise ValueError("driver_command_timeout_ms cannot exceed max_receive_interval_ms")
        if self.max_command_age_ms is not None and (
            isinstance(self.max_command_age_ms, bool)
            or not isinstance(self.max_command_age_ms, int)
            or self.max_command_age_ms <= 0
        ):
            raise ValueError("max_command_age_ms must be a positive integer or null")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SafetyConfig:
        return cls(
            workspace_min_m=tuple(payload["workspace_min_m"]),
            workspace_max_m=tuple(payload["workspace_max_m"]),
            joint_min_rad=tuple(payload["joint_min_rad"]),
            joint_max_rad=tuple(payload["joint_max_rad"]),
            max_translation_step_m=payload["max_translation_step_m"],
            max_rotation_step_rad=payload["max_rotation_step_rad"],
            max_joint_step_rad=payload["max_joint_step_rad"],
            max_translation_velocity_m_s=payload["max_translation_velocity_m_s"],
            max_rotation_velocity_rad_s=payload["max_rotation_velocity_rad_s"],
            max_joint_velocity_rad_s=payload["max_joint_velocity_rad_s"],
            max_joint_acceleration_rad_s2=payload["max_joint_acceleration_rad_s2"],
            max_ik_position_error_m=payload["max_ik_position_error_m"],
            max_ik_orientation_error_rad=payload["max_ik_orientation_error_rad"],
            driver_command_timeout_ms=payload["driver_command_timeout_ms"],
            translation_frame=payload.get("translation_frame", "base"),
            rotation_frame=payload.get("rotation_frame", "tool"),
            require_strictly_increasing_t=payload.get("require_strictly_increasing_t", True),
            max_command_age_ms=payload.get("max_command_age_ms"),
            max_future_skew_ms=payload.get("max_future_skew_ms", 50),
            max_receive_interval_ms=payload.get("max_receive_interval_ms", 250),
        )

    @property
    def dof(self) -> int:
        return len(self.joint_min_rad)

    def validate_delta(self, command: ControlCommand) -> None:
        translation_norm = float(np.linalg.norm(command.translation_m))
        if translation_norm > self.max_translation_step_m:
            raise SafetyViolation(
                f"translation step {translation_norm:.6f} m exceeds {self.max_translation_step_m:.6f} m"
            )
        rotation_norm = float(np.linalg.norm(command.yaw_pitch_roll_rad))
        if rotation_norm > self.max_rotation_step_rad:
            raise SafetyViolation(
                f"rotation step {rotation_norm:.6f} rad exceeds {self.max_rotation_step_rad:.6f} rad"
            )

    def validate_delta_velocity(self, command: ControlCommand, delta_time_s: float | None) -> None:
        """Validate velocity using elapsed receiver-monotonic time."""

        if delta_time_s is None:
            if np.linalg.norm(command.translation_m) > 0 or np.linalg.norm(command.yaw_pitch_roll_rad) > 0:
                raise SafetyViolation("a non-zero command has no positive receiver timing interval")
            return
        if not math.isfinite(delta_time_s) or delta_time_s <= 0:
            raise SafetyViolation("receiver timing interval must be finite and positive")
        translation_velocity = float(np.linalg.norm(command.translation_m)) / delta_time_s
        if translation_velocity > self.max_translation_velocity_m_s:
            raise SafetyViolation(
                f"translation velocity {translation_velocity:.6f} m/s exceeds "
                f"{self.max_translation_velocity_m_s:.6f} m/s"
            )
        rotation_velocity = float(np.linalg.norm(command.yaw_pitch_roll_rad)) / delta_time_s
        if rotation_velocity > self.max_rotation_velocity_rad_s:
            raise SafetyViolation(
                f"rotation velocity {rotation_velocity:.6f} rad/s exceeds "
                f"{self.max_rotation_velocity_rad_s:.6f} rad/s"
            )

    def validate_pose(self, pose: np.ndarray) -> None:
        position = _as_pose(pose, "target pose")[:3, 3]
        lower = np.asarray(self.workspace_min_m)
        upper = np.asarray(self.workspace_max_m)
        if np.any(position < lower) or np.any(position > upper):
            raise SafetyViolation(
                f"target position {position.tolist()} is outside workspace "
                f"[{lower.tolist()}, {upper.tolist()}]"
            )

    def validate_ik_solution(
        self,
        target_pose: np.ndarray,
        achieved_pose: np.ndarray,
    ) -> None:
        _require_pose_residual(
            target_pose,
            achieved_pose,
            max_position_error_m=self.max_ik_position_error_m,
            max_orientation_error_rad=self.max_ik_orientation_error_rad,
            exception_type=SafetyViolation,
        )

    def validate_joint_target(
        self,
        current_joint_positions_rad: Sequence[float],
        target_joint_positions_rad: Sequence[float],
        *,
        delta_time_s: float | None = None,
    ) -> None:
        current = _as_finite_vector(current_joint_positions_rad, "current joints", self.dof)
        target = _as_finite_vector(target_joint_positions_rad, "target joints", self.dof)
        lower = np.asarray(self.joint_min_rad)
        upper = np.asarray(self.joint_max_rad)
        if np.any(current < lower) or np.any(current > upper):
            raise SafetyViolation("measured joints are outside configured joint limits")
        if np.any(target < lower) or np.any(target > upper):
            raise SafetyViolation("IK target is outside configured joint limits")
        largest_step = float(np.max(np.abs(target - current)))
        if largest_step > self.max_joint_step_rad:
            raise SafetyViolation(
                f"joint step {largest_step:.6f} rad exceeds {self.max_joint_step_rad:.6f} rad"
            )
        if delta_time_s is None:
            if largest_step > 0:
                raise SafetyViolation("a non-zero joint step has no positive receiver timing interval")
        else:
            if not math.isfinite(delta_time_s) or delta_time_s <= 0:
                raise SafetyViolation("receiver timing interval must be finite and positive")
            largest_velocity = largest_step / delta_time_s
            if largest_velocity > self.max_joint_velocity_rad_s:
                raise SafetyViolation(
                    f"joint velocity {largest_velocity:.6f} rad/s exceeds "
                    f"{self.max_joint_velocity_rad_s:.6f} rad/s"
                )


class LeRobotKinematicsAdapter:
    """Unit adapter for ``src/lerobot/model/kinematics.py::RobotKinematics``.

    That class currently consumes and returns degrees.  Construct this adapter with
    ``native_joint_unit="deg"`` before using it with the radian-based bridge.  The
    delegate is intentionally duck-typed so this experiment module does not import
    optional ``placo`` dependencies.
    """

    def __init__(
        self,
        delegate: Any,
        *,
        dof: int,
        native_joint_unit: Literal["deg", "rad"] = "deg",
        max_position_error_m: float = 1e-4,
        max_orientation_error_rad: float = 1e-3,
    ) -> None:
        if isinstance(dof, bool) or not isinstance(dof, int) or dof <= 0:
            raise ValueError("dof must be positive")
        if native_joint_unit not in ("deg", "rad"):
            raise ValueError("native_joint_unit must be 'deg' or 'rad'")
        if not callable(getattr(delegate, "forward_kinematics", None)) or not callable(
            getattr(delegate, "inverse_kinematics", None)
        ):
            raise TypeError("delegate must provide forward_kinematics and inverse_kinematics")
        self._delegate = delegate
        self._dof = dof
        self._native_joint_unit = native_joint_unit
        self._max_position_error_m = _config_positive_number(
            max_position_error_m,
            "max_position_error_m",
        )
        self._max_orientation_error_rad = _config_positive_number(
            max_orientation_error_rad,
            "max_orientation_error_rad",
        )

    @property
    def dof(self) -> int:
        return self._dof

    def _to_native(self, joints_rad: np.ndarray) -> np.ndarray:
        return np.rad2deg(joints_rad) if self._native_joint_unit == "deg" else joints_rad.copy()

    def _from_native(self, joints: np.ndarray) -> np.ndarray:
        return np.deg2rad(joints) if self._native_joint_unit == "deg" else joints.copy()

    def forward(self, joint_positions_rad: Sequence[float]) -> np.ndarray:
        joints = _as_finite_vector(joint_positions_rad, "FK joints", self.dof)
        pose = self._delegate.forward_kinematics(self._to_native(joints))
        return _as_pose(pose, "FK result").copy()

    def inverse(self, current_joint_positions_rad: Sequence[float], target_pose: np.ndarray) -> np.ndarray:
        current = _as_finite_vector(current_joint_positions_rad, "IK seed", self.dof)
        pose = _as_pose(target_pose, "IK target")
        result = self._delegate.inverse_kinematics(self._to_native(current), pose)
        native_result = _as_finite_vector(result, "IK result", self.dof)
        result_rad = self._from_native(native_result)
        achieved_pose = self.forward(result_rad)
        _require_pose_residual(
            pose,
            achieved_pose,
            max_position_error_m=self._max_position_error_m,
            max_orientation_error_rad=self._max_orientation_error_rad,
            exception_type=KinematicsError,
        )
        return result_rad


class DryRunCartesianKinematics:
    """Fake, non-physical FK/IK mapping for protocol tests only.

    The first three mock "joints" map directly to XYZ and the next three to
    yaw/pitch/roll.  This is deliberately not a robot model.
    """

    def __init__(self, dof: int = 6) -> None:
        if dof < 6:
            raise ValueError("dry-run Cartesian kinematics requires at least 6 values")
        self._dof = dof

    @property
    def dof(self) -> int:
        return self._dof

    def forward(self, joint_positions_rad: Sequence[float]) -> np.ndarray:
        joints = _as_finite_vector(joint_positions_rad, "dry-run FK joints", self.dof)
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = _rotation_zyx(*joints[3:6])
        pose[:3, 3] = joints[:3]
        return pose

    def inverse(self, current_joint_positions_rad: Sequence[float], target_pose: np.ndarray) -> np.ndarray:
        current = _as_finite_vector(current_joint_positions_rad, "dry-run IK seed", self.dof).copy()
        pose = _as_pose(target_pose, "dry-run IK target")
        current[:3] = pose[:3, 3]
        current[3:6] = _rotation_to_zyx(pose[:3, :3])
        return current


class DryRunRobotDriver:
    """In-memory driver that cannot communicate with physical hardware."""

    def __init__(self, initial_joint_positions_rad: Sequence[float]) -> None:
        initial = _as_finite_vector(initial_joint_positions_rad, "initial dry-run joints")
        if initial.size == 0:
            raise ValueError("initial_joint_positions_rad cannot be empty")
        self._joint_positions = initial.copy()
        self._connected = False
        self._emergency_stopped = False
        self.command_history: list[np.ndarray] = []
        self.motion_constraints_history: list[JointMotionConstraints] = []
        self.stop_count = 0
        self.emergency_stop_reasons: list[str] = []

    @property
    def dof(self) -> int:
        return self._joint_positions.size

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_dry_run(self) -> bool:
        return True

    def connect(self) -> None:
        if self._emergency_stopped:
            raise BridgeStateError("dry-run driver is emergency-stopped; explicitly reset it first")
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def read_joint_positions(self) -> np.ndarray:
        self._require_connected()
        return self._joint_positions.copy()

    def send_joint_positions(
        self,
        target_joint_positions_rad: Sequence[float],
        constraints: JointMotionConstraints,
    ) -> None:
        self._require_connected()
        if self._emergency_stopped:
            raise BridgeStateError("dry-run driver is emergency-stopped")
        if not isinstance(constraints, JointMotionConstraints):
            raise TypeError("constraints must be JointMotionConstraints")
        target = _as_finite_vector(target_joint_positions_rad, "dry-run joint target", self.dof)
        self.motion_constraints_history.append(constraints)
        self.command_history.append(target.copy())
        self._joint_positions = target.copy()

    def stop(self) -> None:
        if self._connected:
            self.stop_count += 1

    def emergency_stop(self, reason: str) -> None:
        self._emergency_stopped = True
        self.emergency_stop_reasons.append(reason)

    def reset_emergency_stop(self) -> None:
        """Reset only this in-memory mock; real hardware needs its documented recovery procedure."""

        self._emergency_stopped = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise BridgeStateError("dry-run driver is not connected")


@dataclass(frozen=True)
class BridgeStepResult:
    """One accepted command and the measured feedback sampled after sending it."""

    command: ControlCommand
    current_joint_positions_rad: np.ndarray
    target_joint_positions_rad: np.ndarray
    measured_joint_positions_rad: np.ndarray
    feedback_message: str


class RobotBridge:
    """Convert validated Cartesian deltas into bounded joint commands."""

    def __init__(
        self,
        *,
        driver: RobotDriver,
        kinematics: KinematicsProvider,
        safety: SafetyConfig,
        recorder: RecordingCallback | None = None,
        allow_hardware: bool = False,
        wall_clock_ms: Callable[[], int] | None = None,
        wall_clock_ns: Callable[[], int] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
    ) -> None:
        if driver.dof != kinematics.dof or driver.dof != safety.dof:
            raise ValueError(
                f"DOF mismatch: driver={driver.dof}, kinematics={kinematics.dof}, safety={safety.dof}"
            )
        recorder_dof = getattr(recorder, "dof", None)
        if recorder_dof is not None and recorder_dof != driver.dof:
            raise ValueError(f"DOF mismatch: recorder={recorder_dof}, driver={driver.dof}")
        if not driver.is_dry_run and not allow_hardware:
            raise HardwareExecutionDisabled(
                "real robot driver rejected; pass allow_hardware=True only after limits "
                "and E-stop are verified"
            )
        if not driver.is_dry_run and safety.max_command_age_ms is None:
            raise HardwareExecutionDisabled(
                "real robot driver requires max_command_age_ms after phone/robot clocks are synchronized"
            )
        if not driver.is_dry_run and not safety.require_strictly_increasing_t:
            raise HardwareExecutionDisabled("real robot driver requires require_strictly_increasing_t=true")
        self.driver = driver
        self.kinematics = kinematics
        self.safety = safety
        self.recorder = recorder or NullRecordingCallback()
        if wall_clock_ms is not None and wall_clock_ns is not None:
            raise ValueError("provide at most one of wall_clock_ms and wall_clock_ns")
        if wall_clock_ns is not None:
            self._wall_clock_ns = wall_clock_ns
        elif wall_clock_ms is not None:
            self._wall_clock_ns = lambda: int(wall_clock_ms()) * 1_000_000
        else:
            self._wall_clock_ns = time.time_ns
        self._monotonic_ns = monotonic_ns or time.monotonic_ns
        self._state_lock = threading.RLock()
        self._safety_state_lock = threading.Lock()
        self._emergency_latched = False
        self._armed = False
        self._last_command_timestamp_ms: int | None = None
        self._last_received_monotonic_ns: int | None = None

    @property
    def is_armed(self) -> bool:
        with self._safety_state_lock:
            return self._armed

    @property
    def emergency_latched(self) -> bool:
        with self._safety_state_lock:
            return self._emergency_latched

    def _set_disarmed(self, *, latch_emergency: bool = False) -> None:
        with self._safety_state_lock:
            self._armed = False
            self._last_received_monotonic_ns = None
            if latch_emergency:
                self._emergency_latched = True

    def _require_not_emergency_latched(self) -> None:
        with self._safety_state_lock:
            if self._emergency_latched:
                raise BridgeStateError(
                    "emergency stop is latched; complete the controller's documented "
                    "reset procedure and construct a new RobotBridge"
                )

    def _arm_state(self, armed_at_ns: int) -> None:
        with self._safety_state_lock:
            if self._emergency_latched:
                raise BridgeStateError(
                    "emergency stop occurred while arming; construct a new bridge "
                    "after the documented controller reset"
                )
            self._last_received_monotonic_ns = armed_at_ns
            self._armed = True

    def _armed_snapshot(self) -> bool:
        with self._safety_state_lock:
            return self._armed

    def _trigger_emergency(self, reason: str, event: str) -> None:
        self._set_disarmed(latch_emergency=True)
        try:
            self.driver.emergency_stop(reason)
        except Exception as exc:
            failure = f"{reason}; emergency stop failed: {type(exc).__name__}: {exc}"
            self._record_event_best_effort("emergency_stop_failure", failure)
            raise SafetyViolation(failure) from exc
        self._record_event_best_effort(event, reason)

    @_synchronized
    def connect(self) -> None:
        """Connect the supplied driver. This deliberately does not arm motion."""

        self._require_not_emergency_latched()
        self.driver.connect()
        self._set_disarmed()
        self._last_command_timestamp_ms = None
        self.recorder.on_event(self._timestamp_s(), "connected", "bridge connected; motion remains disarmed")

    @_synchronized
    def arm(self) -> None:
        """Explicitly enable command processing after checking the measured state."""

        self._require_not_emergency_latched()
        if not self.driver.is_connected:
            raise BridgeStateError("connect the driver before arming")
        current = _as_finite_vector(self.driver.read_joint_positions(), "measured joints", self.driver.dof)
        pose = self.kinematics.forward(current)
        self.safety.validate_joint_target(current, current)
        self.safety.validate_pose(pose)
        armed_at_ns = int(self._monotonic_ns())
        try:
            self.recorder.on_event(
                self._timestamp_s(),
                "armed",
                "bridge command processing enabled",
            )
        except Exception:
            self._set_disarmed()
            self._stop_or_emergency("recording callback failed while arming")
            raise
        self._last_command_timestamp_ms = None
        self._arm_state(armed_at_ns)

    @_synchronized
    def disarm(self, reason: str = "operator request") -> None:
        self._set_disarmed()
        if self.driver.is_connected:
            self._stop_or_emergency(reason)
        self._record_event_best_effort("disarmed", reason)

    def emergency_stop(self, reason: str) -> None:
        """Latch and invoke the independent E-stop without waiting for a stuck send."""

        self._trigger_emergency(reason, "emergency_stop")

    @_synchronized
    def disconnect(self) -> None:
        if self.driver.is_connected:
            self.disarm("disconnect")
            self.driver.disconnect()
        self._set_disarmed()
        self._last_command_timestamp_ms = None

    @_synchronized
    def handle_control_message(self, message: str | bytes | Mapping[str, Any]) -> BridgeStepResult:
        """Validate, solve, and send one DataChannel control message."""

        command = parse_control_message(message)
        received_wall_ns = int(self._wall_clock_ns())
        received_ms = received_wall_ns // 1_000_000
        received_monotonic_ns = int(self._monotonic_ns())
        received_s = received_wall_ns / 1_000_000_000.0
        try:
            self.recorder.on_control(command, received_s)
        except Exception:
            if self._armed_snapshot():
                self._set_disarmed()
                self._stop_or_emergency("recording callback failed while auditing control input")
            raise
        if not self.driver.is_connected:
            raise BridgeStateError("driver is not connected")
        if not self._armed_snapshot():
            raise BridgeStateError("bridge is disarmed")

        try:
            delta_time_s = self._validate_command_timing(
                command,
                received_ms,
                received_monotonic_ns,
            )
            self.safety.validate_delta(command)
            self.safety.validate_delta_velocity(command, delta_time_s)
            current = _as_finite_vector(
                self.driver.read_joint_positions(), "measured joints", self.driver.dof
            )
            current_pose = self.kinematics.forward(current)
            target_pose = compose_delta_pose(
                current_pose,
                command,
                translation_frame=self.safety.translation_frame,
                rotation_frame=self.safety.rotation_frame,
            )
            self.safety.validate_pose(target_pose)
            target = _as_finite_vector(
                self.kinematics.inverse(current, target_pose), "IK target", self.driver.dof
            )
            achieved_pose = self.kinematics.forward(target)
            self.safety.validate_ik_solution(target_pose, achieved_pose)
            self.safety.validate_joint_target(current, target, delta_time_s=delta_time_s)
            constraints = JointMotionConstraints(
                max_velocity_rad_s=self.safety.max_joint_velocity_rad_s,
                max_acceleration_rad_s2=self.safety.max_joint_acceleration_rad_s2,
                deadline_monotonic_ns=(
                    received_monotonic_ns + self.safety.driver_command_timeout_ms * 1_000_000
                ),
            )
            if int(self._monotonic_ns()) > constraints.deadline_monotonic_ns:
                raise SafetyViolation("IK/safety processing exceeded the driver command deadline before send")
            self.driver.send_joint_positions(
                target,
                constraints,
            )
            if not self._armed_snapshot():
                reason = (
                    "command completed after an asynchronous watchdog/emergency stop; "
                    "driver must cancel in-flight sends"
                )
                self._trigger_emergency(reason, "emergency_stop")
                raise SafetyViolation(reason)
            if int(self._monotonic_ns()) > constraints.deadline_monotonic_ns:
                raise SafetyViolation("robot driver returned after the command deadline; motion was stopped")
            applied_timestamp_s = self._timestamp_s()
            self.recorder.on_command_applied(
                applied_timestamp_s,
                command,
                target.copy(),
            )
            measured = _as_finite_vector(
                self.driver.read_joint_positions(), "post-command measured joints", self.driver.dof
            )
            measured_timestamp_s = self._timestamp_s()
            # Record exactly one measured state per accepted command. Recording both
            # pre/post states at the same receive time creates duplicate robot.csv timestamps.
            self.recorder.on_joint_state(measured_timestamp_s, measured.copy())
        except SafetyViolation as exc:
            self._set_disarmed()
            self._stop_or_emergency(f"safety stop: {exc}")
            self._record_event_best_effort("safety_stop", str(exc), timestamp_s=received_s)
            raise
        except Exception as exc:
            reason = f"bridge execution failure: {type(exc).__name__}: {exc}"
            self._trigger_emergency(reason, "emergency_stop")
            raise

        self._last_command_timestamp_ms = command.timestamp_ms
        with self._safety_state_lock:
            if self._armed:
                self._last_received_monotonic_ns = received_monotonic_ns
        return BridgeStepResult(
            command=command,
            current_joint_positions_rad=current.copy(),
            target_joint_positions_rad=target.copy(),
            measured_joint_positions_rad=measured.copy(),
            feedback_message=encode_joint_feedback(measured),
        )

    @_synchronized
    def read_joint_feedback(self) -> str:
        """Read, record, and encode one measured joint-feedback message."""

        if not self.driver.is_connected:
            raise BridgeStateError("driver is not connected")
        self.check_watchdog()
        joints = _as_finite_vector(
            self.driver.read_joint_positions(), "measured feedback joints", self.driver.dof
        )
        timestamp_s = self._timestamp_s()
        try:
            self.recorder.on_joint_state(timestamp_s, joints.copy())
        except Exception:
            if self._armed_snapshot():
                self._set_disarmed()
                self._stop_or_emergency("recording callback failed while recording feedback")
            raise
        return encode_joint_feedback(joints)

    def _validate_command_timing(
        self,
        command: ControlCommand,
        received_ms: int,
        received_monotonic_ns: int,
    ) -> float | None:
        delta_time_s = None
        if (
            self.safety.require_strictly_increasing_t
            and self._last_command_timestamp_ms is not None
            and command.timestamp_ms <= self._last_command_timestamp_ms
        ):
            raise SafetyViolation(
                f"out-of-order/replayed command timestamp {command.timestamp_ms}; "
                f"last accepted timestamp was {self._last_command_timestamp_ms}"
            )
        with self._safety_state_lock:
            last_received_monotonic_ns = self._last_received_monotonic_ns
        if last_received_monotonic_ns is not None:
            elapsed_ns = received_monotonic_ns - last_received_monotonic_ns
            if elapsed_ns < 0:
                raise SafetyViolation("receiver monotonic clock moved backwards")
            if elapsed_ns > 0:
                elapsed_ms = elapsed_ns / 1_000_000.0
                if elapsed_ms > self.safety.max_receive_interval_ms:
                    raise SafetyViolation(
                        f"command watchdog interval {elapsed_ms:.3f} ms exceeds "
                        f"{self.safety.max_receive_interval_ms} ms; re-arm is required"
                    )
                delta_time_s = elapsed_ns / 1_000_000_000.0
        clock_difference_ms = received_ms - command.timestamp_ms
        if (
            self.safety.max_command_age_ms is not None
            and clock_difference_ms > self.safety.max_command_age_ms
        ):
            raise SafetyViolation(
                f"command is stale by {clock_difference_ms} ms; maximum age is "
                f"{self.safety.max_command_age_ms} ms"
            )
        if -clock_difference_ms > self.safety.max_future_skew_ms:
            raise SafetyViolation(
                f"command timestamp is {-clock_difference_ms} ms in the future; maximum "
                f"future skew is {self.safety.max_future_skew_ms} ms"
            )
        return delta_time_s

    def check_watchdog(self) -> None:
        """Fail closed when an armed bridge stops receiving commands.

        A real integration must call this method from an independent periodic control
        task; checking only when another packet arrives cannot stop a disconnected sender.
        """

        now_ns = int(self._monotonic_ns())
        with self._safety_state_lock:
            if not self._armed or self._last_received_monotonic_ns is None:
                return
            elapsed_ms = (now_ns - self._last_received_monotonic_ns) / 1_000_000.0
            if elapsed_ms <= self.safety.max_receive_interval_ms and elapsed_ms >= 0:
                return
            self._armed = False
            self._last_received_monotonic_ns = None
            self._emergency_latched = True
        if elapsed_ms < 0:
            reason = "receiver monotonic clock moved backwards"
            self._trigger_emergency(reason, "clock_failure")
            raise SafetyViolation(reason)
        reason = (
            f"command watchdog interval {elapsed_ms:.3f} ms exceeds {self.safety.max_receive_interval_ms} ms"
        )
        # This path intentionally bypasses the bridge operation lock and uses the
        # emergency channel directly, so a stuck vendor send cannot starve the watchdog.
        # A real driver must make emergency_stop thread-safe and cancel in-flight sends.
        self._trigger_emergency(reason, "watchdog_stop")
        raise SafetyViolation(reason)

    def _stop_or_emergency(self, reason: str) -> None:
        try:
            self.driver.stop()
        except Exception as stop_exc:
            emergency_reason = f"{reason}; controlled stop failed: {type(stop_exc).__name__}: {stop_exc}"
            self._trigger_emergency(emergency_reason, "emergency_stop")

    def _record_event_best_effort(
        self,
        event: str,
        detail: str,
        *,
        timestamp_s: float | None = None,
    ) -> None:
        with suppress(Exception):
            self.recorder.on_event(
                self._timestamp_s() if timestamp_s is None else timestamp_s,
                event,
                detail,
            )
            # Motion is already disarmed/stopped at every call site. A failed
            # diagnostic sink must never re-enable or delay the safety action.

    def _timestamp_s(self) -> float:
        return int(self._wall_clock_ns()) / 1_000_000_000.0


@dataclass(frozen=True)
class RobotBridgeConfig:
    """Configuration used only by the safe dry-run builder."""

    mode: str
    allow_hardware: bool
    joint_names: tuple[str, ...]
    initial_joint_positions_rad: tuple[float, ...]
    driver_type: str
    kinematics_provider: str
    safety: SafetyConfig

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RobotBridgeConfig:
        allow_hardware = payload.get("allow_hardware", False)
        if not isinstance(allow_hardware, bool):
            raise ValueError("allow_hardware must be a boolean")
        raw_joint_names = payload.get("joint_names")
        if not isinstance(raw_joint_names, (list, tuple)) or isinstance(raw_joint_names, (str, bytes)):
            raise ValueError("joint_names must be an array")
        raw_initial = payload.get("initial_joint_positions_rad")
        if not isinstance(raw_initial, (list, tuple)) or isinstance(raw_initial, (str, bytes)):
            raise ValueError("initial_joint_positions_rad must be an array")
        driver_payload = payload.get("driver", {})
        kinematics_payload = payload.get("kinematics", {})
        if not isinstance(driver_payload, Mapping) or not isinstance(
            kinematics_payload,
            Mapping,
        ):
            raise ValueError("driver and kinematics must be objects")
        return cls(
            mode=str(payload.get("mode", "dry_run")),
            allow_hardware=allow_hardware,
            joint_names=tuple(str(value) for value in raw_joint_names),
            initial_joint_positions_rad=tuple(
                _config_finite_number(
                    value,
                    f"initial_joint_positions_rad[{index}]",
                )
                for index, value in enumerate(raw_initial)
            ),
            driver_type=str(driver_payload.get("type", "")),
            kinematics_provider=str(kinematics_payload.get("provider", "")),
            safety=SafetyConfig.from_mapping(payload["safety"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> RobotBridgeConfig:
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("robot bridge config root must be a JSON object")
        return cls.from_mapping(payload)

    def validate_dry_run(self) -> None:
        dof = len(self.joint_names)
        if self.mode != "dry_run" or self.allow_hardware:
            raise HardwareExecutionDisabled(
                "dry-run builder requires mode='dry_run' and allow_hardware=false"
            )
        if self.driver_type != "dry_run_mock":
            raise HardwareExecutionDisabled("dry-run builder only accepts driver.type='dry_run_mock'")
        if self.kinematics_provider != "dry_run_cartesian_mock":
            raise HardwareExecutionDisabled(
                "dry-run builder only accepts kinematics.provider='dry_run_cartesian_mock'"
            )
        if dof < 6 or len(set(self.joint_names)) != dof:
            raise ValueError("dry-run joint_names must contain at least 6 unique names")
        if len(self.initial_joint_positions_rad) != dof:
            raise ValueError("initial_joint_positions_rad length must match joint_names")
        if self.safety.dof != dof:
            raise ValueError("safety joint limits length must match joint_names")


def build_dry_run_bridge(
    config: RobotBridgeConfig,
    *,
    recorder: RecordingCallback | None = None,
    wall_clock_ms: Callable[[], int] | None = None,
    wall_clock_ns: Callable[[], int] | None = None,
    monotonic_ns: Callable[[], int] | None = None,
) -> RobotBridge:
    """Build the only bundled bridge: a fully in-memory, non-hardware mock."""

    config.validate_dry_run()
    driver = DryRunRobotDriver(config.initial_joint_positions_rad)
    kinematics = DryRunCartesianKinematics(len(config.joint_names))
    return RobotBridge(
        driver=driver,
        kinematics=kinematics,
        safety=config.safety,
        recorder=recorder,
        allow_hardware=False,
        wall_clock_ms=wall_clock_ms,
        wall_clock_ns=wall_clock_ns,
        monotonic_ns=monotonic_ns,
    )

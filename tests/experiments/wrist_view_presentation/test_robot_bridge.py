import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.wrist_view_presentation import time_sync
from experiments.wrist_view_presentation.robot_bridge import (
    BridgeStateError,
    CsvEpisodeRecorder,
    DryRunCartesianKinematics,
    DryRunRobotDriver,
    HardwareExecutionDisabled,
    InMemoryRecordingCallback,
    KinematicsError,
    LeRobotKinematicsAdapter,
    ProtocolMessageError,
    RobotBridge,
    RobotBridgeConfig,
    SafetyConfig,
    SafetyViolation,
    build_dry_run_bridge,
    encode_joint_feedback,
    parse_control_message,
)

REPO_ROOT = Path(__file__).parents[3]
EXAMPLE_CONFIG = REPO_ROOT / "experiments" / "wrist_view_presentation" / "robot_bridge_config.example.json"


def make_safety(**overrides) -> SafetyConfig:
    values = {
        "workspace_min_m": (-0.5, -0.5, 0.0),
        "workspace_max_m": (0.5, 0.5, 0.8),
        "joint_min_rad": (-0.5, -0.5, 0.0, -3.2, -3.2, -3.2),
        "joint_max_rad": (0.5, 0.5, 0.8, 3.2, 3.2, 3.2),
        "max_translation_step_m": 0.02,
        "max_rotation_step_rad": 0.2,
        "max_joint_step_rad": 0.2,
        "max_translation_velocity_m_s": 0.5,
        "max_rotation_velocity_rad_s": 2.0,
        "max_joint_velocity_rad_s": 2.0,
        "max_joint_acceleration_rad_s2": 5.0,
        "max_ik_position_error_m": 1e-4,
        "max_ik_orientation_error_rad": 1e-3,
        "driver_command_timeout_ms": 100,
        "max_command_age_ms": 100,
    }
    values.update(overrides)
    return SafetyConfig(**values)


def test_parse_control_message_and_encode_feedback() -> None:
    command = parse_control_message(b'{"t":1234,"d":[0.001,-0.002,0,0.01,-0.02,0.03]}')

    assert command.timestamp_ms == 1234
    assert command.delta == pytest.approx((0.001, -0.002, 0.0, 0.01, -0.02, 0.03))
    assert encode_joint_feedback([0, np.pi / 2]) == f'{{"j":[0.0,{np.pi / 2}]}}'


@pytest.mark.parametrize(
    "message",
    [
        "not json",
        "[]",
        '{"t":1,"d":[0,0,0,0,0]}',
        '{"t":1,"d":[0,0,0,0,0,NaN]}',
        '{"t":true,"d":[0,0,0,0,0,0]}',
        '{"t":-1,"d":[0,0,0,0,0,0]}',
        '{"t":1,"d":[0,0,0,0,0,0],"enabled":true}',
    ],
)
def test_parse_control_message_rejects_malformed_or_unsafe_values(message: str) -> None:
    with pytest.raises(ProtocolMessageError):
        parse_control_message(message)


def test_example_config_builds_an_explicitly_armed_dry_run_bridge() -> None:
    config = RobotBridgeConfig.from_json(EXAMPLE_CONFIG)
    clock_ms = [1_000_000]
    recorder = InMemoryRecordingCallback()
    bridge = build_dry_run_bridge(
        config,
        recorder=recorder,
        wall_clock_ms=lambda: clock_ms[0],
        monotonic_ns=lambda: clock_ms[0] * 1_000_000,
    )

    assert bridge.driver.is_dry_run
    assert not bridge.is_armed
    bridge.connect()
    with pytest.raises(BridgeStateError, match="disarmed"):
        bridge.handle_control_message({"t": clock_ms[0], "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]})

    bridge.arm()
    clock_ms[0] += 20
    result = bridge.handle_control_message({"t": clock_ms[0], "d": [0.001, -0.002, 0.003, 0.01, 0.0, 0.0]})

    assert result.current_joint_positions_rad == pytest.approx([0, 0, 0.2, 0, 0, 0])
    assert result.target_joint_positions_rad == pytest.approx([0.001, -0.002, 0.203, 0.01, 0, 0])
    assert json.loads(result.feedback_message)["j"] == pytest.approx(result.target_joint_positions_rad)
    assert len(bridge.driver.command_history) == 1
    assert len(bridge.driver.motion_constraints_history) == 1
    constraints = bridge.driver.motion_constraints_history[0]
    assert constraints.max_velocity_rad_s == bridge.safety.max_joint_velocity_rad_s
    assert constraints.max_acceleration_rad_s2 == bridge.safety.max_joint_acceleration_rad_s2
    assert len(recorder.controls) == 2  # The rejected disarmed message is still an observed input.
    assert len(recorder.applied_commands) == 1
    assert bridge.read_joint_feedback().startswith('{"j":')


def test_safety_step_violation_stops_and_disarms_without_sending() -> None:
    clock_ms = [10_000]
    recorder = InMemoryRecordingCallback()
    config = RobotBridgeConfig.from_json(EXAMPLE_CONFIG)
    bridge = build_dry_run_bridge(
        config,
        recorder=recorder,
        wall_clock_ms=lambda: clock_ms[0],
        monotonic_ns=lambda: clock_ms[0] * 1_000_000,
    )
    bridge.connect()
    bridge.arm()
    clock_ms[0] += 20

    with pytest.raises(SafetyViolation, match="translation step"):
        bridge.handle_control_message({"t": clock_ms[0], "d": [0.011, 0.0, 0.0, 0.0, 0.0, 0.0]})

    assert not bridge.is_armed
    assert bridge.driver.stop_count == 1
    assert bridge.driver.command_history == []
    assert recorder.events[-1][1] == "safety_stop"


def test_velocity_uses_receiver_clock_and_source_timestamp_only_detects_replay() -> None:
    clock_ms = [20_000]
    config = RobotBridgeConfig.from_json(EXAMPLE_CONFIG)
    bridge = build_dry_run_bridge(
        config,
        wall_clock_ms=lambda: clock_ms[0],
        monotonic_ns=lambda: clock_ms[0] * 1_000_000,
    )
    bridge.connect()
    bridge.arm()
    clock_ms[0] += 10
    bridge.handle_control_message({"t": clock_ms[0], "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]})

    clock_ms[0] += 1
    with pytest.raises(SafetyViolation, match="translation velocity"):
        bridge.handle_control_message({"t": clock_ms[0], "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]})

    bridge.arm()
    clock_ms[0] += 1
    bridge.handle_control_message({"t": clock_ms[0], "d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
    with pytest.raises(SafetyViolation, match="out-of-order/replayed"):
        bridge.handle_control_message({"t": clock_ms[0], "d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})


def test_stale_timestamp_stops_bridge() -> None:
    clock_ms = [50_000]
    config = RobotBridgeConfig.from_json(EXAMPLE_CONFIG)
    bridge = build_dry_run_bridge(
        config,
        wall_clock_ms=lambda: clock_ms[0],
        monotonic_ns=lambda: clock_ms[0] * 1_000_000,
    )
    bridge.connect()
    bridge.arm()
    clock_ms[0] += 10

    with pytest.raises(SafetyViolation, match="stale"):
        bridge.handle_control_message({"t": clock_ms[0] - 1001, "d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})

    assert not bridge.is_armed


def test_future_source_timestamp_cannot_bypass_receiver_velocity_limit() -> None:
    clock_ms = [60_000]
    bridge = build_dry_run_bridge(
        RobotBridgeConfig.from_json(EXAMPLE_CONFIG),
        wall_clock_ms=lambda: clock_ms[0],
        monotonic_ns=lambda: clock_ms[0] * 1_000_000,
    )
    bridge.connect()
    bridge.arm()
    clock_ms[0] += 20
    bridge.handle_control_message({"t": clock_ms[0], "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]})

    clock_ms[0] += 1
    with pytest.raises(SafetyViolation, match="translation velocity"):
        bridge.handle_control_message(
            {
                "t": clock_ms[0] + 49,
                "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )


def test_first_nonzero_command_requires_positive_receiver_interval() -> None:
    clock_ms = [65_000]
    bridge = build_dry_run_bridge(
        RobotBridgeConfig.from_json(EXAMPLE_CONFIG),
        wall_clock_ms=lambda: clock_ms[0],
        monotonic_ns=lambda: clock_ms[0] * 1_000_000,
    )
    bridge.connect()
    bridge.arm()

    with pytest.raises(SafetyViolation, match="no positive receiver timing interval"):
        bridge.handle_control_message({"t": clock_ms[0], "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]})


def test_command_watchdog_stops_and_disarms() -> None:
    clock_ms = [66_000]
    bridge = build_dry_run_bridge(
        RobotBridgeConfig.from_json(EXAMPLE_CONFIG),
        wall_clock_ms=lambda: clock_ms[0],
        monotonic_ns=lambda: clock_ms[0] * 1_000_000,
    )
    bridge.connect()
    bridge.arm()
    clock_ms[0] += bridge.safety.max_receive_interval_ms + 1

    with pytest.raises(SafetyViolation, match="watchdog"):
        bridge.check_watchdog()

    assert not bridge.is_armed
    assert bridge.driver.emergency_stop_reasons


def test_receiver_monotonic_clock_rollback_fails_closed() -> None:
    wall_ms = [66_500]
    monotonic_ns = [5_000_000_000]
    bridge = build_dry_run_bridge(
        RobotBridgeConfig.from_json(EXAMPLE_CONFIG),
        wall_clock_ms=lambda: wall_ms[0],
        monotonic_ns=lambda: monotonic_ns[0],
    )
    bridge.connect()
    bridge.arm()
    monotonic_ns[0] -= 1

    with pytest.raises(SafetyViolation, match="moved backwards"):
        bridge.check_watchdog()

    assert not bridge.is_armed
    assert bridge.driver.emergency_stop_reasons


def test_only_applied_commands_and_one_measured_state_enter_training_stream() -> None:
    clock_ms = [67_000]
    recorder = InMemoryRecordingCallback()
    bridge = build_dry_run_bridge(
        RobotBridgeConfig.from_json(EXAMPLE_CONFIG),
        recorder=recorder,
        wall_clock_ms=lambda: clock_ms[0],
        monotonic_ns=lambda: clock_ms[0] * 1_000_000,
    )
    bridge.connect()
    with pytest.raises(BridgeStateError, match="disarmed"):
        bridge.handle_control_message({"t": clock_ms[0], "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]})
    assert recorder.applied_commands == []
    assert recorder.joint_states == []

    bridge.arm()
    clock_ms[0] += 20
    bridge.handle_control_message({"t": clock_ms[0], "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]})

    assert len(recorder.controls) == 2
    assert len(recorder.applied_commands) == 1
    assert len(recorder.joint_states) == 1


def test_csv_recorder_separates_audit_and_training_streams(tmp_path: Path) -> None:
    wall_ns = [70_000_000_123]
    monotonic_ns = [1_000_000_000]
    recorder = CsvEpisodeRecorder(tmp_path / "episode_000", [f"joint_{i}" for i in range(6)])
    bridge = build_dry_run_bridge(
        RobotBridgeConfig.from_json(EXAMPLE_CONFIG),
        recorder=recorder,
        wall_clock_ns=lambda: wall_ns[0],
        monotonic_ns=lambda: monotonic_ns[0],
    )
    bridge.connect()
    source_ms = wall_ns[0] // 1_000_000
    with pytest.raises(BridgeStateError, match="disarmed"):
        bridge.handle_control_message({"t": source_ms, "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]})

    bridge.arm()
    wall_ns[0] += 20_000_111
    monotonic_ns[0] += 20_000_111
    bridge.handle_control_message(
        {
            "t": wall_ns[0] // 1_000_000,
            "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    recorder.close()

    episode_dir = tmp_path / "episode_000"
    phone = pd.read_csv(episode_dir / "phone.csv")
    applied = pd.read_csv(episode_dir / "applied_actions.csv")
    robot = pd.read_csv(episode_dir / "robot.csv")
    audit = pd.read_csv(episode_dir / "command_audit.csv")
    assert len(phone) == 2
    assert len(applied) == 1
    assert len(robot) == 1
    assert audit["status"].tolist() == ["received", "received", "applied"]
    time_sync._read_ts_csv(episode_dir / "applied_actions.csv")
    time_sync._read_ts_csv(episode_dir / "robot.csv")


def test_csv_recorder_rejects_joint_width_mismatch(tmp_path: Path) -> None:
    recorder = CsvEpisodeRecorder(tmp_path / "episode_000", ["j1", "j2"])
    with pytest.raises(KinematicsError, match="2 values"):
        recorder.on_joint_state(1.0, np.array([0.1, 0.2, 0.3]))
    recorder.close()

    mismatched_recorder = CsvEpisodeRecorder(
        tmp_path / "episode_001",
        ["j1", "j2"],
    )
    try:
        with pytest.raises(ValueError, match="recorder=2"):
            RobotBridge(
                driver=DryRunRobotDriver([0, 0, 0.2, 0, 0, 0]),
                kinematics=DryRunCartesianKinematics(),
                safety=make_safety(),
                recorder=mismatched_recorder,
            )
    finally:
        mismatched_recorder.close()


class TimestampAdvancingDriver(DryRunRobotDriver):
    def __init__(self, wall_ns: list[int]) -> None:
        super().__init__([0, 0, 0.2, 0, 0, 0])
        self.wall_ns = wall_ns
        self._sent = False

    def send_joint_positions(self, target_joint_positions_rad, constraints) -> None:
        super().send_joint_positions(target_joint_positions_rad, constraints)
        self.wall_ns[0] += 50_000_000
        self._sent = True

    def read_joint_positions(self) -> np.ndarray:
        result = super().read_joint_positions()
        if self._sent:
            self.wall_ns[0] += 10_000_000
            self._sent = False
        return result


def test_applied_and_measured_callbacks_use_actual_post_operation_times() -> None:
    wall_ns = [80_000_000_000]
    monotonic_ns = [1_000_000_000]
    recorder = InMemoryRecordingCallback()
    bridge = RobotBridge(
        driver=TimestampAdvancingDriver(wall_ns),
        kinematics=DryRunCartesianKinematics(),
        safety=make_safety(),
        recorder=recorder,
        wall_clock_ns=lambda: wall_ns[0],
        monotonic_ns=lambda: monotonic_ns[0],
    )
    bridge.connect()
    bridge.arm()
    wall_ns[0] += 20_000_000
    monotonic_ns[0] += 20_000_000
    source_ms = wall_ns[0] // 1_000_000

    bridge.handle_control_message({"t": source_ms, "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]})

    received_s = recorder.controls[-1][0]
    applied_s = recorder.applied_commands[-1][0]
    measured_s = recorder.joint_states[-1][0]
    assert applied_s == pytest.approx(received_s + 0.05)
    assert measured_s == pytest.approx(received_s + 0.06)


class ReadTimestampAdvancingDriver(DryRunRobotDriver):
    def __init__(self, wall_ns: list[int]) -> None:
        super().__init__([0, 0, 0.2, 0, 0, 0])
        self.wall_ns = wall_ns

    def read_joint_positions(self) -> np.ndarray:
        result = super().read_joint_positions()
        self.wall_ns[0] += 20_000_000
        return result


def test_feedback_timestamp_is_captured_after_slow_driver_read() -> None:
    wall_ns = [81_000_000_000]
    recorder = InMemoryRecordingCallback()
    bridge = RobotBridge(
        driver=ReadTimestampAdvancingDriver(wall_ns),
        kinematics=DryRunCartesianKinematics(),
        safety=make_safety(),
        recorder=recorder,
        wall_clock_ns=lambda: wall_ns[0],
    )
    bridge.connect()
    before_read_s = wall_ns[0] / 1_000_000_000

    bridge.read_joint_feedback()

    assert recorder.joint_states[-1][0] == pytest.approx(before_read_s + 0.02)


class DeadlineExceedingDriver(DryRunRobotDriver):
    def __init__(self, monotonic_ns: list[int]) -> None:
        super().__init__([0, 0, 0.2, 0, 0, 0])
        self.monotonic_ns = monotonic_ns

    def send_joint_positions(self, target_joint_positions_rad, constraints) -> None:
        super().send_joint_positions(target_joint_positions_rad, constraints)
        self.monotonic_ns[0] = constraints.deadline_monotonic_ns + 1


def test_late_driver_return_is_not_recorded_as_an_applied_command() -> None:
    wall_ms = [82_000]
    monotonic_ns = [3_000_000_000]
    recorder = InMemoryRecordingCallback()
    driver = DeadlineExceedingDriver(monotonic_ns)
    bridge = RobotBridge(
        driver=driver,
        kinematics=DryRunCartesianKinematics(),
        safety=make_safety(),
        recorder=recorder,
        wall_clock_ms=lambda: wall_ms[0],
        monotonic_ns=lambda: monotonic_ns[0],
    )
    bridge.connect()
    bridge.arm()
    wall_ms[0] += 20
    monotonic_ns[0] += 20_000_000

    with pytest.raises(SafetyViolation, match="returned after"):
        bridge.handle_control_message({"t": wall_ms[0], "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]})

    assert len(driver.command_history) == 1
    assert recorder.applied_commands == []
    assert driver.stop_count == 1
    assert not bridge.is_armed


class BlockingDriver(DryRunRobotDriver):
    def __init__(self) -> None:
        super().__init__([0, 0, 0.2, 0, 0, 0])
        self.send_started = threading.Event()
        self.release_send = threading.Event()
        self.emergency_called = threading.Event()

    def send_joint_positions(self, target_joint_positions_rad, constraints) -> None:
        self.send_started.set()
        if not self.release_send.wait(timeout=2):
            raise TimeoutError("test did not release blocked send")
        super().send_joint_positions(target_joint_positions_rad, constraints)

    def emergency_stop(self, reason: str) -> None:
        super().emergency_stop(reason)
        self.emergency_called.set()


def test_watchdog_preempts_a_blocked_command_send() -> None:
    wall_ms = [90_000]
    monotonic_ns = [2_000_000_000]
    driver = BlockingDriver()
    bridge = RobotBridge(
        driver=driver,
        kinematics=DryRunCartesianKinematics(),
        safety=make_safety(max_receive_interval_ms=100),
        wall_clock_ms=lambda: wall_ms[0],
        monotonic_ns=lambda: monotonic_ns[0],
    )
    bridge.connect()
    bridge.arm()
    wall_ms[0] += 20
    monotonic_ns[0] += 20_000_000
    handler_errors = []
    watchdog_errors = []
    handler = threading.Thread(
        target=lambda: _capture_exception(
            handler_errors,
            bridge.handle_control_message,
            {"t": wall_ms[0], "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]},
        )
    )
    handler.start()
    assert driver.send_started.wait(timeout=1)

    wall_ms[0] += 101
    monotonic_ns[0] += 101_000_000
    watchdog = threading.Thread(
        target=lambda: _capture_exception(
            watchdog_errors,
            bridge.check_watchdog,
        )
    )
    watchdog.start()
    watchdog.join(timeout=1)

    assert not watchdog.is_alive()
    assert driver.emergency_called.is_set()
    assert handler.is_alive()
    assert driver.command_history == []

    driver.release_send.set()
    handler.join(timeout=2)

    assert not handler.is_alive()
    assert len(handler_errors) == 1
    assert isinstance(handler_errors[0], BridgeStateError)
    assert len(watchdog_errors) == 1
    assert isinstance(watchdog_errors[0], SafetyViolation)
    assert driver.command_history == []
    np.testing.assert_allclose(driver.read_joint_positions(), [0, 0, 0.2, 0, 0, 0])
    assert not bridge.is_armed


def test_operator_emergency_stop_preempts_a_blocked_send_and_latches() -> None:
    wall_ms = [91_000]
    monotonic_ns = [4_000_000_000]
    driver = BlockingDriver()
    bridge = RobotBridge(
        driver=driver,
        kinematics=DryRunCartesianKinematics(),
        safety=make_safety(),
        wall_clock_ms=lambda: wall_ms[0],
        monotonic_ns=lambda: monotonic_ns[0],
    )
    bridge.connect()
    bridge.arm()
    wall_ms[0] += 20
    monotonic_ns[0] += 20_000_000
    handler_errors: list[Exception] = []
    emergency_errors: list[Exception] = []
    handler = threading.Thread(
        target=lambda: _capture_exception(
            handler_errors,
            bridge.handle_control_message,
            {"t": wall_ms[0], "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]},
        )
    )
    handler.start()
    assert driver.send_started.wait(timeout=1)

    emergency = threading.Thread(
        target=lambda: _capture_exception(
            emergency_errors,
            bridge.emergency_stop,
            "operator button",
        )
    )
    emergency.start()
    emergency.join(timeout=1)

    assert not emergency.is_alive()
    assert emergency_errors == []
    assert driver.emergency_called.is_set()
    assert bridge.emergency_latched
    assert not bridge.is_armed
    assert handler.is_alive()
    assert driver.command_history == []

    driver.release_send.set()
    handler.join(timeout=2)

    assert not handler.is_alive()
    assert len(handler_errors) == 1
    assert isinstance(handler_errors[0], BridgeStateError)
    assert driver.command_history == []
    with pytest.raises(BridgeStateError, match="latched"):
        bridge.arm()


def _capture_exception(errors: list[Exception], callback, *args) -> None:
    try:
        callback(*args)
    except Exception as exc:
        errors.append(exc)


class FailingRecorder(InMemoryRecordingCallback):
    def __init__(self, fail_on: str) -> None:
        super().__init__()
        self.fail_on = fail_on

    def on_event(self, timestamp_s: float, event: str, detail: str) -> None:
        if self.fail_on == event:
            raise OSError(f"cannot record {event}")
        super().on_event(timestamp_s, event, detail)

    def on_command_applied(
        self,
        timestamp_s: float,
        command,
        target_joint_positions_rad: np.ndarray,
    ) -> None:
        if self.fail_on == "applied":
            raise OSError("cannot record applied command")
        super().on_command_applied(timestamp_s, command, target_joint_positions_rad)


def test_recording_failure_while_arming_fails_closed() -> None:
    bridge = build_dry_run_bridge(
        RobotBridgeConfig.from_json(EXAMPLE_CONFIG),
        recorder=FailingRecorder("armed"),
    )
    bridge.connect()

    with pytest.raises(OSError, match="cannot record armed"):
        bridge.arm()

    assert not bridge.is_armed
    assert bridge.driver.stop_count == 1


def test_recording_failure_after_motion_emergency_stops_and_disarms() -> None:
    clock_ms = [68_000]
    bridge = build_dry_run_bridge(
        RobotBridgeConfig.from_json(EXAMPLE_CONFIG),
        recorder=FailingRecorder("applied"),
        wall_clock_ms=lambda: clock_ms[0],
        monotonic_ns=lambda: clock_ms[0] * 1_000_000,
    )
    bridge.connect()
    bridge.arm()
    clock_ms[0] += 20

    with pytest.raises(OSError, match="cannot record applied"):
        bridge.handle_control_message({"t": clock_ms[0], "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]})

    assert not bridge.is_armed
    assert bridge.driver.emergency_stop_reasons


class StopFailingDriver(DryRunRobotDriver):
    def stop(self) -> None:
        raise OSError("controlled stop unavailable")


def test_controlled_stop_failure_escalates_to_emergency_stop() -> None:
    clock_ms = [69_000]
    driver = StopFailingDriver([0, 0, 0.2, 0, 0, 0])
    bridge = RobotBridge(
        driver=driver,
        kinematics=DryRunCartesianKinematics(),
        safety=make_safety(),
        wall_clock_ms=lambda: clock_ms[0],
        monotonic_ns=lambda: clock_ms[0] * 1_000_000,
    )
    bridge.connect()
    bridge.arm()
    clock_ms[0] += 20

    with pytest.raises(SafetyViolation, match="translation step"):
        bridge.handle_control_message({"t": clock_ms[0], "d": [0.021, 0.0, 0.0, 0.0, 0.0, 0.0]})

    assert not bridge.is_armed
    assert driver.emergency_stop_reasons
    assert "controlled stop failed" in driver.emergency_stop_reasons[-1]


@pytest.mark.parametrize(
    ("safety_overrides", "second_delta", "expected_error"),
    [
        (
            {
                "joint_max_rad": (0.0015, 0.5, 0.8, 3.2, 3.2, 3.2),
                "max_translation_velocity_m_s": 100.0,
            },
            [0.001, 0, 0, 0, 0, 0],
            "IK target is outside configured joint limits",
        ),
        (
            {
                "max_translation_velocity_m_s": 100.0,
                "max_joint_velocity_rad_s": 0.5,
            },
            [0.001, 0, 0, 0, 0, 0],
            "joint velocity",
        ),
    ],
)
def test_joint_limit_and_joint_velocity_are_enforced(safety_overrides, second_delta, expected_error) -> None:
    clock_ms = [70_000]
    driver = DryRunRobotDriver([0, 0, 0.2, 0, 0, 0])
    bridge = RobotBridge(
        driver=driver,
        kinematics=DryRunCartesianKinematics(),
        safety=make_safety(**safety_overrides),
        wall_clock_ms=lambda: clock_ms[0],
        monotonic_ns=lambda: clock_ms[0] * 1_000_000,
    )
    bridge.connect()
    bridge.arm()
    clock_ms[0] += 10
    bridge.handle_control_message({"t": clock_ms[0], "d": [0.001, 0, 0, 0, 0, 0]})
    clock_ms[0] += 1

    with pytest.raises(SafetyViolation, match=expected_error):
        bridge.handle_control_message({"t": clock_ms[0], "d": second_delta})

    assert not bridge.is_armed


class FakeHardwareDriver(DryRunRobotDriver):
    @property
    def is_dry_run(self) -> bool:
        return False


def test_real_driver_requires_two_explicit_safety_opt_ins() -> None:
    driver = FakeHardwareDriver([0, 0, 0.2, 0, 0, 0])
    kinematics_config = RobotBridgeConfig.from_json(EXAMPLE_CONFIG)
    mock_bridge = build_dry_run_bridge(kinematics_config)

    with pytest.raises(HardwareExecutionDisabled, match="allow_hardware=True"):
        RobotBridge(
            driver=driver,
            kinematics=mock_bridge.kinematics,
            safety=make_safety(),
        )

    with pytest.raises(HardwareExecutionDisabled, match="max_command_age_ms"):
        RobotBridge(
            driver=driver,
            kinematics=mock_bridge.kinematics,
            safety=make_safety(max_command_age_ms=None),
            allow_hardware=True,
        )

    with pytest.raises(HardwareExecutionDisabled, match="strictly_increasing"):
        RobotBridge(
            driver=driver,
            kinematics=mock_bridge.kinematics,
            safety=make_safety(require_strictly_increasing_t=False),
            allow_hardware=True,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"max_translation_step_m": True}, "max_translation_step_m"),
        ({"workspace_min_m": (False, -0.5, 0.0)}, "workspace_min_m"),
        ({"require_strictly_increasing_t": 1}, "require_strictly_increasing_t"),
        ({"driver_command_timeout_ms": True}, "driver_command_timeout_ms"),
    ],
)
def test_safety_config_rejects_boolean_numeric_values(override, message) -> None:
    with pytest.raises(ValueError, match=message):
        make_safety(**override)


class DegreeKinematicsDelegate:
    def __init__(self) -> None:
        self.forward_input = None
        self.inverse_input = None

    def forward_kinematics(self, joints_deg):
        self.forward_input = np.asarray(joints_deg)
        return np.eye(4)

    def inverse_kinematics(self, current_joints_deg, target_pose):
        self.inverse_input = (np.asarray(current_joints_deg), np.asarray(target_pose))
        return np.array([90.0, -45.0])


def test_lerobot_kinematics_adapter_converts_radians_and_degrees_explicitly() -> None:
    delegate = DegreeKinematicsDelegate()
    adapter = LeRobotKinematicsAdapter(delegate, dof=2, native_joint_unit="deg")

    assert np.array_equal(adapter.forward([np.pi / 2, -np.pi / 4]), np.eye(4))
    assert delegate.forward_input == pytest.approx([90.0, -45.0])
    result = adapter.inverse([np.pi / 2, -np.pi / 4], np.eye(4))

    assert delegate.inverse_input[0] == pytest.approx([90.0, -45.0])
    assert result == pytest.approx([np.pi / 2, -np.pi / 4])


def test_lerobot_kinematics_adapter_rejects_large_fk_backcheck_residual() -> None:
    delegate = DegreeKinematicsDelegate()
    adapter = LeRobotKinematicsAdapter(delegate, dof=2, native_joint_unit="deg")
    target = np.eye(4)
    target[0, 3] = 0.01

    with pytest.raises(KinematicsError, match="FK-backcheck residual"):
        adapter.inverse([0.0, 0.0], target)


class NonConvergingKinematics(DryRunCartesianKinematics):
    def inverse(
        self,
        current_joint_positions_rad,
        target_pose: np.ndarray,
    ) -> np.ndarray:
        del target_pose
        return np.asarray(current_joint_positions_rad, dtype=float)


def test_bridge_rejects_nonconverged_ik_before_motion_or_applied_record() -> None:
    clock_ms = [95_000]
    driver = DryRunRobotDriver([0, 0, 0.2, 0, 0, 0])
    recorder = InMemoryRecordingCallback()
    bridge = RobotBridge(
        driver=driver,
        kinematics=NonConvergingKinematics(),
        safety=make_safety(),
        recorder=recorder,
        wall_clock_ms=lambda: clock_ms[0],
        monotonic_ns=lambda: clock_ms[0] * 1_000_000,
    )
    bridge.connect()
    bridge.arm()
    clock_ms[0] += 20

    with pytest.raises(SafetyViolation, match="FK-backcheck residual"):
        bridge.handle_control_message({"t": clock_ms[0], "d": [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]})

    assert driver.command_history == []
    assert recorder.applied_commands == []
    assert not bridge.is_armed

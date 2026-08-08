#!/usr/bin/env python
"""Forward kinematics: joint angles -> end-effector pose [x, y, z, yaw, pitch, roll].

The robot reports joint angles but not the end-effector pose, while the phone sends
end-effector deltas. To build `observation.state = [joint angles + end-effector pose]`
we compute the pose from the joint angles using the robot's DH parameters.

Fill in your robot's DH table in a JSON config (see dh_params.example.json) and pass it
to convert_raw_to_lerobot.py via --dh_config.

Conventions:
  - Standard (Denavit-Hartenberg) or Modified (Craig) DH, set by "convention".
  - Euler angles are ZYX intrinsic: yaw (Z), pitch (Y), roll (X), in radians.
  - ``length_unit`` is ``m`` or ``mm``; every returned position is converted to metres.
  - ``angle_unit`` describes raw joint input, alpha and theta_offset; every returned
    orientation and normalized joint state is in radians.
  - Optional per-joint ``direction`` must be +1 or -1 and maps sensor signs into the
    kinematic convention.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _dh_matrix(a: float, alpha: float, d: float, theta: float, modified: bool) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    if not modified:
        # Standard DH
        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    # Modified DH (Craig)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _rotation_to_ypr(rotation: np.ndarray) -> np.ndarray:
    """ZYX intrinsic Euler angles -> [yaw, pitch, roll] (radians)."""
    pitch = np.arctan2(
        -rotation[2, 0],
        np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2),
    )
    if np.cos(pitch) > 1e-8:
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    else:  # gimbal lock
        yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
        roll = 0.0
    return np.array([yaw, pitch, roll])


class ForwardKinematics:
    """Compute end-effector pose from joint angles using a DH table."""

    def __init__(self, config: dict):
        if not isinstance(config, Mapping):
            raise TypeError(f"DH config must be a mapping, got {type(config).__name__}")

        convention = str(config.get("convention", "standard")).strip().lower()
        if convention not in {"standard", "modified"}:
            raise ValueError(f"Unsupported DH convention {convention!r}; expected 'standard' or 'modified'")
        self.modified = convention == "modified"

        self.angle_unit = str(config.get("angle_unit", "rad")).strip().lower()
        if self.angle_unit not in {"rad", "deg"}:
            raise ValueError(f"Unsupported angle_unit {self.angle_unit!r}; expected 'rad' or 'deg'")
        self.length_unit = str(config.get("length_unit", "m")).strip().lower()
        if self.length_unit not in {"m", "mm"}:
            raise ValueError(f"Unsupported length_unit {self.length_unit!r}; expected 'm' or 'mm'")
        length_scale = 1.0 if self.length_unit == "m" else 0.001

        joints = config.get("joints")
        if not isinstance(joints, list) or not joints:
            raise ValueError("DH config 'joints' must be a non-empty list")
        self.joints: list[dict[str, float]] = []
        for i, joint in enumerate(joints):
            if not isinstance(joint, Mapping):
                raise TypeError(f"joints[{i}] must be a mapping")
            missing = [name for name in ("a", "alpha", "d") if name not in joint]
            if missing:
                raise ValueError(f"joints[{i}] is missing required field(s): {', '.join(missing)}")
            try:
                normalized = {
                    "a": float(joint["a"]),
                    "alpha": float(joint["alpha"]),
                    "d": float(joint["d"]),
                    "theta_offset": float(joint.get("theta_offset", 0.0)),
                    "direction": float(joint.get("direction", 1.0)),
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(f"joints[{i}] contains a non-numeric DH parameter") from exc
            if not np.isfinite(list(normalized.values())).all():
                raise ValueError(f"joints[{i}] contains a non-finite DH parameter")
            if normalized["direction"] not in (-1.0, 1.0):
                raise ValueError(f"joints[{i}].direction must be +1 or -1")
            normalized["a"] *= length_scale
            normalized["d"] *= length_scale
            if self.angle_unit == "deg":
                normalized["alpha"] = float(np.deg2rad(normalized["alpha"]))
                normalized["theta_offset"] = float(np.deg2rad(normalized["theta_offset"]))
            self.joints.append(normalized)

        configured_names = config.get("joint_names")
        self.has_explicit_joint_names = configured_names is not None
        if configured_names is None:
            self.joint_names = tuple(f"joint_{index + 1}" for index in range(len(self.joints)))
        else:
            if (
                not isinstance(configured_names, list)
                or len(configured_names) != len(self.joints)
                or any(not isinstance(name, str) or not name.strip() for name in configured_names)
            ):
                raise ValueError("DH config joint_names must contain one non-empty string per joint")
            names = tuple(name.strip() for name in configured_names)
            if len(set(names)) != len(names):
                raise ValueError("DH config joint_names must be unique")
            self.joint_names = names

        # Optional fixed base/tool transforms (4x4), default identity.
        self.base = self._validate_transform(
            config.get("base", np.eye(4).tolist()),
            "base",
            length_scale,
        )
        self.tool = self._validate_transform(
            config.get("tool", np.eye(4).tolist()),
            "tool",
            length_scale,
        )

    @staticmethod
    def _validate_transform(
        value: object,
        name: str,
        length_scale: float,
    ) -> np.ndarray:
        try:
            transform = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"DH config '{name}' must be a numeric 4x4 transform") from exc
        if transform.shape != (4, 4):
            raise ValueError(f"DH config '{name}' must have shape (4, 4), got {transform.shape}")
        if not np.isfinite(transform).all():
            raise ValueError(f"DH config '{name}' contains non-finite values")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError(f"DH config '{name}' must have homogeneous bottom row [0,0,0,1]")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(rotation),
            1.0,
            atol=1e-6,
        ):
            raise ValueError(f"DH config '{name}' rotation must be orthonormal with determinant +1")
        normalized = transform.copy()
        normalized[:3, 3] *= length_scale
        return normalized

    @classmethod
    def from_json(cls, path: str | Path) -> "ForwardKinematics":
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"DH config file not found: {config_path}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{config_path}: invalid JSON ({exc.msg})") from exc
        return cls(config)

    @property
    def n_joints(self) -> int:
        return len(self.joints)

    def normalize_joint_angles(self, joint_angles: np.ndarray) -> np.ndarray:
        """Convert raw configured-unit sensor angles/signs to kinematic radians."""

        try:
            values = np.asarray(joint_angles, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("joint_angles must be a numeric array") from exc
        if values.ndim not in (1, 2) or values.shape[-1] != self.n_joints:
            raise ValueError(f"Expected joint_angles final dimension {self.n_joints}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("joint_angles contains non-finite values")
        radians = np.deg2rad(values) if self.angle_unit == "deg" else values.copy()
        directions = np.asarray([joint["direction"] for joint in self.joints])
        return radians * directions

    def pose(self, joint_angles: np.ndarray) -> np.ndarray:
        """joint_angles (n_joints,) -> end-effector pose [x, y, z, yaw, pitch, roll]."""
        q = self.normalize_joint_angles(joint_angles)
        if q.ndim != 1:
            raise ValueError(f"Expected joint_angles shape ({self.n_joints},), got {q.shape}")

        transform = self.base.copy()
        for i, j in enumerate(self.joints):
            theta = q[i] + j["theta_offset"]
            alpha = j["alpha"]
            transform = transform @ _dh_matrix(j["a"], alpha, j["d"], theta, self.modified)
        transform = transform @ self.tool

        pos = transform[:3, 3]
        ypr = _rotation_to_ypr(transform[:3, :3])
        return np.concatenate([pos, ypr])


if __name__ == "__main__":
    # Quick self-check with a trivial 2-link planar arm (no config file needed).
    cfg = {
        "convention": "standard",
        "angle_unit": "rad",
        "joints": [
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta_offset": 0.0},
            {"a": 1.0, "alpha": 0.0, "d": 0.0, "theta_offset": 0.0},
        ],
    }
    fk = ForwardKinematics(cfg)
    print("q=[0,0]      ->", np.round(fk.pose([0.0, 0.0]), 4))  # x~2
    print("q=[pi/2,0]   ->", np.round(fk.pose([np.pi / 2, 0.0]), 4))  # y~2

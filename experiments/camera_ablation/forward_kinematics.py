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
  - Position in the same length unit as your DH `a`/`d` (use meters to match the plan).
"""

import json
from pathlib import Path

import numpy as np


def _dh_matrix(a: float, alpha: float, d: float, theta: float, modified: bool) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    if not modified:
        # Standard DH
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,      sa,       ca,      d],
            [0.0,     0.0,      0.0,    1.0],
        ])
    # Modified DH (Craig)
    return np.array([
        [ct,        -st,       0.0,       a],
        [st * ca,  ct * ca,    -sa,  -d * sa],
        [st * sa,  ct * sa,     ca,   d * ca],
        [0.0,          0.0,    0.0,      1.0],
    ])


def _rotation_to_ypr(R: np.ndarray) -> np.ndarray:
    """ZYX intrinsic Euler angles -> [yaw, pitch, roll] (radians)."""
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if np.cos(pitch) > 1e-8:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:  # gimbal lock
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
    return np.array([yaw, pitch, roll])


class ForwardKinematics:
    """Compute end-effector pose from joint angles using a DH table."""

    def __init__(self, config: dict):
        self.joints = config["joints"]              # list of {a, alpha, d, theta_offset}
        self.modified = config.get("convention", "standard").lower() == "modified"
        self.angle_unit = config.get("angle_unit", "rad").lower()
        # Optional fixed base/tool transforms (4x4), default identity.
        self.base = np.array(config.get("base", np.eye(4).tolist()), dtype=float)
        self.tool = np.array(config.get("tool", np.eye(4).tolist()), dtype=float)

    @classmethod
    def from_json(cls, path: str | Path) -> "ForwardKinematics":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @property
    def n_joints(self) -> int:
        return len(self.joints)

    def pose(self, joint_angles: np.ndarray) -> np.ndarray:
        """joint_angles (n_joints,) -> end-effector pose [x, y, z, yaw, pitch, roll]."""
        q = np.asarray(joint_angles, dtype=float)
        if self.angle_unit == "deg":
            q = np.deg2rad(q)
        if q.shape[0] != self.n_joints:
            raise ValueError(f"Expected {self.n_joints} joint angles, got {q.shape[0]}")

        T = self.base.copy()
        for i, j in enumerate(self.joints):
            theta = q[i] + j.get("theta_offset", 0.0)
            offset = np.deg2rad(theta) if self.angle_unit == "deg" else theta
            # theta_offset shares the angle unit of the joint reading
            T = T @ _dh_matrix(j["a"], j["alpha"], j["d"], offset, self.modified)
        T = T @ self.tool

        pos = T[:3, 3]
        ypr = _rotation_to_ypr(T[:3, :3])
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
    print("q=[0,0]      ->", np.round(fk.pose([0.0, 0.0]), 4))            # x~2
    print("q=[pi/2,0]   ->", np.round(fk.pose([np.pi / 2, 0.0]), 4))     # y~2

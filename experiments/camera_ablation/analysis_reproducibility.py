"""Small reproducibility helpers shared by the camera-ablation analyses."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_path(value: str | Path | None) -> dict[str, Any] | None:
    """Return a content fingerprint for a file or directory input.

    Directory fingerprints include every regular file in deterministic relative-path
    order. This deliberately hashes file contents (not mtimes) so a saved sidecar can
    establish the exact local checkpoint/dataset tree used by an analysis.
    """

    if value is None:
        return None
    path = Path(value).expanduser()
    result: dict[str, Any] = {"path": str(path)}
    if not path.exists():
        result["status"] = "not_local_or_missing"
        return result
    resolved = path.resolve()
    result["resolved_path"] = str(resolved)
    if resolved.is_file():
        result.update(
            {
                "kind": "file",
                "sha256": _sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
        return result
    if not resolved.is_dir():
        result["status"] = "unsupported_path_type"
        return result

    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for child in sorted(
        (item for item in resolved.rglob("*") if item.is_file()), key=lambda item: item.as_posix()
    ):
        relative = child.relative_to(resolved).as_posix()
        child_hash = _sha256_file(child)
        size = child.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(child_hash.encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        total_size += size
    result.update(
        {
            "kind": "directory",
            "sha256": digest.hexdigest(),
            "file_count": file_count,
            "size_bytes": total_size,
        }
    )
    return result


def fingerprint_inputs(inputs: Mapping[str, str | Path | None]) -> dict[str, Any]:
    """Fingerprint every explicitly supplied local input path."""

    return {name: fingerprint_path(value) for name, value in inputs.items()}


def runtime_environment() -> dict[str, Any]:
    """Capture runtime/package versions needed to reproduce numerical results."""

    packages: dict[str, str | None] = {}
    for display_name, distribution in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("torch", "torch"),
        ("lerobot", "lerobot"),
    ):
        try:
            packages[display_name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[display_name] = None
    return {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def git_state(start: str | Path | None = None) -> dict[str, Any]:
    """Return repository HEAD and dirty state without mutating the worktree."""

    cwd = Path(start).resolve() if start is not None else Path.cwd().resolve()
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {
            "root": root,
            "head": head,
            "dirty": bool(status.strip()),
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc)}


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a consistently formatted UTF-8 JSON sidecar."""

    output = Path(path)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    output.write_text(f"{serialized}\n", encoding="utf-8")
    return output

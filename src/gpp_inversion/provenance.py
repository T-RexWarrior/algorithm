"""Experiment configuration hashing and manifest persistence."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import torch


def _jsonable(value: Any):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def config_payload(config, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = _jsonable(config)
    if extra:
        payload = {"experiment": payload, "run": _jsonable(extra)}
    return payload


def config_hash(config, *, extra: dict[str, Any] | None = None) -> str:
    canonical = json.dumps(
        config_payload(config, extra=extra),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_experiment_manifest(
    config,
    data_files: Iterable[str | Path],
    output_dir: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = config_hash(config, extra=extra)
    files = []
    for raw_path in sorted(Path(path) for path in data_files):
        stat = raw_path.stat()
        files.append(
            {
                "path": str(raw_path.resolve()),
                "size_bytes": stat.st_size,
                "modified_time_ns": stat.st_mtime_ns,
            }
        )
    manifest = {
        "schema_version": 1,
        "config_hash": digest,
        "status": "running",
        "started_at_utc": _utc_now(),
        "completed_at_utc": None,
        "config": config_payload(config, extra=extra),
        "data_files": files,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        },
        "result": None,
        "artifacts": {},
    }
    path = output_dir / "experiment_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest, path


def finalize_experiment_manifest(
    path: str | Path,
    *,
    result: dict[str, Any],
    artifacts: dict[str, Any],
    status: str = "completed",
) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest["completed_at_utc"] = _utc_now()
    manifest["result"] = _jsonable(result)
    manifest["artifacts"] = _jsonable(artifacts)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest

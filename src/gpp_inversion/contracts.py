"""Versioned training/serving contracts for production GPP models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


CONTRACT_VERSION = "1.0"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    unit: str
    source: str
    temporal_semantics: str
    spatial_sampling: str
    missing_semantics: str


@dataclass(frozen=True)
class FeatureContract:
    forcing: tuple[FeatureSpec, ...]
    state: tuple[FeatureSpec, ...]
    static: tuple[FeatureSpec, ...]
    target: FeatureSpec
    history_hours: int = 96
    timestamp_semantics: str = "interval_end_utc"
    satellite_assignment: str = "observation in (t-1h,t] is assigned to t"
    version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.history_hours != 96:
            raise ValueError("Production FeatureContract requires 96 history hours")
        names = [item.name for item in (*self.forcing, *self.state, *self.static)]
        if len(names) != len(set(names)):
            raise ValueError("FeatureContract contains duplicate feature names")

    @property
    def schema_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["schema_hash"] = self.schema_hash
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


@dataclass
class ModelBatch:
    """Canonical tensor batch shared by training and global inference."""

    hourly_forcing: torch.Tensor
    state_values: torch.Tensor
    state_valid: torch.Tensor
    state_age: torch.Tensor
    time_features: torch.Tensor
    static_xyz: torch.Tensor
    veg_id: torch.Tensor
    daily_context: torch.Tensor | None = None

    def validate(self, *, history_hours: int = 96) -> None:
        batch, steps, _ = self.hourly_forcing.shape
        if steps != history_hours:
            raise ValueError(f"Expected {history_hours} forcing hours, got {steps}")
        expected_prefix = (batch, steps)
        for name, value in (
            ("state_values", self.state_values),
            ("state_valid", self.state_valid),
            ("state_age", self.state_age),
            ("time_features", self.time_features),
        ):
            if tuple(value.shape[:2]) != expected_prefix:
                raise ValueError(f"{name} does not align with forcing history")
        if self.static_xyz.shape != (batch, 3):
            raise ValueError("static_xyz must have shape [batch, 3]")
        if self.veg_id.shape[0] != batch:
            raise ValueError("veg_id does not align with batch")
        if self.state_valid.dtype is not torch.bool:
            raise ValueError("state_valid must be boolean")

    def to(self, device: torch.device | str) -> "ModelBatch":
        values: dict[str, Any] = {}
        for name, value in vars(self).items():
            values[name] = value.to(device) if value is not None else None
        return ModelBatch(**values)


def spherical_xyz(lat_deg, lon_deg) -> np.ndarray:
    """Encode latitude/longitude without a discontinuity at the dateline."""

    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    cos_lat = np.cos(lat)
    return np.column_stack(
        (cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat))
    ).astype(np.float32)


def observation_metadata(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return validity, causal age and observation count for dense histories."""

    valid = np.asarray(mask) > 0.5
    if valid.ndim == 1:
        valid = valid[None, :]
    age = np.full(valid.shape, np.inf, dtype=np.float32)
    last = np.full(valid.shape[0], -1, dtype=np.int64)
    for step in range(valid.shape[1]):
        observed = valid[:, step]
        last[observed] = step
        known = last >= 0
        age[known, step] = step - last[known]
    counts = valid.sum(axis=1).astype(np.int16)
    return valid, age, counts


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_model_package_manifest(
    destination: str | Path,
    *,
    contract: FeatureContract,
    checkpoint: str | Path,
    scaler: str | Path,
    model_kind: str,
    split_hash: str,
    code_commit: str,
) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint)
    scaler = Path(scaler)
    payload = {
        "package_version": "1",
        "contract_version": contract.version,
        "schema_hash": contract.schema_hash,
        "model_kind": model_kind,
        "history_hours": contract.history_hours,
        "split_hash": split_hash,
        "code_commit": code_commit,
        "files": {
            checkpoint.name: sha256_file(checkpoint),
            scaler.name: sha256_file(scaler),
        },
    }
    contract.save(destination / "feature_contract.json")
    path = destination / "model_package.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

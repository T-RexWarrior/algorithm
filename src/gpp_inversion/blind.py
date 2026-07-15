"""Deterministic target-blind station selection for a locked external test set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import sha256_file, spherical_xyz


SELECTION_FORCING = ("SW_IN_F", "TA_F", "VPD_F", "P_F", "SWC_F_MDS_1")
MODEL_FORCING = (
    "SW_IN_F", "SW_IN_POT", "CO2_F_MDS", "P_F", "VPD_F", "TA_F",
    "TS_F_MDS_1", "SWC_F_MDS_1", "WS_F",
)
DESCRIPTOR_COLUMNS = tuple(dict.fromkeys((
    *MODEL_FORCING, "EPIC_Available_Mask", "Lat", "Long", "Veg_ID",
    "GPP_DT_VUT_REF",
)))


@dataclass(frozen=True)
class StationDescriptor:
    file: str
    site: str
    rows: int
    valid_target_rows: int
    lat: float
    lon: float
    veg_id: int
    epic_fraction: float
    forcing_means: tuple[float, ...]
    valid_model_rows: int | None = None


def _site_name(path: Path) -> str:
    stem = path.stem
    return stem[:-7] if stem.endswith("_Merged") else stem


def scan_station(path: Path, *, chunksize: int = 200_000) -> StationDescriptor:
    sums = np.zeros(6, dtype=np.float64)
    counts = np.zeros(6, dtype=np.int64)
    rows = 0
    valid_targets = 0
    valid_model_rows = 0
    lat = lon = np.nan
    veg_id = -1
    for frame in pd.read_csv(
        path, usecols=list(DESCRIPTOR_COLUMNS), chunksize=chunksize, low_memory=False
    ):
        frame = frame.replace([-9999, -999], np.nan)
        rows += len(frame)
        target = pd.to_numeric(frame["GPP_DT_VUT_REF"], errors="coerce")
        valid_targets += int(target.notna().sum())
        model_values = frame[[*MODEL_FORCING, "GPP_DT_VUT_REF"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        valid_model_rows += int(np.isfinite(model_values).all(axis=1).sum())
        values = frame[[*SELECTION_FORCING, "EPIC_Available_Mask"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        sums += np.nansum(values, axis=0)
        counts += finite.sum(axis=0)
        if not np.isfinite(lat):
            lat_values = pd.to_numeric(frame["Lat"], errors="coerce").dropna()
            lon_values = pd.to_numeric(frame["Long"], errors="coerce").dropna()
            veg_values = pd.to_numeric(frame["Veg_ID"], errors="coerce").dropna()
            if not lat_values.empty and not lon_values.empty:
                lat, lon = float(lat_values.iloc[0]), float(lon_values.iloc[0])
            if not veg_values.empty:
                veg_id = int(veg_values.mode().iloc[0])
    means = np.divide(sums, counts, out=np.full(6, np.nan), where=counts > 0)
    return StationDescriptor(
        file=path.name,
        site=_site_name(path),
        rows=rows,
        valid_target_rows=valid_targets,
        lat=lat,
        lon=lon,
        veg_id=veg_id,
        epic_fraction=float(means[-1]),
        forcing_means=tuple(float(value) for value in means[:-1]),
        valid_model_rows=valid_model_rows,
    )


def _previous_sites(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "config" in payload and "experiment" in payload["config"]:
        payload = payload["config"]["experiment"]
    values = []
    for key in ("train_sites", "val_sites", "test_sites"):
        values.extend(payload.get(key, []))
    return {str(value) for value in values}


def select_blind_stations(
    descriptors: list[StationDescriptor],
    *,
    previous_sites: set[str],
    count: int = 60,
    minimum_targets: int = 96,
) -> tuple[list[StationDescriptor], list[StationDescriptor]]:
    candidates = [
        item for item in descriptors
        if item.valid_target_rows >= minimum_targets
        and (item.valid_model_rows is None or item.valid_model_rows >= minimum_targets)
        and item.site not in previous_sites
        and np.isfinite(item.lat)
        and np.isfinite(item.lon)
        and 0 <= item.veg_id < 13
    ]
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible unused stations for {count} blind sites")
    xyz = spherical_xyz(
        [item.lat for item in candidates], [item.lon for item in candidates]
    )
    forcing = np.asarray([item.forcing_means for item in candidates], dtype=np.float64)
    epic = np.asarray([item.epic_fraction for item in candidates], dtype=np.float64)[:, None]
    matrix = np.column_stack([xyz, forcing, epic])
    median = np.nanmedian(matrix, axis=0)
    scale = np.nanmedian(np.abs(matrix - median), axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    normalized = np.nan_to_num((matrix - median) / scale)
    distance = np.sqrt(np.mean(normalized**2, axis=1))
    order_near = np.argsort(distance, kind="stable")
    order_far = order_near[::-1]
    half = count // 2

    def diversified(order, wanted, excluded=frozenset()):
        selected = []
        selected_ids = set(excluded)
        by_class: dict[int, list[int]] = {}
        for index in order:
            by_class.setdefault(candidates[int(index)].veg_id, []).append(int(index))
        classes = sorted(by_class)
        while len(selected) < wanted:
            progressed = False
            for veg in classes:
                while by_class[veg] and by_class[veg][0] in selected_ids:
                    by_class[veg].pop(0)
                if by_class[veg]:
                    index = by_class[veg].pop(0)
                    selected.append(index)
                    selected_ids.add(index)
                    progressed = True
                    if len(selected) == wanted:
                        break
            if not progressed:
                break
        return selected

    representative_indices = diversified(order_near, half)
    ood_indices = diversified(order_far, count - half, set(representative_indices))
    return (
        [candidates[index] for index in representative_indices],
        [candidates[index] for index in ood_indices],
    )


def build_blind_lock(
    data_dir: str | Path,
    output: str | Path,
    *,
    previous_config: str | Path | None = None,
    count: int = 60,
) -> Path:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv"))
    descriptors = [scan_station(path) for path in files]
    previous_path = Path(previous_config) if previous_config else None
    previous_sites = _previous_sites(previous_path)
    representative, ood = select_blind_stations(
        descriptors,
        previous_sites=previous_sites,
        count=count,
    )
    selected = representative + ood
    payload = {
        "protocol_version": 2,
        "selection_program": "gpp_inversion.blind.build_blind_lock/v2",
        "selection_uses_target_magnitude": False,
        "previous_sites_excluded": len(previous_sites),
        "previous_config": str(previous_path.resolve()) if previous_path else None,
        "previous_config_sha256": sha256_file(previous_path) if previous_path else None,
        "eligibility_minimum_valid_targets": 96,
        "eligibility_minimum_valid_model_rows": 96,
        "representative_count": len(representative),
        "ood_count": len(ood),
        "eligible_stations": [
            {
                "file": item.file,
                "site": item.site,
                "veg_id": item.veg_id,
                "valid_target_rows": item.valid_target_rows,
                "valid_model_rows": item.valid_model_rows,
            }
            for item in descriptors
            if item.valid_target_rows >= 96
            and (item.valid_model_rows is None or item.valid_model_rows >= 96)
            and 0 <= item.veg_id < 13
        ],
        "stations": [
            {
                **asdict(item),
                "group": "representative" if item in representative else "ood",
                "sha256": sha256_file(data_dir / item.file),
            }
            for item in selected
        ],
    }
    canonical = json.dumps(payload["stations"], ensure_ascii=False, sort_keys=True)
    payload["split_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output

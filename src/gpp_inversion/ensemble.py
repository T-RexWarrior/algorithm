"""Strict, chunked equal-weight ensembling for saved prediction files."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RegressionAccumulator:
    count: int = 0
    target_sum: float = 0.0
    target_square_sum: float = 0.0
    square_error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    error_sum: float = 0.0

    def update(self, target, prediction) -> None:
        target = np.asarray(target, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        valid = np.isfinite(target) & np.isfinite(prediction)
        target = target[valid]
        prediction = prediction[valid]
        error = prediction - target
        self.count += int(target.size)
        self.target_sum += float(target.sum())
        self.target_square_sum += float(np.dot(target, target))
        self.square_error_sum += float(np.dot(error, error))
        self.absolute_error_sum += float(np.abs(error).sum())
        self.error_sum += float(error.sum())

    def metrics(self) -> dict[str, float | int]:
        if not self.count:
            return {
                "mse": float("nan"), "rmse": float("nan"),
                "mae": float("nan"), "bias": float("nan"),
                "r2": float("nan"), "count": 0,
            }
        mse = self.square_error_sum / self.count
        denominator = (
            self.target_square_sum
            - self.target_sum * self.target_sum / self.count
        )
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": self.absolute_error_sum / self.count,
            "bias": self.error_sum / self.count,
            "r2": (
                1.0 - self.square_error_sum / denominator
                if denominator > 0 else float("nan")
            ),
            "count": self.count,
        }


def _update_groups(groups, labels, target, prediction) -> None:
    labels = np.asarray(labels)
    for label in np.unique(labels):
        mask = labels == label
        groups[str(label)].update(target[mask], prediction[mask])


def _write_group_csv(path: Path, column: str, groups) -> None:
    rows = []
    for label in sorted(groups):
        row = {column: label}
        row.update(groups[label].metrics())
        rows.append(row)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ensemble_prediction_files(
    prediction_files,
    output_dir: str | Path,
    *,
    prefix: str = "val",
    high_target_threshold: float | None = None,
    chunk_size: int = 200_000,
) -> dict:
    paths = [Path(path) for path in prediction_files]
    if len(paths) < 2:
        raise ValueError("At least two prediction files are required")
    output_dir = Path(output_dir)
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    output_prediction = evaluation_dir / f"{prefix}_predictions.csv"
    usecols = ["station", "date", "land_cover_id", "target", "prediction"]
    readers = [
        pd.read_csv(path, usecols=usecols, chunksize=chunk_size)
        for path in paths
    ]
    global_metrics = RegressionAccumulator()
    high_metrics = RegressionAccumulator()
    station_metrics = defaultdict(RegressionAccumulator)
    land_cover_metrics = defaultdict(RegressionAccumulator)
    year_metrics = defaultdict(RegressionAccumulator)
    first_chunk = True

    with output_prediction.open("w", encoding="utf-8-sig", newline="") as handle:
        for chunk_number, chunks in enumerate(zip_longest(*readers), start=1):
            if any(chunk is None for chunk in chunks):
                raise ValueError("Prediction files contain different row counts")
            reference = chunks[0].reset_index(drop=True)
            reference_target = reference["target"].to_numpy(dtype=np.float64)
            predictions = []
            for source_index, chunk in enumerate(chunks):
                chunk = chunk.reset_index(drop=True)
                for column in ("station", "date", "land_cover_id"):
                    if not np.array_equal(
                        reference[column].to_numpy(), chunk[column].to_numpy()
                    ):
                        raise ValueError(
                            f"Prediction alignment mismatch in chunk {chunk_number}, "
                            f"source {source_index}, column {column}"
                        )
                if not np.allclose(
                    reference_target,
                    chunk["target"].to_numpy(dtype=np.float64),
                    rtol=0.0,
                    atol=1e-6,
                ):
                    raise ValueError(
                        f"Target mismatch in chunk {chunk_number}, source {source_index}"
                    )
                predictions.append(chunk["prediction"].to_numpy(dtype=np.float64))
            stacked = np.vstack(predictions)
            prediction = stacked.mean(axis=0)
            prediction_std = stacked.std(axis=0, ddof=0)
            output = reference.copy()
            output["prediction"] = prediction
            output["prediction_std"] = prediction_std
            output["residual"] = prediction - reference_target
            output.to_csv(handle, index=False, header=first_chunk)
            first_chunk = False

            global_metrics.update(reference_target, prediction)
            _update_groups(
                station_metrics,
                reference["station"].to_numpy(),
                reference_target,
                prediction,
            )
            _update_groups(
                land_cover_metrics,
                reference["land_cover_id"].to_numpy(),
                reference_target,
                prediction,
            )
            years = pd.to_datetime(reference["date"], errors="coerce").dt.year
            valid_year = years.notna().to_numpy()
            _update_groups(
                year_metrics,
                years[valid_year].astype(int).to_numpy(),
                reference_target[valid_year],
                prediction[valid_year],
            )
            if high_target_threshold is not None:
                high = reference_target >= high_target_threshold
                high_metrics.update(reference_target[high], prediction[high])

    global_path = evaluation_dir / f"{prefix}_metrics_global.json"
    global_path.write_text(
        json.dumps(global_metrics.metrics(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    station_path = evaluation_dir / f"{prefix}_metrics_by_station.csv"
    land_cover_path = evaluation_dir / f"{prefix}_metrics_by_land_cover.csv"
    year_path = evaluation_dir / f"{prefix}_metrics_by_year.csv"
    _write_group_csv(station_path, "station", station_metrics)
    _write_group_csv(land_cover_path, "land_cover_id", land_cover_metrics)
    _write_group_csv(year_path, "year", year_metrics)
    high_path = None
    if high_target_threshold is not None:
        high_path = evaluation_dir / f"{prefix}_metrics_high_target.json"
        high_path.write_text(
            json.dumps(
                {
                    "threshold": float(high_target_threshold),
                    "metrics": high_metrics.metrics(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "equal_weight_mean",
        "source_count": len(paths),
        "sources": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "modified_time_ns": path.stat().st_mtime_ns,
            }
            for path in paths
        ],
        "high_target_threshold": high_target_threshold,
        "metrics": global_metrics.metrics(),
        "artifacts": {
            "predictions": str(output_prediction),
            "global_metrics": str(global_path),
            "station_metrics": str(station_path),
            "land_cover_metrics": str(land_cover_path),
            "year_metrics": str(year_path),
            "high_target_metrics": str(high_path) if high_path else None,
        },
    }
    manifest_path = output_dir / "ensemble_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest["manifest"] = str(manifest_path)
    return manifest

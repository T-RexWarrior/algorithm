"""Evaluation tables, prediction files and per-station diagnostic plots."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EvaluationConfig
from .engine import EvaluationResult
from .metrics import regression_metrics


@dataclass(frozen=True)
class EvaluationArtifacts:
    predictions: Path | None
    global_metrics: Path
    station_metrics: Path
    land_cover_metrics: Path
    yearly_metrics: Path
    high_target_metrics: Path | None
    plot_directory: Path | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            name: str(value) if value is not None else None
            for name, value in self.__dict__.items()
        }


def evaluation_frame(result: EvaluationResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": result.station_names,
            "date": pd.to_datetime(result.dates, errors="coerce"),
            "land_cover_id": result.land_cover_ids,
            "target": result.targets,
            "prediction": result.predictions,
            "residual": result.predictions - result.targets,
        }
    )


def _group_metrics(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for group, values in frame.groupby(column, dropna=False, sort=True):
        row = {column: group}
        row.update(regression_metrics(values["target"], values["prediction"]))
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "station"


def _plot_station(frame: pd.DataFrame, output_dir: Path, config: EvaluationConfig) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    frame = frame.sort_values("date").dropna(subset=["date"])
    if frame.empty:
        return
    station = str(frame["station"].iloc[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    rolling_target = frame["target"].rolling(
        config.moving_average_window, min_periods=1
    ).mean()
    rolling_prediction = frame["prediction"].rolling(
        config.moving_average_window, min_periods=1
    ).mean()

    plt.figure(figsize=(15, 6))
    plt.plot(frame["date"], frame["target"], alpha=0.2, linewidth=0.5, label="Actual raw")
    plt.plot(frame["date"], frame["prediction"], alpha=0.2, linewidth=0.5, label="Predicted raw")
    plt.plot(frame["date"], rolling_target, linewidth=1.5, label="Actual moving average")
    plt.plot(frame["date"], rolling_prediction, linewidth=1.5, linestyle="--", label="Predicted moving average")
    plt.title(f"{station}: GPP prediction trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "trend_moving_average.png", dpi=300)
    plt.close()

    peak_index = int(np.nanargmax(rolling_target.to_numpy()))
    peak_date = frame["date"].iloc[peak_index]
    half_window = pd.Timedelta(days=config.zoom_days / 2)
    zoom = frame[
        (frame["date"] >= peak_date - half_window)
        & (frame["date"] <= peak_date + half_window)
    ]
    if not zoom.empty:
        plt.figure(figsize=(15, 5))
        plt.plot(zoom["date"], zoom["target"], label="Actual")
        plt.plot(zoom["date"], zoom["prediction"], linestyle="--", label="Predicted")
        plt.title(f"{station}: {config.zoom_days}-day detail")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "zoom.png", dpi=300)
        plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(frame["target"], frame["prediction"], alpha=0.6, s=15)
    low = min(frame["target"].min(), frame["prediction"].min())
    high = max(frame["target"].max(), frame["prediction"].max())
    plt.plot([low, high], [low, high], "r--", label="1:1")
    plt.xlabel("Actual GPP")
    plt.ylabel("Predicted GPP")
    plt.title(f"{station}: actual vs predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "scatter.png", dpi=300)
    plt.close()

    years = frame["date"].dt.year
    if years.notna().any():
        year = int(years.value_counts().idxmax())
        selected = frame[years == year]
        plt.figure(figsize=(15, 5))
        plt.plot(selected["date"], selected["target"], label="Actual")
        plt.plot(selected["date"], selected["prediction"], linestyle="--", label="Predicted")
        plt.title(f"{station}: {year} time series")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"single_year_{year}.png", dpi=300)
        plt.close()


def save_evaluation_artifacts(
    result: EvaluationResult,
    output_dir: str | Path,
    *,
    prefix: str,
    config: EvaluationConfig,
    high_target_threshold: float | None = None,
) -> EvaluationArtifacts:
    output_dir = Path(output_dir)
    report_dir = output_dir / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = evaluation_frame(result)

    predictions_path = report_dir / f"{prefix}_predictions.csv"
    if config.save_predictions:
        frame.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    else:
        predictions_path = None

    global_path = report_dir / f"{prefix}_metrics_global.json"
    global_path.write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    station_path = report_dir / f"{prefix}_metrics_by_station.csv"
    land_cover_path = report_dir / f"{prefix}_metrics_by_land_cover.csv"
    yearly_path = report_dir / f"{prefix}_metrics_by_year.csv"
    _group_metrics(frame, "station").to_csv(station_path, index=False, encoding="utf-8-sig")
    _group_metrics(frame, "land_cover_id").to_csv(land_cover_path, index=False, encoding="utf-8-sig")
    yearly = frame.assign(year=frame["date"].dt.year)
    _group_metrics(yearly, "year").to_csv(yearly_path, index=False, encoding="utf-8-sig")

    high_target_path = None
    if high_target_threshold is not None:
        high_target_path = report_dir / f"{prefix}_metrics_high_target.json"
        high = frame[frame["target"] >= high_target_threshold]
        payload = {
            "threshold": float(high_target_threshold),
            "metrics": regression_metrics(high["target"], high["prediction"]),
        }
        high_target_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    plot_directory = None
    if config.save_plots:
        plot_directory = report_dir / f"{prefix}_station_plots"
        for station, station_frame in frame.groupby("station", sort=True):
            _plot_station(
                station_frame,
                plot_directory / _safe_name(str(station)),
                config,
            )
    return EvaluationArtifacts(
        predictions=predictions_path,
        global_metrics=global_path,
        station_metrics=station_path,
        land_cover_metrics=land_cover_path,
        yearly_metrics=yearly_path,
        high_target_metrics=high_target_path,
        plot_directory=plot_directory,
    )

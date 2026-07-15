"""Paired development/blind promotion gates for candidate GPP models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["station", "date", "land_cover_id", "target"]


def _metrics(frame: pd.DataFrame, prediction: str) -> dict:
    error = frame[prediction].to_numpy() - frame["target"].to_numpy()
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "count": int(len(frame)),
    }


def evaluate_promotion(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    high_target_threshold: float,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict:
    baseline = pd.read_csv(baseline_path)
    candidate = pd.read_csv(candidate_path)
    required = set(KEYS + ["prediction"])
    for name, frame in (("baseline", baseline), ("candidate", candidate)):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} predictions missing columns: {sorted(missing)}")
    merged = baseline[KEYS + ["prediction"]].merge(
        candidate[KEYS + ["prediction"]], on=KEYS, how="inner",
        suffixes=("_baseline", "_candidate"), validate="one_to_one",
    )
    if len(merged) != len(baseline) or len(merged) != len(candidate):
        raise ValueError("Prediction files are not strictly aligned")
    overall_baseline = _metrics(merged, "prediction_baseline")
    overall_candidate = _metrics(merged, "prediction_candidate")
    station_rows = []
    for station, group in merged.groupby("station"):
        first = _metrics(group, "prediction_baseline")
        second = _metrics(group, "prediction_candidate")
        station_rows.append({
            "station": station,
            "baseline_rmse": first["rmse"], "candidate_rmse": second["rmse"],
            "baseline_mae": first["mae"], "candidate_mae": second["mae"],
        })
    stations = pd.DataFrame(station_rows)
    macro_baseline = float(stations["baseline_rmse"].mean())
    macro_candidate = float(stations["candidate_rmse"].mean())
    macro_mae_baseline = float(stations["baseline_mae"].mean())
    macro_mae_candidate = float(stations["candidate_mae"].mean())
    high = merged[merged["target"] >= high_target_threshold]
    high_baseline = _metrics(high, "prediction_baseline") if len(high) else {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "count": 0}
    high_candidate = _metrics(high, "prediction_candidate") if len(high) else dict(high_baseline)
    rng = np.random.default_rng(seed)
    deltas = np.empty(bootstrap_samples, dtype=np.float64)
    station_delta = stations["candidate_rmse"].to_numpy() - stations["baseline_rmse"].to_numpy()
    for index in range(bootstrap_samples):
        sample = rng.integers(0, len(station_delta), len(station_delta))
        deltas[index] = station_delta[sample].mean()
    land_cover_degradation = {}
    for land_cover, group in merged.groupby("land_cover_id"):
        if group["station"].nunique() < 5:
            continue
        first = _metrics(group, "prediction_baseline")["rmse"]
        second = _metrics(group, "prediction_candidate")["rmse"]
        land_cover_degradation[str(int(land_cover))] = (second / first - 1.0) if first else 0.0
    checks = {
        "micro_rmse_improves_0p5pct": overall_candidate["rmse"] <= overall_baseline["rmse"] * 0.995,
        "macro_rmse_improves_1pct": macro_candidate <= macro_baseline * 0.99,
        "micro_mae_noninferior": overall_candidate["mae"] <= overall_baseline["mae"] * 1.005,
        "macro_mae_noninferior": macro_mae_candidate <= macro_mae_baseline * 1.005,
        "high_rmse_noninferior": high_candidate["rmse"] <= high_baseline["rmse"] if len(high) else True,
        "high_abs_bias_improves_10pct": abs(high_candidate["bias"]) <= abs(high_baseline["bias"]) * 0.9 if len(high) else True,
        "station_win_fraction_55pct": float(np.mean(station_delta < 0)) >= 0.55,
        "no_major_land_cover_degrades_3pct": not land_cover_degradation or max(land_cover_degradation.values()) <= 0.03,
        "bootstrap_not_clearly_negative": float(np.quantile(deltas, 0.025)) <= 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "baseline": {"micro": overall_baseline, "macro_rmse": macro_baseline, "macro_mae": macro_mae_baseline, "high": high_baseline},
        "candidate": {"micro": overall_candidate, "macro_rmse": macro_candidate, "macro_mae": macro_mae_candidate, "high": high_candidate},
        "station_win_fraction": float(np.mean(station_delta < 0)),
        "paired_macro_rmse_delta_ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "land_cover_rmse_degradation": land_cover_degradation,
    }


def write_promotion_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

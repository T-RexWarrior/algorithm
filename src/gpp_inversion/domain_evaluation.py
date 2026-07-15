"""Evaluate one trained checkpoint under tower and deployment-style inputs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DomainConfig, ExperimentConfig
from .data import BatchedWindowLoader, MultiStationWindowDataset, ScalingStats
from .engine import evaluate_model
from .experiments import build_model
from .metrics import regression_metrics
from .reporting import evaluation_frame, save_evaluation_artifacts
from .splits import split_files_by_sites


def _metrics(frame: pd.DataFrame, *, high_threshold: float | None = None) -> dict:
    micro = regression_metrics(frame["target"], frame["prediction"])
    station = [
        regression_metrics(group["target"], group["prediction"])
        for _, group in frame.groupby("station", sort=True)
    ]
    high_threshold = (
        float(frame["target"].quantile(0.95))
        if high_threshold is None else float(high_threshold)
    )
    high = frame[frame["target"] >= high_threshold]
    return {
        "count": int(len(frame)),
        "stations": int(frame["station"].nunique()),
        "micro": micro,
        "macro_rmse": float(np.mean([row["rmse"] for row in station])),
        "macro_mae": float(np.mean([row["mae"] for row in station])),
        "high_target_threshold": high_threshold,
        "high_target_bias": float((high["prediction"] - high["target"]).mean()),
        "high_target_rmse": regression_metrics(
            high["target"], high["prediction"]
        )["rmse"],
    }


def evaluate_checkpoint_domain(
    config: ExperimentConfig,
    checkpoint: str | Path,
    scaler_path: str | Path,
    output_dir: str | Path,
    *,
    domain: DomainConfig,
    data_dir: str | Path | None = None,
    sites: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Run hourly validation without refitting either model or scaler."""
    data_dir = Path(data_dir or config.data_dir)
    sites = tuple(sites or config.val_sites)
    files = split_files_by_sites(
        data_dir.glob("*.csv"), (), sites, (), strict=False
    ).val
    if not files:
        raise ValueError(f"No evaluation CSV files matched {data_dir}")
    scaler = ScalingStats.load(scaler_path)
    window = replace(config.window, endpoint_stride=1, endpoint_phase=0)
    dataset = MultiStationWindowDataset(
        files, config.features, window, scaler=scaler,
        split_name="domain_evaluation", domain=domain,
    )
    loader = BatchedWindowLoader(
        dataset, batch_size=config.training.batch_size, shuffle=False,
        pin_memory=torch.cuda.is_available(), metadata="full",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config.model, config.features, seq_len=config.window.seq_len,
        time_feature_dim=dataset.time_feature_dim,
    ).to(device)
    if hasattr(model, "configure_scaling"):
        model.configure_scaling(scaler.forcing_offset, scaler.forcing_scale)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload))
    result = evaluate_model(
        model, loader, device, scaler=scaler,
        minimum_target=config.evaluation.minimum_target,
        amp=config.training.amp,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_artifacts(
        result, output_dir, prefix="domain", config=replace(
            config.evaluation, save_predictions=True, save_plots=False
        ),
    )
    frame = evaluation_frame(result)
    (output_dir / "domain_metrics_extended.json").write_text(
        json.dumps(_metrics(frame), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return frame


def compare_on_common_rows(
    frames: dict[str, pd.DataFrame], output_path: str | Path
) -> dict:
    """Compare domains only on station-hours shared by every input variant."""
    common = None
    prepared: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        values = frame.copy()
        values["date"] = pd.to_datetime(values["date"], utc=True)
        values = values.drop_duplicates(["station", "date"], keep="last")
        prepared[name] = values
        keys = values[["station", "date"]]
        common = keys if common is None else common.merge(keys, on=["station", "date"])
    if common is None or common.empty:
        raise ValueError("Domain evaluations have no common station-hours")
    reference = next(iter(prepared.values()))
    target = common.merge(
        reference[["station", "date", "target"]], on=["station", "date"]
    )
    threshold = float(target["target"].quantile(0.95))
    domains = {}
    for name, frame in prepared.items():
        paired = target.merge(
            frame[["station", "date", "prediction", "land_cover_id"]],
            on=["station", "date"], validate="one_to_one",
        )
        domains[name] = _metrics(paired, high_threshold=threshold)
    payload = {
        "common_rows": int(len(target)),
        "common_stations": int(target["station"].nunique()),
        "domains": domains,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload

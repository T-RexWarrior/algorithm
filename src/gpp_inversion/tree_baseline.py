"""Station-safe feature engineering and HistGradientBoosting baseline."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import EvaluationConfig, FeatureColumns, ScalingMethod, WindowConfig
from .data import MultiStationWindowDataset
from .engine import EvaluationResult
from .metrics import regression_metrics
from .reporting import save_evaluation_artifacts


@dataclass(frozen=True)
class TreeBaselineConfig:
    max_windows_per_station: int = 10_000
    batch_size: int = 4096
    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    l2_regularization: float = 1e-4
    seed: int = 42


def _uniform_station_indices(dataset, maximum: int) -> np.ndarray:
    """Take evenly spaced valid windows from every station."""
    selected = []
    offset = 0
    for count in dataset.station_window_counts:
        count = int(count)
        take = min(count, maximum)
        if take:
            local = (
                np.arange(count, dtype=np.int64)
                if take == count
                else np.linspace(0, count - 1, num=take, dtype=np.int64)
            )
            selected.append(local + offset)
        offset += count
    return np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)


def _tree_feature_names(features: FeatureColumns, land_cover_classes: int) -> list[str]:
    names = [f"current_forcing__{name}" for name in features.forcing]
    for horizon in (6, 24, 96):
        for statistic in ("mean", "std", "min", "max"):
            names.extend(
                f"forcing_{horizon}h_{statistic}__{name}"
                for name in features.forcing
            )
        names.append(f"precipitation_{horizon}h_sum")
    state_names = [*features.state, *features.spectral_indices]
    names.extend(f"current_state__{name}" for name in state_names)
    names.extend(f"latest_valid_epic__{name}" for name in state_names)
    names.append("hours_since_latest_valid_epic")
    names.extend(("hour_sin", "hour_cos", "year_sin", "year_cos"))
    names.extend(f"static__{name}" for name in features.static)
    names.extend(f"land_cover__{index}" for index in range(land_cover_classes))
    return names


def tree_features_from_batch(
    batch,
    dataset: MultiStationWindowDataset,
    *,
    land_cover_classes: int,
) -> np.ndarray:
    forcing, state, time_features, static, land_cover = (
        value.detach().cpu().numpy() for value in batch[:5]
    )
    blocks = [forcing[:, -1, :]]
    precipitation_index = (
        dataset.features.forcing.index("P_F")
        if "P_F" in dataset.features.forcing
        else 0
    )
    for horizon in (6, 24, 96):
        values = forcing[:, -min(horizon, forcing.shape[1]) :, :]
        blocks.extend(
            (
                values.mean(axis=1),
                values.std(axis=1),
                values.min(axis=1),
                values.max(axis=1),
                values[:, :, precipitation_index].sum(axis=1, keepdims=True),
            )
        )
    blocks.append(state[:, -1, :])

    if "EPIC_Available_Mask" not in dataset.features.state:
        raise ValueError("Tree baseline requires EPIC_Available_Mask in state features")
    mask_index = dataset.features.state.index("EPIC_Available_Mask")
    raw_mask = (
        state[:, :, mask_index] * dataset.scaler.state_scale[mask_index]
        + dataset.scaler.state_offset[mask_index]
    ) > 0.5
    reverse_position = np.argmax(raw_mask[:, ::-1], axis=1)
    has_valid = raw_mask.any(axis=1)
    last_index = state.shape[1] - 1 - reverse_position
    rows = np.arange(state.shape[0])
    latest = state[rows, last_index].copy()
    latest[~has_valid] = 0.0
    age = (state.shape[1] - 1 - last_index).astype(np.float32)
    age[~has_valid] = float(state.shape[1])
    blocks.extend((latest, age[:, None]))
    blocks.extend(
        (
            time_features[:, -1, :4],
            static[:, -1, :],
            np.eye(land_cover_classes, dtype=np.float32)[
                np.clip(land_cover[:, -1], 0, land_cover_classes - 1)
            ],
        )
    )
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def _matrix_for_indices(dataset, indices, batch_size, land_cover_classes):
    pieces = []
    targets = []
    for offset in range(0, len(indices), batch_size):
        batch = dataset.get_batch(indices[offset : offset + batch_size], metadata="none")
        pieces.append(
            tree_features_from_batch(
                batch, dataset, land_cover_classes=land_cover_classes
            )
        )
        targets.append(batch[5].numpy())
    return np.vstack(pieces), np.concatenate(targets)


def _evaluate_tree(
    model,
    dataset,
    output_dir,
    *,
    prefix,
    batch_size,
    land_cover_classes,
    threshold,
):
    predictions, targets, dates, stations, land_cover_ids = [], [], [], [], []
    indices = np.arange(len(dataset), dtype=np.int64)
    for offset in range(0, len(indices), batch_size):
        batch = dataset.get_batch(indices[offset : offset + batch_size], metadata="full")
        matrix = tree_features_from_batch(
            batch, dataset, land_cover_classes=land_cover_classes
        )
        predictions.append(dataset.scaler.inverse_target(model.predict(matrix)))
        targets.append(dataset.scaler.inverse_target(batch[5].numpy()))
        dates.extend(batch[6])
        stations.extend(batch[7])
        land_cover_ids.append(batch[4][:, -1].numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    result = EvaluationResult(
        metrics=regression_metrics(target, prediction),
        predictions=prediction,
        targets=target,
        dates=tuple(dates),
        station_names=tuple(stations),
        land_cover_ids=np.concatenate(land_cover_ids),
    )
    artifacts = save_evaluation_artifacts(
        result,
        output_dir,
        prefix=prefix,
        config=EvaluationConfig(save_predictions=True, save_plots=False),
        high_target_threshold=threshold,
    )
    return result.metrics, artifacts.as_dict()


def run_tree_baseline(
    *,
    train_files,
    val_files,
    output_dir: str | Path,
    features: FeatureColumns,
    window: WindowConfig,
    land_cover_classes: int,
    config: TreeBaselineConfig = TreeBaselineConfig(),
    test_files=(),
    evaluate_test: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = MultiStationWindowDataset(
        train_files, features, window, scaling=ScalingMethod.ZSCORE, split_name="train"
    )
    validation = MultiStationWindowDataset(
        val_files, features, window, scaler=train.scaler, split_name="val"
    )
    train_targets = train.raw_window_targets()
    threshold = float(np.quantile(train_targets, 0.9))
    indices = _uniform_station_indices(train, config.max_windows_per_station)
    matrix, target = _matrix_for_indices(
        train, indices, config.batch_size, land_cover_classes
    )
    model = HistGradientBoostingRegressor(
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        max_leaf_nodes=config.max_leaf_nodes,
        l2_regularization=config.l2_regularization,
        early_stopping=True,
        random_state=config.seed,
    )
    model.fit(matrix, target)
    del matrix, target
    model_path = output_dir / "hist_gradient_boosting.joblib"
    joblib.dump(model, model_path)
    validation_metrics, validation_artifacts = _evaluate_tree(
        model,
        validation,
        output_dir,
        prefix="val",
        batch_size=config.batch_size,
        land_cover_classes=land_cover_classes,
        threshold=threshold,
    )
    test_payload = None
    if evaluate_test:
        test = MultiStationWindowDataset(
            test_files, features, window, scaler=train.scaler, split_name="test"
        )
        metrics, artifacts = _evaluate_tree(
            model,
            test,
            output_dir,
            prefix="test",
            batch_size=config.batch_size,
            land_cover_classes=land_cover_classes,
            threshold=threshold,
        )
        test_payload = {"metrics": metrics, "artifacts": artifacts}

    feature_names = _tree_feature_names(features, land_cover_classes)
    configuration = {
        "tree": asdict(config),
        "features": {
            "forcing": list(features.forcing),
            "state": list(features.state),
            "static": list(features.static),
            "spectral_indices": list(features.spectral_indices),
            "target": features.target,
            "time": features.time,
            "land_cover": features.land_cover,
        },
        "window": {
            **asdict(window),
            "time_features": window.time_features.value,
        },
        "land_cover_classes": land_cover_classes,
        "evaluate_test": evaluate_test,
        "train_files": [str(Path(path).resolve()) for path in train_files],
        "val_files": [str(Path(path).resolve()) for path in val_files],
        "test_files": (
            [str(Path(path).resolve()) for path in test_files]
            if evaluate_test else []
        ),
    }
    configuration_hash = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_set_locked": not evaluate_test,
        "configuration": configuration,
        "config_hash": configuration_hash,
        "input_files": [
            {
                "path": str(Path(path).resolve()),
                "size_bytes": Path(path).stat().st_size,
                "modified_time_ns": Path(path).stat().st_mtime_ns,
                "split": split,
            }
            for split, paths in (("train", train_files), ("val", val_files))
            for path in paths
        ],
        "features": feature_names,
        "feature_count": len(feature_names),
        "training_windows": int(indices.size),
        "training_stations": len(train.station_names),
        "high_target_threshold": threshold,
        "validation": {
            "metrics": validation_metrics,
            "artifacts": validation_artifacts,
        },
        "test": test_payload,
        "model": str(model_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = output_dir / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload

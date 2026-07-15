"""Training, station-aware model selection, and evaluation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import ScalingStats
from .metrics import regression_metrics


@dataclass
class TrainingResult:
    best_val_loss: float
    best_selection_score: float
    selection_metric: str
    epochs_completed: int
    history: dict[str, list[float]]
    best_checkpoint: Path
    latest_checkpoint: Path
    config_hash: str | None = None


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    predictions: np.ndarray
    targets: np.ndarray
    dates: tuple[str, ...]
    station_names: tuple[str, ...]
    land_cover_ids: np.ndarray


@dataclass(frozen=True)
class EpochStatistics:
    loss: float
    micro_rmse: float = float("nan")
    macro_rmse: float = float("nan")


def _move_batch(batch, device: torch.device):
    if len(batch) != 8:
        raise ValueError("Expected dataset batch with 8 fields")
    forcing, state, time_features, static, land_cover, target, dates, stations = batch
    non_blocking = device.type == "cuda"
    return (
        forcing.to(device, non_blocking=non_blocking),
        state.to(device, non_blocking=non_blocking),
        time_features.to(device, non_blocking=non_blocking),
        static.to(device, non_blocking=non_blocking),
        land_cover.to(device, non_blocking=non_blocking),
        target.to(device, non_blocking=non_blocking),
        dates,
        stations,
    )


def _forward(model, moved_batch):
    forcing, state, time_features, static, land_cover, target, dates, stations = moved_batch
    prediction = model(forcing, state, time_features, static, land_cover)
    return prediction, target, dates, stations, land_cover[:, -1]


def _run_epoch(
    model,
    loader,
    criterion,
    device,
    *,
    optimizer=None,
    grad_scaler=None,
    amp: bool = False,
    station_metrics: bool = False,
) -> EpochStatistics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    total_squared_error = 0.0
    station_squared_error: dict[str, float] = defaultdict(float)
    station_counts: dict[str, int] = defaultdict(int)
    amp_enabled = amp and device.type == "cuda"
    grad_context = torch.enable_grad() if training else torch.no_grad()

    with grad_context:
        for batch in loader:
            moved = _move_batch(batch, device)
            target = moved[5]
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction, target, _, stations, _ = _forward(model, moved)
                loss = criterion(prediction, target)
            if training:
                if grad_scaler is not None and amp_enabled:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            batch_count = int(target.numel())
            total_loss += float(loss.detach().float().cpu()) * batch_count
            total_count += batch_count
            if station_metrics:
                squared_error = (
                    (prediction.detach().float() - target.detach().float()) ** 2
                ).cpu().numpy()
                station_array = np.asarray(stations, dtype=object)
                total_squared_error += float(np.sum(squared_error))
                for station in np.unique(station_array):
                    mask = station_array == station
                    station_squared_error[str(station)] += float(
                        np.sum(squared_error[mask])
                    )
                    station_counts[str(station)] += int(np.sum(mask))

    if total_count == 0:
        raise ValueError("DataLoader produced no batches")
    if not station_metrics:
        return EpochStatistics(loss=total_loss / total_count)
    station_rmse = [
        math.sqrt(station_squared_error[name] / count)
        for name, count in station_counts.items()
        if count
    ]
    return EpochStatistics(
        loss=total_loss / total_count,
        micro_rmse=math.sqrt(total_squared_error / total_count),
        macro_rmse=float(np.mean(station_rmse)),
    )


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    output_dir: str | Path,
    *,
    epochs: int = 100,
    patience: int = 10,
    resume: bool = True,
    scaler: ScalingStats | None = None,
    config_hash: str | None = None,
    selection_metric: str = "val_loss",
    amp: bool = False,
) -> TrainingResult:
    if selection_metric not in {"val_loss", "micro_rmse", "macro_rmse"}:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "checkpoint_latest.pth"
    best_path = output_dir / "checkpoint_best.pth"
    history_path = output_dir / "training_history.json"

    start_epoch = 0
    best_val_loss = float("inf")
    best_selection_score = float("inf")
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_micro_rmse": [],
        "val_macro_rmse": [],
        "selection_score": [],
    }
    if resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        checkpoint_hash = checkpoint.get("config_hash")
        if config_hash and checkpoint_hash != config_hash:
            raise ValueError(
                "Checkpoint configuration hash does not match this experiment: "
                f"{checkpoint_hash} != {config_hash}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        best_selection_score = float(
            checkpoint.get("best_selection_score", best_val_loss)
        )
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        restored_history = checkpoint.get("history", {})
        for key in history:
            history[key] = list(restored_history.get(key, history[key]))

    amp_enabled = amp and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    completed = start_epoch
    for epoch in range(start_epoch, epochs):
        train_stats = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            amp=amp,
        )
        val_stats = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            amp=amp,
            station_metrics=True,
        )
        scores = {
            "val_loss": val_stats.loss,
            "micro_rmse": val_stats.micro_rmse,
            "macro_rmse": val_stats.macro_rmse,
        }
        selection_score = scores[selection_metric]
        history["train_loss"].append(train_stats.loss)
        history["val_loss"].append(val_stats.loss)
        history["val_micro_rmse"].append(val_stats.micro_rmse)
        history["val_macro_rmse"].append(val_stats.macro_rmse)
        history["selection_score"].append(selection_score)
        completed = epoch + 1

        improved = selection_score < best_selection_score
        if improved:
            best_val_loss = val_stats.loss
            best_selection_score = selection_score
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_selection_score": best_selection_score,
                    "selection_metric": selection_metric,
                    "config_hash": config_hash,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_selection_score": best_selection_score,
                "selection_metric": selection_metric,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "config_hash": config_hash,
            },
            latest_path,
        )
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"epoch={epoch + 1}/{epochs} "
            f"train_loss={train_stats.loss:.6f} "
            f"val_loss={val_stats.loss:.6f} "
            f"micro_rmse={val_stats.micro_rmse:.6f} "
            f"macro_rmse={val_stats.macro_rmse:.6f} "
            f"selected={selection_metric}:{selection_score:.6f} "
            f"best={best_selection_score:.6f}",
            flush=True,
        )
        if epochs_without_improvement >= patience:
            break

    if best_path.exists():
        best_checkpoint = torch.load(
            best_path, map_location=device, weights_only=False
        )
        if isinstance(best_checkpoint, dict) and "model_state_dict" in best_checkpoint:
            best_checkpoint_hash = best_checkpoint.get("config_hash")
            if config_hash and best_checkpoint_hash != config_hash:
                raise ValueError(
                    "Best checkpoint configuration hash does not match this experiment: "
                    f"{best_checkpoint_hash} != {config_hash}"
                )
            best_state = best_checkpoint["model_state_dict"]
        else:
            best_state = best_checkpoint
        model.load_state_dict(best_state)
    if scaler is not None:
        scaler.save(output_dir / "scaler.npz")
    return TrainingResult(
        best_val_loss=best_val_loss,
        best_selection_score=best_selection_score,
        selection_metric=selection_metric,
        epochs_completed=completed,
        history=history,
        best_checkpoint=best_path,
        latest_checkpoint=latest_path,
        config_hash=config_hash,
    )


def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    scaler: ScalingStats | None = None,
    minimum_target: float | None = 0.0,
    amp: bool = False,
) -> EvaluationResult:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    all_dates: list[str] = []
    all_stations: list[str] = []
    all_land_cover: list[np.ndarray] = []
    amp_enabled = amp and device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction, target, dates, stations, land_cover = _forward(
                    model, _move_batch(batch, device)
                )
            predictions.append(prediction.detach().float().cpu().numpy())
            targets.append(target.detach().float().cpu().numpy())
            all_dates.extend(str(value) for value in dates)
            all_stations.extend(str(value) for value in stations)
            all_land_cover.append(land_cover.detach().cpu().numpy())
    if not predictions:
        raise ValueError("DataLoader produced no evaluation batches")
    prediction_values = np.concatenate(predictions)
    target_values = np.concatenate(targets)
    land_cover_values = np.concatenate(all_land_cover).astype(np.int64)
    if scaler is not None:
        prediction_values = scaler.inverse_target(prediction_values)
        target_values = scaler.inverse_target(target_values)
    valid = np.isfinite(prediction_values) & np.isfinite(target_values)
    if minimum_target is not None:
        valid &= target_values >= minimum_target
    prediction_values = prediction_values[valid]
    target_values = target_values[valid]
    dates = tuple(value for value, keep in zip(all_dates, valid) if keep)
    stations = tuple(value for value, keep in zip(all_stations, valid) if keep)
    land_cover_values = land_cover_values[valid]
    return EvaluationResult(
        metrics=regression_metrics(target_values, prediction_values),
        predictions=prediction_values,
        targets=target_values,
        dates=dates,
        station_names=stations,
        land_cover_ids=land_cover_values,
    )

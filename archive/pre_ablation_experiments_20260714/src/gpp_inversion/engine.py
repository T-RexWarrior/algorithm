"""Training and evaluation shared by all integrated model families."""

from __future__ import annotations

import json
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


def _move_batch(batch, device: torch.device):
    if len(batch) != 8:
        raise ValueError("Expected dataset batch with 8 fields")
    forcing, state, time_features, static, land_cover, target, dates, stations = batch
    return (
        forcing.to(device),
        state.to(device),
        time_features.to(device),
        static.to(device),
        land_cover.to(device),
        target.to(device),
        dates,
        stations,
    )


def _forward(model, moved_batch):
    forcing, state, time_features, static, land_cover, target, dates, stations = moved_batch
    prediction = model(forcing, state, time_features, static, land_cover)
    return prediction, target, dates, stations, land_cover[:, -1]


def _mean_epoch_loss(model, loader, criterion, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            prediction, target, *_ = _forward(model, _move_batch(batch, device))
            loss = criterion(prediction, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("DataLoader produced no batches")
    return float(np.mean(losses))


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
) -> TrainingResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "checkpoint_latest.pth"
    best_path = output_dir / "checkpoint_best.pth"
    history_path = output_dir / "training_history.json"

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}
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
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history = checkpoint.get("history", history)

    completed = start_epoch
    for epoch in range(start_epoch, epochs):
        train_loss = _mean_epoch_loss(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss = _mean_epoch_loss(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        completed = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
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
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "config_hash": config_hash,
            },
            latest_path,
        )
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
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
) -> EvaluationResult:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    all_dates: list[str] = []
    all_stations: list[str] = []
    all_land_cover: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            prediction, target, dates, stations, land_cover = _forward(
                model, _move_batch(batch, device)
            )
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            all_dates.extend(str(value) for value in dates)
            all_stations.extend(str(value) for value in stations)
            all_land_cover.append(land_cover.detach().cpu().numpy())
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

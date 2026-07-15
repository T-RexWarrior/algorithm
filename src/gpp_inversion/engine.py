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
    steps: int = 0


def _move_batch(batch, device: torch.device):
    if len(batch) == 8:
        forcing, state, time_features, static, land_cover, target, dates, stations = batch
        daily_context = None
    elif len(batch) == 9:
        (
            forcing, state, time_features, static, land_cover,
            daily_context, target, dates, stations,
        ) = batch
    else:
        raise ValueError("Expected dataset batch with 8 or 9 fields")
    non_blocking = device.type == "cuda"
    return (
        forcing.to(device, non_blocking=non_blocking),
        state.to(device, non_blocking=non_blocking),
        time_features.to(device, non_blocking=non_blocking),
        static.to(device, non_blocking=non_blocking),
        land_cover.to(device, non_blocking=non_blocking),
        (
            daily_context.to(device, non_blocking=non_blocking)
            if daily_context is not None else None
        ),
        target.to(device, non_blocking=non_blocking),
        dates,
        stations,
    )


def _forward(model, moved_batch):
    (
        forcing, state, time_features, static, land_cover,
        daily_context, target, dates, stations,
    ) = moved_batch
    if daily_context is None:
        prediction = model(forcing, state, time_features, static, land_cover)
    else:
        prediction = model(
            forcing, state, time_features, static, land_cover,
            daily_context=daily_context,
        )
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
    scheduler=None,
    max_batches: int | None = None,
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
        steps = 0
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            moved = _move_batch(batch, device)
            target = moved[6]
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
                optimizer_stepped = True
                if grad_scaler is not None and amp_enabled:
                    scale_before = grad_scaler.get_scale()
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer_stepped = grad_scaler.get_scale() >= scale_before
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                if scheduler is not None and optimizer_stepped:
                    scheduler.step()
                steps += 1

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
        return EpochStatistics(loss=total_loss / total_count, steps=steps)
    station_rmse = [
        math.sqrt(station_squared_error[name] / count)
        for name, count in station_counts.items()
        if count
    ]
    return EpochStatistics(
        loss=total_loss / total_count,
        micro_rmse=math.sqrt(total_squared_error / total_count),
        macro_rmse=float(np.mean(station_rmse)),
        steps=steps,
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
    max_steps: int | None = None,
    warmup_steps: int = 0,
    eval_interval_steps: int = 1000,
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
        "global_step": [],
    }
    global_step = 0
    scheduler = None
    if max_steps is not None:
        def lr_factor(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return max(1e-8, (step + 1) / warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
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
        global_step = int(checkpoint.get("global_step", 0))
        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restored_history = checkpoint.get("history", {})
        for key in history:
            history[key] = list(restored_history.get(key, history[key]))

    amp_enabled = amp and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    completed = start_epoch
    effective_epochs = (
        math.ceil(max_steps / eval_interval_steps)
        if max_steps is not None else epochs
    )
    for epoch in range(start_epoch, effective_epochs):
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        remaining = None if max_steps is None else max_steps - global_step
        if remaining is not None and remaining <= 0:
            break
        max_batches = (
            min(eval_interval_steps, remaining) if remaining is not None else None
        )
        train_stats = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            amp=amp,
            scheduler=scheduler,
            max_batches=max_batches,
        )
        global_step += train_stats.steps
        val_stats = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            amp=amp,
            station_metrics=True,
        )
        target_rmse_scale = (
            float(scaler.target_scale)
            if scaler is not None and scaler.scale_target else 1.0
        )
        val_micro_rmse = val_stats.micro_rmse * target_rmse_scale
        val_macro_rmse = val_stats.macro_rmse * target_rmse_scale
        scores = {
            "val_loss": val_stats.loss,
            "micro_rmse": val_micro_rmse,
            "macro_rmse": val_macro_rmse,
        }
        selection_score = scores[selection_metric]
        history["train_loss"].append(train_stats.loss)
        history["val_loss"].append(val_stats.loss)
        history["val_micro_rmse"].append(val_micro_rmse)
        history["val_macro_rmse"].append(val_macro_rmse)
        history["selection_score"].append(selection_score)
        history["global_step"].append(global_step)
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
                    "global_step": global_step,
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
                "global_step": global_step,
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            },
            latest_path,
        )
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"epoch={epoch + 1}/{effective_epochs} step={global_step} "
            f"train_loss={train_stats.loss:.6f} "
            f"val_loss={val_stats.loss:.6f} "
            f"micro_rmse={val_micro_rmse:.6f} "
            f"macro_rmse={val_macro_rmse:.6f} "
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

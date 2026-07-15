"""End-to-end orchestration used by both CLI and the integrated Notebook."""

from __future__ import annotations

import random
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import MultiStationWindowDataset
from .engine import evaluate_model, train_model
from .experiments import build_model
from .losses import build_loss
from .splits import split_files_by_sites


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(config: ExperimentConfig) -> dict:
    _seed_everything(config.training.seed)
    all_files = sorted(config.data_dir.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files in {config.data_dir}")
    files = split_files_by_sites(
        all_files,
        config.train_sites,
        config.val_sites,
        config.test_sites,
        strict=True,
    )

    train_dataset = MultiStationWindowDataset(
        files.train,
        config.features,
        config.window,
        scaling=config.scaling,
        scale_target=config.scale_target,
        split_name="train",
    )
    val_dataset = MultiStationWindowDataset(
        files.val,
        config.features,
        config.window,
        scaler=train_dataset.scaler,
        split_name="val",
    )
    test_dataset = MultiStationWindowDataset(
        files.test,
        config.features,
        config.window,
        scaler=train_dataset.scaler,
        split_name="test",
    )

    loader_options = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config.model,
        config.features,
        seq_len=config.window.seq_len,
        time_feature_dim=train_dataset.time_feature_dim,
    ).to(device)
    criterion = build_loss(config.loss, **config.loss_options)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.learning_rate
    )
    training_result = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        config.output_dir,
        epochs=config.training.epochs,
        patience=config.training.patience,
        resume=config.training.resume,
        scaler=train_dataset.scaler,
    )
    evaluation = evaluate_model(
        model, test_loader, device, scaler=train_dataset.scaler
    )
    return {
        "device": str(device),
        "split_counts": {
            "train_files": len(files.train),
            "val_files": len(files.val),
            "test_files": len(files.test),
            "ignored_files": len(files.ignored),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "test_windows": len(test_dataset),
        },
        "training": {
            "best_val_loss": training_result.best_val_loss,
            "epochs_completed": training_result.epochs_completed,
            "best_checkpoint": str(training_result.best_checkpoint),
        },
        "test_metrics": evaluation.metrics,
    }

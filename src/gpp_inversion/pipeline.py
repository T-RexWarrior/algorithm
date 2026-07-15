"""End-to-end holdout and stratified cross-validation orchestration."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .architecture_diagnostics import (
    parameter_count,
    profile_inference,
    save_fixed_window_diagnostics,
)
from .data import (
    BatchedWindowLoader,
    MultiStationWindowDataset,
    StationBalancedSampler,
    StationTargetBalancedSampler,
)
from .engine import evaluate_model, train_model
from .experiments import build_model
from .losses import build_loss
from .provenance import (
    finalize_experiment_manifest,
    write_experiment_manifest,
)
from .reporting import save_evaluation_artifacts
from .splits import (
    FileSplits,
    infer_site_land_cover_labels,
    split_files_by_sites,
    stratified_site_folds,
    validate_site_splits,
)


def _seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _loader_options(config: ExperimentConfig, device: torch.device) -> dict:
    options = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.training.num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return options


def _aggregate_fold_metrics(metrics: list[dict[str, float]]) -> dict:
    summary = {"folds": len(metrics)}
    for key in ("mse", "rmse", "mae", "r2"):
        values = np.asarray([row[key] for row in metrics], dtype=float)
        values = values[np.isfinite(values)]
        summary[key] = {
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "std": float(np.std(values)) if values.size else float("nan"),
        }
    summary["total_count"] = int(sum(row.get("count", 0) for row in metrics))
    return summary


def _run_training_split(
    config: ExperimentConfig,
    files: FileSplits,
    output_dir: Path,
    *,
    manifest_extra: dict,
    evaluate_test: bool,
) -> dict:
    relevant_files = (
        *files.train,
        *files.val,
        *(files.test if evaluate_test else ()),
    )
    manifest, manifest_path = write_experiment_manifest(
        config, relevant_files, output_dir, extra=manifest_extra
    )
    digest = manifest["config_hash"]
    try:
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
        test_dataset = (
            MultiStationWindowDataset(
                files.test,
                config.features,
                config.window,
                scaler=train_dataset.scaler,
                split_name="test",
            )
            if evaluate_test and files.test else None
        )
        train_targets = train_dataset.raw_window_targets()
        high_target_threshold = float(np.quantile(train_targets, 0.9))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        samples_per_epoch = config.training.samples_per_epoch
        if samples_per_epoch is None and config.training.max_steps is not None:
            samples_per_epoch = (
                config.training.eval_interval_steps * config.training.batch_size
            )
        if config.training.target_balanced:
            train_sampler = StationTargetBalancedSampler(
                train_dataset,
                num_samples=int(samples_per_epoch or len(train_dataset)),
                seed=config.training.seed,
            )
        elif config.training.station_balanced:
            train_sampler = StationBalancedSampler(
                train_dataset,
                num_samples=samples_per_epoch,
                seed=config.training.seed,
            )
        else:
            train_sampler = None
        if config.training.num_workers == 0:
            train_loader = BatchedWindowLoader(
                train_dataset,
                batch_size=config.training.batch_size,
                shuffle=train_sampler is None,
                sampler=train_sampler,
                seed=config.training.seed,
                pin_memory=device.type == "cuda",
                metadata="none",
            )
            val_training_loader = BatchedWindowLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                pin_memory=device.type == "cuda",
                metadata="stations",
            )
            val_loader = BatchedWindowLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                pin_memory=device.type == "cuda",
                metadata="full",
            )
            test_loader = (
                BatchedWindowLoader(
                    test_dataset,
                    batch_size=config.training.batch_size,
                    shuffle=False,
                    pin_memory=device.type == "cuda",
                    metadata="full",
                )
                if test_dataset is not None else None
            )
        else:
            options = _loader_options(config, device)
            train_loader = DataLoader(
                train_dataset,
                shuffle=train_sampler is None,
                sampler=train_sampler,
                **options,
            )
            val_training_loader = DataLoader(
                val_dataset, shuffle=False, **options
            )
            val_loader = val_training_loader
            test_loader = (
                DataLoader(test_dataset, shuffle=False, **options)
                if test_dataset is not None else None
            )

        if config.model.kind.value == "tcn_multiscale":
            if config.window.context_days != 30:
                raise ValueError("tcn_multiscale requires window.context_days=30")
            if len(config.window.daily_context_columns) != config.model.daily_context_features:
                raise ValueError("daily context feature count does not match model config")
        if config.model.kind.value == "hybrid_lue_tcn" and config.scale_target:
            raise ValueError("hybrid_lue_tcn requires scale_target=false for physical units")
        model = build_model(
            config.model,
            config.features,
            seq_len=config.window.seq_len,
            time_feature_dim=train_dataset.time_feature_dim,
        ).to(device)
        loss_options = dict(config.loss_options)
        if config.loss.value == "tail_aware" and not {"p50", "p80", "p95"}.issubset(loss_options):
            raw_thresholds = np.quantile(train_targets, [0.5, 0.8, 0.95])
            if train_dataset.scaler.scale_target:
                raw_thresholds = (
                    raw_thresholds - train_dataset.scaler.target_offset
                ) / train_dataset.scaler.target_scale
            loss_options.update(
                p50=float(raw_thresholds[0]),
                p80=float(raw_thresholds[1]),
                p95=float(raw_thresholds[2]),
            )
        criterion = build_loss(config.loss, **loss_options).to(device)
        if hasattr(model, "configure_scaling"):
            model.configure_scaling(
                train_dataset.scaler.forcing_offset,
                train_dataset.scaler.forcing_scale,
            )
        if config.training.pretrained_checkpoint and not (
            config.training.resume and (output_dir / "checkpoint_latest.pth").exists()
        ):
            pretrained = torch.load(
                config.training.pretrained_checkpoint, map_location=device,
                weights_only=False,
            )
            state_dict = pretrained.get("model_state_dict", pretrained)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise ValueError(f"Unexpected pretrained parameters: {unexpected}")
        optimizer_class = (
            torch.optim.AdamW
            if config.training.optimizer == "adamw"
            else torch.optim.Adam
        )
        optimizer = optimizer_class(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        training_result = train_model(
            model,
            train_loader,
            val_training_loader,
            optimizer,
            criterion,
            device,
            output_dir,
            epochs=config.training.epochs,
            patience=config.training.patience,
            resume=config.training.resume,
            scaler=train_dataset.scaler,
            config_hash=digest,
            selection_metric=config.training.selection_metric,
            amp=config.training.amp,
            max_steps=config.training.max_steps,
            warmup_steps=config.training.warmup_steps,
            eval_interval_steps=config.training.eval_interval_steps,
        )

        architecture_profile, profile_path = profile_inference(
            model, val_loader, device, output_dir
        )
        architecture_diagnostics = save_fixed_window_diagnostics(
            model, val_dataset, device, output_dir
        )

        val_evaluation = evaluate_model(
            model,
            val_loader,
            device,
            scaler=train_dataset.scaler,
            minimum_target=config.evaluation.minimum_target,
            amp=config.training.amp,
        )
        val_artifacts = save_evaluation_artifacts(
            val_evaluation,
            output_dir,
            prefix="val",
            config=config.evaluation,
            high_target_threshold=high_target_threshold,
        )
        test_metrics = None
        test_artifacts = None
        if test_loader is not None:
            test_evaluation = evaluate_model(
                model,
                test_loader,
                device,
                scaler=train_dataset.scaler,
                minimum_target=config.evaluation.minimum_target,
                amp=config.training.amp,
            )
            test_metrics = test_evaluation.metrics
            test_artifacts = save_evaluation_artifacts(
                test_evaluation,
                output_dir,
                prefix="test",
                config=config.evaluation,
                high_target_threshold=high_target_threshold,
            )

        result = {
            "mode": manifest_extra["mode"],
            "config_hash": digest,
            "device": str(device),
            "split_counts": {
                "train_files": len(files.train),
                "val_files": len(files.val),
                "test_files": len(files.test),
                "ignored_files": len(files.ignored),
                "train_windows": len(train_dataset),
                "val_windows": len(val_dataset),
                "test_windows": len(test_dataset) if test_dataset is not None else 0,
            },
            "training": {
                "best_val_loss": training_result.best_val_loss,
                "best_selection_score": training_result.best_selection_score,
                "selection_metric": training_result.selection_metric,
                "epochs_completed": training_result.epochs_completed,
                "best_checkpoint": str(training_result.best_checkpoint),
            },
            "val_metrics": val_evaluation.metrics,
            "test_metrics": test_metrics,
            "high_target_threshold": high_target_threshold,
            "architecture_profile": architecture_profile,
        }
        artifacts = {
            "manifest": str(manifest_path),
            "validation": val_artifacts.as_dict(),
            "test": test_artifacts.as_dict() if test_artifacts else None,
            "architecture_profile": str(profile_path),
            "architecture_diagnostics": architecture_diagnostics,
        }
        summary_path = output_dir / "result_summary.json"
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        artifacts["result_summary"] = str(summary_path)
        finalize_experiment_manifest(
            manifest_path, result=result, artifacts=artifacts
        )
        result["artifacts"] = artifacts
        return result
    except Exception as exc:
        finalize_experiment_manifest(
            manifest_path,
            result={"error": f"{type(exc).__name__}: {exc}"},
            artifacts={},
            status="failed",
        )
        raise


def _run_holdout(config: ExperimentConfig, all_files: list[Path]) -> dict:
    files = split_files_by_sites(
        all_files,
        config.train_sites,
        config.val_sites,
        config.test_sites,
        strict=True,
    )
    return _run_training_split(
        config,
        files,
        config.output_dir,
        manifest_extra={"mode": "holdout"},
        evaluate_test=config.evaluation.evaluate_test,
    )


def _run_cross_validation(config: ExperimentConfig, all_files: list[Path]) -> dict:
    if config.features.land_cover is None:
        raise ValueError("Stratified cross-validation requires a land-cover column")
    development_sites = tuple(
        dict.fromkeys((*config.train_sites, *config.val_sites))
    )
    validate_site_splits(development_sites, (), config.test_sites)
    labels = infer_site_land_cover_labels(
        all_files, development_sites, config.features.land_cover
    )
    folds = tuple(
        stratified_site_folds(
            development_sites,
            labels,
            n_splits=config.cross_validation.n_splits,
            seed=config.cross_validation.seed,
        )
    )

    pool = split_files_by_sites(
        all_files, development_sites, (), config.test_sites, strict=True
    )
    root_manifest, root_manifest_path = write_experiment_manifest(
        config,
        (*pool.train, *pool.test),
        config.output_dir,
        extra={
            "mode": "stratified_kfold",
            "n_splits": config.cross_validation.n_splits,
        },
    )
    try:
        fold_results = []
        fold_manifests = []
        for fold_number, (train_sites, val_sites) in enumerate(folds, start=1):
            test_sites = (
                config.test_sites
                if config.cross_validation.evaluate_test_each_fold else ()
            )
            files = split_files_by_sites(
                all_files, train_sites, val_sites, test_sites, strict=True
            )
            fold_dir = config.output_dir / f"fold_{fold_number:02d}"
            fold_result = _run_training_split(
                config,
                files,
                fold_dir,
                manifest_extra={
                    "mode": "stratified_kfold",
                    "fold": fold_number,
                    "n_splits": config.cross_validation.n_splits,
                    "train_sites": train_sites,
                    "val_sites": val_sites,
                },
                evaluate_test=config.cross_validation.evaluate_test_each_fold,
            )
            fold_result["fold"] = fold_number
            fold_result["train_sites"] = list(train_sites)
            fold_result["val_sites"] = list(val_sites)
            fold_results.append(fold_result)
            fold_manifests.append(fold_result["artifacts"]["manifest"])

        validation_summary = _aggregate_fold_metrics(
            [result["val_metrics"] for result in fold_results]
        )
        test_rows = [
            result["test_metrics"]
            for result in fold_results
            if result["test_metrics"] is not None
        ]
        result = {
            "mode": "stratified_kfold",
            "config_hash": root_manifest["config_hash"],
            "n_splits": config.cross_validation.n_splits,
            "folds": fold_results,
            "validation_summary": validation_summary,
            "test_summary": _aggregate_fold_metrics(test_rows) if test_rows else None,
        }
        summary_path = config.output_dir / "cross_validation_summary.json"
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        artifacts = {
            "manifest": str(root_manifest_path),
            "summary": str(summary_path),
            "fold_manifests": fold_manifests,
        }
        finalize_experiment_manifest(
            root_manifest_path, result=result, artifacts=artifacts
        )
        result["artifacts"] = artifacts
        return result
    except Exception as exc:
        finalize_experiment_manifest(
            root_manifest_path,
            result={"error": f"{type(exc).__name__}: {exc}"},
            artifacts={},
            status="failed",
        )
        raise


def run_experiment(config: ExperimentConfig) -> dict:
    _seed_everything(
        config.training.seed,
        deterministic=config.training.deterministic,
    )
    all_files = sorted(config.data_dir.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files in {config.data_dir}")
    if config.cross_validation.enabled:
        return _run_cross_validation(config, all_files)
    return _run_holdout(config, all_files)

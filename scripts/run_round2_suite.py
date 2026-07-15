"""Run the locked-test, one-factor round-two GPP screening suite."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from gpp_inversion.config import (
    EvaluationConfig,
    ExperimentConfig,
    FeatureColumns,
    LossKind,
    ModelConfig,
    ModelKind,
    ScalingMethod,
    TimeFeatureMode,
    TrainingConfig,
    WindowConfig,
)
from gpp_inversion.metrics import regression_metrics
from gpp_inversion.pipeline import run_experiment
from gpp_inversion.splits import split_files_by_sites
from gpp_inversion.tree_baseline import TreeBaselineConfig, run_tree_baseline


VARIANTS = (
    "huber_delta1",
    "adamw_weight_decay",
    "learned_lag_embedding",
    "static_film",
    "ndvi_nirv",
    "layernorm_lstm",
    "hist_gradient_boosting",
)


def _summary(path: Path, threshold: float) -> dict:
    frame = pd.read_csv(path)
    station_rows = []
    for station, group in frame.groupby("station", sort=True):
        row = {"station": station}
        row.update(regression_metrics(group.target, group.prediction))
        station_rows.append(row)
    high = frame[frame.target >= threshold]
    return {
        "micro": regression_metrics(frame.target, frame.prediction),
        "macro": {
            f"{key}_mean": float(np.nanmean([row[key] for row in station_rows]))
            for key in ("rmse", "mae", "r2", "bias")
        },
        "high_target": regression_metrics(high.target, high.prediction),
        "station_metrics": station_rows,
    }


def _gate(candidate: dict, baseline: dict) -> dict:
    baseline_station = {row["station"]: row for row in baseline["station_metrics"]}
    station_wins = sum(
        row["rmse"] < baseline_station[row["station"]]["rmse"]
        for row in candidate["station_metrics"]
        if row["station"] in baseline_station
    )
    checks = {
        "macro_rmse_improves_at_least_0_5pct": candidate["macro"]["rmse_mean"]
        <= baseline["macro"]["rmse_mean"] * 0.995,
        "micro_rmse_guardrail": candidate["micro"]["rmse"]
        <= baseline["micro"]["rmse"] * 1.005,
        "micro_mae_guardrail": candidate["micro"]["mae"]
        <= baseline["micro"]["mae"] * 1.005,
        "macro_mae_guardrail": candidate["macro"]["mae_mean"]
        <= baseline["macro"]["mae_mean"] * 1.005,
        "high_gpp_mae_guardrail": candidate["high_target"]["mae"]
        <= baseline["high_target"]["mae"] * 1.01,
        "station_rmse_wins_at_least_22": station_wins >= 22,
    }
    return {"passed": all(checks.values()), "station_rmse_wins": station_wins, "checks": checks}


def _base(args, legacy) -> ExperimentConfig:
    features = FeatureColumns(
        forcing=tuple(legacy["forcing_cols"]),
        state=tuple(legacy["state_cols"]),
        static=tuple(legacy["static_cols"]),
        target=legacy["target_col"],
        time=legacy["time_col"],
        land_cover=legacy["lc_col"],
    )
    return ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_sites=tuple(legacy["train_sites"]),
        val_sites=tuple(legacy["val_sites"]),
        test_sites=tuple(legacy["test_sites"]),
        features=features,
        window=WindowConfig(
            seq_len=int(legacy["seq_len"]),
            time_features=TimeFeatureMode.CYCLIC,
            require_regular=True,
            max_gap_hours=1.0,
            max_span_hours=95.0,
        ),
        scaling=ScalingMethod.ZSCORE,
        scale_target=True,
        model=ModelConfig(
            kind=ModelKind.TCN,
            d_model=64,
            nhead=4,
            dropout=0.1,
            dim_feedforward=128,
            num_layers=2,
            num_land_cover_classes=int(legacy["num_lc_classes"]),
            land_cover_embedding_dim=int(legacy["lc_embed_dim"]),
            tcn_layers=6,
        ),
        loss=LossKind.MSE,
        training=TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
            resume=False,
            selection_metric="macro_rmse",
            amp=args.amp,
            deterministic=True,
        ),
        evaluation=EvaluationConfig(
            save_predictions=True,
            save_plots=False,
            minimum_target=None,
            evaluate_test=False,
        ),
    )


def _variant(base, name):
    config = replace(base, output_dir=base.output_dir / name)
    if name == "huber_delta1":
        return replace(config, loss=LossKind.HUBER, loss_options={"delta": 1.0})
    if name == "adamw_weight_decay":
        return replace(config, training=replace(config.training, weight_decay=1e-4))
    if name == "learned_lag_embedding":
        return replace(config, model=replace(config.model, lag_encoding="embedding"))
    if name == "static_film":
        return replace(config, model=replace(config.model, static_context_mode="film"))
    if name == "ndvi_nirv":
        return replace(
            config,
            features=replace(config.features, spectral_indices=("NDVI", "NIRv")),
        )
    if name == "layernorm_lstm":
        return replace(
            config,
            model=replace(
                config.model,
                kind=ModelKind.LSTM,
                d_model=64,
                lstm_hidden_size=64,
                lstm_layers=2,
                dropout=0.1,
            ),
        )
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_config", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("baseline_predictions", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    legacy = json.loads(args.legacy_config.read_text(encoding="utf-8"))
    base = _base(args, legacy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suite_path = args.output_dir / "round2_screening_summary.json"
    suite = {
        "protocol": {
            "test_set_locked": True,
            "seed": args.seed,
            "epochs": args.epochs,
            "variants": list(args.variants),
        },
        "variants": {},
    }
    if suite_path.exists():
        suite["variants"].update(json.loads(suite_path.read_text(encoding="utf-8")).get("variants", {}))

    for name in args.variants:
        output = args.output_dir / name
        manifest_path = output / "experiment_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "completed":
            print(f"variant={name} status=already_completed", flush=True)
            continue
        print(f"variant={name} status=starting", flush=True)
        started = time.perf_counter()
        if name == "hist_gradient_boosting":
            splits = split_files_by_sites(
                args.data_dir.glob("*.csv"), base.train_sites, base.val_sites, base.test_sites
            )
            features = replace(base.features, spectral_indices=("NDVI", "NIRv"))
            result = run_tree_baseline(
                train_files=splits.train,
                val_files=splits.val,
                test_files=splits.test,
                output_dir=output,
                features=features,
                window=base.window,
                land_cover_classes=base.model.num_land_cover_classes,
                config=TreeBaselineConfig(seed=args.seed, batch_size=args.batch_size),
                evaluate_test=False,
            )
            prediction_path = Path(result["validation"]["artifacts"]["predictions"])
            threshold = result["high_target_threshold"]
            config_hash = result["config_hash"]
        else:
            config = _variant(base, name)
            result = run_experiment(config)
            prediction_path = Path(result["artifacts"]["validation"]["predictions"])
            threshold = result["high_target_threshold"]
            config_hash = result["config_hash"]
        candidate = _summary(prediction_path, threshold)
        baseline = _summary(args.baseline_predictions, threshold)
        suite["variants"][name] = {
            "elapsed_seconds": time.perf_counter() - started,
            "config_hash": config_hash,
            "high_target_threshold": threshold,
            "validation": candidate,
            "baseline": baseline,
            "promotion_gate": _gate(candidate, baseline),
            "prediction_file": str(prediction_path),
            "manifest": str(manifest_path),
        }
        suite_path.write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"variant={name} status=completed passed={suite['variants'][name]['promotion_gate']['passed']}", flush=True)
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "summary": str(suite_path),
                "variants": {
                    name: {
                        "passed": values["promotion_gate"]["passed"],
                        "macro_rmse": values["validation"]["macro"]["rmse_mean"],
                    }
                    for name, values in suite["variants"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Run isolated one-factor-at-a-time GPP model screening experiments."""

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


VARIANT_ORDER = (
    "baseline",
    "cross_residual",
    "lag_only",
    "norm_tcn5_only",
    "station_balanced_only",
    "cross_lag_exploratory",
)


def _macro_metrics(frame: pd.DataFrame) -> dict[str, float]:
    rows = []
    for _, group in frame.groupby("station", sort=True):
        rows.append(regression_metrics(group["target"], group["prediction"]))
    result = {"station_count": len(rows)}
    for key in ("rmse", "mae", "r2"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[f"{key}_mean"] = float(np.nanmean(values))
        result[f"{key}_median"] = float(np.nanmedian(values))
    return result


def _prediction_summary(path: Path) -> dict:
    frame = pd.read_csv(path)
    # Preserve physical zeros affected only by inverse-scaling roundoff.
    nonnegative = frame[frame["target"] >= -1e-6]
    return {
        "all_targets": {
            "micro": regression_metrics(frame["target"], frame["prediction"]),
            "macro": _macro_metrics(frame),
        },
        "nonnegative_targets": {
            "micro": regression_metrics(
                nonnegative["target"], nonnegative["prediction"]
            ),
            "macro": _macro_metrics(nonnegative),
        },
    }


def _base_config(args, legacy: dict) -> ExperimentConfig:
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
            normalized_tcn=False,
            cross_attention_residual=False,
            lag_encoding="none",
        ),
        loss=LossKind.MSE,
        training=TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            num_workers=args.num_workers,
            seed=args.seed,
            # Formal ablations must never silently resume without restoring the
            # complete RNG and AMP state; incomplete runs are restarted.
            resume=False,
            selection_metric="macro_rmse",
            station_balanced=False,
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


def _variant_config(base: ExperimentConfig, name: str) -> ExperimentConfig:
    model = base.model
    training = base.training
    if name == "cross_residual":
        model = replace(model, cross_attention_residual=True)
    elif name == "lag_only":
        model = replace(model, lag_encoding="continuous")
    elif name == "norm_tcn5_only":
        model = replace(model, tcn_layers=5, normalized_tcn=True)
    elif name == "station_balanced_only":
        training = replace(training, station_balanced=True)
    elif name == "cross_lag_exploratory":
        model = replace(
            model,
            cross_attention_residual=True,
            lag_encoding="continuous",
        )
    return replace(
        base,
        output_dir=base.output_dir / name,
        model=model,
        training=training,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_config", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_ORDER,
        default=list(VARIANT_ORDER),
    )
    args = parser.parse_args()
    legacy = json.loads(args.legacy_config.read_text(encoding="utf-8"))
    base = _base_config(args, legacy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suite_path = args.output_dir / "suite_summary.json"
    suite = {
        "protocol": {
            "test_set_locked": True,
            "variant_order": list(args.variants),
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "amp": args.amp,
        },
        "variants": {},
    }
    if suite_path.exists():
        prior = json.loads(suite_path.read_text(encoding="utf-8"))
        suite["variants"].update(prior.get("variants", {}))

    for name in args.variants:
        config = _variant_config(base, name)
        result_path = config.output_dir / "result_summary.json"
        manifest_path = config.output_dir / "experiment_manifest.json"
        if result_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "completed":
                print(f"variant={name} status=already_completed", flush=True)
                continue
        print(f"variant={name} status=starting", flush=True)
        started = time.perf_counter()
        result = run_experiment(config)
        prediction_path = Path(
            result["artifacts"]["validation"]["predictions"]
        )
        suite["variants"][name] = {
            "elapsed_seconds": time.perf_counter() - started,
            "model": {
                "cross_attention_residual": config.model.cross_attention_residual,
                "lag_encoding": config.model.lag_encoding,
                "tcn_layers": config.model.tcn_layers,
                "normalized_tcn": config.model.normalized_tcn,
                "station_balanced": config.training.station_balanced,
            },
            "training": result["training"],
            "validation": _prediction_summary(prediction_path),
            "result_summary": str(result_path),
        }
        suite_path.write_text(
            json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"variant={name} status=completed", flush=True)
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(json.dumps(suite, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

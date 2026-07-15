"""Evaluate the historical best checkpoint with station-aware tables."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gpp_inversion.config import (
    EvaluationConfig,
    FeatureColumns,
    ScalingMethod,
    TimeFeatureMode,
    WindowConfig,
)
from gpp_inversion.data import MultiStationWindowDataset, ScalingStats
from gpp_inversion.engine import evaluate_model
from gpp_inversion.metrics import regression_metrics
from gpp_inversion.models import TCNTransformerCrossAttention
from gpp_inversion.reporting import evaluation_frame, save_evaluation_artifacts
from gpp_inversion.splits import split_files_by_sites


def _macro_metrics(frame: pd.DataFrame) -> dict[str, float]:
    rows = [
        regression_metrics(group["target"], group["prediction"])
        for _, group in frame.groupby("station", sort=True)
    ]
    return {
        f"{key}_{summary}": float(getattr(np, summary)([row[key] for row in rows]))
        for key in ("rmse", "mae", "r2")
        for summary in ("mean", "median")
    } | {"station_count": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_output", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    legacy_config = json.loads(
        (args.legacy_output / "model_input_config.json").read_text(
            encoding="utf-8"
        )
    )
    full_checkpoint = torch.load(
        args.legacy_output / "checkpoint_best_full.pth",
        map_location="cpu",
        weights_only=False,
    )
    scaler = ScalingStats(
        method=ScalingMethod.ZSCORE,
        forcing_offset=np.asarray(full_checkpoint["feat_mean_f"]),
        forcing_scale=np.asarray(full_checkpoint["feat_std_f"]),
        state_offset=np.asarray(full_checkpoint["feat_mean_s"]),
        state_scale=np.asarray(full_checkpoint["feat_std_s"]),
        static_offset=np.asarray(full_checkpoint["static_mean"]),
        static_scale=np.asarray(full_checkpoint["static_std"]),
        target_offset=float(full_checkpoint["target_mean"]),
        target_scale=float(full_checkpoint["target_std"]),
        scale_target=True,
    )
    features = FeatureColumns(
        forcing=tuple(legacy_config["forcing_cols"]),
        state=tuple(legacy_config["state_cols"]),
        static=tuple(legacy_config["static_cols"]),
        target=legacy_config["target_col"],
        time=legacy_config["time_col"],
        land_cover=legacy_config["lc_col"],
    )
    splits = split_files_by_sites(
        sorted(args.data_dir.glob("*.csv")),
        legacy_config["train_sites"],
        legacy_config["val_sites"],
        legacy_config["test_sites"],
        strict=True,
    )
    selected_files = splits.val if args.split == "val" else splits.test
    started = time.perf_counter()
    dataset = MultiStationWindowDataset(
        selected_files,
        features,
        WindowConfig(
            seq_len=int(legacy_config["seq_len"]),
            time_features=TimeFeatureMode.CYCLIC,
            require_regular=True,
            max_gap_hours=1.0,
            max_span_hours=95.0,
        ),
        scaler=scaler,
        split_name=args.split,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_options["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_options)
    model = TCNTransformerCrossAttention(
        len(features.forcing),
        len(features.state),
        int(legacy_config["seq_len"]),
        num_static=len(features.static),
        time_feature_dim=4,
        num_lc_classes=int(legacy_config["num_lc_classes"]),
        lc_embed_dim=int(legacy_config["lc_embed_dim"]),
    ).to(device)
    state = torch.load(
        args.legacy_output / "checkpoint_best.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    result = evaluate_model(
        model,
        loader,
        device,
        scaler=scaler,
        minimum_target=None,
        amp=args.amp,
    )
    artifacts = save_evaluation_artifacts(
        result,
        args.output_dir,
        prefix=f"legacy_{args.split}",
        config=EvaluationConfig(
            save_predictions=True,
            save_plots=False,
            minimum_target=None,
        ),
    )
    frame = evaluation_frame(result)
    # Inverse scaling can turn an exact zero into a tiny negative float.
    nonnegative = frame[frame["target"] >= -1e-6]
    summary = {
        "split": args.split,
        "device": str(device),
        "files": len(selected_files),
        "windows": len(dataset),
        "loaded_files": [str(path) for path in dataset.loaded_files],
        "skipped_files": [
            {"path": str(path), "reason": reason}
            for path, reason in dataset.skipped_files
        ],
        "all_targets": {
            "micro": result.metrics,
            "macro": _macro_metrics(frame),
        },
        "nonnegative_targets": {
            "micro": regression_metrics(
                nonnegative["target"], nonnegative["prediction"]
            ),
            "macro": _macro_metrics(nonnegative),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": artifacts.as_dict(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"legacy_{args.split}_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

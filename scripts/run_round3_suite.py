"""Run locked-test third-round architecture screening and confirmation."""

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
    EvaluationConfig, ExperimentConfig, FeatureColumns, LossKind, ModelConfig,
    ModelKind, ScalingMethod, TimeFeatureMode, TrainingConfig, WindowConfig,
)
from gpp_inversion.ensemble import ensemble_prediction_files
from gpp_inversion.experiments import build_model
from gpp_inversion.metrics import regression_metrics
from gpp_inversion.pipeline import run_experiment


DETAIL_VARIANTS = (
    "pre_ln_encoder", "zero_init_gated_residual",
    "bidirectional_cross_attention", "gpp_query_pooling",
)
NEW_ARCHITECTURES = ("timexer", "modern_tcn", "time_mixer_pp")
SCREENING_VARIANTS = ("adamw_tcn_control", *DETAIL_VARIANTS, *NEW_ARCHITECTURES)
PARAMETER_CAP = int(226_537 * 1.5)


def summarize_predictions(path: Path, threshold: float) -> dict:
    frame = pd.read_csv(path)
    stations = []
    for station, group in frame.groupby("station", sort=True):
        row = {"station": station}
        row.update(regression_metrics(group.target, group.prediction))
        stations.append(row)
    high = frame[frame.target >= threshold]
    return {
        "micro": regression_metrics(frame.target, frame.prediction),
        "macro": {
            f"{key}_mean": float(np.nanmean([row[key] for row in stations]))
            for key in ("rmse", "mae", "r2", "bias")
        },
        "high_target": regression_metrics(high.target, high.prediction),
        "station_metrics": stations,
    }


def promotion_gate(candidate: dict, baseline: dict) -> dict:
    baseline_station = {row["station"]: row for row in baseline["station_metrics"]}
    station_wins = sum(
        row["rmse"] < baseline_station[row["station"]]["rmse"]
        for row in candidate["station_metrics"]
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
    return {
        "passed": bool(all(checks.values())),
        "station_rmse_wins": int(station_wins),
        "checks": checks,
    }


def base_config(args, legacy, output_dir, seed):
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
        output_dir=output_dir,
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
            learning_rate=1e-3,
            patience=args.patience,
            seed=seed,
            resume=False,
            selection_metric="macro_rmse",
            amp=args.amp,
            weight_decay=1e-4,
            deterministic=True,
            optimizer="adamw",
        ),
        evaluation=EvaluationConfig(
            save_predictions=True,
            save_plots=False,
            minimum_target=None,
            evaluate_test=False,
        ),
    )


def variant_config(base, name):
    model = base.model
    if name == "pre_ln_encoder":
        model = replace(model, state_norm_first=True)
    elif name == "zero_init_gated_residual":
        model = replace(model, cross_fusion_mode="zero_init_gated")
    elif name == "bidirectional_cross_attention":
        model = replace(
            model, cross_fusion_mode="bidirectional_gated", cross_direction="bidirectional"
        )
    elif name == "gpp_query_pooling":
        model = replace(model, temporal_pooling="gpp_query")
    elif name == "timexer":
        model = replace(model, kind=ModelKind.TIMEXER)
    elif name == "modern_tcn":
        model = replace(model, kind=ModelKind.MODERN_TCN)
    elif name == "time_mixer_pp":
        model = replace(model, kind=ModelKind.TIME_MIXER_PP)
    elif name.startswith("refined_tcn__"):
        selected = name.split("__", 1)[1].split("+")
        if "pre_ln_encoder" in selected:
            model = replace(model, state_norm_first=True)
        if "gpp_query_pooling" in selected:
            model = replace(model, temporal_pooling="gpp_query")
        zero = "zero_init_gated_residual" in selected
        bidirectional = "bidirectional_cross_attention" in selected
        if zero and bidirectional:
            model = replace(
                model,
                cross_fusion_mode="zero_init_bidirectional",
                cross_direction="bidirectional",
            )
        elif zero:
            model = replace(model, cross_fusion_mode="zero_init_gated")
        elif bidirectional:
            model = replace(
                model,
                cross_fusion_mode="bidirectional_gated",
                cross_direction="bidirectional",
            )
    config = replace(base, output_dir=base.output_dir.parent / name, model=model)
    candidate = build_model(
        model, base.features,
        seq_len=base.window.seq_len,
        time_feature_dim=base.window.time_feature_dim,
    )
    count = sum(parameter.numel() for parameter in candidate.parameters())
    if count > PARAMETER_CAP and model.kind is not ModelKind.TCN:
        model = replace(model, d_model=48)
        config = replace(config, model=model)
        candidate = build_model(
            model, base.features,
            seq_len=base.window.seq_len,
            time_feature_dim=base.window.time_feature_dim,
        )
        count = sum(parameter.numel() for parameter in candidate.parameters())
    if count > PARAMETER_CAP:
        raise ValueError(f"{name} has {count} parameters, above cap {PARAMETER_CAP}")
    del candidate
    return config, int(count)


def is_complete(directory: Path):
    path = directory / "experiment_manifest.json"
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"


def run_one(config, name, parameter_count):
    if is_complete(config.output_dir):
        manifest = json.loads((config.output_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
        result = {**manifest["result"], "artifacts": manifest["artifacts"]}
        elapsed = None
    else:
        started = time.perf_counter()
        result = run_experiment(config)
        elapsed = time.perf_counter() - started
    prediction = Path(result["artifacts"]["validation"]["predictions"])
    threshold = float(result["high_target_threshold"])
    return {
        "name": name,
        "elapsed_seconds": elapsed,
        "parameter_count": parameter_count,
        "config_hash": result["config_hash"],
        "high_target_threshold": threshold,
        "prediction_file": str(prediction),
        "experiment_dir": str(config.output_dir),
        "validation": summarize_predictions(prediction, threshold),
        "profile": result.get("architecture_profile"),
    }


def save_seed_summary(path, seed, records):
    control = records["adamw_tcn_control"]["validation"]
    for name, item in records.items():
        item["promotion_gate"] = (
            {"passed": True, "station_rmse_wins": 43, "checks": {}}
            if name == "adamw_tcn_control" else promotion_gate(item["validation"], control)
        )
    payload = {"test_set_locked": True, "seed": seed, "variants": records}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for name, item in records.items():
        value, gate = item["validation"], item["promotion_gate"]
        rows.append({
            "seed": seed, "variant": name, "config_hash": item["config_hash"],
            "parameter_count": item["parameter_count"],
            "elapsed_seconds": item["elapsed_seconds"],
            "micro_rmse": value["micro"]["rmse"],
            "micro_mae": value["micro"]["mae"],
            "macro_rmse": value["macro"]["rmse_mean"],
            "macro_mae": value["macro"]["mae_mean"],
            "high_gpp_rmse": value["high_target"]["rmse"],
            "high_gpp_mae": value["high_target"]["mae"],
            "station_rmse_wins": gate["station_rmse_wins"],
            "passed": gate["passed"],
            "failed_checks": ";".join(key for key, ok in gate["checks"].items() if not ok),
        })
    pd.DataFrame(rows).sort_values("macro_rmse").to_csv(
        path.with_suffix(".csv"), index=False, encoding="utf-8-sig"
    )
    return payload


def screening(args, legacy):
    root = args.output_dir / "screening_seed42"
    root.mkdir(parents=True, exist_ok=True)
    base = base_config(args, legacy, root / "adamw_tcn_control", 42)
    records = {}
    for name in SCREENING_VARIANTS:
        print(f"stage=screening seed=42 variant={name} status=starting", flush=True)
        config, count = variant_config(base, name)
        records[name] = run_one(config, name, count)
        save_seed_summary(root / "seed42_summary.json", 42, records)
        print(f"stage=screening seed=42 variant={name} status=completed", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    detail_passed = [
        name for name in DETAIL_VARIANTS
        if promotion_gate(records[name]["validation"], records["adamw_tcn_control"]["validation"])["passed"]
    ]
    if len(detail_passed) >= 2:
        detail_passed.sort(key=lambda name: records[name]["validation"]["macro"]["rmse_mean"])
        refined = "refined_tcn__" + "+".join(detail_passed[:2])
        config, count = variant_config(base, refined)
        print(f"stage=screening seed=42 variant={refined} status=starting", flush=True)
        records[refined] = run_one(config, refined, count)
        print(f"stage=screening seed=42 variant={refined} status=completed", flush=True)
    payload = save_seed_summary(root / "seed42_summary.json", 42, records)
    promoted = [
        name for name, item in payload["variants"].items()
        if name != "adamw_tcn_control" and item["promotion_gate"]["passed"]
    ]
    promoted.sort(key=lambda name: records[name]["validation"]["macro"]["rmse_mean"])
    selection = {"top_two": promoted[:2], "all_promoted": promoted}
    (root / "promotion_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(selection, ensure_ascii=False), flush=True)


def confirmation(args, legacy):
    if not args.candidates:
        selection = json.loads(
            (args.output_dir / "screening_seed42" / "promotion_selection.json").read_text(encoding="utf-8")
        )
        args.candidates = selection["top_two"]
    if not args.candidates:
        print("confirmation skipped: no promoted candidates", flush=True)
        return
    for seed in (7, 2026):
        root = args.output_dir / f"confirmation_seed{seed}"
        root.mkdir(parents=True, exist_ok=True)
        base = base_config(args, legacy, root / "adamw_tcn_control", seed)
        records = {}
        for name in ("adamw_tcn_control", *args.candidates):
            print(f"stage=confirmation seed={seed} variant={name} status=starting", flush=True)
            config, count = variant_config(base, name)
            records[name] = run_one(config, name, count)
            save_seed_summary(root / f"seed{seed}_summary.json", seed, records)
            print(f"stage=confirmation seed={seed} variant={name} status=completed", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        save_seed_summary(root / f"seed{seed}_summary.json", seed, records)


def finalize(args):
    selection = json.loads(
        (args.output_dir / "screening_seed42" / "promotion_selection.json").read_text(encoding="utf-8")
    )
    candidates = args.candidates or selection["top_two"]
    if not candidates:
        payload = {"final_candidate": None, "reason": "no_seed42_candidate_passed", "test_set_read": False}
        (args.output_dir / "final_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload), flush=True)
        return
    summaries = {
        42: json.loads((args.output_dir / "screening_seed42" / "seed42_summary.json").read_text(encoding="utf-8")),
        7: json.loads((args.output_dir / "confirmation_seed7" / "seed7_summary.json").read_text(encoding="utf-8")),
        2026: json.loads((args.output_dir / "confirmation_seed2026" / "seed2026_summary.json").read_text(encoding="utf-8")),
    }
    ensemble_root = args.output_dir / "validation_ensembles"
    ensemble_root.mkdir(parents=True, exist_ok=True)
    threshold = summaries[42]["variants"]["adamw_tcn_control"]["high_target_threshold"]
    control_files = [
        Path(summaries[seed]["variants"]["adamw_tcn_control"]["prediction_file"])
        for seed in (42, 7, 2026)
    ]
    control_manifest = ensemble_prediction_files(
        control_files, ensemble_root / "adamw_tcn_control", high_target_threshold=threshold
    )
    control_summary = summarize_predictions(Path(control_manifest["artifacts"]["predictions"]), threshold)
    original_summary = summarize_predictions(args.original_ensemble_predictions, threshold)
    candidate_results = {}
    for name in candidates:
        files = [Path(summaries[seed]["variants"][name]["prediction_file"]) for seed in (42, 7, 2026)]
        manifest = ensemble_prediction_files(
            files, ensemble_root / name, high_target_threshold=threshold
        )
        summary = summarize_predictions(Path(manifest["artifacts"]["predictions"]), threshold)
        seed_improvements = sum(
            summaries[seed]["variants"][name]["validation"]["macro"]["rmse_mean"]
            < summaries[seed]["variants"]["adamw_tcn_control"]["validation"]["macro"]["rmse_mean"]
            for seed in (42, 7, 2026)
        )
        gate_control = promotion_gate(summary, control_summary)
        gate_original = promotion_gate(summary, original_summary)
        passed = seed_improvements >= 2 and gate_control["passed"] and gate_original["passed"]
        candidate_results[name] = {
            "seed_macro_rmse_improvements": int(seed_improvements),
            "ensemble": summary,
            "gate_vs_adamw_control": gate_control,
            "gate_vs_original_tcn_ensemble": gate_original,
            "passed": passed,
            "manifest": manifest["manifest"],
        }
    passed = [name for name, item in candidate_results.items() if item["passed"]]
    passed.sort(key=lambda name: candidate_results[name]["ensemble"]["macro"]["rmse_mean"])
    payload = {
        "final_candidate": passed[0] if passed else None,
        "test_set_read": False,
        "control_ensemble": control_summary,
        "original_tcn_ensemble": original_summary,
        "candidates": candidate_results,
    }
    (args.output_dir / "final_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"final_candidate": payload["final_candidate"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_config", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("original_ensemble_predictions", type=Path)
    parser.add_argument("--stage", choices=("screening", "confirmation", "finalize"), required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--candidates", nargs="*")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    legacy = json.loads(args.legacy_config.read_text(encoding="utf-8"))
    if args.stage == "screening":
        screening(args, legacy)
    elif args.stage == "confirmation":
        confirmation(args, legacy)
    else:
        finalize(args)


if __name__ == "__main__":
    main()

"""Evaluate completed experiment checkpoints on one locked split without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from gpp_inversion.config import EvaluationConfig, ExperimentConfig
from gpp_inversion.data import BatchedWindowLoader, MultiStationWindowDataset, ScalingStats
from gpp_inversion.engine import evaluate_model
from gpp_inversion.experiments import build_model
from gpp_inversion.reporting import save_evaluation_artifacts
from gpp_inversion.splits import split_files_by_sites


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--high-target-threshold", type=float)
    args = parser.parse_args()

    source_manifest_path = args.experiment_dir / "experiment_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "completed":
        raise ValueError("Source experiment is not completed")
    config = ExperimentConfig.from_dict(
        {**source_manifest["config"]["experiment"], "data_dir": str(args.data_dir)}
    )
    splits = split_files_by_sites(
        args.data_dir.glob("*.csv"),
        config.train_sites,
        config.val_sites,
        config.test_sites,
        strict=True,
    )
    selected_files = splits.test if args.split == "test" else splits.val
    scaler = ScalingStats.load(args.experiment_dir / "scaler.npz")
    dataset = MultiStationWindowDataset(
        selected_files,
        config.features,
        config.window,
        scaler=scaler,
        split_name=args.split,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = BatchedWindowLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        metadata="full",
    )
    model = build_model(
        config.model,
        config.features,
        seq_len=config.window.seq_len,
        time_feature_dim=config.window.time_feature_dim,
    ).to(device)
    checkpoint_path = args.experiment_dir / "checkpoint_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("config_hash") != source_manifest.get("config_hash"):
        raise ValueError("Checkpoint and experiment manifest configuration hashes differ")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
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
        prefix=args.split,
        config=EvaluationConfig(save_predictions=True, save_plots=False, minimum_target=None),
        high_target_threshold=args.high_target_threshold,
    )
    payload = {
        "schema_version": 1,
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "source_experiment": str(args.experiment_dir.resolve()),
        "source_config_hash": source_manifest["config_hash"],
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
        },
        "input_files": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "modified_time_ns": path.stat().st_mtime_ns,
            }
            for path in selected_files
        ],
        "windows": len(dataset),
        "high_target_threshold": args.high_target_threshold,
        "metrics": result.metrics,
        "artifacts": artifacts.as_dict(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "split": payload["split"],
                "windows": payload["windows"],
                "metrics": payload["metrics"],
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

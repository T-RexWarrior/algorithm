from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gpp_inversion.config import ExperimentConfig
from gpp_inversion.data import BatchedWindowLoader, MultiStationWindowDataset
from gpp_inversion.experiments import build_model
from gpp_inversion.pretraining import pretrain_model
from gpp_inversion.splits import split_files_by_sites


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain production encoders without blind sites")
    parser.add_argument("config")
    parser.add_argument("output_dir")
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    files = split_files_by_sites(
        sorted(config.data_dir.glob("*.csv")),
        config.train_sites, (), (*config.val_sites, *config.test_sites),
        strict=True,
    )
    dataset = MultiStationWindowDataset(
        files.train, config.features, config.window,
        scaling=config.scaling, scale_target=config.scale_target,
        split_name="pretraining_train_only",
    )
    loader = BatchedWindowLoader(
        dataset, batch_size=config.training.batch_size,
        shuffle=True, seed=config.training.seed, metadata="none",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config.model, config.features, seq_len=config.window.seq_len,
        time_feature_dim=dataset.time_feature_dim,
    ).to(device)
    result = pretrain_model(model, loader, device, args.output_dir, max_steps=args.steps)
    print(result["checkpoint"])


if __name__ == "__main__":
    main()

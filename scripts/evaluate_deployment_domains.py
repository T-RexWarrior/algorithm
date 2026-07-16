from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gpp_inversion.config import DomainConfig, ExperimentConfig
from gpp_inversion.domain_evaluation import compare_on_common_rows, evaluate_checkpoint_domain


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint under four input domains")
    parser.add_argument("config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--stress-manifest", required=True)
    parser.add_argument("--modis-manifest", required=True)
    parser.add_argument("--exact-era-dir", required=True)
    parser.add_argument("--calibrated-era-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    root = Path(args.output)
    modis = str(Path(args.modis_manifest))
    domains = {
        "tower_tower": (DomainConfig(), config.data_dir),
        "tower_modis": (DomainConfig(land_cover_mode="modis", land_cover_manifest=modis), config.data_dir),
        "era_stress_modis": (
            DomainConfig(
                forcing_mode="era_stress", forcing_manifest=str(Path(args.stress_manifest)),
                land_cover_mode="modis", land_cover_manifest=modis,
            ),
            config.data_dir,
        ),
        "exact_era_modis": (
            DomainConfig(land_cover_mode="modis", land_cover_manifest=modis),
            Path(args.exact_era_dir),
        ),
    }
    if args.calibrated_era_dir:
        domains["calibrated_era_modis"] = (
            DomainConfig(land_cover_mode="modis", land_cover_manifest=modis),
            Path(args.calibrated_era_dir),
        )
    frames: dict[str, pd.DataFrame] = {}
    for name, (domain, data_dir) in domains.items():
        frames[name] = evaluate_checkpoint_domain(
            config, args.checkpoint, args.scaler, root / name,
            domain=domain, data_dir=data_dir,
        )
    result = compare_on_common_rows(frames, root / "common_domain_comparison.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gpp_inversion.config import ExperimentConfig
from gpp_inversion.domain import EraCalibrationTransform, fit_era_calibration_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize target-blind D4 calibrated ERA validation CSVs"
    )
    parser.add_argument("config")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--exact-era-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = fit_era_calibration_manifest(
        args.pairs, output / "era_to_tower_calibration.json"
    )
    transform = EraCalibrationTransform.load(manifest_path)
    forcing = list(config.features.forcing)
    report = {
        "uses_target": False,
        "source_directory": str(Path(args.exact_era_dir).resolve()),
        "calibration_manifest": str(manifest_path.resolve()),
        "files": {},
    }
    for source in sorted(Path(args.exact_era_dir).glob("*.csv")):
        frame = pd.read_csv(source, low_memory=False)
        missing = sorted(set(forcing) - set(frame.columns))
        if missing:
            raise ValueError(f"{source} misses forcing columns: {missing}")
        values = frame[forcing].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        frame.loc[:, forcing] = transform.apply(values, forcing)
        destination = output / source.name
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
        report["files"][source.name] = {
            "rows": int(len(frame)),
            "source_sha256": _sha256(source),
            "output_sha256": _sha256(destination),
        }
    report["file_count"] = len(report["files"])
    (output / "materialization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"file_count": report["file_count"]}))


if __name__ == "__main__":
    main()

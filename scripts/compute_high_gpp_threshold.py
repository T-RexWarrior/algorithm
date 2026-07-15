"""Compute the locked training-target 90th percentile without reading test data."""

import argparse
import json
from pathlib import Path

import numpy as np

from gpp_inversion.config import FeatureColumns, ScalingMethod, TimeFeatureMode, WindowConfig
from gpp_inversion.data import MultiStationWindowDataset
from gpp_inversion.splits import split_files_by_sites


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_config", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    legacy = json.loads(args.legacy_config.read_text(encoding="utf-8"))
    features = FeatureColumns(
        forcing=tuple(legacy["forcing_cols"]), state=tuple(legacy["state_cols"]),
        static=tuple(legacy["static_cols"]), target=legacy["target_col"],
        time=legacy["time_col"], land_cover=legacy["lc_col"],
    )
    splits = split_files_by_sites(
        args.data_dir.glob("*.csv"), legacy["train_sites"], legacy["val_sites"], legacy["test_sites"]
    )
    dataset = MultiStationWindowDataset(
        splits.train,
        features,
        WindowConfig(seq_len=int(legacy["seq_len"]), time_features=TimeFeatureMode.CYCLIC, require_regular=True, max_gap_hours=1.0, max_span_hours=95.0),
        scaling=ScalingMethod.ZSCORE,
        split_name="train",
    )
    payload = {
        "quantile": 0.9,
        "threshold": float(np.quantile(dataset.raw_window_targets(), 0.9)),
        "training_windows": len(dataset),
        "training_stations": len(dataset.station_names),
        "test_data_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

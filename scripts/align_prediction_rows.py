from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = ["station", "date", "land_cover_id", "target"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align two prediction files to identical station-hours"
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    columns = [*KEYS, "prediction"]
    baseline = pd.read_csv(args.baseline, usecols=columns)
    candidate = pd.read_csv(args.candidate, usecols=columns)
    if baseline.duplicated(KEYS).any() or candidate.duplicated(KEYS).any():
        raise ValueError("Prediction keys must be unique")
    merged = baseline.merge(
        candidate,
        on=KEYS,
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    ).sort_values(["station", "date"], kind="stable")
    if merged.empty:
        raise ValueError("Prediction files have no common station-hours")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_output = args.output / "baseline_common.csv"
    candidate_output = args.output / "candidate_common.csv"
    merged[[*KEYS, "prediction_baseline"]].rename(
        columns={"prediction_baseline": "prediction"}
    ).to_csv(baseline_output, index=False, encoding="utf-8-sig")
    merged[[*KEYS, "prediction_candidate"]].rename(
        columns={"prediction_candidate": "prediction"}
    ).to_csv(candidate_output, index=False, encoding="utf-8-sig")
    manifest = {
        "baseline_rows": int(len(baseline)),
        "candidate_rows": int(len(candidate)),
        "common_rows": int(len(merged)),
        "common_stations": int(merged["station"].nunique()),
        "baseline_common": str(baseline_output),
        "candidate_common": str(candidate_output),
    }
    (args.output / "alignment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

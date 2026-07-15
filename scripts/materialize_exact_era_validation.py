from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from gpp_inversion.config import ExperimentConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create target-preserving six-day CSVs with exact ERA forcing"
    )
    parser.add_argument("config")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--modis-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    pair_path = Path(args.pairs)
    pairs = pd.read_csv(pair_path)
    pairs = pairs[pairs["split"] == args.split].copy()
    pairs["date"] = pd.to_datetime(pairs["date"], utc=True)
    wide = pairs.pivot_table(
        index=["station", "date"], columns="feature", values="era", aggfunc="last"
    ).reset_index()
    forcing = list(config.features.forcing)
    missing = sorted(set(forcing) - set(wide.columns))
    if missing:
        raise ValueError(f"Pair manifest misses forcing features: {missing}")
    mapping_payload = json.loads(Path(args.modis_manifest).read_text(encoding="utf-8"))
    mapping = mapping_payload["sites"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sites = config.val_sites if args.split == "val" else config.train_sites
    report = {"split": args.split, "source_pairs_sha256": _sha256(pair_path), "sites": {}}

    for site in sites:
        site_pairs = wide[wide["station"] == site].copy()
        modis = int(mapping.get(site, {}).get("modis_veg_id", -1))
        if site_pairs.empty or modis < 0:
            continue
        matches = sorted(config.data_dir.glob(f"{site}*_Merged.csv"))
        if not matches:
            continue
        source = matches[0]
        frame = pd.read_csv(source, low_memory=False)
        frame["_date_utc"] = pd.to_datetime(
            frame[config.features.time], errors="coerce", utc=True
        )
        era = site_pairs[["date", *forcing]].rename(columns={"date": "_date_utc"})
        merged = frame.merge(era, on="_date_utc", how="inner", suffixes=("", "_era"))
        for feature in forcing:
            merged[feature] = merged.pop(f"{feature}_era")
        if config.features.land_cover:
            merged[config.features.land_cover] = modis
        merged = merged.sort_values("_date_utc").drop(columns=["_date_utc"])
        destination = output / source.name
        merged.to_csv(destination, index=False, encoding="utf-8-sig")
        report["sites"][site] = {
            "rows": int(len(merged)), "source": str(source),
            "source_sha256": _sha256(source), "output": str(destination),
        }
    report["site_count"] = len(report["sites"])
    report["row_count"] = sum(row["rows"] for row in report["sites"].values())
    (output / "materialization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"site_count": report["site_count"], "row_count": report["row_count"]}))


if __name__ == "__main__":
    main()

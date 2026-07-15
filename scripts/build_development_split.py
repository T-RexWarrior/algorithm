from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Build development/validation split around locked blind sites")
    parser.add_argument("blind_lock")
    parser.add_argument("previous_config")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    lock = json.loads(Path(args.blind_lock).read_text(encoding="utf-8"))
    previous = json.loads(Path(args.previous_config).read_text(encoding="utf-8"))
    blind = {row["site"] for row in lock["stations"]}
    # A previously generated split can be supplied directly when refreshing
    # the blind lock; preserve the original 42-site historical comparison set.
    legacy_test = set(
        previous.get("legacy_test_sites", previous.get("test_sites", []))
    )
    by_class: dict[int, list[str]] = {}
    for row in lock["eligible_stations"]:
        site = str(row["site"])
        if site in blind or site in legacy_test:
            continue
        by_class.setdefault(int(row["veg_id"]), []).append(site)
    rng = np.random.default_rng(args.seed)
    train, val = [], []
    for veg_id in sorted(by_class):
        sites = sorted(set(by_class[veg_id]))
        rng.shuffle(sites)
        val_count = max(1, int(round(len(sites) * args.validation_fraction)))
        val.extend(sites[:val_count])
        train.extend(sites[val_count:])
    payload = {
        "protocol_version": 1,
        "blind_split_hash": lock["split_hash"],
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "train_sites": sorted(train),
        "val_sites": sorted(val),
        "blind_test_sites": sorted(blind),
        "legacy_test_sites": sorted(legacy_test),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    payload["split_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

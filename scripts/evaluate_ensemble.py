"""Create a strict equal-weight ensemble from saved prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpp_inversion.ensemble import ensemble_prediction_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("prediction_files", type=Path, nargs="+")
    parser.add_argument("--prefix", default="val")
    parser.add_argument("--high-target-threshold", type=float)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()
    result = ensemble_prediction_files(
        args.prediction_files,
        args.output_dir,
        prefix=args.prefix,
        high_target_threshold=args.high_target_threshold,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

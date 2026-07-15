from __future__ import annotations

import argparse

from gpp_inversion.blind import build_blind_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the locked 60-station blind split")
    parser.add_argument("data_dir")
    parser.add_argument("output")
    parser.add_argument("--previous-config")
    parser.add_argument("--count", type=int, default=60)
    args = parser.parse_args()
    path = build_blind_lock(
        args.data_dir,
        args.output,
        previous_config=args.previous_config,
        count=args.count,
    )
    print(path)


if __name__ == "__main__":
    main()

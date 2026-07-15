from __future__ import annotations

import argparse

from gpp_inversion.packaging import export_model_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a production GPP model package")
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("scaler")
    parser.add_argument("destination")
    parser.add_argument("--split-hash", required=True)
    args = parser.parse_args()
    print(export_model_package(
        args.config, args.checkpoint, args.scaler, args.destination,
        split_hash=args.split_hash,
    ))


if __name__ == "__main__":
    main()

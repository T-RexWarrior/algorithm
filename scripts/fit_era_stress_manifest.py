from __future__ import annotations

import argparse

from gpp_inversion.domain import fit_era_stress_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit target-blind ERA forcing stress transform")
    parser.add_argument("pair_csv")
    parser.add_argument("output")
    args = parser.parse_args()
    print(fit_era_stress_manifest(args.pair_csv, args.output))


if __name__ == "__main__":
    main()

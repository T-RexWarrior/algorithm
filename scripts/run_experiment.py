"""Run one integrated GPP experiment from a JSON configuration."""

from __future__ import annotations

import argparse
import json

from gpp_inversion import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to an experiment JSON file")
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    result = run_experiment(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert locked station splits into a production config")
    parser.add_argument("template")
    parser.add_argument("split")
    parser.add_argument("output")
    args = parser.parse_args()
    config = json.loads(Path(args.template).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    config["train_sites"] = split["train_sites"]
    config["val_sites"] = split["val_sites"]
    config["test_sites"] = split["blind_test_sites"]
    config["split_protocol"] = {
        "name": "blind_v1",
        "split_hash": split["split_hash"],
        "blind_split_hash": split["blind_split_hash"],
        "legacy_test_sites": split["legacy_test_sites"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

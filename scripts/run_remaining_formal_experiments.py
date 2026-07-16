from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _run(script: Path, config: Path, root: Path, stage: str, extra: list[str]) -> None:
    command = [
        sys.executable, str(script), str(config), "--root", str(root),
        "--stage", stage, *extra,
    ]
    print(f"START {stage}: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)
    print(f"DONE {stage}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all remaining formal stages in order")
    parser.add_argument("config", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stress-manifest", type=Path, required=True)
    parser.add_argument("--modis-manifest", type=Path, required=True)
    parser.add_argument("--wait-hours", type=float, default=12.0)
    args = parser.parse_args()
    formal = Path(__file__).with_name("run_formal_experiments.py")
    expected_architecture_runs = 3 * 2
    deadline = time.monotonic() + args.wait_hours * 3600.0
    summary = args.root / "architecture_full_summary.json"
    while len(_rows(summary)) < expected_architecture_runs:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"architecture_full did not produce {expected_architecture_runs} rows"
            )
        print(
            f"WAIT architecture_full: {len(_rows(summary))}/{expected_architecture_runs}",
            flush=True,
        )
        time.sleep(60)

    domain_extra = [
        "--stress-manifest", str(args.stress_manifest),
        "--modis-manifest", str(args.modis_manifest),
    ]
    for stage, extra in (
        ("pretraining_proxy", []),
        ("pretraining_full", []),
        ("domain_proxy", domain_extra),
        ("domain_full", domain_extra),
        ("final_cv", []),
    ):
        _run(formal, args.config, args.root, stage, extra)
    print("ALL QUEUED FORMAL STAGES COMPLETE", flush=True)


if __name__ == "__main__":
    main()

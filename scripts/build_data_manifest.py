"""Build a lightweight, versionable inventory for external station CSV data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def csv_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return tuple(next(csv.reader(handle)))


def build_manifest(data_dir: Path) -> dict:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    records = []
    schemas: Counter[tuple[str, ...]] = Counter()
    for path in files:
        stat = path.stat()
        header = csv_header(path)
        schemas[header] += 1
        records.append(
            {
                "name": path.name,
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(),
                "column_count": len(header),
            }
        )

    schema_rows = sorted(
        (
            {"file_count": count, "columns": list(columns)}
            for columns, count in schemas.items()
        ),
        key=lambda row: (-row["file_count"], row["columns"]),
    )
    return {
        "manifest_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir.resolve()),
        "file_pattern": "*.csv",
        "file_count": len(records),
        "total_size_bytes": sum(row["size_bytes"] for row in records),
        "schema_consistent": len(schema_rows) == 1,
        "schemas": schema_rows,
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=Path(r"D:\实验五\全波段全变量DT"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real_data_manifest.json"),
    )
    args = parser.parse_args()
    manifest = build_manifest(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {args.output}: {manifest['file_count']} files, "
        f"{manifest['total_size_bytes']} bytes, "
        f"schema_consistent={manifest['schema_consistent']}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize three-seed age experiments")
    parser.add_argument("root")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root)
    names = ("tcn_baseline", "a0_mask_only", "a3_age_count_recency")
    seeds = (42, 7, 2026)
    summary: dict[str, dict] = {}
    station_frames = {}
    for name in names:
        rows = []
        for seed in seeds:
            directory = root / name / f"seed_{seed}"
            result = json.loads(
                (directory / "result_summary.json").read_text(encoding="utf-8")
            )
            high = json.loads(
                (directory / "evaluation" / "val_metrics_high_target.json").read_text(
                    encoding="utf-8"
                )
            )
            station = pd.read_csv(
                directory / "evaluation" / "val_metrics_by_station.csv"
            ).set_index("station")
            station_frames[(name, seed)] = station
            rows.append(
                {
                    "seed": seed,
                    "macro_rmse": float(station["rmse"].mean()),
                    "micro_rmse": float(result["val_metrics"]["rmse"]),
                    "micro_mae": float(result["val_metrics"]["mae"]),
                    "bias": float(result["val_metrics"]["bias"]),
                    "high_rmse": float(high["metrics"]["rmse"]),
                    "high_mae": float(high["metrics"]["mae"]),
                    "high_bias": float(high["metrics"]["bias"]),
                }
            )
        metric_names = tuple(key for key in rows[0] if key != "seed")
        summary[name] = {
            "seeds": rows,
            "mean": {
                key: float(np.mean([row[key] for row in rows]))
                for key in metric_names
            },
            "std_macro_rmse": float(
                np.std([row["macro_rmse"] for row in rows], ddof=1)
            ),
        }
    baseline = "tcn_baseline"
    for name in names[1:]:
        wins = []
        for seed in seeds:
            base, candidate = station_frames[(baseline, seed)].align(
                station_frames[(name, seed)], join="inner"
            )
            wins.append(
                {
                    "seed": seed,
                    "wins": int((candidate["rmse"] < base["rmse"]).sum()),
                    "stations": int(len(base)),
                }
            )
        summary[name]["station_wins_vs_tcn"] = wins
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

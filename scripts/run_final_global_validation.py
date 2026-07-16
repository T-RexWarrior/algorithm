from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from gpp_inversion.contracts import sha256_file
from gpp_inversion.packaging import export_model_package


def _wait(path: Path, wait_hours: float) -> None:
    deadline = time.monotonic() + wait_hours * 3600.0
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        print(f"WAIT postformal completion: {path}", flush=True)
        time.sleep(60)


def _run(command: list[str], *, cwd: Path) -> None:
    print(f"START {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)
    print(f"DONE {' '.join(command)}", flush=True)


def _export_package(experiment: Path, destination: Path, split_hash: str) -> Path:
    manifest = destination / "model_package.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("files", {}).get("checkpoint.pth") == sha256_file(
            experiment / "checkpoint_best.pth"
        ):
            return manifest
    return export_model_package(
        experiment / "resolved_config.json",
        experiment / "checkpoint_best.pth",
        experiment / "scaler.npz",
        destination,
        split_hash=split_hash,
    )


def _write_global_config(
    base: dict,
    path: Path,
    *,
    name: str,
    package: Path,
    output: Path,
    diagnostic_points: int | None,
) -> Path:
    payload = json.loads(json.dumps(base))
    payload["project"].update(
        name=name,
        start="2022-07-19 00:00:00",
        end="2022-07-19 23:00:00",
        timezone="UTC",
    )
    payload["paths"].update(
        model_package=str(package),
        checkpoint=str(package / "checkpoint.pth"),
        scaler=str(package / "global_scalers.npz"),
        output_dir=str(output),
        regular_zarr=str(output / "gpp_0p1deg_hourly.zarr"),
    )
    processing = payload["processing"]
    processing.update(
        mode="regular_grid_hourly",
        resume=True,
        overwrite_outputs=False,
        use_epic_history=True,
        build_missing_epic_history_cache=True,
        history_hours=96,
        grid_resolution_degrees=0.1,
        grid_tile_rows=128,
        grid_tile_cols=256,
        batch_size=4096,
        inference_precision="fp16",
    )
    processing.pop("diagnostic_sample_points", None)
    processing.pop("diagnostic_sample_seed", None)
    if diagnostic_points is not None:
        processing["diagnostic_sample_points"] = int(diagnostic_points)
        processing["diagnostic_sample_seed"] = 42
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def _global_run(global_repo: Path, config: Path, output: Path) -> dict:
    summary = output / "regular_grid_summary.json"
    _run(
        [sys.executable, str(global_repo / "scripts" / "inspect_inputs.py"), "--config", str(config)],
        cwd=global_repo,
    )
    if not summary.exists():
        _run(
            [sys.executable, str(global_repo / "scripts" / "run_inversion.py"), "--config", str(config)],
            cwd=global_repo,
        )
        _run(
            [
                sys.executable,
                str(global_repo / "scripts" / "summarize_regular_grid.py"),
                str(output / "gpp_0p1deg_hourly.zarr"),
                str(summary),
            ],
            cwd=global_repo,
        )
    return json.loads(summary.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export final packages and run 100k/full global 24-hour checks"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--global-repo", type=Path, required=True)
    parser.add_argument("--wait-hours", type=float, default=72.0)
    args = parser.parse_args()
    completion = args.root / "postformal" / "oof_ensemble" / "oof_summary.json"
    _wait(completion, args.wait_hours)

    reference = args.root / "age_full" / "tcn_baseline" / "seed_42"
    candidate = args.root / "pretraining_full" / "seed_42" / "supervised"
    config_payload = json.loads((reference / "resolved_config.json").read_text(encoding="utf-8"))
    split_hash = str(config_payload["split_protocol"]["split_hash"])
    packages = {
        "reference": args.root / "model_packages" / "reference",
        "masked_pretraining": args.root / "model_packages" / "masked_pretraining",
    }
    _export_package(reference, packages["reference"], split_hash)
    _export_package(candidate, packages["masked_pretraining"], split_hash)

    base = yaml.safe_load(
        (args.global_repo / "configs" / "global_regular_hourly_production.yaml").read_text(
            encoding="utf-8"
        )
    )
    reports = {"blind_test_opened": False, "runs": {}}
    for mode, points in (("diagnostic_100k", 100_000), ("full_global", None)):
        for model_name, package in packages.items():
            output = args.root / "global_validation" / mode / model_name
            config = _write_global_config(
                base,
                args.global_repo / "configs" / f"formal_{mode}_{model_name}.yaml",
                name=f"formal_{mode}_{model_name}",
                package=package,
                output=output,
                diagnostic_points=points,
            )
            reports["runs"][f"{mode}/{model_name}"] = _global_run(
                args.global_repo, config, output
            )
    destination = args.root / "global_validation" / "global_validation_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("FINAL GLOBAL VALIDATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()

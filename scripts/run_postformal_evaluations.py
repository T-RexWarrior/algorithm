from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from gpp_inversion.ensemble import ensemble_prediction_files, fit_nonnegative_oof_weights
from gpp_inversion.promotion import evaluate_promotion, write_promotion_report


DOMAINS = (
    "tower_tower",
    "tower_modis",
    "era_stress_modis",
    "exact_era_modis",
    "calibrated_era_modis",
)


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _wait_for_final_cv(root: Path, expected: int, wait_hours: float) -> None:
    deadline = time.monotonic() + wait_hours * 3600.0
    path = root / "final_cv_summary.json"
    while len(_rows(path)) < expected:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"final CV did not produce {expected} completed runs")
        print(f"WAIT final_cv: {len(_rows(path))}/{expected}", flush=True)
        time.sleep(60)


def _run_domain_evaluation(
    project: Path,
    experiment: Path,
    output: Path,
    *,
    stress_manifest: Path,
    modis_manifest: Path,
    exact_era_dir: Path,
    calibrated_era_dir: Path,
) -> None:
    completed = output / "common_domain_comparison.json"
    if completed.exists():
        print(f"SKIP deployment evaluation: {output}", flush=True)
        return
    required = {
        "config": experiment / "resolved_config.json",
        "checkpoint": experiment / "checkpoint_best.pth",
        "scaler": experiment / "scaler.npz",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Experiment misses deployable artifacts: {missing}")
    command = [
        sys.executable,
        str(project / "scripts" / "evaluate_deployment_domains.py"),
        str(required["config"]),
        "--checkpoint", str(required["checkpoint"]),
        "--scaler", str(required["scaler"]),
        "--stress-manifest", str(stress_manifest),
        "--modis-manifest", str(modis_manifest),
        "--exact-era-dir", str(exact_era_dir),
        "--calibrated-era-dir", str(calibrated_era_dir),
        "--output", str(output),
    ]
    print(f"START deployment evaluation: {experiment}", flush=True)
    subprocess.run(command, cwd=project, check=True)
    print(f"DONE deployment evaluation: {experiment}", flush=True)


def _prediction(root: Path, domain: str) -> Path:
    return root / domain / "evaluation" / "domain_predictions.csv"


def _paired_report(baseline: Path, candidate: Path, output: Path) -> dict:
    target = pd.read_csv(baseline, usecols=["target"])["target"]
    threshold = float(target.quantile(0.95))
    report = evaluate_promotion(
        baseline,
        candidate,
        high_target_threshold=threshold,
        bootstrap_samples=1000,
        seed=42,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_promotion_report(report, output)
    return report


def _combine_fold_predictions(seed_root: Path, output: Path) -> Path:
    sources = sorted(seed_root.glob("fold_*/evaluation/val_predictions.csv"))
    if len(sources) != 5:
        raise ValueError(f"Expected five OOF fold files under {seed_root}, found {len(sources)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as destination:
        for index, source in enumerate(sources):
            with source.open("rb") as handle:
                if index:
                    handle.readline()
                shutil.copyfileobj(handle, destination, length=4 * 1024 * 1024)
    return output


def _build_oof_ensemble(root: Path, output: Path) -> dict:
    candidate_files: dict[str, list[Path]] = {}
    for name in ("reference", "masked_pretraining"):
        paths = []
        for seed in (42, 7, 2026):
            paths.append(
                _combine_fold_predictions(
                    root / "final_cv" / name / f"seed_{seed}",
                    output / "combined" / f"{name}_seed_{seed}.csv",
                )
            )
        candidate_files[name] = paths

    threshold = float(
        pd.read_csv(candidate_files["reference"][0], usecols=["target"])["target"].quantile(0.95)
    )
    candidate_ensembles = {}
    for name, paths in candidate_files.items():
        manifest = ensemble_prediction_files(
            paths,
            output / name,
            prefix="oof",
            high_target_threshold=threshold,
        )
        candidate_ensembles[name] = Path(manifest["artifacts"]["predictions"])

    aligned = [candidate_ensembles["reference"], candidate_ensembles["masked_pretraining"]]
    weights = fit_nonnegative_oof_weights(aligned, seed=42)
    final_manifest = ensemble_prediction_files(
        aligned,
        output / "nonnegative_final",
        prefix="oof",
        high_target_threshold=threshold,
        weights=weights,
    )
    promotion = _paired_report(
        candidate_ensembles["reference"],
        candidate_ensembles["masked_pretraining"],
        output / "pretraining_vs_reference_promotion.json",
    )
    payload = {
        "blind_test_opened": False,
        "candidate_seed_ensembles": {
            name: str(path) for name, path in candidate_ensembles.items()
        },
        "nonnegative_weights": {
            "reference": float(weights[0]),
            "masked_pretraining": float(weights[1]),
        },
        "final_manifest": final_manifest,
        "pretraining_promotion": promotion,
    }
    (output / "oof_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run post-formal deployment replay, paired gates and OOF ensemble"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stress-manifest", type=Path, required=True)
    parser.add_argument("--modis-manifest", type=Path, required=True)
    parser.add_argument("--exact-era-dir", type=Path, required=True)
    parser.add_argument("--calibrated-era-dir", type=Path, required=True)
    parser.add_argument("--wait-hours", type=float, default=30.0)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    _wait_for_final_cv(args.root, expected=6, wait_hours=args.wait_hours)

    formal = project / "scripts" / "run_formal_experiments.py"
    base_config = project / "configs" / "production_observation_aware.json"
    domain_extra = [
        "--stress-manifest", str(args.stress_manifest),
        "--modis-manifest", str(args.modis_manifest),
    ]
    for stage in ("domain_corrected_proxy", "domain_corrected_full"):
        subprocess.run(
            [
                sys.executable, str(formal), str(base_config),
                "--root", str(args.root), "--stage", stage, *domain_extra,
            ],
            cwd=project,
            check=True,
        )

    evaluations = args.root / "postformal" / "deployment"
    experiments: dict[str, Path] = {}
    for name in ("d0_tower_tower", "d1_tower_modis", "d2_era_stress_modis", "d3_mixed_modis"):
        experiments[f"domain_corrected_proxy/{name}"] = args.root / "domain_corrected_proxy" / name
    for seed in (42, 7, 2026):
        experiments[f"reference/seed_{seed}"] = (
            args.root / "age_full" / "tcn_baseline" / f"seed_{seed}"
        )
        experiments[f"masked_pretraining/seed_{seed}"] = (
            args.root / "pretraining_full" / f"seed_{seed}" / "supervised"
        )
    for name, experiment in experiments.items():
        _run_domain_evaluation(
            project,
            experiment,
            evaluations / name,
            stress_manifest=args.stress_manifest,
            modis_manifest=args.modis_manifest,
            exact_era_dir=args.exact_era_dir,
            calibrated_era_dir=args.calibrated_era_dir,
        )

    paired = {}
    proxy_base = evaluations / "domain_corrected_proxy" / "d0_tower_tower"
    for candidate in ("d1_tower_modis", "d2_era_stress_modis", "d3_mixed_modis"):
        for domain in DOMAINS:
            key = f"domain_corrected_proxy/{candidate}/{domain}"
            paired[key] = _paired_report(
                _prediction(proxy_base, domain),
                _prediction(evaluations / "domain_corrected_proxy" / candidate, domain),
                args.root / "postformal" / "promotion" / f"{candidate}__{domain}.json",
            )
    for seed in (42, 7, 2026):
        baseline = evaluations / "reference" / f"seed_{seed}"
        candidate = evaluations / "masked_pretraining" / f"seed_{seed}"
        for domain in DOMAINS:
            key = f"masked_pretraining/seed_{seed}/{domain}"
            paired[key] = _paired_report(
                _prediction(baseline, domain),
                _prediction(candidate, domain),
                args.root / "postformal" / "promotion" / f"pretraining_seed_{seed}__{domain}.json",
            )
    (args.root / "postformal" / "paired_promotion_summary.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _build_oof_ensemble(args.root, args.root / "postformal" / "oof_ensemble")
    print("POSTFORMAL EVALUATIONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()

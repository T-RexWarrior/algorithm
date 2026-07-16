from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, replace
from enum import Enum
from pathlib import Path

import numpy as np
import torch

from gpp_inversion.config import DomainConfig, ExperimentConfig, LossKind, ModelKind
from gpp_inversion.data import BatchedWindowLoader, MultiStationWindowDataset
from gpp_inversion.experiments import build_model
from gpp_inversion.pipeline import run_experiment
from gpp_inversion.pretraining import pretrain_model
from gpp_inversion.splits import split_files_by_sites


AGE_VARIANTS = {
    "a0_mask_only": (False, False, False),
    "a1_endpoint_age": (True, False, False),
    "a2_age_count": (True, True, False),
    "a3_age_count_recency": (True, True, True),
}


def _default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(type(value).__name__)


def _write_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(config), default=_default, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _completed(output: Path) -> dict | None:
    for name in ("result_summary.json", "cross_validation_summary.json"):
        path = output / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _run(config: ExperimentConfig) -> dict:
    _write_config(config, config.output_dir / "resolved_config.json")
    existing = _completed(config.output_dir)
    if existing is not None:
        print(f"skip completed: {config.output_dir}")
        result = existing
    else:
        result = run_experiment(config)
    station_metrics = config.output_dir / "evaluation" / "val_metrics_by_station.csv"
    if station_metrics.exists():
        with station_metrics.open("r", encoding="utf-8-sig", newline="") as handle:
            values = [float(row["rmse"]) for row in csv.DictReader(handle)]
        if values:
            result["comparison_macro_rmse"] = sum(values) / len(values)
    return result


def _training(base: ExperimentConfig, *, steps: int, seed: int):
    return replace(
        base.training,
        batch_size=1024, optimizer="adamw", learning_rate=1e-3,
        weight_decay=1e-4, max_steps=steps, warmup_steps=500,
        eval_interval_steps=1000, patience=3, seed=seed, resume=True,
        target_balanced=True, station_balanced=False,
        samples_per_epoch=1024 * 1000, amp=True,
    )


def _base(base: ExperimentConfig, output: Path, *, steps: int, seed: int):
    return replace(
        base, output_dir=output,
        window=replace(base.window, endpoint_stride=3, endpoint_phase=0, context_days=0),
        training=_training(base, steps=steps, seed=seed),
        evaluation=replace(
            base.evaluation, save_predictions=True, save_plots=False,
            evaluate_test=False,
        ),
        cross_validation=replace(base.cross_validation, enabled=False),
        domain=DomainConfig(),
        loss=LossKind.MSE, loss_options={}, scale_target=True,
    )


def _age_config(base: ExperimentConfig, output: Path, name: str, *, steps: int, seed: int):
    config = _base(base, output, steps=steps, seed=seed)
    if name == "tcn_baseline":
        return replace(config, model=replace(base.model, kind=ModelKind.TCN))
    endpoint_age, count, recency = AGE_VARIANTS[name]
    return replace(
        config,
        model=replace(
            base.model, kind=ModelKind.TCN_OBSERVATION_AWARE,
            use_endpoint_observation_age=endpoint_age,
            use_observation_count=count, use_token_recency=recency,
        ),
    )


def _score(result: dict) -> float:
    return float(
        result.get("comparison_macro_rmse", result["training"]["best_selection_score"])
    )


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def run_age_proxy(base: ExperimentConfig, root: Path) -> list[dict]:
    rows = []
    for name in ("tcn_baseline", *AGE_VARIANTS):
        result = _run(_age_config(base, root / "age_proxy" / name, name, steps=3000, seed=42))
        rows.append({"name": name, "seed": 42, "steps": 3000, "score": _score(result)})
        _write_summary(root / "age_proxy_summary.json", rows)
    return rows


def _best_age(root: Path) -> str:
    path = root / "age_proxy_summary.json"
    if not path.exists():
        raise ValueError("age_proxy must complete before later stages")
    rows = json.loads(path.read_text(encoding="utf-8"))
    age = [row for row in rows if row["name"] in AGE_VARIANTS]
    return min(age, key=lambda row: row["score"])["name"]


def run_age_full(base: ExperimentConfig, root: Path) -> list[dict]:
    best = _best_age(root)
    rows = []
    for name in ("tcn_baseline", "a0_mask_only", best):
        for seed in (42, 7, 2026):
            output = root / "age_full" / name / f"seed_{seed}"
            result = _run(_age_config(base, output, name, steps=12000, seed=seed))
            rows.append({"name": name, "seed": seed, "steps": 12000, "score": _score(result)})
            _write_summary(root / "age_full_summary.json", rows)
    return rows


def _winning_age_config(base: ExperimentConfig, root: Path, output: Path, *, steps: int, seed: int):
    return _age_config(base, output, _best_age(root), steps=steps, seed=seed)


def run_architecture_proxy(base: ExperimentConfig, root: Path) -> list[dict]:
    reference = _winning_age_config(base, root, root / "architecture_proxy" / "reference", steps=3000, seed=42)
    candidates = {
        "reference": reference,
        "multiscale_30d": replace(
            reference, output_dir=root / "architecture_proxy" / "multiscale_30d",
            window=replace(reference.window, context_days=30),
            model=replace(reference.model, kind=ModelKind.TCN_MULTISCALE),
        ),
        "hybrid_lue": replace(
            reference, output_dir=root / "architecture_proxy" / "hybrid_lue",
            model=replace(reference.model, kind=ModelKind.HYBRID_LUE_TCN),
            scale_target=False,
        ),
        "tail_aware": replace(
            reference, output_dir=root / "architecture_proxy" / "tail_aware",
            loss=LossKind.TAIL_AWARE,
            loss_options={"weights": [1.0, 1.0, 1.5, 2.5], "underprediction_weight": 0.25},
        ),
    }
    rows = []
    for name, config in candidates.items():
        result = _run(config)
        rows.append({"name": name, "seed": 42, "steps": 3000, "score": _score(result)})
        _write_summary(root / "architecture_proxy_summary.json", rows)
    return rows


def _architecture_config(
    base: ExperimentConfig, root: Path, name: str, output: Path, *, steps: int, seed: int
) -> ExperimentConfig:
    reference = _winning_age_config(base, root, output, steps=steps, seed=seed)
    if name == "reference":
        return reference
    if name == "multiscale_30d":
        return replace(
            reference, window=replace(reference.window, context_days=30),
            model=replace(reference.model, kind=ModelKind.TCN_MULTISCALE),
        )
    if name == "hybrid_lue":
        return replace(
            reference, model=replace(reference.model, kind=ModelKind.HYBRID_LUE_TCN),
            scale_target=False,
        )
    if name == "tail_aware":
        return replace(
            reference, loss=LossKind.TAIL_AWARE,
            loss_options={"weights": [1.0, 1.0, 1.5, 2.5], "underprediction_weight": 0.25},
        )
    raise ValueError(f"Unknown architecture candidate: {name}")


def _architecture_survivors(root: Path) -> list[str]:
    rows = json.loads((root / "architecture_proxy_summary.json").read_text(encoding="utf-8"))
    reference = next(row["score"] for row in rows if row["name"] == "reference")
    return [
        row["name"] for row in rows
        if row["name"] == "reference" or row["score"] <= reference * 1.02
    ]


def run_architecture_full(base: ExperimentConfig, root: Path) -> list[dict]:
    rows = []
    for name in _architecture_survivors(root):
        for seed in (42, 7, 2026):
            output = root / "architecture_full" / name / f"seed_{seed}"
            result = _run(_architecture_config(base, root, name, output, steps=12000, seed=seed))
            rows.append({"name": name, "seed": seed, "steps": 12000, "score": _score(result)})
            _write_summary(root / "architecture_full_summary.json", rows)
    return rows


def _grouped_mean(path: Path) -> dict[str, float]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get("status") == "eliminated_at_proxy":
            continue
        grouped.setdefault(row["name"], []).append(float(row["score"]))
    return {
        name: float(sum(values) / len(values)) for name, values in grouped.items()
    }


def _final_candidates(root: Path) -> list[str]:
    architecture = _grouped_mean(root / "architecture_full_summary.json")
    reference = architecture["reference"]
    eligible = {
        name: score for name, score in architecture.items()
        if name != "reference" and score <= reference * 0.99
    }
    pretraining_path = root / "pretraining_full_summary.json"
    if pretraining_path.exists():
        pretraining = _grouped_mean(pretraining_path)
        score = pretraining.get("masked_pretraining")
        if score is not None and score <= reference * 0.99:
            eligible["masked_pretraining"] = score
    # The baseline is mandatory. At most one genuinely promoted candidate is
    # carried into the expensive five-fold/three-seed confirmation.
    selected = min(eligible, key=eligible.get) if eligible else None
    return ["reference", *([selected] if selected else [])]


def _final_candidate_config(
    base: ExperimentConfig, root: Path, name: str, output: Path, *, seed: int
) -> ExperimentConfig:
    if name == "masked_pretraining":
        config = _architecture_config(
            base, root, "reference", output, steps=12000, seed=seed
        )
        return replace(
            config,
            training=replace(
                config.training,
                pretrained_checkpoint=None,
                pretraining_steps=3000,
            ),
        )
    return _architecture_config(base, root, name, output, steps=12000, seed=seed)


def run_final_cv(base: ExperimentConfig, root: Path) -> list[dict]:
    rows = []
    for name in _final_candidates(root):
        for seed in (42, 7, 2026):
            output = root / "final_cv" / name / f"seed_{seed}"
            config = _final_candidate_config(base, root, name, output, seed=seed)
            config = replace(
                config,
                cross_validation=replace(
                    config.cross_validation, enabled=True, n_splits=5,
                    seed=42, evaluate_test_each_fold=False,
                ),
            )
            result = _run(config)
            rows.append({
                "name": name, "seed": seed, "steps": 12000,
                "macro_rmse": result["validation_summary"]["rmse"]["mean"],
                "micro_count": result["validation_summary"]["total_count"],
            })
            _write_summary(root / "final_cv_summary.json", rows)
    return rows


def _ensure_pretraining(
    config: ExperimentConfig, output: Path, *, steps: int, seed: int
) -> Path:
    checkpoint = output / "pretrained_encoder.pth"
    if checkpoint.exists() and (output / "pretraining_manifest.json").exists():
        return checkpoint
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    files = split_files_by_sites(
        config.data_dir.glob("*.csv"), config.train_sites, (),
        (*config.val_sites, *config.test_sites), strict=True,
    )
    dataset = MultiStationWindowDataset(
        files.train, config.features, config.window, scaling=config.scaling,
        scale_target=config.scale_target, split_name="pretraining_train_only",
        domain=config.domain,
    )
    loader = BatchedWindowLoader(
        dataset, batch_size=config.training.batch_size, shuffle=True,
        seed=seed, pin_memory=torch.cuda.is_available(), metadata="none",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config.model, config.features, seq_len=config.window.seq_len,
        time_feature_dim=dataset.time_feature_dim,
    ).to(device)
    pretrain_model(model, loader, device, output, max_steps=steps)
    return checkpoint


def run_pretraining_proxy(base: ExperimentConfig, root: Path) -> list[dict]:
    output = root / "pretraining_proxy" / "supervised"
    config = _winning_age_config(base, root, output, steps=3000, seed=42)
    checkpoint = _ensure_pretraining(
        config, root / "pretraining_proxy" / "encoder", steps=3000, seed=42
    )
    config = replace(
        config, training=replace(config.training, pretrained_checkpoint=str(checkpoint))
    )
    result = _run(config)
    rows = [{"name": "masked_pretraining", "seed": 42, "steps": 3000, "score": _score(result)}]
    _write_summary(root / "pretraining_proxy_summary.json", rows)
    return rows


def run_pretraining_full(base: ExperimentConfig, root: Path) -> list[dict]:
    proxy = json.loads((root / "pretraining_proxy_summary.json").read_text(encoding="utf-8"))[0]
    architecture = json.loads((root / "architecture_proxy_summary.json").read_text(encoding="utf-8"))
    reference = next(row["score"] for row in architecture if row["name"] == "reference")
    if proxy["score"] > reference * 1.02:
        rows = [{**proxy, "status": "eliminated_at_proxy"}]
        _write_summary(root / "pretraining_full_summary.json", rows)
        return rows
    rows = []
    for seed in (42, 7, 2026):
        output = root / "pretraining_full" / f"seed_{seed}" / "supervised"
        config = _winning_age_config(base, root, output, steps=12000, seed=seed)
        checkpoint = _ensure_pretraining(
            config, root / "pretraining_full" / f"seed_{seed}" / "encoder",
            steps=3000, seed=seed,
        )
        config = replace(
            config, training=replace(config.training, pretrained_checkpoint=str(checkpoint))
        )
        result = _run(config)
        rows.append({"name": "masked_pretraining", "seed": seed, "steps": 12000, "score": _score(result)})
        _write_summary(root / "pretraining_full_summary.json", rows)
    return rows


def _mapped_sites(base: ExperimentConfig, mapping_path: Path):
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))["sites"]
    mapped = {site for site, row in payload.items() if int(row["modis_veg_id"]) >= 0}
    return (
        tuple(site for site in base.train_sites if site in mapped),
        tuple(site for site in base.val_sites if site in mapped),
    )


def run_domain_proxy(
    base: ExperimentConfig, root: Path,
    stress_manifest: Path, modis_manifest: Path,
) -> list[dict]:
    train_sites, val_sites = _mapped_sites(base, modis_manifest)
    reference = _winning_age_config(base, root, root / "domain_proxy" / "d0_tower_tower", steps=3000, seed=42)
    reference = replace(reference, train_sites=train_sites, val_sites=val_sites)
    configs = {
        "d0_tower_tower": reference,
        "d1_tower_modis": replace(
            reference, output_dir=root / "domain_proxy" / "d1_tower_modis",
            domain=DomainConfig(
                land_cover_mode="modis", land_cover_manifest=str(modis_manifest), seed=42
            ),
        ),
        "d2_era_stress_modis": replace(
            reference, output_dir=root / "domain_proxy" / "d2_era_stress_modis",
            domain=DomainConfig(
                forcing_mode="era_stress", forcing_manifest=str(stress_manifest),
                land_cover_mode="modis", land_cover_manifest=str(modis_manifest), seed=42,
            ),
        ),
        "d3_mixed_modis": replace(
            reference, output_dir=root / "domain_proxy" / "d3_mixed_modis",
            domain=DomainConfig(
                forcing_mode="mixed", forcing_manifest=str(stress_manifest), mixed_probability=0.5,
                land_cover_mode="modis", land_cover_manifest=str(modis_manifest), seed=42,
            ),
        ),
    }
    rows = []
    for name, config in configs.items():
        result = _run(config)
        rows.append({"name": name, "seed": 42, "steps": 3000, "score": _score(result)})
        _write_summary(root / "domain_proxy_summary.json", rows)
    return rows


def _domain_config(
    base: ExperimentConfig, root: Path, name: str, output: Path, *, steps: int,
    seed: int, stress_manifest: Path, modis_manifest: Path,
) -> ExperimentConfig:
    train_sites, val_sites = _mapped_sites(base, modis_manifest)
    config = _winning_age_config(base, root, output, steps=steps, seed=seed)
    config = replace(config, train_sites=train_sites, val_sites=val_sites)
    if name == "d0_tower_tower":
        return config
    if name == "d1_tower_modis":
        domain = DomainConfig(
            land_cover_mode="modis", land_cover_manifest=str(modis_manifest), seed=seed
        )
    elif name == "d2_era_stress_modis":
        domain = DomainConfig(
            forcing_mode="era_stress", forcing_manifest=str(stress_manifest),
            land_cover_mode="modis", land_cover_manifest=str(modis_manifest), seed=seed,
        )
    elif name == "d3_mixed_modis":
        domain = DomainConfig(
            forcing_mode="mixed", forcing_manifest=str(stress_manifest), mixed_probability=0.5,
            land_cover_mode="modis", land_cover_manifest=str(modis_manifest), seed=seed,
        )
    else:
        raise ValueError(f"Unknown domain candidate: {name}")
    return replace(config, domain=domain)


def _domain_survivors(root: Path) -> list[str]:
    rows = json.loads((root / "domain_proxy_summary.json").read_text(encoding="utf-8"))
    reference = next(row["score"] for row in rows if row["name"] == "d0_tower_tower")
    return [
        row["name"] for row in rows
        if row["name"] == "d0_tower_tower" or row["score"] <= reference * 1.02
    ]


def run_domain_full(
    base: ExperimentConfig, root: Path, stress_manifest: Path, modis_manifest: Path,
) -> list[dict]:
    rows = []
    for name in _domain_survivors(root):
        for seed in (42, 7, 2026):
            output = root / "domain_full" / name / f"seed_{seed}"
            config = _domain_config(
                base, root, name, output, steps=12000, seed=seed,
                stress_manifest=stress_manifest, modis_manifest=modis_manifest,
            )
            result = _run(config)
            rows.append({"name": name, "seed": seed, "steps": 12000, "score": _score(result)})
            _write_summary(root / "domain_full_summary.json", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable formal GPP experiment stages")
    parser.add_argument("config")
    parser.add_argument("--root", default=r"D:\全局反演\gpp_production_experiments_v2")
    parser.add_argument(
        "--stage", required=True,
        choices=[
            "age_proxy", "age_full", "architecture_proxy", "architecture_full",
            "pretraining_proxy", "pretraining_full", "domain_proxy", "domain_full",
            "final_cv",
        ],
    )
    # Keep the production default in a separate assignment so legacy files
    # written under a non-UTF-8 Windows code page cannot corrupt it silently.
    parser.set_defaults(root="D:/\u5168\u5c40\u53cd\u6f14/gpp_production_experiments_v2")
    parser.add_argument("--stress-manifest")
    parser.add_argument("--modis-manifest")
    args = parser.parse_args()
    base = ExperimentConfig.from_json(args.config)
    root = Path(args.root)
    if args.stage == "age_proxy":
        run_age_proxy(base, root)
    elif args.stage == "age_full":
        run_age_full(base, root)
    elif args.stage == "architecture_proxy":
        run_architecture_proxy(base, root)
    elif args.stage == "architecture_full":
        run_architecture_full(base, root)
    elif args.stage == "pretraining_proxy":
        run_pretraining_proxy(base, root)
    elif args.stage == "pretraining_full":
        run_pretraining_full(base, root)
    elif args.stage == "final_cv":
        run_final_cv(base, root)
    else:
        if not args.stress_manifest or not args.modis_manifest:
            parser.error("domain stages require --stress-manifest and --modis-manifest")
        if args.stage == "domain_proxy":
            run_domain_proxy(base, root, Path(args.stress_manifest), Path(args.modis_manifest))
        else:
            run_domain_full(base, root, Path(args.stress_manifest), Path(args.modis_manifest))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from gpp_inversion.config import ExperimentConfig, LossKind, ModelKind
from gpp_inversion.pipeline import run_experiment


def candidate_configs(base: ExperimentConfig):
    common_training = replace(
        base.training,
        optimizer="adamw", batch_size=1024, learning_rate=1e-3,
        weight_decay=1e-4, max_steps=3000, warmup_steps=500,
        eval_interval_steps=1000, patience=3, station_balanced=False,
        target_balanced=True, samples_per_epoch=1024 * 1000,
    )
    baseline = replace(
        base,
        output_dir=base.output_dir / "proxy_baseline",
        window=replace(base.window, endpoint_stride=3, endpoint_phase=0, context_days=0),
        model=replace(base.model, kind=ModelKind.TCN),
        loss=LossKind.MSE, loss_options={}, training=common_training,
    )
    observation = replace(
        baseline,
        output_dir=base.output_dir / "proxy_observation_aware",
        model=replace(base.model, kind=ModelKind.TCN_OBSERVATION_AWARE),
    )
    tail = replace(
        observation,
        output_dir=base.output_dir / "proxy_observation_tail",
        loss=LossKind.TAIL_AWARE,
        loss_options={"weights": [1.0, 1.0, 1.5, 2.5], "underprediction_weight": 0.25},
    )
    multiscale = replace(
        observation,
        output_dir=base.output_dir / "proxy_multiscale_30d",
        window=replace(observation.window, context_days=30),
        model=replace(
            base.model, kind=ModelKind.TCN_MULTISCALE,
            daily_context_features=5, daily_context_hidden=32,
        ),
    )
    hybrid = replace(
        observation,
        output_dir=base.output_dir / "proxy_hybrid_lue",
        scale_target=False,
        model=replace(base.model, kind=ModelKind.HYBRID_LUE_TCN),
    )
    return [baseline, observation, tail, multiscale, hybrid]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed-budget production GPP funnel")
    parser.add_argument("config")
    parser.add_argument("--full", action="store_true", help="Promote <=2% proxy survivors to 12k steps and three seeds")
    args = parser.parse_args()
    base = ExperimentConfig.from_json(args.config)
    results = []
    candidates = candidate_configs(base)
    baseline_score = None
    survivors = []
    for candidate in candidates:
        result = run_experiment(candidate)
        score = float(result["training"]["best_selection_score"])
        results.append({"stage": "proxy", "name": candidate.output_dir.name, "score": score})
        if baseline_score is None:
            baseline_score = score
        if score <= baseline_score * 1.02:
            survivors.append(candidate)
    if args.full:
        for candidate in survivors:
            for seed in (42, 7, 2026):
                full = replace(
                    candidate,
                    output_dir=base.output_dir / "full" / candidate.output_dir.name / f"seed_{seed}",
                    training=replace(
                        candidate.training, seed=seed, max_steps=12000,
                        samples_per_epoch=1024 * 1000, resume=False,
                    ),
                )
                result = run_experiment(full)
                results.append({
                    "stage": "full", "name": candidate.output_dir.name,
                    "seed": seed, "score": float(result["training"]["best_selection_score"]),
                })
    summary = base.output_dir / "production_suite_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

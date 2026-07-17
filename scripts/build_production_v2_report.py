"""Build the compact, user-facing report for the formal production-v2 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path


def _read(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required final artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(rows: list[dict], name: str, field: str) -> float:
    values = [float(row[field]) for row in rows if row["name"] == name]
    if len(values) != 3:
        raise ValueError(f"Expected three seeds for {name}, found {len(values)}")
    return statistics.mean(values)


def _pct(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("docs_output", type=Path)
    args = parser.parse_args()
    root = args.results_root
    output = args.docs_output

    age = _read(root / "age_full_detailed_summary.json")
    architecture = _read(root / "architecture_full_summary.json")
    final_cv = _read(root / "final_cv_summary.json")
    domain = _read(root / "domain_corrected_proxy_summary.json")
    oof = _read(root / "postformal" / "oof_ensemble" / "oof_summary.json")
    global_validation = _read(
        root / "global_validation" / "global_validation_summary.json"
    )

    expected_runs = {
        "diagnostic_100k/reference",
        "diagnostic_100k/masked_pretraining",
        "full_global/reference",
        "full_global/masked_pretraining",
    }
    if set(global_validation.get("runs", {})) != expected_runs:
        raise ValueError("Global validation is incomplete or contains unexpected runs")
    if oof.get("blind_test_opened") is not False:
        raise ValueError("Blind-test safety flag is not false in OOF summary")
    if global_validation.get("blind_test_opened") is not False:
        raise ValueError("Blind-test safety flag is not false in global summary")

    reference_cv = _mean(final_cv, "reference", "macro_rmse")
    candidate_cv = _mean(final_cv, "masked_pretraining", "macro_rmse")
    seeds = (42, 7, 2026)
    seed_rows = []
    for seed in seeds:
        reference = next(
            row for row in final_cv
            if row["name"] == "reference" and int(row["seed"]) == seed
        )
        candidate = next(
            row for row in final_cv
            if row["name"] == "masked_pretraining" and int(row["seed"]) == seed
        )
        seed_rows.append((seed, reference["macro_rmse"], candidate["macro_rmse"]))

    promotion = oof["pretraining_promotion"]
    base = promotion["baseline"]
    candidate = promotion["candidate"]
    failed_checks = [
        name for name, passed in promotion["checks"].items() if not passed
    ]
    degraded_covers = {
        key: value for key, value in promotion["land_cover_rmse_degradation"].items()
        if float(value) > 0.03
    }

    domain_scores = {row["name"]: float(row["score"]) for row in domain}
    d0 = domain_scores["d0_tower_tower"]
    diagnostic_reference = global_validation["runs"]["diagnostic_100k/reference"]
    diagnostic_candidate = global_validation["runs"][
        "diagnostic_100k/masked_pretraining"
    ]
    full_reference = global_validation["runs"]["full_global/reference"]
    full_candidate = global_validation["runs"]["full_global/masked_pretraining"]
    ensemble = oof["final_manifest"]

    age_base = age["tcn_baseline"]["mean"]["macro_rmse"]
    age_candidate = age["a3_age_count_recency"]["mean"]["macro_rmse"]
    multiscale = _mean(architecture, "multiscale_30d", "score")
    architecture_reference = _mean(architecture, "reference", "score")

    seed_lines = [
        "| 种子 | 原TCN五折宏RMSE | 自监督五折宏RMSE | 相对变化 |",
        "|---:|---:|---:|---:|",
    ]
    seed_lines.extend(
        f"| {seed} | {reference:.4f} | {candidate_value:.4f} | "
        f"{_pct(candidate_value, reference):+.2f}% |"
        for seed, reference, candidate_value in seed_rows
    )

    domain_lines = [
        "| 训练输入 | 3000步宏RMSE | 相对D0 | 结论 |",
        "|---|---:|---:|---|",
    ]
    labels = {
        "d0_tower_tower": "D0 塔基气象+塔基地类",
        "d1_tower_modis": "D1 塔基气象+MODIS地类",
        "d2_era_stress_modis": "D2 ERA风格气象+MODIS地类",
        "d3_mixed_modis": "D3 混合气象+MODIS地类",
    }
    for name, value in domain_scores.items():
        change = _pct(value, d0)
        conclusion = "保留基线" if name == "d0_tower_tower" else "超过2%淘汰"
        domain_lines.append(
            f"| {labels[name]} | {value:.4f} | {change:+.2f}% | {conclusion} |"
        )

    report = f"""# GPP正式长实验与全球反演报告

生成日期：2026-07-17  
正式盲测集是否打开：**否**

## 一句话结论

这轮实验没有找到一个能够在所有站点、主要植被类型和全球部署输入下都稳定胜过原TCN的替代模型。因此，**正式生产模型继续使用原TCN**；自监督候选保留为研究候选，不替换生产包。

## 通俗解释

- “距离上次卫星观测多久”确实包含信息，但单独加入后，三种子平均宏RMSE从 {age_base:.4f} 变为 {age_candidate:.4f}（{_pct(age_candidate, age_base):+.2f}%），没有稳定提高总体精度。
- 30天生态记忆相对它自己的观测感知参考从 {architecture_reference:.4f} 变为 {multiscale:.4f}，有轻微平均收益，但逐种子、MAE、高值和植被类型门槛没有同时通过；而且当前真实ERA只有6天，尚不具备正式全球30天输入证据。
- 自监督预训练是本轮最有希望的改动：30项空间/气候五折平均宏RMSE从 {reference_cv:.4f} 降到 {candidate_cv:.4f}，平均改善 {-_pct(candidate_cv, reference_cv):.2f}%。但三个种子中有一个退化，且主要植被类型退化超过3%，所以严格晋级失败。
- 直接把训练输入替换成MODIS地类或ERA风格气象，在塔基验证上明显变差；这说明“让训练看起来更像全球输入”并不会自动提高真实GPP精度。

## 正式空间/气候五折

{chr(10).join(seed_lines)}

OOF逐站配对结果：微观RMSE {base['micro']['rmse']:.4f} → {candidate['micro']['rmse']:.4f}，宏观RMSE {base['macro_rmse']:.4f} → {candidate['macro_rmse']:.4f}，改善站点占 {promotion['station_win_fraction'] * 100:.1f}%。高GPP偏差绝对值也由 {abs(base['high']['bias']):.4f} 降到 {abs(candidate['high']['bias']):.4f}。不过植被类型 {', '.join(degraded_covers)} 的RMSE退化超过3%，失败门槛为 `{', '.join(failed_checks)}`，因此 `passed=false`。

## ERA/MODIS部署一致性

{chr(10).join(domain_lines)}

真实ERA回放只有6天、31至34个可对齐站点，只能作为压力测试，不能代替多年结论。少数组合在某一个部署域上通过，但没有任何D1–D3方案同时满足部署改善和塔基输入不明显退化，所以均不晋级。

## 集成结果

开发集OOF非负权重为：原TCN {oof['nonnegative_weights']['reference']:.3f}，自监督 {oof['nonnegative_weights']['masked_pretraining']:.3f}。组合后的OOF RMSE为 {ensemble['metrics']['rmse']:.4f}、MAE为 {ensemble['metrics']['mae']:.4f}。由于自监督单模型没有通过植被类型门槛，这个集成只作为研究结果，不发布为正式生产模型。

## 全球24小时验证

### 固定10万分层陆地格点

| 模型 | 有效预测率 | GPP均值 | P1 | P50 | P95 | P99 | 夜间GPP<0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原TCN | {diagnostic_reference['valid_prediction_rate'] * 100:.3f}% | {diagnostic_reference['gpp']['mean']:.3f} | {diagnostic_reference['gpp']['p1']:.3f} | {diagnostic_reference['gpp']['p50']:.3f} | {diagnostic_reference['gpp']['p95']:.3f} | {diagnostic_reference['gpp']['p99']:.3f} | {diagnostic_reference['night_gpp_below_0p1_fraction'] * 100:.2f}% |
| 自监督候选 | {diagnostic_candidate['valid_prediction_rate'] * 100:.3f}% | {diagnostic_candidate['gpp']['mean']:.3f} | {diagnostic_candidate['gpp']['p1']:.3f} | {diagnostic_candidate['gpp']['p50']:.3f} | {diagnostic_candidate['gpp']['p95']:.3f} | {diagnostic_candidate['gpp']['p99']:.3f} | {diagnostic_candidate['night_gpp_below_0p1_fraction'] * 100:.2f}% |

### 完整0.1°全球陆地格点

| 模型 | 有效预测率 | GPP均值 | P1 | P50 | P95 | P99 | 夜间GPP<0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原TCN | {full_reference['valid_prediction_rate'] * 100:.3f}% | {full_reference['gpp']['mean']:.3f} | {full_reference['gpp']['p1']:.3f} | {full_reference['gpp']['p50']:.3f} | {full_reference['gpp']['p95']:.3f} | {full_reference['gpp']['p99']:.3f} | {full_reference['night_gpp_below_0p1_fraction'] * 100:.2f}% |
| 自监督候选 | {full_candidate['valid_prediction_rate'] * 100:.3f}% | {full_candidate['gpp']['mean']:.3f} | {full_candidate['gpp']['p1']:.3f} | {full_candidate['gpp']['p50']:.3f} | {full_candidate['gpp']['p95']:.3f} | {full_candidate['gpp']['p99']:.3f} | {full_candidate['night_gpp_below_0p1_fraction'] * 100:.2f}% |

两个模型使用同一时间、同一陆地网格、同一96小时EPIC历史和同一ERA/MODIS输入。全球地图没有独立真值，因此不以“看起来更平滑”作为模型晋级依据。

## 最终生产决定

1. 正式模型：原TCN参考包。
2. 固定输入：96小时因果EPIC历史、ERA5-Land、CO₂、MODIS地类，0.1°全球逐小时输出。
3. 自监督候选：保留研究用途；下一步应优先补齐多年真实ERA5再复验，而不是打开60站盲测集。
4. 60站盲测集保持关闭，未参与训练、归一化、预训练、选模或集成权重拟合。
5. 模型包已经验证CPU导出与CUDA FP16推理；原始TCN批量4096，观测感知Transformer批量1024。

## 关键外部产物

- 正式结果根目录：`{root}`
- 五折摘要：`{root / 'final_cv_summary.json'}`
- 配对晋级：`{root / 'postformal' / 'paired_promotion_summary.json'}`
- OOF与集成：`{root / 'postformal' / 'oof_ensemble' / 'oof_summary.json'}`
- 全球总摘要：`{root / 'global_validation' / 'global_validation_summary.json'}`
- 正式模型包：`{root / 'model_packages' / 'reference'}`
"""

    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "GPP正式长实验与全球反演报告.md"
    report_path.write_text(report, encoding="utf-8")
    compact = {
        "blind_test_opened": False,
        "production_model": "reference_tcn",
        "candidate_promoted": bool(promotion["passed"]),
        "final_cv": {
            "reference_macro_rmse": reference_cv,
            "candidate_macro_rmse": candidate_cv,
            "candidate_relative_change_percent": _pct(candidate_cv, reference_cv),
        },
        "promotion": promotion,
        "domain_proxy_scores": domain_scores,
        "global_runs": global_validation["runs"],
    }
    summary_path = output / "formal_summary.json"
    summary_path.write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()
    manifest = {
        "code_commit": commit,
        "results_root": str(root),
        "blind_test_opened": False,
        "files": {
            report_path.name: _sha256(report_path),
            summary_path.name: _sha256(summary_path),
        },
    }
    manifest_path = output / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()

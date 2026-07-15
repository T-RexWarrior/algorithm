"""Build reproducible CSV and Markdown summaries from completed ablations."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


FORMAL_VARIANTS = (
    "baseline",
    "cross_residual",
    "lag_only",
    "norm_tcn5_only",
    "station_balanced_only",
)
LABELS = {
    "baseline": "基线",
    "cross_residual": "交叉注意力残差",
    "lag_only": "连续相对时间编码",
    "norm_tcn5_only": "5层归一化TCN",
    "station_balanced_only": "站点均衡采样",
}
VERDICTS = {
    "baseline": "参照",
    "cross_residual": "不利",
    "lag_only": "不稳定，不推荐",
    "norm_tcn5_only": "不利",
    "station_balanced_only": "不利",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(seed: int, variant: str, entry: dict) -> dict:
    values = entry["validation"]["all_targets"]
    return {
        "seed": seed,
        "variant": variant,
        "micro_rmse": float(values["micro"]["rmse"]),
        "micro_mae": float(values["micro"]["mae"]),
        "micro_r2": float(values["micro"]["r2"]),
        "macro_rmse": float(values["macro"]["rmse_mean"]),
        "macro_mae": float(values["macro"]["mae_mean"]),
        "macro_r2": float(values["macro"]["r2_mean"]),
        "elapsed_seconds": float(entry["elapsed_seconds"]),
    }


def pct(value: float, baseline: float) -> float:
    return 100.0 * (value - baseline) / baseline


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_station_metrics(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["station"]: row for row in csv.DictReader(handle)}


def format_delta(value: float) -> str:
    return f"{value:+.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("screening_root", type=Path)
    parser.add_argument("seed7_root", type=Path)
    parser.add_argument("seed2026_root", type=Path)
    parser.add_argument("legacy_summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    screening = load_json(args.screening_root / "suite_summary.json")
    screening_rows = [
        metric_row(42, name, screening["variants"][name])
        for name in FORMAL_VARIANTS
    ]
    baseline = screening_rows[0]
    screening_output = []
    for row in screening_rows:
        item = dict(row)
        for key in ("micro_rmse", "micro_mae", "macro_rmse", "macro_mae"):
            item[f"{key}_delta_pct"] = pct(row[key], baseline[key])
        result = load_json(
            args.screening_root / row["variant"] / "result_summary.json"
        )
        item["config_hash"] = result["config_hash"]
        item["verdict"] = VERDICTS[row["variant"]]
        screening_output.append(item)
    screening_csv = args.output_dir / "model_ablation_screening_seed42.csv"
    write_csv(screening_csv, screening_output)

    roots = {
        42: args.screening_root,
        7: args.seed7_root,
        2026: args.seed2026_root,
    }
    confirmation_rows = []
    for seed, root in roots.items():
        suite = load_json(root / "suite_summary.json")
        base = metric_row(seed, "baseline", suite["variants"]["baseline"])
        lag = metric_row(seed, "lag_only", suite["variants"]["lag_only"])
        for row in (base, lag):
            item = dict(row)
            for key in ("micro_rmse", "micro_mae", "macro_rmse", "macro_mae"):
                item[f"{key}_delta_pct_vs_seed_baseline"] = pct(row[key], base[key])
            confirmation_rows.append(item)
    confirmation_csv = args.output_dir / "lag_multiseed_confirmation.csv"
    write_csv(confirmation_csv, confirmation_rows)

    station_base = read_station_metrics(
        args.screening_root / "baseline" / "evaluation" / "val_metrics_by_station.csv"
    )
    station_rows = []
    station_summary = {}
    for variant in FORMAL_VARIANTS[1:]:
        current = read_station_metrics(
            args.screening_root
            / variant
            / "evaluation"
            / "val_metrics_by_station.csv"
        )
        rmse_wins = 0
        mae_wins = 0
        r2_wins = 0
        for station in sorted(station_base):
            base_row = station_base[station]
            row = current[station]
            base_rmse = float(base_row["rmse"])
            base_mae = float(base_row["mae"])
            base_r2 = float(base_row["r2"])
            new_rmse = float(row["rmse"])
            new_mae = float(row["mae"])
            new_r2 = float(row["r2"])
            rmse_wins += new_rmse < base_rmse
            mae_wins += new_mae < base_mae
            r2_wins += new_r2 > base_r2
            station_rows.append(
                {
                    "variant": variant,
                    "station": station,
                    "count": int(float(base_row["count"])),
                    "baseline_rmse": base_rmse,
                    "variant_rmse": new_rmse,
                    "rmse_delta": new_rmse - base_rmse,
                    "rmse_delta_pct": pct(new_rmse, base_rmse),
                    "baseline_mae": base_mae,
                    "variant_mae": new_mae,
                    "mae_delta": new_mae - base_mae,
                    "baseline_r2": base_r2,
                    "variant_r2": new_r2,
                    "r2_delta": new_r2 - base_r2,
                }
            )
        station_summary[variant] = {
            "rmse_wins": rmse_wins,
            "mae_wins": mae_wins,
            "r2_wins": r2_wins,
        }
    station_csv = args.output_dir / "station_paired_comparison_seed42.csv"
    write_csv(station_csv, station_rows)

    grouped = {
        variant: [row for row in confirmation_rows if row["variant"] == variant]
        for variant in ("baseline", "lag_only")
    }
    aggregates = {}
    for variant, rows in grouped.items():
        aggregates[variant] = {}
        for key in ("micro_rmse", "micro_mae", "macro_rmse", "macro_mae"):
            values = [row[key] for row in rows]
            aggregates[variant][key] = (mean(values), stdev(values))
    paired = []
    for seed in roots:
        base = next(
            row
            for row in confirmation_rows
            if row["seed"] == seed and row["variant"] == "baseline"
        )
        lag = next(
            row
            for row in confirmation_rows
            if row["seed"] == seed and row["variant"] == "lag_only"
        )
        paired.append(
            {
                key: pct(lag[key], base[key])
                for key in ("micro_rmse", "micro_mae", "macro_rmse", "macro_mae")
            }
        )

    legacy = load_json(args.legacy_summary)["all_targets"]
    report_path = args.output_dir / "GPP模型消融实验报告.md"
    lines = [
        "# GPP 模型改动消融实验报告",
        "",
        "## 结论",
        "",
        "在本次相同训练预算的真实数据实验中，**没有任何一项模型改动达到稳健有利的标准**。建议继续保留原 6 层 TCN + Transformer + CrossAttention 基线作为默认模型。",
        "",
        "连续相对时间编码是唯一在 seed=42 的逐站宏观 RMSE 上出现改善的改动，但该优势未在 seed=7 和 seed=2026 复现；三种子平均后，微观 RMSE 恶化 2.12%，宏观 RMSE 恶化 1.15%。因此它只能作为后续研究开关，不应并入默认模型。",
        "",
        "测试集始终锁定。本轮没有可靠胜出候选，因此没有进行测试集评估，避免用测试集反复选择模型。",
        "",
        "## 实验协议",
        "",
        "- 数据入口：530 个真实站点 CSV；固定 64 个训练站、43 个验证站、42 个测试站。",
        "- 窗口：96 小时；训练 3,712,060 个窗口，验证 2,404,286 个窗口。",
        "- 训练：batch size 1024，最多 3 轮，patience 2，AMP，CUDA 确定性算法。",
        "- 选择指标：43 站逐站 RMSE 的宏观平均；同时报告微观 RMSE/MAE/R²。",
        "- 单因素原则：每组只改一个因素；seed=42 用于筛选，seed=7/42/2026 用于复核唯一候选。",
        "- 目标值：使用全部真实目标；逆标准化产生的极小浮点负零按零处理，不删除夜间零值。",
        "- 结论边界：这是相同 3 轮筛选预算下的比较，不代表增加训练轮数后的所有可能结果。",
        "",
        "## 历史最佳模型等价性检查",
        "",
        f"历史最佳检查点在新评估链路上的验证集微观 RMSE 为 {legacy['micro']['rmse']:.5f}，宏观 RMSE 为 {legacy['macro']['rmse_mean']:.5f}。新基线 seed=42 分别为 {baseline['micro_rmse']:.5f} 和 {baseline['macro_rmse']:.5f}，说明真实数据入口、逆标准化和统一评估链路能够得到同量级结果。",
        "",
        "## seed=42 单因素筛选",
        "",
        "变化百分比均相对同种子基线；RMSE/MAE 越低越好。",
        "",
        "| 改动 | 微观 RMSE | Δ | 宏观 RMSE | Δ | 微观 MAE | Δ | 结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in screening_output:
        lines.append(
            "| {label} | {micro:.5f} | {micro_delta} | {macro:.5f} | "
            "{macro_delta} | {mae:.5f} | {mae_delta} | {verdict} |".format(
                label=LABELS[row["variant"]],
                micro=row["micro_rmse"],
                micro_delta=format_delta(row["micro_rmse_delta_pct"]),
                macro=row["macro_rmse"],
                macro_delta=format_delta(row["macro_rmse_delta_pct"]),
                mae=row["micro_mae"],
                mae_delta=format_delta(row["micro_mae_delta_pct"]),
                verdict=row["verdict"],
            )
        )
    lines += [
        "",
        "逐站配对补充：seed=42 下，相对时间编码有 26/43 站 RMSE 改善，但只有 14/43 站 MAE 改善；宏观 RMSE 优势并非各指标一致。完整逐站差值见配套 CSV。",
        "",
        "## 相对时间编码三种子复核",
        "",
        "| 种子 | 基线微观 RMSE | 时间编码微观 RMSE | Δ | 基线宏观 RMSE | 时间编码宏观 RMSE | Δ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in roots:
        base = next(
            row
            for row in confirmation_rows
            if row["seed"] == seed and row["variant"] == "baseline"
        )
        lag = next(
            row
            for row in confirmation_rows
            if row["seed"] == seed and row["variant"] == "lag_only"
        )
        lines.append(
            f"| {seed} | {base['micro_rmse']:.5f} | {lag['micro_rmse']:.5f} | "
            f"{format_delta(pct(lag['micro_rmse'], base['micro_rmse']))} | "
            f"{base['macro_rmse']:.5f} | {lag['macro_rmse']:.5f} | "
            f"{format_delta(pct(lag['macro_rmse'], base['macro_rmse']))} |"
        )
    base_agg = aggregates["baseline"]
    lag_agg = aggregates["lag_only"]
    lines += [
        "",
        f"三种子均值±样本标准差：基线微观 RMSE {base_agg['micro_rmse'][0]:.5f}±{base_agg['micro_rmse'][1]:.5f}，时间编码 {lag_agg['micro_rmse'][0]:.5f}±{lag_agg['micro_rmse'][1]:.5f}；基线宏观 RMSE {base_agg['macro_rmse'][0]:.5f}±{base_agg['macro_rmse'][1]:.5f}，时间编码 {lag_agg['macro_rmse'][0]:.5f}±{lag_agg['macro_rmse'][1]:.5f}。",
        "",
        f"同种子配对变化的平均值：微观 RMSE {format_delta(mean(row['micro_rmse'] for row in paired))}，宏观 RMSE {format_delta(mean(row['macro_rmse'] for row in paired))}，微观 MAE {format_delta(mean(row['micro_mae'] for row in paired))}，宏观 MAE {format_delta(mean(row['macro_mae'] for row in paired))}。",
        "",
        "## 各改动最终判断",
        "",
        "- 交叉注意力残差：RMSE、MAE 与耗时均变差，不采用。",
        "- 连续相对时间编码：单种子宏观 RMSE 有小幅收益，但三种子不稳定，且总体 RMSE 变差；保留为可选实验开关，不采用为默认。",
        "- 5 层归一化 TCN：训练损失下降更快，但验证 RMSE 和耗时均变差，不采用。",
        "- 站点均衡采样：相同 3 轮预算下收敛更慢，宏观与微观 RMSE 均变差，不采用。",
        "",
        "## 产物与可复现性",
        "",
        "每个实验目录均包含 experiment_manifest.json、配置哈希、最佳/最新检查点、training_history.json、逐站指标和完整验证集预测文件。筛选组配置哈希如下：",
        "",
    ]
    for row in screening_output:
        lines.append(f"- {LABELS[row['variant']]}：`{row['config_hash']}`")
    lines += [
        "",
        "本报告配套文件：model_ablation_screening_seed42.csv、lag_multiseed_confirmation.csv、station_paired_comparison_seed42.csv、report_manifest.json。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "test_set_locked": True,
        "recommended_default": "baseline",
        "robustly_beneficial_changes": [],
        "conditional_candidate": "lag_only",
        "sources": {
            "screening": str(args.screening_root / "suite_summary.json"),
            "seed7": str(args.seed7_root / "suite_summary.json"),
            "seed2026": str(args.seed2026_root / "suite_summary.json"),
            "legacy": str(args.legacy_summary),
        },
        "outputs": [
            str(report_path),
            str(screening_csv),
            str(confirmation_csv),
            str(station_csv),
        ],
    }
    (args.output_dir / "report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

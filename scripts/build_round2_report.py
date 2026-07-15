"""Build compact round-two CSV deliverables and the final Markdown report."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _macro(path: Path) -> dict:
    frame = pd.read_csv(path)
    return {
        "macro_rmse": float(frame.rmse.mean()),
        "macro_mae": float(frame.mae.mean()),
        "macro_r2": float(frame.r2.mean()),
        "station_count": len(frame),
    }


def _high(path: Path) -> dict:
    payload = _json(path)
    return payload.get("metrics", payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("report_output", type=Path)
    parser.add_argument("docs_output", type=Path)
    args = parser.parse_args()
    results = args.results_root
    output = args.report_output
    docs = args.docs_output
    output.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    screening_root = results / "round2_screening_seed42_e3_b1024"
    screening = _json(screening_root / "round2_screening_summary.json")
    baseline = next(iter(screening["variants"].values()))["baseline"]
    rows = []
    station_rows = []
    for name, item in screening["variants"].items():
        value = item["validation"]
        gate = item["promotion_gate"]
        rows.append(
            {
                "variant": name,
                "config_hash": item["config_hash"],
                "elapsed_seconds": item["elapsed_seconds"],
                "micro_rmse": value["micro"]["rmse"],
                "micro_mae": value["micro"]["mae"],
                "micro_bias": value["micro"]["bias"],
                "macro_rmse": value["macro"]["rmse_mean"],
                "macro_mae": value["macro"]["mae_mean"],
                "high_gpp_rmse": value["high_target"]["rmse"],
                "high_gpp_mae": value["high_target"]["mae"],
                "high_gpp_bias": value["high_target"]["bias"],
                "station_rmse_wins": gate["station_rmse_wins"],
                "passed": gate["passed"],
                "failed_checks": ";".join(
                    check for check, passed in gate["checks"].items() if not passed
                ),
                "prediction_file": item["prediction_file"],
                "manifest": item["manifest"],
            }
        )
        base_station = {row["station"]: row for row in item["baseline"]["station_metrics"]}
        for candidate in value["station_metrics"]:
            base = base_station[candidate["station"]]
            delta = candidate["rmse"] - base["rmse"]
            station_rows.append(
                {
                    "variant": name,
                    "station": candidate["station"],
                    "baseline_rmse": base["rmse"],
                    "candidate_rmse": candidate["rmse"],
                    "rmse_delta": delta,
                    "rmse_delta_percent": delta / base["rmse"] * 100,
                    "outcome": "win" if delta < 0 else ("tie" if delta == 0 else "loss"),
                }
            )
    screening_frame = pd.DataFrame(rows).sort_values("macro_rmse")
    screening_frame.to_csv(output / "screening_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(station_rows).to_csv(
        output / "screening_station_win_loss.csv", index=False, encoding="utf-8-sig"
    )

    validation_sources = {
        "seed42": results / "screening_deterministic_seed42_e3_b1024" / "baseline" / "evaluation",
        "seed7": results / "confirmation_seed7_e3_b1024" / "baseline" / "evaluation",
        "seed2026": results / "confirmation_seed2026_e3_b1024" / "baseline" / "evaluation",
        "three_seed_ensemble": results / "round2_baseline_ensemble" / "evaluation",
    }
    validation_rows = []
    for name, directory in validation_sources.items():
        metrics = _json(directory / "val_metrics_global.json")
        macro = _macro(directory / "val_metrics_by_station.csv")
        validation_rows.append({"model": name, **metrics, **macro})
    validation_frame = pd.DataFrame(validation_rows)
    validation_frame.to_csv(
        output / "baseline_multiseed_validation.csv", index=False, encoding="utf-8-sig"
    )

    high_rows = [
        {
            "split": "validation",
            "model": "seed42_baseline",
            "threshold": 11.099723815917969,
            **baseline["high_target"],
        }
    ]
    for name, item in screening["variants"].items():
        high_rows.append(
            {
                "split": "validation",
                "model": name,
                "threshold": item["high_target_threshold"],
                **item["validation"]["high_target"],
            }
        )
    ensemble_high = _high(
        results / "round2_baseline_ensemble" / "evaluation" / "val_metrics_high_target.json"
    )
    high_rows.append(
        {
            "split": "validation",
            "model": "three_seed_ensemble",
            "threshold": 11.099723815917969,
            **ensemble_high,
        }
    )

    test_root = results / "round2_final_test"
    test_sources = {
        "seed42": test_root / "baseline_seed42" / "evaluation",
        "seed7": test_root / "baseline_seed7" / "evaluation",
        "seed2026": test_root / "baseline_seed2026" / "evaluation",
        "three_seed_ensemble": test_root / "baseline_three_seed_ensemble" / "evaluation",
    }
    test_rows = []
    for name, directory in test_sources.items():
        metrics = _json(directory / "test_metrics_global.json")
        macro = _macro(directory / "test_metrics_by_station.csv")
        test_rows.append({"model": name, **metrics, **macro})
        high_rows.append(
            {
                "split": "test",
                "model": name,
                "threshold": 11.099723815917969,
                **_high(directory / "test_metrics_high_target.json"),
            }
        )
    test_frame = pd.DataFrame(test_rows)
    test_frame.to_csv(output / "final_test_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(high_rows).to_csv(
        output / "high_gpp_metrics.csv", index=False, encoding="utf-8-sig"
    )

    seed_station = pd.read_csv(test_sources["seed42"] / "test_metrics_by_station.csv")
    ensemble_station = pd.read_csv(
        test_sources["three_seed_ensemble"] / "test_metrics_by_station.csv"
    )
    test_station = seed_station[["station", "rmse", "mae"]].merge(
        ensemble_station[["station", "rmse", "mae"]],
        on="station",
        suffixes=("_seed42", "_ensemble"),
        validate="one_to_one",
    )
    test_station["rmse_delta"] = test_station.rmse_ensemble - test_station.rmse_seed42
    test_station["mae_delta"] = test_station.mae_ensemble - test_station.mae_seed42
    test_station["rmse_outcome"] = np.where(test_station.rmse_delta < 0, "win", "loss")
    test_station.to_csv(
        output / "final_test_station_win_loss.csv", index=False, encoding="utf-8-sig"
    )

    seed42_val = validation_frame.set_index("model").loc["seed42"]
    ensemble_val = validation_frame.set_index("model").loc["three_seed_ensemble"]
    seed42_test = test_frame.set_index("model").loc["seed42"]
    ensemble_test = test_frame.set_index("model").loc["three_seed_ensemble"]
    validation_improvement = (seed42_val.rmse - ensemble_val.rmse) / seed42_val.rmse * 100
    test_rmse_improvement = (seed42_test.rmse - ensemble_test.rmse) / seed42_test.rmse * 100
    test_mae_improvement = (seed42_test.mae - ensemble_test.mae) / seed42_test.mae * 100
    test_wins = int((test_station.rmse_delta < 0).sum())

    table_lines = [
        "| 候选 | 宏观 RMSE | 微观 RMSE | 微观 MAE | 高 GPP MAE | 站点胜数 | 晋级 |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in screening_frame.itertuples():
        table_lines.append(
            f"| {row.variant} | {row.macro_rmse:.5f} | {row.micro_rmse:.5f} | "
            f"{row.micro_mae:.5f} | {row.high_gpp_mae:.5f} | {row.station_rmse_wins}/43 | "
            f"{'是' if row.passed else '否'} |"
        )
    report = f"""# GPP 第二轮优化与替代模型测试报告

## 结论

本轮七个新方向均未同时通过全部晋级护栏，因此没有启动 seed=7 和 seed=2026 的候选复核。最终方案按预案确定为原 TCN 基线的 seed=42、7、2026 三种子等权集成，不拟合验证集权重。

验证集三种子集成 RMSE 为 {ensemble_val.rmse:.5f}、宏观 RMSE 为 {ensemble_val.macro_rmse:.5f}，相对 seed=42 单模型 RMSE 改善 {validation_improvement:.2f}%。最终测试集只在方案锁定后评估一次：集成 RMSE 为 {ensemble_test.rmse:.5f}、MAE 为 {ensemble_test.mae:.5f}、宏观 RMSE 为 {ensemble_test.macro_rmse:.5f}，相对 seed=42 单模型分别改善 {test_rmse_improvement:.2f}% 和 {test_mae_improvement:.2f}%，并在 {test_wins}/42 个测试站降低 RMSE。

## 实验协议

- 固定 64/43/42 训练、验证、测试站点划分；窗口长度 96 小时。
- seed=42 单因素三轮筛选，宏观 RMSE 优先；测试集在筛选期间锁定。
- 晋级要求：宏观 RMSE 改善至少 0.5%，三项常规护栏恶化不超过 0.5%，高 GPP MAE恶化不超过 1%，且至少 22/43 站改善。
- 高 GPP阈值为训练目标第 90 百分位 {11.099723815917969:.5f}，由 3,712,060 个训练窗口计算，未读取测试数据。
- CUDA 确定性复验最大参数差为 0；失败实验整组重启，不使用缺少完整 RNG/AMP 状态的检查点续训。

## seed=42 筛选结果

{chr(10).join(table_lines)}

最接近晋级的是 AdamW weight decay：宏观 RMSE、常规微观/宏观护栏和站点胜数均通过，但高 GPP MAE未通过。NDVI/NIRv 通过宏观 RMSE、微观 RMSE、高 GPP和站点胜数，但微观与宏观 MAE未通过。其余方向至少有两项关键门槛失败。

## 最终测试结果

| 模型 | 微观 RMSE | 微观 MAE | 宏观 RMSE | 宏观 MAE |
|---|---:|---:|---:|---:|
| seed=42 | {seed42_test.rmse:.5f} | {seed42_test.mae:.5f} | {seed42_test.macro_rmse:.5f} | {seed42_test.macro_mae:.5f} |
| seed=7 | {test_frame.set_index('model').loc['seed7'].rmse:.5f} | {test_frame.set_index('model').loc['seed7'].mae:.5f} | {test_frame.set_index('model').loc['seed7'].macro_rmse:.5f} | {test_frame.set_index('model').loc['seed7'].macro_mae:.5f} |
| seed=2026 | {test_frame.set_index('model').loc['seed2026'].rmse:.5f} | {test_frame.set_index('model').loc['seed2026'].mae:.5f} | {test_frame.set_index('model').loc['seed2026'].macro_rmse:.5f} | {test_frame.set_index('model').loc['seed2026'].macro_mae:.5f} |
| 三种子等权集成 | {ensemble_test.rmse:.5f} | {ensemble_test.mae:.5f} | {ensemble_test.macro_rmse:.5f} | {ensemble_test.macro_mae:.5f} |

测试结果没有反向参与模型选择。三种子集成预测文件同时保存逐行种子标准差，可用于后续不确定性诊断。

## 实现与验证

已实现严格等权集成、固定 Huber 参数、AdamW 权重衰减、离散 lag embedding、静态 FiLM、NDVI/NIRv、双层 LayerNorm-LSTM、HistGradientBoosting 基线，以及训练 P90 高 GPP指标。每项实验均保存配置哈希、实验清单、完整预测和逐站指标。

自动化测试共 15 项通过，另有 7 个模型子用例通过，覆盖光谱指数掩码与训练集缩放、FiLM/LSTM 形状、旧 TCN 状态字典严格加载、损失参数哈希、集成行对齐和树模型小型端到端产物。

## 交付文件

- `screening_summary.csv`：七方向筛选与护栏。
- `screening_station_win_loss.csv`：所有候选逐站胜负。
- `baseline_multiseed_validation.csv`：三个单种子和验证集成比较。
- `high_gpp_metrics.csv`：验证与测试高 GPP RMSE、MAE、偏差。
- `final_test_summary.csv`：最终测试汇总。
- `final_test_station_win_loss.csv`：测试集成相对 seed=42 的逐站胜负。

原始实验产物位于 `{results}`，包括所有完整预测、模型检查点、scaler、配置哈希和实验清单。
"""
    report_path = output / "GPP第二轮优化与替代模型测试报告.md"
    report_path.write_text(report, encoding="utf-8")
    shutil.copy2(report_path, docs / report_path.name)
    for csv_path in output.glob("*.csv"):
        shutil.copy2(csv_path, docs / csv_path.name)
    print(json.dumps({"report": str(report_path), "docs": str(docs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

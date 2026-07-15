"""Build CSV deliverables and a readable Markdown report for round three."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DISPLAY_NAMES = {
    "adamw_tcn_control": "AdamW-TCN 控制组",
    "pre_ln_encoder": "Pre-LN 状态编码器",
    "zero_init_gated_residual": "零初始化门控残差",
    "bidirectional_cross_attention": "双向交叉注意力",
    "gpp_query_pooling": "GPP query 汇聚",
    "timexer": "TimeXerGPP",
    "modern_tcn": "ModernTCNGPP",
    "time_mixer_pp": "TimeMixerPlusPlusGPP",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def macro_metrics(path: Path):
    frame = pd.read_csv(path)
    return {
        "macro_rmse": float(frame.rmse.mean()),
        "macro_mae": float(frame.mae.mean()),
        "macro_r2": float(frame.r2.mean()),
        "station_count": int(len(frame)),
    }


def diagnostic_rows(root: Path):
    rows = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        path = directory / "architecture_diagnostics.npz"
        if not path.exists():
            continue
        with np.load(path) as values:
            for key in values.files:
                if key == "prediction_scaled":
                    continue
                array = np.asarray(values[key], dtype=float)
                rows.append({
                    "variant": directory.name,
                    "diagnostic": key,
                    "shape": "x".join(str(value) for value in array.shape),
                    "minimum": float(np.nanmin(array)),
                    "mean": float(np.nanmean(array)),
                    "maximum": float(np.nanmax(array)),
                })
            if "pooling_weights" in values.files:
                weights = np.asarray(values["pooling_weights"], dtype=float)
                rows.extend([
                    {
                        "variant": directory.name,
                        "diagnostic": "pooling_max_weight_per_head",
                        "shape": "derived",
                        "minimum": float(weights.max(axis=-1).min()),
                        "mean": float(weights.max(axis=-1).mean()),
                        "maximum": float(weights.max(axis=-1).max()),
                    },
                    {
                        "variant": directory.name,
                        "diagnostic": "pooling_last_step_weight",
                        "shape": "derived",
                        "minimum": float(weights[..., -1].min()),
                        "mean": float(weights[..., -1].mean()),
                        "maximum": float(weights[..., -1].max()),
                    },
                ])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("round3_root", type=Path)
    parser.add_argument("report_output", type=Path)
    parser.add_argument("docs_output", type=Path)
    parser.add_argument("original_val_ensemble", type=Path)
    parser.add_argument("original_test_ensemble", type=Path)
    args = parser.parse_args()

    screening_root = args.round3_root / "screening_seed42"
    output = args.report_output
    docs = args.docs_output
    output.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    screening = pd.read_csv(screening_root / "seed42_summary.csv")
    screening["display_name"] = screening.variant.map(DISPLAY_NAMES)
    control_macro = float(
        screening.loc[screening.variant == "adamw_tcn_control", "macro_rmse"].iloc[0]
    )
    screening["macro_rmse_change_pct_vs_control"] = (
        screening.macro_rmse / control_macro - 1.0
    ) * 100.0
    screening = screening.sort_values("macro_rmse")
    screening.to_csv(output / "architecture_screening.csv", index=False, encoding="utf-8-sig")

    seed_summary = read_json(screening_root / "seed42_summary.json")
    control_stations = {
        row["station"]: row
        for row in seed_summary["variants"]["adamw_tcn_control"]["validation"]["station_metrics"]
    }
    station_rows = []
    for variant, item in seed_summary["variants"].items():
        if variant == "adamw_tcn_control":
            continue
        for candidate in item["validation"]["station_metrics"]:
            control = control_stations[candidate["station"]]
            delta = candidate["rmse"] - control["rmse"]
            station_rows.append({
                "variant": variant,
                "station": candidate["station"],
                "control_rmse": control["rmse"],
                "candidate_rmse": candidate["rmse"],
                "rmse_delta": delta,
                "rmse_delta_pct": delta / control["rmse"] * 100.0,
                "outcome": "win" if delta < 0 else ("tie" if delta == 0 else "loss"),
            })
    pd.DataFrame(station_rows).to_csv(
        output / "station_win_loss.csv", index=False, encoding="utf-8-sig"
    )

    profile_rows = []
    for row in screening.itertuples():
        path = screening_root / row.variant / "architecture_profile.json"
        profile = read_json(path)
        profile_rows.append({
            "variant": row.variant,
            "display_name": DISPLAY_NAMES[row.variant],
            "parameter_count": profile["parameter_count"],
            "mean_batch_latency_ms": profile["mean_batch_latency_ms"],
            "samples_per_second": profile["samples_per_second"],
            "peak_allocated_memory_mb": (
                profile["peak_allocated_memory_bytes"] / 1024**2
                if profile["peak_allocated_memory_bytes"] is not None else np.nan
            ),
            "training_elapsed_minutes": row.elapsed_seconds / 60.0,
        })
    profiles = pd.DataFrame(profile_rows)
    control_profile = profiles.set_index("variant").loc["adamw_tcn_control"]
    profiles["latency_change_pct_vs_control"] = (
        profiles.mean_batch_latency_ms / control_profile.mean_batch_latency_ms - 1.0
    ) * 100.0
    profiles.to_csv(output / "speed_memory_comparison.csv", index=False, encoding="utf-8-sig")

    diagnostics = pd.DataFrame(diagnostic_rows(screening_root))
    diagnostics.to_csv(output / "attention_diagnostics.csv", index=False, encoding="utf-8-sig")

    original_val_global = read_json(args.original_val_ensemble / "val_metrics_global.json")
    original_val_macro = macro_metrics(args.original_val_ensemble / "val_metrics_by_station.csv")
    original_val_high = read_json(args.original_val_ensemble / "val_metrics_high_target.json")["metrics"]
    original_test_global = read_json(args.original_test_ensemble / "test_metrics_global.json")
    original_test_macro = macro_metrics(args.original_test_ensemble / "test_metrics_by_station.csv")
    original_test_high = read_json(args.original_test_ensemble / "test_metrics_high_target.json")["metrics"]
    reused_test = pd.DataFrame([{
        "source": "round2_existing_original_tcn_three_seed_ensemble",
        **original_test_global,
        **original_test_macro,
        "high_gpp_rmse": original_test_high["rmse"],
        "high_gpp_mae": original_test_high["mae"],
        "high_gpp_bias": original_test_high["bias"],
        "round3_test_data_read": False,
    }])
    reused_test.to_csv(output / "reused_final_test_result.csv", index=False, encoding="utf-8-sig")

    table = [
        "| 候选 | 宏观 RMSE | 微观 RMSE | 微观 MAE | 高 GPP MAE | 改善站 | 相对控制组宏观变化 | 晋级 |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in screening.itertuples():
        table.append(
            f"| {row.display_name} | {row.macro_rmse:.4f} | {row.micro_rmse:.4f} | "
            f"{row.micro_mae:.4f} | {row.high_gpp_mae:.4f} | "
            f"{int(row.station_rmse_wins)}/43 | {row.macro_rmse_change_pct_vs_control:+.2f}% | "
            f"{'是' if row.passed and row.variant != 'adamw_tcn_control' else '否'} |"
        )

    speed_table = [
        "| 架构 | 参数量 | 256 样本延迟 | 吞吐 | 峰值显存 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in profiles.sort_values("mean_batch_latency_ms").itertuples():
        speed_table.append(
            f"| {row.display_name} | {int(row.parameter_count):,} | "
            f"{row.mean_batch_latency_ms:.2f} ms | {row.samples_per_second:,.0f}/s | "
            f"{row.peak_allocated_memory_mb:.1f} MB |"
        )

    best_candidate = screening[screening.variant != "adamw_tcn_control"].iloc[0]
    control = screening[screening.variant == "adamw_tcn_control"].iloc[0]
    control_vs_original_macro = (control.macro_rmse / original_val_macro["macro_rmse"] - 1) * 100
    control_vs_original_micro = (control.micro_rmse / original_val_global["rmse"] - 1) * 100
    final_selection = read_json(args.round3_root / "final_selection.json")

    report = f"""# GPP 第三轮模型架构优化与前沿架构测试报告

## 最终结论

本轮四个原 TCN 细节改造和三个前沿替代架构均未通过 seed=42 晋级门槛，因此没有启动 seed=7/2026 复核、没有生成 `refined_tcn`，也没有读取测试集。最终生产方案保持不变：**原 TCN 的 seed=42/7/2026 三次独立训练 + 三种子等权平均预测**。

新的 8 轮 AdamW-TCN 控制组验证集宏观 RMSE 为 {control.macro_rmse:.4f}、微观 RMSE 为 {control.micro_rmse:.4f}。相对现有原 TCN 三种子集成，它的宏观 RMSE变化 {control_vs_original_macro:+.2f}%、微观 RMSE变化 {control_vs_original_micro:+.2f}%，但只有单一 seed=42，且微观 MAE为 {control.micro_mae:.4f}，不能替代已验证的三种子集成。

表现最接近控制组的架构候选是 **{DISPLAY_NAMES[best_candidate.variant]}**，宏观 RMSE 为 {best_candidate.macro_rmse:.4f}，仍比控制组恶化 {best_candidate.macro_rmse_change_pct_vs_control:.2f}%，仅 {int(best_candidate.station_rmse_wins)}/43 个站改善。因此本轮没有“接近门槛但被单个护栏卡住”的候选，而是总体精度均不足。

## 实验协议

- 固定 64/43/42 训练、验证、测试站和 96 小时窗口。
- 全部候选统一使用真正的 `torch.optim.AdamW`、`weight_decay=1e-4`、MSE、学习率 1e-3、batch size 1024、AMP 和 CUDA 确定性。
- 最多 8 轮，按验证站宏观 RMSE 选最佳 checkpoint，patience=3。
- 训练目标第 90 百分位 11.09972 作为高 GPP 阈值。
- 测试集在筛选期间保持锁定；最终选择文件记录 `test_set_read=false`。
- 所有候选均未超过 339,805 参数上限，无需把 `d_model` 从 64 降到 48。

## seed=42 筛选结果

{chr(10).join(table)}

晋级要求是宏观 RMSE 至少改善 0.5%，微观 RMSE、微观 MAE、宏观 MAE恶化不超过 0.5%，高 GPP MAE恶化不超过 1%，并至少改善 22/43 个站。七个候选没有一个同时满足这些条件。

## 架构诊断

- **Pre-LN**：宏观 RMSE 3.1494，仅 16/43 站改善。理论上的优化稳定性没有转化为当前数据和预算下的精度收益。
- **零初始化门控残差**：注意力门约 -0.144、FFN 门约 -0.171，说明门控确实学会开启；但宏观 RMSE 3.1456、17/43 站改善，失败不是因为门始终为零。
- **双向交叉注意力**：融合门均值约 0.637；气象→状态注意力熵均值约 4.41，接近 96 步最大熵 4.56，反向注意力过于分散。宏观 RMSE 3.2362，仅 10/43 站改善。
- **GPP query 汇聚**：汇聚熵均值约 4.42，单头最大时刻权重均值仅约 1.60%，末时刻平均权重约 0.83%。它接近对 96 步平均，削弱了末时刻信息；宏观 RMSE 3.2547。
- **TimeXerGPP**：全局 token 两层注意力熵总体均值约 1.80，未出现简单的全均匀注意力，但 patch/global-token 表征仍使宏观 RMSE恶化到 3.4859。
- **ModernTCNGPP**：四块激活 RMS 随深度放大，最高约 4.35；宏观 RMSE 3.3408，说明 patch + 全局汇聚不如原 TCN 的逐小时因果末端表征。
- **TimeMixerPlusPlusGPP**：三尺度周期/趋势建模未能适配单点末时刻 GPP 任务，最终也未通过任何总体晋级条件。

共同现象很清楚：当前任务不是典型的多步时间序列预测，而是用过去 96 小时预测末时刻 GPP。原架构保留逐小时分辨率、用因果 TCN处理气象、并读取末时刻融合 token，恰好是有效的任务归纳偏置。patch、全局平均或额外双向融合都倾向于稀释末时刻信息。

## 速度与资源

{chr(10).join(speed_table)}

速度结果用于解释成本，不参与放宽精度门槛。更快的小模型仍因精度不足被淘汰。

## 测试集与最终生产结果

本轮没有新候选晋级，因此遵照预案不重新读取测试数据。沿用第二轮已经锁定并只评估一次的原 TCN 三种子集成测试结果：微观 RMSE {original_test_global['rmse']:.4f}、微观 MAE {original_test_global['mae']:.4f}、宏观 RMSE {original_test_macro['macro_rmse']:.4f}、宏观 MAE {original_test_macro['macro_mae']:.4f}，高 GPP MAE {original_test_high['mae']:.4f}。

## 实现与验证

- 新增 `ModelKind.MODERN_TCN`、`TIMEXER`、`TIME_MIXER_PP`，以及 TCN 的 Pre-LN、融合方向、融合模式和时序汇聚配置。
- 默认配置不增加新参数键；旧 226,537 参数 checkpoint 已通过 `strict=True` 加载。
- 明确区分 Adam 与 AdamW，旧配置默认 Adam 保持可复现，第三轮明确使用 AdamW。
- 每项实验保存配置哈希、实验清单、完整验证预测、逐站指标、高 GPP 指标、参数量、推理速度、峰值显存和固定窗口诊断。
- CUDA 确定性复核最大参数差为 0。
- 自动化测试覆盖旧 checkpoint、零门控恒等性、双向形状、query 汇聚、patch 边界、多尺度长度、参数上限、确定性和三个新架构的小型端到端产物。

## 交付文件

- `architecture_screening.csv`：全部候选指标和晋级结果。
- `station_win_loss.csv`：七个候选相对控制组的逐站胜负。
- `speed_memory_comparison.csv`：参数、延迟、吞吐、显存和训练耗时。
- `attention_diagnostics.csv`：注意力、门控、汇聚和多尺度诊断统计。
- `reused_final_test_result.csv`：明确标注未在第三轮读取测试数据的既有最终测试结果。
- `final_selection.json`：最终候选为空、测试集未读取的机器可读决策。

原始结果位于 `{args.round3_root}`。最终选择状态：`final_candidate={final_selection.get('final_candidate')}`，`test_set_read={final_selection.get('test_set_read')}`。
"""
    report_path = output / "GPP第三轮模型架构优化与前沿架构测试报告.md"
    report_path.write_text(report, encoding="utf-8")

    for path in output.iterdir():
        if path.is_file():
            shutil.copy2(path, docs / path.name)
    shutil.copy2(args.round3_root / "final_selection.json", docs / "final_selection.json")
    print(json.dumps({"report": str(report_path), "docs": str(docs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

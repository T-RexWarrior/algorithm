# 外部实验产物索引

完整模型检查点、scaler、逐行预测和中间诊断文件体积较大，不提交 Git。本索引把仓库中的报告与本机外部产物连接起来。

## 产物根目录

| 路径 | 当前状态 | 内容 |
|---|---:|---|
| `D:\全局反演\gpp_ablation_results` | 存在；418 个文件，6,375,983,282 字节 | 三轮消融、基准、三种子确认、最终测试和报告源产物 |
| `D:\全局反演\gpp_ablation_results\round3_architecture` | 存在；130 个文件，1,467,868,086 字节 | 第三轮架构筛选、诊断、速度/显存结果和最终选择 |

以上统计于 2026-07-15 建立索引时读取。目录内容更新后应同步更新本页；这里的数量用于发现缺失或误挂载，不代替文件级校验。

## 关键目录映射

| 外部目录 | 仓库内摘要 | 说明 |
|---|---|---|
| `screening_deterministic_seed42_e3_b1024` | `docs/reports/20260714_gpp_ablation/` | 第一轮确定性单因素消融 |
| `confirmation_seed7_e3_b1024`、`confirmation_seed2026_e3_b1024` | 第一轮报告中的多种子复核 | 连续相对时间编码稳定性确认 |
| `round2_screening_seed42_e3_b1024` | `docs/reports/20260714_gpp_ablation_round2/` | 第二轮七方向筛选 |
| `round2_baseline_ensemble` | 第二轮验证集集成表 | 三种子等权验证集预测 |
| `round2_final_test` | 第二轮 `final_test_*.csv` | 方案锁定后的唯一外部测试评估；含三个单模型和等权集成 |
| `round3_architecture/screening_seed42` | `docs/reports/20260714_gpp_architecture_round3/` | AdamW-TCN 控制组和七个架构候选 |
| `round3_architecture/report` | 第三轮报告目录 | 汇总表与诊断报告源文件 |
| `round3_architecture/final_selection.json` | `docs/reports/.../final_selection.json` | `final_candidate=null`、`test_set_read=false` |

## 正式生产方案

当前锁定方案是原 TCN 的 seed=42、7、2026 三次独立训练与逐行等权平均。正式测试集摘要：微观 RMSE 3.28001、微观 MAE 1.52565、宏观 RMSE 3.02756、宏观 MAE 1.52351。第三轮没有候选晋级，因此没有产生新的正式测试读取或替代 checkpoint。

## 保存与迁移要求

迁移或备份外部产物时，至少保留每次正式运行的：

- `experiment_manifest.json` 与配置哈希；
- `checkpoint_best.pth` 和对应 scaler；
- `training_history.json`、`result_summary.json`；
- 完整预测 CSV 与逐站指标；
- 集成 manifest、最终选择文件和训练目标 P90 阈值文件。

不要只复制仓库内的汇总 CSV 后删除外部目录；汇总结果不足以恢复模型或验证逐行对齐。

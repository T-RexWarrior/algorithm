# 实验配置

`experiment.example.json` 是统一入口使用的完整配置模板。它默认指向当前真实数据目录 `D:/实验五/全波段全变量DT`，复制后仍需修改输出目录和互斥的站点清单。

默认模型是经过三轮真实数据筛选后保留的正式基线：6 层 TCN + Transformer + Cross-Attention、96 小时规则窗口、周期时间特征、全局 Z-score、MSE，并按验证站宏观 RMSE 选择最佳 checkpoint。Mamba、Neural CDE 和第三轮候选架构只作为显式实验开关，不是生产默认值。

模型通过 `model.kind` 选择：

- `tcn`：TCN + Transformer + Cross-Attention。
- `mamba`：Time-aware Mamba；未安装 `mamba_ssm` 时默认使用 Notebook 原有的门控因果卷积替代块。
- `neural_cde`：Notebook 中实现的离散 Euler Neural CDE。
- `lstm`：LayerNorm LSTM 替代基线。
- `modern_tcn`、`timexer`、`time_mixer_pp`：第三轮架构候选；当前真实数据结果均未通过晋级门槛。

时间特征通过 `window.time_features` 选择：`cyclic` 为 4 维，`irregular` 为 6 维，`cde` 为 7 维。缩放策略支持 `zscore` 和 `minmax`；所有验证/测试数据都会复用训练集缩放器。

评估产物通过 `evaluation` 控制：

- `save_predictions`：保存带站点和日期的逐样本预测 CSV。
- `save_plots`：保存逐站趋势、峰值局部、散点和单年时间序列图。
- `moving_average_window`、`zoom_days`：控制趋势平滑窗口和峰值局部图范围。

交叉验证通过 `cross_validation` 控制。`enabled=true` 时，`train_sites` 与 `val_sites` 合并为开发站点池，按照每站 CSV 中 `features.land_cover` 列的主导类别做分层 K 折；模板的 `n_splits=5`。`test_sites` 不参与折叠，默认也不会在每折重复评估；如需该行为可设置 `evaluate_test_each_fold=true`。

每个输出目录都有 `experiment_manifest.json`。配置哈希由完整解析后配置生成，因此修改模型、窗口、训练参数、站点或输出路径后都将得到新哈希；启用断点恢复时，哈希必须与检查点一致。

## 全球生产候选配置

- `production.example.json`：96 小时观测感知 TCN 的完整 12,000 步模板，需由锁定清单物化。
- `production_observation_aware.json`：已写入当前开发/验证/盲测站点和清单哈希的生产配置。盲测只允许在最终决策时运行一次，默认 `evaluate_test=false`。
- `production_smoke.example.json`：两个真实站点、两步优化的工程烟雾配置，只验证数据—模型—GPU—评估链路，结果不得用于模型比较。

可选生产模型为 `tcn_observation_aware`、`tcn_multiscale` 和 `hybrid_lue_tcn`。长期记忆模型要求 `window.context_days=30`；LUE 混合头要求 `scale_target=false`。固定预算漏斗由 `scripts/run_production_suite.py` 执行，详细协议见 `docs/PRODUCTION_WORKFLOW.md`。

正式长实验使用 `scripts/run_formal_experiments.py`。`model.use_endpoint_observation_age`、`model.use_observation_count` 和 `model.use_token_recency` 可独立控制年龄消融；`domain` 段控制塔基/ERA扰动气象和塔基/MODIS地类。未配置 `domain` 的旧配置仍保持原塔基输入行为。

# 实验配置

`experiment.example.json` 是统一入口使用的完整配置模板。复制后修改本机数据目录、输出目录和互斥的站点清单即可。

模型通过 `model.kind` 选择：

- `tcn`：TCN + Transformer + Cross-Attention。
- `mamba`：Time-aware Mamba；未安装 `mamba_ssm` 时默认使用 Notebook 原有的门控因果卷积替代块。
- `neural_cde`：Notebook 中实现的离散 Euler Neural CDE。

时间特征通过 `window.time_features` 选择：`cyclic` 为 4 维，`irregular` 为 6 维，`cde` 为 7 维。缩放策略支持 `zscore` 和 `minmax`；所有验证/测试数据都会复用训练集缩放器。

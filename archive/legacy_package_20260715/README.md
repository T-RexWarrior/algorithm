# 旧包内模块快照

此目录保存 2026-07-15 从 `src/gpp_inversion/` 隔离出的三个旧模块：

- `dataset.py`：旧 Min-Max 数据集和六元素样本接口；
- `trainer.py`：旧训练、恢复和逐站评估入口；
- `visualization.py`：旧绘图及 SHAP 辅助函数。

它们只用于追溯历史 Notebook/脚本行为，不属于可安装包，也不应被新代码导入。正式替代分别是 `src/gpp_inversion/data.py`、`engine.py`、`reporting.py` 和 `explain.py`。

# GPP 站点时序反演实验

本项目研究基于通量站、气象强迫、卫星反射率、经纬度和植被类型等变量的 GPP（总初级生产力）时序反演。核心实验路线由 TCN + Transformer + Cross-Attention 基线，逐步扩展到全局归一化/标准化、损失函数对比、站点级交叉验证、人工地类划分，以及面向不等距观测的 Mamba 和 Neural CDE 模型。

## 当前状态

- `src/gpp_inversion/` 是从 2026-06-01 模块化脚本中整理出的可复用基线组件。
- `notebooks/` 保存完整实验历史，按研究问题分类；Notebook 仍是当前最新实验的主要载体。
- 2026-07-02 的 Mamba 与 Neural CDE Notebook 是时间上最新的模型实验，尚未强行合并为唯一“正式模型”。
- 原始顶层 Python 文件完整保存在 `archive/legacy_python_20260601/`，便于核对迁移前行为。

## 目录

```text
.
├── src/gpp_inversion/              # 可复用的 TCN 基线组件
├── notebooks/
│   ├── 01_baseline/                # 跑数据系列与早期基线
│   ├── 02_normalization/           # 全局归一化、标准化
│   ├── 03_loss_functions/          # MSE、目标标准化、Huber、MAE
│   ├── 04_sampling_validation/     # 地类采样、五折验证、人工站点划分
│   ├── 05_irregular_time/          # 不等距输入、Mamba、Neural CDE
│   └── 90_tools/                   # 辅助检查 Notebook
├── configs/                        # 后续从 Notebook 外置的实验配置
├── data/                           # 仅放本地数据说明，不提交大型数据
├── outputs/                        # 图片、检查点和评估产物
├── archive/                        # 原始脚本快照
├── docs/EXPERIMENTS.md             # Notebook 实验索引
├── tests/                          # 不依赖真实数据的烟雾测试
└── pyproject.toml                  # Python 包与依赖声明
```

## 环境

PyCharm 当前实际配置使用 `D:\miniconda3\envs\dl_env`（Python 3.10），其中已有 PyTorch、Pandas、NumPy、scikit-learn、Matplotlib 和 SHAP。项目原有 `.venv` 是 Python 3.14 的空环境，不包含上述依赖，不建议继续混用。

```powershell
conda activate dl_env
python -m pip install -e ".[notebook,explainability]"
python -m unittest discover -s tests -v
```

## 使用边界

旧 `main.py` 把同一批 CSV 同时作为训练、验证和测试数据，而且数据集构造函数曾删除 `split_type` 参数后仍被旧入口传入。为避免误用，这个入口只保留在 `archive/`，没有作为正式命令暴露。新的实验应明确提供互斥的站点级训练/验证/测试划分，并只用训练集拟合标准化参数。

详细实验关系、已知问题和下一步拆分建议见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

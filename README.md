# GPP 站点时序反演实验

本项目研究基于通量站、气象强迫、卫星反射率、经纬度和植被类型等变量的 GPP（总初级生产力）时序反演。任务使用过去 96 小时的站点序列预测窗口末端 GPP。核心实验路线由 TCN + Transformer + Cross-Attention 基线，逐步扩展到缩放与损失函数对比、站点级交叉验证、Mamba、Neural CDE、LSTM、ModernTCN、TimeXer 和 TimeMixer++。

## 当前状态

- `src/gpp_inversion/` 是唯一正式代码入口，包含统一数据集、站点划分、缩放、模型、训练评估引擎和完整实验管线。
- `notebooks/` 保存按研究问题分类的历史实验，不再作为正式公共实现来源。
- `notebooks/00_integrated/统一实验入口.ipynb` 是新的轻量入口，不再复制公共实现。
- 2026-07-14 至 2026-07-15 已完成三轮真实数据消融和架构筛选；正式方案仍是原 TCN 的 seed=42/7/2026 三种子等权集成。
- Mamba、Neural CDE、LSTM、ModernTCN、TimeXer 和 TimeMixer++ 均保留为可选研究模型，不是默认生产方案。
- 旧数据集、训练器、可视化和顶层入口已移至 `archive/`，不会再从 `gpp_inversion` 公共接口导出。

## 目录

```text
.
├── src/gpp_inversion/              # 可复用的 TCN 基线组件
├── notebooks/
│   ├── 00_integrated/              # 基于配置的统一轻量入口
│   ├── 01_baseline/                # 跑数据系列与早期基线
│   ├── 02_normalization/           # 全局归一化、标准化
│   ├── 03_loss_functions/          # MSE、目标标准化、Huber、MAE
│   ├── 04_sampling_validation/     # 地类采样、五折验证、人工站点划分
│   ├── 05_irregular_time/          # 不等距输入、Mamba、Neural CDE
│   └── 90_tools/                   # 辅助检查 Notebook
├── configs/                        # 后续从 Notebook 外置的实验配置
├── data/                           # 数据说明与轻量清单，不提交大型 CSV
├── outputs/                        # 图片、检查点和评估产物
├── archive/                        # 原始脚本快照
├── docs/EXPERIMENTS.md             # Notebook 实验索引
├── docs/EXTERNAL_ARTIFACTS.md      # 外部检查点和预测产物索引
├── tests/                          # 不依赖真实数据的烟雾测试
└── pyproject.toml                  # Python 包与依赖声明
```

## 环境

PyCharm 当前实际配置使用 `D:\miniconda3\envs\dl_env`（Python 3.10），其中已有 PyTorch、Pandas、NumPy、scikit-learn、Matplotlib 和 SHAP。项目原有 `.venv` 是 Python 3.14 的空环境，不包含上述依赖，不建议继续混用。

```powershell
conda activate dl_env
python -m pip install -e ".[notebook,explainability,test]"
$env:MPLBACKEND = "Agg"
python -m pytest
```

pytest 已在 `pyproject.toml` 中限定只收集 `tests/`，并由 `tests/conftest.py` 固定使用无界面的 Matplotlib `Agg` 后端。`archive/` 中的同名历史测试不会被收集。

Windows 下不要绕过 Conda 直接调用 `D:\miniconda3\envs\dl_env\python.exe` 运行绘图任务；这种方式不会把环境的 `Library\bin` 加入 DLL 搜索路径，Matplotlib Agg 可能在渲染时原生退出。未激活终端可使用：

```powershell
conda run -n dl_env python -m pytest
```

## 真实数据

当前模型输入位于 `D:\实验五\全波段全变量DT`：530 个站点 CSV，总大小约 7.49 GB。CSV 含日期、9 个气象强迫变量、EPIC 掩膜、14 个反射率/观测几何状态变量、经纬度、植被类别和 `GPP_DT_VUT_REF`。

- `data/REAL_DATA.md` 说明数据边界、命名和清单生成方式。
- `data/real_data_manifest.json` 保存逐文件名称、字节数、修改时间和列结构检查结果，不复制原始数据。
- `scripts/build_data_manifest.py` 可在数据更新后重建清单。

## 使用边界

旧 `main.py` 曾把同一批 CSV 同时作为训练、验证和测试数据。旧入口和与之配套的数据集、训练器、可视化模块只保留在 `archive/`。新的统一管线会先检查站点互斥性，只用训练集拟合缩放器，再将同一缩放器传给验证和测试。

## 统一运行方式

复制并修改 `configs/experiment.example.json` 中的输出目录和互斥站点清单后运行：

```powershell
python scripts/run_experiment.py configs/your_experiment.json
```

模板默认使用正式 TCN 基线：96 小时规则窗口、周期时间特征、全局 Z-score、6 层 TCN、MSE，并按验证站宏观 RMSE 选最佳 checkpoint。其他模型、时间特征、缩放和损失函数必须通过配置显式选择。

默认使用人工指定的训练/验证/测试站点。把 `cross_validation.enabled` 设为 `true` 后，统一入口会把 `train_sites + val_sites` 作为开发站点池，按站点主导地类进行分层 K 折（模板默认 5 折）；`test_sites` 始终作为外部测试站点，不参与折叠。若确实需要每折都评估外部测试集，再开启 `cross_validation.evaluate_test_each_fold`。

每次运行会在输出目录保存：

- `experiment_manifest.json`：完整配置、64 位配置哈希、输入文件大小/修改时间、运行环境、状态与产物清单。
- `checkpoint_latest.pth` / `checkpoint_best.pth`：检查点携带配置哈希；恢复训练时会拒绝加载不同配置的检查点。
- `evaluation/*_predictions.csv`：站点、日期、地类、真实值、预测值和残差。
- `evaluation/*_metrics_by_station.csv`、`*_metrics_by_land_cover.csv`、`*_metrics_by_year.csv`：逐站、逐地类和逐年指标。
- `evaluation/*_station_plots/`：逐站移动平均趋势、峰值局部、预测散点和单年序列图。
- 五折模式额外保存 `fold_01` 至 `fold_05` 的独立产物和 `cross_validation_summary.json`。

## 当前正式结果

第二轮锁定方案后只评估一次外部测试集。原 TCN 三种子等权集成取得微观 RMSE 3.28001、微观 MAE 1.52565、宏观 RMSE 3.02756、宏观 MAE 1.52351，并在 35/42 个测试站降低 RMSE。第三轮全部架构候选均未通过预设晋级门槛，因此没有重新读取测试集，也没有替换正式模型。

汇总报告已提交在 `docs/reports/`；完整检查点、scaler 和逐行预测保留在外部实验目录，位置与内容见 `docs/EXTERNAL_ARTIFACTS.md`。

详细实验关系见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)，外部产物见 [docs/EXTERNAL_ARTIFACTS.md](docs/EXTERNAL_ARTIFACTS.md)。

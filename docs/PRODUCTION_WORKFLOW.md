# 全球逐小时 GPP 模型生产工作流

## 已锁定的数据协议

生产输入窗口固定为目标时刻 `t` 的 `t-95h … t`。EPIC 只有实际观测小时为有效状态；缺测槽保持 `EPIC_Available_Mask=0`，不进行未来回填或最后观测前向复制。训练和全球推理共享特征合同，静态位置统一使用球面 `Coord_X/Coord_Y/Coord_Z`，旧模型仍可使用原经纬度接口。

当前清单：

- 真实 CSV：`D:/实验五/全波段全变量DT`，共 530 个文件。
- 盲测锁：`data/blind_split_v1.json`，60 站，文件 SHA256 随清单保存；代表性/OOD 各 30 站。
- 盲测哈希：`b50665afe62c84c68a223b55f59142c63fc0ac41aa398bb303c9c802cc7dc4af`。
- 开发划分：200 个训练站、51 个验证站；原 42 站仅作历史比较。
- 开发划分哈希：`5842ca0d08ef333ce4cdbd6df9b6d734f40457dd131a6fe14d8c16557ecb7013`。

盲测选择不使用 GPP 数值大小。资格检查只使用记录有效性，并要求目标及九个生产强迫变量至少有 96 行共同有效，避免锁入无法完成模型评价的站点。历史实验 manifest 中的 149 个已用站点在选择前全部排除，最终交集为 0；该 manifest 的 SHA256 为 `911240a22d4e74ad0fc25e3758bc7b6babf271fdf5f94eae392b4ba4aac56a66`。盲测站不得进入训练、自监督、缩放统计、OOF 权重拟合或候选选择。

## 生产入口

```powershell
conda run -n dl_env python scripts/build_blind_split.py "D:/实验五/全波段全变量DT" data/blind_split_v1.json --count 60
conda run -n dl_env python scripts/build_development_split.py data/blind_split_v1.json data/development_split_v1.json data/development_split_v1.json
conda run -n dl_env python scripts/materialize_production_config.py configs/production.example.json data/development_split_v1.json configs/production_observation_aware.json
conda run -n dl_env python scripts/run_production_suite.py configs/production_observation_aware.json
```

默认漏斗先以 seed 42 运行 3,000 步的同预算基线、观测感知、尾部损失、30 天记忆和 LUE 混合头。相对基线差超过 2% 的候选淘汰；加 `--full` 后，幸存者运行 12,000 步和 seed 42/7/2026。正式五折 × 三种子只对最多两个候选运行，不能用烟雾配置替代。

其他入口：

```powershell
conda run -n dl_env python scripts/pretrain_production.py configs/production_observation_aware.json outputs/pretraining --steps 3000
conda run -n dl_env python scripts/evaluate_promotion.py baseline.csv candidate.csv report.json --high-threshold 15
conda run -n dl_env python scripts/export_model_package.py configs/production_observation_aware.json checkpoint_best.pth scaler.npz model_package --split-hash 5842ca0d08ef333ce4cdbd6df9b6d734f40457dd131a6fe14d8c16557ecb7013
```

模型包包含 TorchScript、原 checkpoint、训练 scaler、全球 scaler、FeatureContract、模型/划分/代码版本和逐文件 SHA256。全球端加载前会验证哈希，避免模型定义和缩放器独立漂移。

## 已实现的候选与门槛

- `tcn_observation_aware`：仅对有效 EPIC 槽注意；全缺测使用学习 token；显式使用观测年龄和 96 小时计数。
- `tcn_multiscale`：30 天因果日统计经 32 维 GRU 编码，以 FiLM 调制小时 TCN。
- `hybrid_lue_tcn`：可学习 `辐射 × 植被效率 × 温度/VPD/SWC 胁迫` 基线加神经残差，输出非负。
- `tail_aware`：四目标分位权重 `1/1/1.5/2.5`，高值低估额外惩罚 0.25。
- 自监督：15% 掩码重建和气象预测下一次有效 EPIC，只使用非盲测站。
- 集成：最多三模型，使用开发集 OOF 拟合非负、和为 1 的权重。

`scripts/evaluate_promotion.py` 固化宏/微 RMSE、MAE、高值偏差、站点胜率、主要植被类型和站点配对 bootstrap 门槛。最终盲测只打开一次，且生产候选不得依赖全球端不可获得变量。

## 工程验收

2026-07-15 的真实烟雾测试读取 AU-Cow 与 AU-Cpr 两站，构造 27,581/28,993 个 96 小时窗口，在 CUDA 上完成两步训练、逐步验证、checkpoint、架构性能统计和评估清单。该运行只证明生产链路可执行，指标不得用于科学比较。

测试命令：

```powershell
$env:PYTHONPATH = "src"
$env:MPLBACKEND = "Agg"
conda run -n dl_env python -m pytest -p no:cacheprovider -q
```

必须通过 Conda 启动器运行 Windows 绘图测试，以便加入 `Library/bin` DLL 搜索路径；直接调用环境内 `python.exe` 可能导致 Matplotlib 原生退出。

# 实验索引

## 主线概览

项目的共同任务是用长度通常为 96 的时间窗口预测窗口末端的 `GPP_DT_VUT_REF`。常见输入分为：

- 气象强迫：短波辐射、潜在短波辐射、CO₂、降水、VPD、气温、土壤温湿度、风速等。
- 状态变量：EPIC 可用性掩膜、多个卫星光谱波段、太阳/观测几何角度。
- 静态或地类变量：经纬度与 `Veg_ID`。
- 时间编码：小时和年内日的正余弦；不等距版本增加相邻时间间隔及目标时刻距离。

基线用 TCN 编码气象强迫、Transformer 编码状态与时间，再用 Cross-Attention 融合并回归 GPP。后续实验主要改变缩放策略、损失函数、站点划分和时序编码器。

## Notebook 分类

| 目录 | Notebook | 主要目的 |
|---|---|---|
| `01_baseline` | `跑数据.ipynb` | 最早的大型综合基线，含多次迭代代码与输出 |
| `01_baseline` | `跑数据2.ipynb` | 加入植被类型与分层交叉验证实验 |
| `01_baseline` | `跑数据3.ipynb` | NT 数据实验，并混入 ERA5-Land 到 FLUXNET 格式转换工具 |
| `01_baseline` | `跑数据4.ipynb` | 地类独热/自然开阔植被等变体 |
| `02_normalization` | `全局归一化.ipynb` | 全站点 Min-Max 归一化基线 |
| `02_normalization` | `全局标准化.ipynb` | 全局 Z-score 标准化；文件含两代实现 |
| `03_loss_functions` | `损失函数1 目标变量归一化.ipynb` | 目标变量缩放/标准化对比 |
| `03_loss_functions` | `损失函数2 huberloss.ipynb` | 加权 Huber Loss |
| `03_loss_functions` | `损失函数3 MAE.ipynb` | MAE / L1 Loss |
| `04_sampling_validation` | `各地类站点不超过8.ipynb` | 控制各植被类型站点数量并设置对照 |
| `04_sampling_validation` | `五折交叉验证.ipynb` | 按地类分层的五折站点交叉验证 |
| `04_sampling_validation` | `自己定义测试训练验证集  农田.ipynb` | 人工指定农田/森林等站点划分 |
| `05_irregular_time` | `全局标准化-不等距输入版.ipynb` | 用间隔与跨度约束构造不等距时间窗口 |
| `05_irregular_time` | `不等距输入 mamba.ipynb` | Time-aware Mamba 编码器；无 `mamba_ssm` 时有替代块 |
| `05_irregular_time` | `不等距输入 Neural CDE.ipynb` | 离散 Neural CDE 风格编码器与 Cross-Attention |
| `90_tools` | `云覆盖度图片.ipynb` | 很小的环境/辅助检查 Notebook |

## 已确认的结构与方法问题

1. Notebook 通常只有 1–4 个超长代码单元，模型、数据、训练、评估和画图重复粘贴，难以比较单一变量的影响。
2. 多数实验把本机绝对数据路径和输出目录写在代码中，换机器后不能直接复现。
3. 部分 Notebook 在一个文件中保留多代完整实现，后定义会覆盖前定义，实际运行版本不直观。
4. 旧模块化入口重复使用同一批站点作为训练、验证和测试，存在数据泄漏，不能把其指标当作独立测试结果。
5. 缩放器必须只在训练站点上拟合，再传给验证和测试；部分后期 Notebook 已开始这样做，但需要统一。
6. 模型检查点、图和 Notebook 输出混在源码附近，且项目原先没有根级 `.gitignore` 或依赖声明。

## 建议的下一阶段

先选择一个“正式实验基线”（建议从人工站点划分 + 全局 Z-score 版本开始），再把 Notebook 中的公共部分依次抽到 `src/gpp_inversion/`：

1. `splits.py`：站点匹配、互斥检查、五折划分。
2. `preprocessing.py`：缺失值处理、缩放器拟合/保存、连续与不等距窗口。
3. `models/`：TCN、Mamba、Neural CDE 编码器及统一融合头。
4. `engine.py`：训练、早停、恢复、评估和指标汇总。
5. `configs/*.json`：数据路径以外的可复现实验参数；本机路径通过命令行或环境变量传入。

在完成等价性测试前，不应直接删除 Notebook 中的历史实现或覆盖已有实验输出。

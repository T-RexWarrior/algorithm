# 真实站点数据清单

## 数据位置与边界

- 本机目录：`D:\实验五\全波段全变量DT`
- 文件格式：每站一个 `*_Merged.csv`
- 当前文件数：530
- 当前总大小：7,489,379,642 字节（约 6.97 GiB / 7.49 GB）
- 原始 CSV 不进入 Git；仓库只保存轻量元数据清单。

文件名通常由站点、主导地类和 `Merged` 后缀组成，例如 `AR-CCa_CRO_Merged.csv`。实验配置中的站点 ID 通过文件名匹配，修改命名规则前必须复核 `src/gpp_inversion/splits.py`。

## 当前列结构

清单生成器检查每个 CSV 的首行表头。当前主结构为 32 列：

- 时间：`date`
- 气象强迫：`SW_IN_F`、`SW_IN_POT`、`CO2_F_MDS`、`P_F`、`VPD_F`、`TA_F`、`TS_F_MDS_1`、`SWC_F_MDS_1`、`WS_F`
- 卫星状态：`EPIC_Available_Mask`、317–780 nm 反射率波段、`Mean_SZA`、`Mean_VZA`、`Mean_RAA`
- 静态与类别：`Lat`、`Long`、`Veg`、`Veg_ID` 以及三维坐标
- 目标：`GPP_DT_VUT_REF`

数据使用 `-9999`/`-999` 表示部分缺失值。正式数据集入口会将这些哨兵值转为缺失值，按时间排序、去重，并在构造窗口前删除所需列不完整的行。

## 更新清单

数据内容变化后，在项目根目录运行：

```powershell
python scripts/build_data_manifest.py "D:\实验五\全波段全变量DT"
```

生成的 `real_data_manifest.json` 记录：

- 数据目录和生成时间；
- 文件数与总字节数；
- 所有表头变体及对应文件数；
- 每个 CSV 的文件名、大小、修改时间和列数。

清单不会逐行扫描 7.49 GB 数据，也不会记录敏感内容或复制观测值。训练运行自身还会在 `experiment_manifest.json` 中记录实际进入该次实验的文件大小和修改时间。

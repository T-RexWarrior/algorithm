# 统一实验入口

`统一实验入口.ipynb` 不再复制模型和训练代码。它只负责加载 JSON 配置并调用 `gpp_inversion.pipeline.run_experiment`。

原有 16 个 Notebook 继续保留在其他分类目录中，作为实验历史和已生成结果的依据。

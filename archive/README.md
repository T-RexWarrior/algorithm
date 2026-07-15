# 归档

`legacy_python_20260601/` 保存整理前位于项目根目录的五个 Python 文件。它们用于追溯历史行为，不作为推荐入口。

其中旧 `main.py` 存在同站点数据重复用于训练、验证和测试的问题；可复用模块的整理版本位于 `src/gpp_inversion/`。

`legacy_package_20260715/` 保存曾残留在正式 Python 包中的旧 `dataset.py`、`trainer.py` 和 `visualization.py`。它们依赖旧的 Min-Max 数据接口，现已从 `src/gpp_inversion/` 和公共导出中移除；正式代码统一使用 `data.py`、`engine.py` 和 `reporting.py`。

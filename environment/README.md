# 环境记录

原实验使用 Linux、Python 3.12，以及两个独立 Python 环境：

- `requirements-3d-lock.txt`：WiLoR、3D 训练、FK 和轨迹优化环境。
- `requirements-rfdetr-lock.txt`：RF-DETR 检测与实例分割环境。

这两个文件是**已运行环境的包版本清单**，包含基础镜像中的系统包；不是保证跨平台可直接全部安装的最小 requirements。CUDA/PyTorch、手跟踪工具包、WiLoR 和 SAM2 按上游要求准备。使用源码时将重定位后的 `code/` 加入 `PYTHONPATH`。

原比较全部使用同一 GPU、BF16、相机与时间戳。历史控制器包含物理 GPU3 的资源保护规则；迁移时须在新的独立源码版本中更新本地 GPU 约定，不直接修改正在运行的旧快照。

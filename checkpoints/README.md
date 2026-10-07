> 权重文件待 Git LFS 上传；本次代码发布先提供清单和版本说明。

# 权重清单

所有模型均由 Git LFS 存储，原始字节不改动。大小和 SHA-256 见根目录 `snapshot_manifest.json`；运行 `python tools/verify_release.py` 可以核验。

| 文件 | 用途 | 状态 |
| --- | --- | --- |
| `yolo26_hot3d.pt` | 原手框检测器 | 旧无 mask 前端 |
| `yolo26_multidomain_control.pt` | 多域 YOLO | 辅助带 mask 对照 |
| `rfdetr_hand_v53_ema.pth` | RF-DETR SegSmall 576，阈值 .2，框 NMS .7 | 检测开发集准入；完整 3D 尚未准入 |
| `regression_v39.pt` | v42 稳定流程的直接回归头 | 历史参考 |
| `dit_v43.pt` | 参数时序 DiT | 旧无 mask 参考 |
| `dit_v48_protected.pt` | 在线视觉＋准确点保护 DiT | 旧无 mask 参考 |
| `dit_v51_instance.pt` | v53 复用的实例完整后端 | 开发集权重 |
| `dit_v54_rf_best.pt` | RF 条件适配，第 80 步 | 开发集 best；完整评估 pending |
| `dit_v54_yolo_control_best.pt` | 同预算带 mask YOLO，第 160 步 | 辅助对照；完整评估 pending |
| `rgb_reference_v16.pt` | 冻结 RGB 定位参考 | 运行依赖 |
| `risk_v16.pt` | 预测可靠性模型 | 运行依赖 |
| `rgb_probe.pt` | 原空间探针 | 运行依赖 |
| `risk_projection.pt` | 可靠性特征投影 | 运行依赖 |
| `shape_space.pt` | 固定形状空间 | FK 几何依赖 |
| `kinematic_template.pt` | 固定几何模板 | FK 几何依赖 |
| `mask_localizer_v51.pt` | 实例空间定位器 | 完整后端依赖 |

WiLoR 原始主干、手侧 helper、MANO、SAM2 基础资产没有复制到此目录，按上游说明单独准备。best 文件不等同于完整训练恢复文件；恢复相同 optimizer、scheduler 和 RNG 需原实验的 resume 状态。

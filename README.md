> 发布进度：代码和说明正在发布；16 个权重已在本地校验，Git LFS 上传等待本机 GitHub 登录。当前尚未宣称权重可下载，详见 [PUBLISHING.md](PUBLISHING.md)。

# Ego-Hand

面向离线数据标注的 RGB 手部检测、实例分割和时序 3D 补全研究代码。

当前流程：**YOLO26 / RF-DETR → 手轨迹 → WiLoR 初始 3D → RGB 条件时序 DiT → 手参数与 FK → 整段运动优化 → 待复核的 3D 标注**。

这是 2026-10-08 的研究快照：保留无 mask 的 v43/v48 基线，以及 RF-DETR、实例 mask 和 v54 条件适配。v54 两组各 160 步训练完成，完整公平对比仍在运行，尚未产生替换旧版的结论。详情见 [版本与结果](docs/RESULTS.md)、[开发记录](docs/DEVELOPMENT.md) 和 [导出时状态](evidence/v54/status_at_export.json)。

## 方法

| 模块 | 输入和作用 | 主要代码 |
| --- | --- | --- |
| 前端 | YOLO 手框，或 RF-DETR 手框＋手实例 mask | [RF 推理](research/predict_instances_v53.py)、[RF 训练](research/train_rfdetr_multidata_v53_r2.py) |
| 初始重建 | 原始 RGB、预测手框和预测手侧 → WiLoR 3D；相机坐标统一 | [重建适配](research/complete_instance_v51_r5.py)、[WiLoR 适配](research/wilor_eval_common.py) |
| 视觉条件 | 保留每帧 16×12＝192 个空间位置、1280 维特征；在线版本训练最后四个视觉块和 Norm | [在线视觉](research/online_parameter_model_v47.py) |
| 时序补全 | 17 帧双向上下文，近密远疏，最大 ±1.6 秒；结合 RGB、初始 3D、时间、相机与预测可靠性 | [参数 DiT](research/semantic_parameter_model_v36.py)、[时间窗口](research/temporal_window_v44.py) |
| 实例条件 | 自己的 mask、其他手的 mask 和质量；辅助可见性预测 | [实例模型](research/instance_parameter_model_v51.py)、[可见性分支](research/instance_parameter_model_v51_r2.py) |
| 手型约束 | 3 根平移＋6 旋转＋20 关节角＋5 形状参数，共 34 维；FK 解码为 20 个 3D 点 | [参数编码](research/parameter_codec_v31.py) |
| 时序优化 | 在同一手轨迹内共享形状，拟合整段运动；保持速度限制，软硬加速度限制均放宽两倍 | [轨迹优化](research/stability_trajectory_v43.py)、[加速度配置](research/complete_hand_tracks_acceleration_v46.py) |
| 准确点保护 | 教师已准确的点只在训练损失中用于保护 | [保护损失](research/protected_parameter_model_v48.py) |

输出是 **3D**，单位米，HOT3D 20 点顺序，手腕索引 5。2D 热图和可见性是辅助分支。当前 FK 使用 UmeTrack 兼容的手几何；WiLoR 的 MANO 位于前端，不能将后续 FK 称为 MANO 解码。

RF-DETR 与 3D 后端分阶段训练。RGB 末四层、DiT 和 FK 损失支持在线联合训练；检测、关联、初始 IK 和最终轨迹求解仍是独立步骤。结构与运动约束通过不代表遮挡处的真实姿态一定准确。

## 获取代码和权重

```bash
git lfs install
git clone https://github.com/Mikehuntisbald/Ego-Hand.git
cd Ego-Hand
git lfs pull
python tools/verify_release.py
```

16 个模型及固定运行依赖约 1.64 GB，均通过 Git LFS 管理。具体用途、原始大小和 SHA-256 见 [checkpoint 清单](snapshot_manifest.json) 与 [权重说明](checkpoints/README.md)。v54 导出的是开发集选择的 best，第 80 / 160 步，未把未经准入的末步当成安全版本。

## 环境与运行

使用 Linux、Python 3.12 和 CUDA。RF-DETR 与 3D 后端原实验使用独立环境，版本记录在 [environment](environment/README.md)。先按上游说明准备 WiLoR、MANO、手跟踪工具包和所需数据，再按 [复现说明](docs/REPRODUCING.md) 建立运行目录。

研究脚本保留原实验的目录约定；`tools/prepare_workspace.py` 可以把代码复制并重定位到新目录，安装已导出的权重，默认只预览操作。它不会启动训练或定时任务。

```bash
python tools/prepare_workspace.py --workspace /work/ego-hand
python tools/prepare_workspace.py --workspace /work/ego-hand --apply
```

`research/` 保留从早期 2D 到当前 3D 的完整实验历史；当前入口和监督方式见 [架构说明](docs/ARCHITECTURE.md)。旧实验的缓存、数据切分和训练现场需要另外准备，未声称仅克隆仓库即可精确恢复全部历史实验。

## 评估原则

完整输出先封存，再读取回放测试标签。比较必须同时报告漏手、共同匹配点上的 3D 误差、坏点恢复、原正确点损伤、来源序列配对置信区间、跳变、骨长和快速动作保留。GT、人工 mask 和教师预测不得成为自动推理条件。当前训练使用真实自然遮挡，不增加人工遮挡。

公开仓库包含研究源码、模型和汇总证据。数据图像、视频、样本级标签、WiLoR/MANO/SAM2 上游资产需按各自来源与许可另行获取；100DOH 参与训练，原作者说明其数据限非商业研究。见 [来源和使用说明](docs/THIRD_PARTY.md)。

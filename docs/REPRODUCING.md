# 复现与运行

## 本次发布能核验什么

`tools/verify_release.py` 不需要 GPU 或 PyTorch；核验冻结源码、Python 语法及 16 个 LFS 文件的大小/哈希。它不宣称 GPU 推理或重训练已经在一个全新的机器环境复现成功。

主模型 checkpoint 保留原始训练选择文件。完整恢复训练仍需各实验的 resume checkpoint、optimizer/scheduler/RNG、批次计划和输入缓存，本次没有上传这些现场数据。

## 建立独立目录

原始代码使用 `/mnt/why/HOT3D` 与 `/mnt/why/hot3d_hand_residual` 路径。仓库保持这些快照不变；工具把代码复制到独立工作区时替换路径，并按清单链接已导出的 checkpoint。已有不同文件会导致报错，不会覆盖。

```bash
python tools/prepare_workspace.py --workspace /work/ego-hand
python tools/prepare_workspace.py --workspace /work/ego-hand --apply \
  --wilor-source /work/upstream/WiLoR \
  --wilor-assets /work/assets/wilor \
  --hand-toolkit /work/upstream/hand_tracking_toolkit
export PYTHONPATH=/work/ego-hand/code:$PYTHONPATH
```

WiLoR 资产目录约定：`model_config.yaml`、`wilor_final.mirror.ckpt`、`MANO_RIGHT.pkl`、`mano_mean_params.npz`；`wilor_detector.pt` 是完整模型仍使用的预测手侧 helper。按上游下载后安排这些文件名。可能还需要 WiLoR 上游配置引用的其他辅助资产，按原配置准备。

SAM2 只用于历史或 YOLO 带 mask 辅助对照。该路径需要 `--sam2-source`，并另行放置上游权重 `HOT3D/domain_data_v51/sam2.1_hiera_small.pt`。RF 完整路径使用封存的 RF masks 替代 segmenter，不调用 SAM2。

## 输入与输出

旧无 mask 的通用入口接受一个 JSON：根字段 `image_size: [1408, 1408]` 和 `tracks`，每个轨迹有 `id`、`frames`；帧包含 `image`、真实 `timestamp_s`、`box_xyxy`、与手跟踪工具包一致的 `camera` 标定和 `T_world_from_camera`。使用 `--prepared` 时还提供预测的 `xyz_camera_m`、可用点标志及预测手侧。坐标以米表示，20 点、手腕索引 5。

```bash
python /work/ego-hand/code/complete_hand_tracks_stable_v42.py \
  --input your_tracks.json --output outputs/stable.json --device cuda:0
```

以上是 v42 通用入口示例。当前 v54 和 RF 完整路径的命令与缓存约定在冻结的 `infer_fair_v54.py` 中；它是特定实验的入口，需要按 `prepare_fair_v54.py`、`cache_fair_v54.py` 的记录另行准备相同数据、source manifests 和封存预测，不是任意视频一行命令的产品接口。

RF 前端独立入口接受 `frames` JSON，每帧记录原 RGB 路径及观察元数据：

```bash
CUDA_VISIBLE_DEVICES=3 python /work/ego-hand/code/predict_instances_v53.py \
  --input observation_frames.json --output outputs/rf_instances \
  --checkpoint checkpoints/rfdetr_hand_v53_ema.pth \
  --threshold 0.2 --policy box_nms_0.7
```

RF 原快照含 GPU3 保护断言。原控制器也检查 GPU 所有权；其他机器的设备编号应在新实验源码版本中明确调整，不能盲目启动所有控制器。分割输出为真实预测 masks/boxes/scores；后续按预测实例构建轨迹，生成 WiLoR/IK 观测，再调用完整补全核心。

## 训练和评分

- RF 多数据训练：`prepare_rfdetr_multidata_v53.py`、`train_rfdetr_multidata_v53_r2.py`、`train_stageB_v53.py`。原数据和切分自行按记录准备。
- 在线 RGB/3D：`train_online_parameter_v47.py`、`train_online_parameter_v48.py`。
- mask 完整后端：`train_full_instance_v51_r1.py`、`train_full_surgical_v51_r2.py`。
- 当前适配：`train_fair_v54.py`。它产生独立 trainer revision；修改训练应另立实验目录。
- 封存和评分：`infer_fair_v54.py`、`score_fair_v54.py`、`historical_reference_fair_v54.py`。所有推理输出完成封存后再打开测试标签。

历史开发说明在 `docs/history/`。不要将既有回放、开发集准入或约束可行性，当成新增数据上的泛化和准确性证据。

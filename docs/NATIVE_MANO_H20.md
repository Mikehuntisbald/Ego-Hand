# 原生 MANO DiT、头部 VO 与固定相机 BA

2026-10-19 H20 研究快照。它是公开资源实现，不是私有产线的数值复现，也没有通过生产替换准入。本分支保留原有 UmeTrack 研究代码；新增代码放在 `research/native_mano_h20_20261019/`。

## 当前链路

```text
左右跟踪相机 → 共同视场整流、CLAHE → cuVSLAM17 双目 VO → 相机轨迹 ─┐
双目 → LightESS → UniDepth 补盲 → 表面到腕深度锚 ──────────────────┤
RGB → RF 实例检测 → 轨迹/手侧筛选 → WiLoR 初始 MANO ───────────────┤
                             ↓                                  │
             17 帧 RGB 空间特征与相机对齐后的姿态                   │
                             ↓                                  │
             原生 MANO 旋转残差 DiT → FP32 MANO FK                 │
                             ↓                                  ↓
                 独立 RGB 二维观测 → 固定相机整段 MANO BA
                             ↓
                 轨迹参数保护、重新 FK、JSON/H5
```

头部支路使用真实图像运行 cuVSLAM，没有融合 IMU，因此称双目 VO。它把左右跟踪相机的轨迹通过物理外参转换到 RGB 相机。DiT 用相机轨迹把邻帧腕位置和全局朝向对齐到中心相机；BA 使用固定相机轨迹，不反向优化 SLAM。六段补全模块评测使用数据提供的参考轨迹；只有完整片段回放实际运行了头部 VO、物理深度和 BA，二者不能混报。

## 网络及输出契约

- 冻结 WiLoR 视觉主干，每帧保留 `16×12×1280` 特征，192 个空间位置。
- 17 个非因果时间槽，近密远疏，最大约 ±1.6 秒；空间 cross-attention、两层时序 Transformer、四层 DiT，宽 128、四头，余弦日程及 v-prediction。
- 17 个三维残差 token：腕平移、全局旋转、15 个局部关节旋转。腕平移 token 关闭，形状沿用预测值；旋转残差经 SO(3) 组合。默认四采样、每个十步，生成中心帧，不是整段联合扩散。
- 输出原生 `global_orient`、`hand_pose`、`betas`、`transl`，另保留左右手反射与物理相机腕位置。`state_camera51` 的前三维是腕 XYZ，**不是 MANO transl**。
- MANO FK 使用 21 点、腕索引 0；不经 UmeTrack 20 点 IK 桥接。BA 可调整手部平移、姿态与轨迹共享形状，使用重投影、腕深度锚和真实时间运动项。
- RF mask 用于深度采样的实例排除。本轮 DiT 没有新增 mask 输入通道。

原生监督采用数据的 MANO PCA 展开目标，当前可靠训练范围仅右手。左手新姿态生成原样通过，BA 耦合仍可能改变左手。现有 RIGHT 资产与官方解码数值核对通过，但原标注资产版本尚未独立核验。MANO 参数化保证 FK 结构，不能保证遮挡姿态真实，也没有严格碰撞或解剖限位保证。

## 源码定位

| 范围 | 主要入口 |
|---|---|
| 头部 VO、深度与几何 | `public/src/run_headpose.py`、`run_depth.py`、`run_unidepth.py`、`protect_depth.py` |
| 原生 DiT、FP32 FK 与导出 | `v2/src/native_dit.py`、`native_mano.py`、`infer_native.py` |
| 一致手侧前端与原生 BA | `v3/src/run_wilor_coherent.py`、`v2/src/native_ba.py` |
| 原生右手监督及实际采样训练 | `v5/src/`、`v7/src/train_native_v7.py` |
| 多数据推理 | `v7/eval_src/infer_v7.py`、`infer_full_right_v7.py` |
| 独立 RGB 观测和风险 | `v9/src/rgb_head.py`、`v9/eval_src/native_ba_rgb.py` |
| 整段二阶候选选择（未准入） | `v10/src/sequence_solver.py`、`select_candidates.py` |
| 三维候选风险（无新增收益） | `v11/src/risk_model.py`、`v11/error_only_r1/src/risk_model_r1.py` |

表中的相对路径以 `research/native_mano_h20_20261019/` 为根。保留了独立版本及失败修订，不把旧控制器当成推荐的一键运行入口。

## 当前结果，不能只看均值

六段历史录制、894 个匹配右手帧，原生 21 点腕对齐后排除腕，共 20 点：WiLoR 17.54 mm，v7 受保护均值 16.81 mm。v7 恢复 903/6019 个困难点，同时损伤 1642/5302 个原准确点（误差增加 >1 mm），未准入。六段有历史评估暴露，不称新鲜测试。

完整历史片段 150 个右手帧，v7 受保护 DiT+BA 的相机/相对误差为 16.897/10.859 mm；加 v9 独立 RGB 观测后为 16.782/10.996 mm。绝对误差略降而相对姿态略差，仍不能宣称全面改善。

v10 的 RGB+真实时间二阶候选选择把六段相对误差从 16.808 变为 17.152 mm，录制配对 95% CI 为 [+0.161,+0.588] mm，拒绝晋级。v11 两版、各三组评分器完成训练，七段封存推理均选择候选 0，尚无新增收益；不能以训练完成宣称改善。v11 没有完成新的整线 BA 验收，未发布其模型作为推荐权重。

原始 5/10/30 mm 版本位移及 +0.2 px 重投影保护线不变，它们限制版本变化，不是 GT 误差保证。GT 和准确点掩码只进入训练损失，预测先封存后评分。没有人工遮挡，没有修改生产默认或恢复定时任务。

## 已解决的问题

1. MANO 模板腕不在原点：按 LBS 根位置恢复真正 transl，导出后重新 FK 核对；不能把腕 XYZ 直接写成 transl。
2. BF16 污染几何：旋转组合、FK、参数导出固定 FP32，旧错误版本保留。
3. 手侧标记与实际前向不一致：用预测证据作轨迹投票，实际重跑受影响 WiLoR 帧，而非只改标签。
4. 旧 UmeTrack 点目标与 MANO 几何不一致：改为原生参数监督；形状解耦另做控制，没有因损失更合理就晋级。
5. WiLoR 自投影不是独立二维证据：增加 RGB 热图与风险头；保留无新观测的 BA 逐值一致控制。
6. 候选平滑不等于姿态更准：保留失败选择器、原正确点损伤及恢复统计，不用 GT best-of-K 作推理输出。

## 获取与核验

四个新检查点作为普通 Git 文件提交，每个不足 10 MB，只有学习头，不包含 WiLoR 主干或 MANO 资产：v7 大/小数据 DiT，v9 空间/池化 RGB 控制。GitHub 账号 LFS 配额已满，本分支的新头不依赖 LFS；历史 RF 等权重仍需原有实体或恢复 LFS 额度。v7 导出只保留推理字段；重新训练和完整断点恢复仍需原数据、输入缓存、批次与 optimizer/RNG，不由本快照提供。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch codex/native-mano-vo-20261019 \
  https://github.com/Mikehuntisbald/Ego-Hand.git
cd Ego-Hand
python tools/verify_native_mano_snapshot.py
python tools/prepare_native_mano_workspace.py --workspace /work/native-mano
```

核验工具检查文件 SHA、语法和四个模型实体；工作区工具默认仅预览。显式加 `--apply` 才创建独立目录、复制并重定位源码；已有目录会被拒绝。重定位记录原始与新文件 SHA，不改冻结快照。

```bash
python tools/prepare_native_mano_workspace.py \
  --workspace /work/native-mano --apply \
  --wilor-source /work/upstream/WiLoR \
  --wilor-assets /work/assets/wilor \
  --hand-toolkit /work/upstream/hand_tracking_toolkit
```

手部原训练环境是 Python 3.12、Torch 2.8/CUDA 12.8；头部 cuVSLAM17 为独立 Python 3.10，深度需要独立可用的 TensorRT 环境。仍须准备相机、原图/时间戳、WiLoR 及许可 MANO、LightESS 与匹配插件、UniDepth、cuVSLAM；不得跨硬件直接复用 TRT 引擎。公开依赖入口见冻结的 `public/src/`，其中旧失败尝试仅作历史，不保证所有旧下载入口仍可用。

完成资产准备后，已验证接口的原生推理与 BA 示例：

```bash
W=/work/native-mano
export PYTHONPATH="$W/v7/eval_src:$W/v7/src:$W/v2/src:$W/v4/src:$W/public/src:$W/public/vendor/Ego-Hand/research"
CUDA_VISIBLE_DEVICES=3 python "$W/v7/eval_src/infer_full_right_v7.py" \
  --public-root /work/new-observations \
  --checkpoint "$W/v7/runs/full/best_development.pt" \
  --output /work/new-native --protect

PUBLIC_HEADHAND_ROOT=/work/new-observations CUDA_VISIBLE_DEVICES=3 \
python "$W/v9/eval_src/native_ba_rgb.py" \
  --depth depth_protected_clip2102 --native /work/new-native/native_parameters.npz \
  --output /work/new-native-ba
```

`new-observations` 必须满足当前研究接口（包括 `outputs/wilor_clip2102`、`head_clip2102` 和深度目录）。这里未传 `--image-observations`，示例使用原始 WiLoR 二维观测；接入 v9 时应先用独立 RGB 头生成 NPZ，再显式传入。它不是任意视频一行命令，也不代表在全新机器上已复现。

原始源文件、模型、协议和验证回执的 SHA 在 `evidence/native_mano_h20_20261019/manifest.json`。未包含 RGBTrack 参考源码、私有产线源码、数据、GT、相机观测和第三方许可资产。

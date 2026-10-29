# 独立双目与原生 MANO runtime

基于 `3a3284b8f61c8b83b54b79a432cb37ecc6620fd1`，本地分支 `codex/ego-hand-runtime-alignment`。原 `research/`、checkpoint、evidence 和 manifest 不改写。HaMR4D `f98db4511a2e6984053e7d02b06513d5e529d0f1` 的 SSH 连接超时、HTTPS 连接中断，未取得私有源码；这里是公开依赖下的等效功能实现，不能称逐行移植或数值复现。

## 执行链

```text
episode manifest：原始 ns 时钟、图像、真实逐帧物理标定
    ├─ 左右 tracking → CPU 解码/共同视场重映射 → cuVSLAM17 双目 VO
    └─ RGB → RF 实例检测/排除重叠 mask → 手轨迹与手侧投票 → 实际 WiLoR 前向
                  └─ LightESS 物理双目 → MANO 表面交点 → 腕深度锚
                       └─ 可选 UniDepth Torch 补盲、与附近物理深度冲突检查
固定相机轨迹 + 原生 MANO 初值/二维观测/腕锚
    → 同一个 MANO BA：真实时间、轨迹共享形状，可选世界根/腕运动/观测保护
    → 相对匹配 BA 控制组的 5/10/30 mm 与 +0.2 px 参数保护
    → 原始源时钟上的 MANO/SO(3) 插值、重新 FK → JSON/H5 → SHA 封存
独立 score 入口先核验封存，再打开标签
```

不读取 GT 轨迹、人工好坏点掩码或评分决定推理。WiLoR 自投影明确标为 `wilor_self_projection`，不能当独立 RGB 观测。多相机原始观测具有相机 ID、来源时钟和模型来源，进入同一个求解图；当前实际 live 只启用 RGB 与 tracking 双目，不假造侧目 RGB。owner 保留 unknown。

DiT 默认关闭；原研究 opt-in 入口完整保留在 `docs/NATIVE_MANO_H20.md`。本 runtime 对 `dit.enabled=true` 明确报错，不把尚未接入/准入的研究头悄悄当成正式后端。future-seed、多假设选择和 MeGo 正式 exporter 还没有迁移，不能以已有工具函数称它们已上线。

## 数据与几何契约

- MANO21/OpenPose，腕 0，米制。统一保存物理手侧、腕 XYZ、根/15 局部旋转、betas、真正 transl 和显式左右反射。几何/FK/导出不使用 FP16；相机数值计算可使用更高精度。
- `state_camera51[:3]`、`state_world51[:3]` 是腕位置，不能直接写为 MANO transl。导出通过实际 SMPLX 前向重新构造同一骨架。
- 整段共享形状只在预测轨迹内，不假定未知操作者的双手属于同一人。相机轨迹固定，BA 不反向改 SLAM。
- 时序项按每一对/三帧的真实 Δt，并归一到参考 60 Hz；不跨 reset、身份改变和长缺口。没有硬碰撞、解剖限位或“绝不跳变”的保证。
- 完整源时钟不由 60 FPS 猜测。短缺口插值 MANO 参数/SO(3) 后重新 FK，显式无效帧继续无效；输出区间外不伪造有效姿态。
- Aria 的 tracking 单色相机不是侧目 RGB。当前输出是有 schema 的内部/Aria 研究 MANO21 JSON/H5，不冒充 MeGo `raw_joints.h5`。

## 通用入口

所有路径通过参数/配置提供，无固定 clip、150 帧、GPU3 或 `/mnt/why` 运行依赖。完整片段是默认；区间用整数 `--start-ns/--end-ns`。

```bash
python -m ego_hand_runtime prepare-hot3d --input /data/extracted-clip --output /work/episode.json
python -m ego_hand_runtime doctor --config /work/runtime.json --output /work/doctor.json
python -m ego_hand_runtime infer --episode /work/episode.json --config /work/runtime.json --output /work/run --dry-run
python -m ego_hand_runtime infer --episode /work/episode.json --config /work/runtime.json --output /work/run
python -m ego_hand_runtime verify --output /work/run
python -m ego_hand_runtime infer --episode /work/episode.json --config /work/runtime.json --output /work/run --resume
```

`--dry-run` 不加载模型、不创建推理目录，`execution_ready=false` 就不能称可以运行。真实推理缺资产返回非零并保留失败记录；不会用随机权重或 CPU VO 顶替。

`--resume` 校验源码、配置、资产、原始输入字节和阶段 SHA；已封存结果只核验/返回，不重新求解。中断前的 head/检测/原始观测阶段可复用，未完成 BA 重新求解；这不是训练 optimizer/RNG 恢复。`--rerun` 在新 attempt 目录运行，旧结果不覆盖。历史预测回放必须显式 `backend=replay`，使用已封存的预测束，不能算 live 检测/深度/VO 验收。

每项模型资产配置必须包含 `path`、预期 `sha256` 和来源。TRT engine 还需要本机 SM、TRT 版本、模型 SHA、shape、精度和 plugin 绑定。LFS 指针拒绝执行。`configs/runtime/` 是资源模板，空 assets 的模板故意不能执行；4090 模板仍从单 worker/单模型槽起步，不能承诺十并发。

## 常驻服务

```bash
python -m ego_hand_runtime serve --config /work/runtime.json --port 8765 --receipts /work/service
python -m ego_hand_runtime submit --episode /work/episode.json --output /work/run-001 --wait
python -m ego_hand_runtime status
python -m ego_hand_runtime stop-service
```

仅监听本机 loopback，不做云端上报。多个客户端共享 RF/WiLoR/UniDepth/headpose/BA 槽；相同槽独占，VO tracker 每个 episode 新建，SDK/模型常驻。队列有界，资源准入超时明确失败；正常停服先排空自己的任务再释放模型。服务源码/配置改变要求重启，避免旧内存模型和新文件混用。

输入、episode、request、worker ID 和父子 span 保留关联。供帧、等待、导入/加载、模型、深度、BA、写出和退出分别计时。线程 CPU、进程 CPU、墙钟及监测开销分开；嵌套 CPU 不相加。服务墙钟包含空闲等待，不能当有效吞吐成绩。资源 hard 是协作边界保护，不是 CUDA allocator 或容器硬配额。

环境只增量恢复；完整匹配的 K8s service-link 组才清理，API/代理/不完整组保留。继承的 `HAMR4D_*` 清理后才应用显式配置。实际解码由 `decode` 配置及运行回执确认，不靠环境变量宣称生效；当前只准入 CPU 图像/FFmpeg 有界缓存，NVDEC 配置会报不支持。

## 镜像与本机验收

`tools/prepare_runtime_image.py` 生成独立离线构建上下文；依赖 wheel 和源码逐个 SHA 固定，第三方源码与许可按需烘焙，许可模型只读挂载。`docker/runtime/Dockerfile` 不在运行时联网取源码/权重，校验失败即停。`tools/build_runtime_ess.py` 从指定 SHA 的 ONNX/plugin 在空闲目标 GPU 上重建，不转换外机 engine。

镜像默认入口先核验烘焙源码 manifest，再导入推理模块；已在一次性容器中改动源码，确认在模型导入前失败。资产另在首次执行按完整 SHA 校验，常驻进程内只对未变化的只读实体复用哈希结果。

本机 Windows Python 是隔离的 CPU 验证环境；WSL/Ubuntu22 容器已实跑公开 cuVSLAM GPU 后端。使用本机已有基础镜像构建的镜像只供本地审计，没有发布；新机器还需审计自己的基础镜像、依赖和许可证。

最新执行状态见 `evidence/runtime_alignment_20261029/validation.json`。程序成功、几何检查、质量准入、性能准入分别记录。没有独立 GT，本次不报告绝对精度提升、坏点恢复或原正确点损伤；也不拿回放/单元测试代替整线 GPU 验收。

### 2026-10-29 本机实跑

RTX 4080 Laptop 12 GiB，WSL 容器，公开 HOT3D Aria 的一段完整 150 RGB 帧（约 30 FPS）。这是一次真实前端运行，不是历史预测回放。

- cuVSLAM GPU 双目 VO、RF、实际 WiLoR、LightESS TRT、UniDepthV2 Torch、原生 MANO BA 全部参与。150 个源时刻完整交付，双手各 149 个有效帧；首帧没有可插值的头部姿态，保留无效。
- 300 个手部深度支持中，147 个直接 LightESS、2 个直接 UniDepth，其余 151 个为有时间边界的同轨迹锚插值；不能称逐帧独立测深。
- 短测、全片和恢复请求在同一常驻服务执行；六类模型槽各加载一次。全片请求墙钟 110.258 秒，属于已热加载的诊断运行，不能当冷服务批次吞吐。
- 完整离线依赖镜像重新冷加载同片运行成功，推理回执 185.641 秒；与前一全片运行的 6,258 个共同预测点逐值相同，源时钟和有效掩码一致。此项仅证明本次打包一致性，不是与私有产线或 GT 的比较。RF/WiLoR 仍为 PyTorch，尚未追齐产线吞吐。
- CPU 独立解码导出 MANO 参数，6,258 个点的最大 FK 差异为 `8.94e-8 m`；只证明参数与骨架一致，不是真值误差。
- 原快照 190 个源码和 4 个 checkpoint 实体 SHA 核验通过。实际 CUDA/许可 MANO 测试另有日志；源码、配置、SDK、engine/plugin、最终镜像和回执分别记录。

本机 ToDesk 图形后台仍在，未停止其他任务。离线完整依赖镜像为 `ego-hand-runtime:alignment-complete-20261029`，带启动完整性核验的最终镜像为 `ego-hand-runtime:alignment-verified-20261029`；只在本机保存，没有推送仓库或发布镜像。侧目、MeGo、独立 RGB 头、custom-near SDK、CPU ORB 回退、RF/WiLoR C++ TRT 和多 episode 冷启动吞吐尚未准入。详见验收目录 README 和原始回执索引。

## 遇到的问题与处理

| 问题 | 处理 |
|---|---|
| 私有参考不可访问 | 使用公开闭包，明确不声称源码/数值一致 |
| 研究入口混有固定片段/路径/设备 | 新建 runtime CLI、数据契约和常驻服务，旧快照不改 |
| 模板腕不在 MANO 原点、左右标签与前向可能不同 | 按真实 LBS 腕恢复 transl，明确反射，重新 FK 核验 |
| Aria 在片段中更新标定，短测未覆盖 | 保留逐帧标定，重映射到固定虚拟双目；原点漂移门限仍检查，保留失败日志 |
| 原始像素/时钟与导出可能错位 | 保留 source index 和整数 ns，JSON/H5/FK 联合核验，显式无效不补成有效 |
| Torch 首次下载截断、SHA 不符 | 拒绝安装并保留错误，再从官方源完整下载；没有跳过哈希 |
| Docker 在线仓库不可达 | 使用本机已审计缓存和离线依赖上下文，完整保存来源/SHA，不临时注入源码 |
| GPU engine/SDK 与机器不匹配 | 实际加载库 SHA 检查，本机重建 engine，完整后端验收另列 |
| RF 导入把全局矩阵精度改成 high，影响 MANO 导出 | 几何/BA 使用局部 IEEE 精度并恢复视觉设置；Torch 后端共享精度锁，不放宽 FK 核验门限 |
| 基础镜像的系统包不能被 pip 卸载 | 在镜像内新建隔离 venv，不修改 ROS/系统 Python 包 |
| 每个二维观测分别发 GPU 小算子开销大 | 按相机模型成批投影，仍保留每帧真实内外参；与冻结目标函数对照 |

CPU 测试：`EGO_RUNTIME_TEST_MANO=/readonly/MANO_RIGHT.pkl python -m unittest discover -s tests/runtime -v`。缺许可实体时相关真实几何测试会 skip；skip 不能称完整验证通过。冻结研究快照仍用 `tools/verify_native_mano_snapshot.py` 单独核验。

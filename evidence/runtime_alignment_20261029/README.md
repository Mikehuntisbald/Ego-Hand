# 双目 runtime 本机验收（2026-10-29）

本目录是新增证据，原研究 manifest 不改。固定基础为 `3a3284b8f61c8b83b54b79a432cb37ecc6620fd1`，当前实现说明在 `docs/RUNTIME_ALIGNMENT.md`。私有参考未获得；公开等效功能不代表 HaMR4D 数值复现。

## 已执行

- 本机 RTX 4080 Laptop，公开 cuVSLAM17 GPU 双目 VO、RF/WiLoR、真实 LightESS TRT/UniDepthV2 Torch、原生 MANO BA 和完整源时钟导出。
- 一个公开 HOT3D Aria 片段，完整 150 个 RGB 源时刻（约 30 FPS）；300 个手观测中 298 个进入 BA，首帧缺头部插值支持仍无效。
- 同一服务先短测再全片、再恢复：六类模型槽各加载一次；正常排空退出，原始监测尾段保留。
- 36 项测试，实际 MANO 和显式 CUDA 测试启用，无 skip。190 个原源码与 4 个 checkpoint 实体核验通过。
- 封存结果后，CPU 独立 MANO 重新生成 6,258 个点，最大 FK 差异 `8.94e-8 m`。这证明参数与输出一致，不是真值精度。
- 完整离线镜像冷加载运行也通过（推理回执 185.641 秒）；6,258 个预测点与先前热服务运行逐值相同，时钟/有效掩码一致。最终启动校验镜像成功恢复同一封存输出；源码篡改探针则在模型导入前失败。

`validation.json` 汇总结果及边界；`effective_config.json` 是实际生效配置；`environment.json` 记录实际导入位置和版本；`local_artifacts.json` 索引本机保留的原始日志、输出、监测和自然图像叠图及 SHA。本目录不复制权重或图像。

新增源码/文档/验收清单使用独立 manifest；与原研究清单分别核验：

```bash
python tools/verify_runtime_release.py --manifest evidence/runtime_alignment_20261029/release_manifest.json
python tools/verify_native_mano_snapshot.py
```

## 复现方式

模型和标定需要实体及精确 SHA。配置中的 `/assets`、`/ess`、`/data`、`/output` 是只读资产/输入和可写输出的容器挂载点，不是宿主机预置路径。许可 MANO 不随 Git 或镜像分发。资产源位置见本机 `runtime_local/asset_inventory_pinned.json`。

通用入口：

```bash
python -m ego_hand_runtime prepare-hot3d --input /data/extracted-clip --output /output/episode.json
python -m ego_hand_runtime infer --episode /output/episode.json --config /output/runtime.json --output /output/new-run --dry-run
python -m ego_hand_runtime serve --config /output/runtime.json --port 18765 --receipts /output/service
# 在同一容器的另一个客户端提交；没有外部网络或云端服务。
python -m ego_hand_runtime submit --url http://127.0.0.1:18765 --episode /output/episode.json --output /output/new-run --wait
python -m ego_hand_runtime submit --url http://127.0.0.1:18765 --episode /output/episode.json --output /output/new-run --resume --wait
python -m ego_hand_runtime stop-service --url http://127.0.0.1:18765
```

容器使用 `--network=none` 仍可在容器内部 loopback 提交；客户端可通过 `docker exec` 运行。内部端口不对外暴露。实际本机输入、配置和结果在 `runtime_local/container_runs/`，重新执行应使用新输出目录，或者显式 `--resume` 核验旧结果。

完整离线构建由 `tools/prepare_runtime_image.py` 配合 `docker/runtime/Dockerfile` 执行。当前闭包为 Python3.10/Ubuntu22，121 个固定依赖文件及额外 dill wheel；WiLoR、UniDepth 源码和各自 license 烘焙，运行时不取源码/权重。构建日志和源码清单留在 `runtime_local/image_context_complete/`、`runtime_local/image_build_complete.log`。基础镜像必须按 `validation.json` 的本机审计 ID 核验，不能只信可变的 tag。

```bash
python tools/prepare_runtime_image.py --output /work/context \
  --wheelhouse /readonly/wheelhouse --extra-wheel /readonly/dill-0.4.0-py3-none-any.whl \
  --wilor-source /readonly/WiLoR --unidepth-source /readonly/UniDepth
docker build --network=none --build-arg BASE_IMAGE=<audited-local-base> -t ego-hand-runtime:local /work/context
```

本机 source/config/资产 seal、恢复和 JSON/H5/FK 核验已执行；更换模型、GPU、SDK 或像素链路要重新验证。不会把 H20 引擎作为本机 SM89 引擎使用。

`alignment-complete-20261029` 由完整离线依赖配方构建；最终 `alignment-verified-20261029` 在构建时加入源码启动核验层，推理包内容完全相同。实际模型/输入只读挂载。默认入口先核验源码 manifest；一次性容器的源码篡改探针已确认失败退出。完整依赖与源码 SHA 分别见 `dependencies.lock.json`、镜像源码清单及新 release manifest。没有向镜像或仓库分发许可权重。

## 未准入

没有打开独立 GT：不能报告绝对精度提升、原正确点损伤/困难点恢复或跳变误差。原始预测运动仍有长尾，不能保证不跳。全片 110.258 秒为短测后热模型诊断请求；服务 562.482 秒包括空闲等待与检查，两者都不是冷启动批次吞吐。ToDesk 图形后台保留，因此也不是排除外部干扰的性能对照。

DiT 默认关闭；原 v7 研究入口完整保留，新 runtime 尚不提供准入的 DiT profile。侧目、MeGo 正式格式、future-seed/完整多候选路径、独立 RGB 头、CPU ORB 恢复、RF/WiLoR C++ TRT、NVDEC 和多 episode 性能另行验收。详见 `validation.json`，不据单片结果替换生产默认。

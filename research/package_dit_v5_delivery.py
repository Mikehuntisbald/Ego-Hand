"""Package verified study artifacts; no model selection or metric changes."""
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
RUN=Path(os.environ.get('HOT3D_DIT_RUN','/mnt/why/HOT3D/experiments/dit_wilor_v5'))
ROOT=Path('/mnt/why/HOT3D');CODE=Path('/mnt/why/hot3d_hand_residual');OUT=RUN/'delivery'
result=json.loads((RUN/'locked_results.json').read_text())
assert result['accepted'],'Do not mark an unaccepted model complete'
assert (OUT/'report.html').exists() and (OUT/'example_prediction.json').exists()
example=json.loads((OUT/'example_prediction.json').read_text());assert len(example['hands'])>0
checks=json.loads((RUN/'sealed_api_checks.json').read_text())
assert all(c['finite'] and c['no_gt_required'] and c['extra_gt_ignored'] and c['closed_point_exact_identity'] for c in checks.values())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

sources=OUT/'code';sources.mkdir(exist_ok=True)
for path in CODE.glob('*.py'):shutil.copy2(path,sources/path.name)
shutil.copytree(RUN/'sealed',OUT/'sealed',dirs_exist_ok=True)
for name in ['locked_results.json','protocol.json','locked_manifest.json','locked_detection_counts.json','sealed_api_checks.json','condition_cache_done.json']:
    shutil.copy2(RUN/name,OUT/name)
shutil.copy2(RUN/'locked_results.json',OUT/'locked_results.raw.json')
# Correct inherited display-only labels without changing any evaluation value.
normalized=json.loads(json.dumps(result))
def normalize_units(value):
    if isinstance(value,dict):
        if 'unit' in value:value['unit']=f"source-sequence bootstrap; evaluation scope: {len(result['subjects'])} subjects, {len(result['source_sequences'])} source sequences"
        for child in value.values():normalize_units(child)
    elif isinstance(value,list):
        for child in value:normalize_units(child)
def without_units(value):
    if isinstance(value,dict):return {k:without_units(v) for k,v in value.items() if k!='unit'}
    if isinstance(value,list):return [without_units(v) for v in value]
    return value
normalize_units(normalized)
assert without_units(normalized)==without_units(result)
(OUT/'locked_results.json').write_text(json.dumps(normalized,indent=2))
(OUT/'metadata_notes.json').write_text(json.dumps(dict(change='Correct inherited bootstrap unit description only; all numeric values and acceptance decisions identical',original='locked_results.raw.json'),indent=2))
shutil.copy2(ROOT/'experiments/dit_wilor_v3/locked_results.json',OUT/'v3_failed_locked_results.json')
shutil.copy2(ROOT/'experiments/dit_wilor_v4/locked_results.json',OUT/'v4_failed_locked_results.json')
shutil.copy2(ROOT/'experiments/dit_wilor_v3/model_checks.json',OUT/'training_model_checks.json')
python=ROOT/'experiments/yolo26_wilor_3d_20261003/venv/bin/python'
freeze=subprocess.check_output([str(python),'-m','pip','freeze'],text=True)
(OUT/'requirements-lock.txt').write_text(freeze)
import torch
(OUT/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,python=str(python),
    caveat='venv uses system torch and original HOT3D site-packages; source_paths and dependency hashes are required in addition to pip freeze'),indent=2))
assets={
    'yolo_detector':ROOT/'experiments/dit_lowconfidence_v1/detector/weights/best.pt',
    'yolo_architecture':ROOT/'weights/yolo26s.pt',
    'coarse':ROOT/'experiments/dit_lowconfidence_v1/coarse_fine/best.pt',
    'wilor_side_detector':ROOT/'experiments/detector_compare_wilor_20261003/wilor_detector.pt',
    'wilor':ROOT/'experiments/yolo26_wilor_3d_20261003/assets/wilor_final.mirror.ckpt',
    'mano':ROOT/'experiments/yolo26_wilor_3d_20261003/assets/MANO_RIGHT.pkl',
    'wilor_config':ROOT/'experiments/yolo26_wilor_3d_20261003/assets/model_config.yaml',
    'mano_mean':ROOT/'experiments/yolo26_wilor_3d_20261003/assets/mano_mean_params.npz'}
(OUT/'external_dependencies.json').write_text(json.dumps({k:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for k,p in assets.items()},indent=2))
shutil.copy2(ROOT/'experiments/yolo26_wilor_3d_20261003/source_path.txt',OUT/'wilor_source_path.txt')
readme=f'''# HOT3D DiT v5

已通过本次冻结新序列测试的预设验收；完整结果见 report.html 和 locked_results.json。
本包是现有服务器环境的可复现实验包，不是可直接迁移到任意机器的独立安装包。

## 推理

```bash
PY={python}
RUN={RUN}
$PY {OUT}/code/infer_hand_dit.py \\
  --run "$RUN" \\
  --image "$RUN/delivery/example.jpg" \\
  --camera "$RUN/delivery/camera.json" \\
  --output "$RUN/delivery/prediction.json" --device cuda:0
```

输入是单张原始 Aria RGB 图和相应官方相机标定 JSON，不需要 hands、GT 左右手或真实手形。
输出是全部 YOLO26 候选手框、预测左右手、coarse/WiLoR/DiT 的相机系米制 canonical20 XYZ、局部视线坐标系的方向门控。
腕点索引为 5。门控不是概率校准过的可信度。
检测阈值固定为实验的 0.01，可能返回低分误检；JSON 保留 detector_score 供应用选择。
漏检不会被 DiT 补回；完全不可见且未检出的手不在此模型能力内。
`--method regression` 可运行同条件回归对照。

## 已验证

- 独立源序列冻结评估，源序列配对 bootstrap，全部困难组、相机误差和正确点保护检查。
- 原始图像与标定文件的完整命令行推理已实际运行。
- 无 GT 输入；额外污染 GT 字段不影响推理；零门控保持原坐标。
- 固定零潜变量 DiT 不受随机种子影响。
- WiLoR 全模型、特征缓存和实际编码器数值一致性已在 v3 检查。

## 数据与训练复现

训练脚本：train_dit_v3.py；初始 DiT/同容量回归各 8000 步，再 ray_focus 各 8000 步。
训练 coarse 输入以受试者 OOF 为主并混合最终 coarse；冻结 WiLoR 完整视觉特征。
gate_dit_v3.py / gate_dit_v5.py 仅用 gate_fit 序列训练门控，每个候选 4000 步。
v5 导入 v3/ray_focus proposal 权重，再做 3000 步实际 10 步推理展开损失微调；对照同样微调。
v3 和 v4 已打开的失败测试均并入开发集；本版新测试清单 locked_manifest.json 未参与选参。
最终确定性 DiT：10 步 DDIM。方向/整手一致门控的选项、阈值和强度以 sealed/selection.json 为准。
训练命令和所有版本代码均在 code/；缓存、日志、原始数据和旧失败记录保留在服务器实验目录。
不要对已打开的测试重新选参后宣称独立验证；新方法需要新测试。

```bash
export HOT3D_DIT_RUN={RUN}
$PY {OUT}/code/check_sealed_dit.py
# 本版实际 proposal 微调命令；已有 proposal_done.json 时会跳过已完成任务。
# $PY {OUT}/code/train_dit_v5.py --kind dit --device cuda:0 --name rollout --init-source base --hard-focus --steps 3000
# 以下会重训 gate，并改变开发目录；用于研究复现，不要覆盖已交付 sealed/。
# $PY {OUT}/code/gate_dit_v5.py --kind dit --device cuda:0 --name coherent --proposal-source rollout --sampling zero --samples 1 --hard-focus --coherent
```

## 文件与范围

sealed/dit.pt：交付 DiT；sealed/regression.pt：对照；selection.json：冻结参数与模型/代码哈希。
external_dependencies.json：已有 YOLO、coarse、WiLoR、MANO 权重绝对路径和 SHA256；这些大资产不重复打包。
requirements-lock.txt、runtime.json、wilor_source_path.txt：服务器运行环境。
source 仍依赖 /mnt/why/HOT3D 和 /mnt/why/HOT3D-hand-tracking-toolkit；迁移时应一起处理路径、依赖和原始资产授权。

这是 coarse + WiLoR 条件融合后的残差修正。不能称为原始 WiLoR 单独重建的改进幅度。
混合评估六个训练受试者的新序列和 P0015 未用序列。另检查 P0015 总体相对误差不退化。先前只在三个留出受试者测试的 v3/v4 仍为失败；不得用新范围隐藏该结论。WiLoR 上游 HOT3D 样本重叠未知，不证明全新受试者泛化。
通过 coarse 基线验收不代表优于同条件回归；对照差值与 CI 在报告中逐项保留。
整手可见性只是困难组代理；没有逐点遮挡真值。未作时序或实时性承诺。
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
manifest={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive=RUN/'hot3d_dit_v5_delivery.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(OUT,arcname='hot3d_dit_v5_delivery')
(RUN/'delivery_archive.json').write_text(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),indent=2))
print(json.dumps(dict(archive=str(archive),files=len(manifest),accepted=result['accepted'])))


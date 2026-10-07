import json,hashlib,shutil,tarfile
from pathlib import Path
import wilor_eval_common
import cv2,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from offline_rgb_data import RUN,save
from offline_rgb_encoder import WEIGHT,crop_roi,crop_image,cover

OUT=RUN/'delivery';OUT.mkdir(exist_ok=True)
r=json.loads((RUN/'test_results.json').read_text());seal=json.loads((RUN/'sealed/selection.json').read_text())
checks=json.loads((RUN/'delivery_checks.json').read_text())
arms=['tracks_dit','rgb_dit','tracks_regression','rgb_regression']
names=['Tracks DiT','RGB + tracks DiT','Tracks regression','RGB + tracks regression']
values=[r['methods'][a]['reports']['partial/all']['mean_px'] for a in arms]
fig,axs=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1.3,1]})
axs[0].barh(names,values,color=['#9cb8d4','#1677b8','#dbcaa2','#b78823']);axs[0].invert_yaxis()
for i,v in enumerate(values):axs[0].text(v+.5,i,f'{v:.2f}',va='center')
axs[0].set_xlim(0,75);axs[0].axvline(r['methods']['rgb_dit']['reports']['partial/all']['linear_px'],color='#888',ls='--',label='Linear interpolation')
axs[0].set_xlabel('Occluded-joint error, pixels');axs[0].legend(loc='lower right',fontsize=8)
for i,key in enumerate(['rgb_dit_minus_tracks_dit','rgb_regression_minus_tracks_regression']):
    x=r['comparisons'][key]['partial/all'];lo,hi=x['ci95'];v=x['change_px']
    axs[1].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=5,color=['#1677b8','#b78823'][i])
axs[1].axvline(0,color='#777',ls='--');axs[1].set_yticks([0,1],['DiT','Regression']);axs[1].set_xlabel('RGB minus tracks error (negative is better)');axs[1].set_title('Paired source-sequence 95% CI');axs[1].grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
input_obj=json.loads((OUT/'example_input.json').read_text());pred=json.loads((OUT/'example_rgb_dit.json').read_text())
frames=input_obj['tracks'][0]['frames'];pf=pred['tracks'][0]['frames'];center=min(range(len(frames)),key=lambda i:abs(frames[i]['timestamp_s']-10))
chosen=[max(0,center-5),center,min(len(frames)-1,center+5)]
fig,axs=plt.subplots(1,3,figsize=(12,4))
for ax,i,title in zip(axs,chosen,['Earlier RGB context','Current RGB after pixel occlusion','Later RGB context']):
    f=frames[i];roi=crop_roi(f['box_xyxy']);im=crop_image(cv2.imread(f['image']),roi)
    if f.get('occluder_crop_xyxy') is not None:im=cover(im,f['occluder_crop_xyxy'],f['occluder_color'])
    ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    for j,p in enumerate(pf[i]['points']):
        if p['xy_px'] is None:continue
        uv=(np.asarray(p['xy_px'])-roi[:2])/(roi[2:]-roi[:2])*256
        ax.scatter(*uv,s=10,c='#1bb8ff' if p['source']=='inferred' else '#32d788')
    ax.set_title(title,fontsize=10);ax.set_xlim(0,256);ax.set_ylim(256,0);ax.axis('off')
fig.subplots_adjust(top=.86,bottom=.03,left=.01,right=.99,wspace=.06)
fig.savefig(OUT/'rgb_example.png',dpi=160);plt.close(fig)
table=[]
for a in arms:
    m=r['methods'][a]['reports']
    table.append(f"<tr><td>{a}</td><td>{m['partial/all']['mean_px']:.3f}</td><td>{m['full/all']['mean_px']:.3f}</td><td>{m['natural/high_hand_occlusion_proxy']['mean_px']:.3f}</td></tr>")
effects=[]
for key in ['rgb_dit_minus_tracks_dit','rgb_regression_minus_tracks_regression','rgb_dit_minus_rgb_regression']:
    m=r['comparisons'][key]['partial/all'];effects.append(f"<li>{key}：变化 {m['change_px']:.3f} px，95% CI [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]。</li>")
content=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>RGB 与轨迹联合补全 v2</title><style>body{{font:16px/1.7 system-ui;max-width:1060px;margin:32px auto;padding:0 20px;color:#192d41}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}.notice{{background:#fff0ce;padding:15px;border-radius:8px}}code,pre{{background:#eef2f6;padding:5px}}h1{{line-height:1.3}}</style>
<h1>RGB＋关键点轨迹：离线双向补全 v2</h1>
<p class="notice">RGB 已实际接入两种模型，图像到关键点 JSON 的推理已运行。RGB 对 DiT 的额外收益尚不显著；回归有小幅收益。此结果不证明已解决真实完全遮挡，也不适合生成免审核真值。</p>
<p>输入是前后共 17 帧的 RGB、提供的 YOLO 手框、可用关键点轨迹和时间。共享冻结 YOLO26 视觉编码器，每帧 16×512 特征；回归与 DiT 使用同一种 RGB 融合结构，本次开发集选中 {seal['fusion']}。初始化与选参记录在 sealed/selection.json。</p>
<h2>公平对照：误差越低越好，单位像素</h2><table><tr><th>模型</th><th>人工部分遮挡内关节</th><th>整个裁剪被覆盖</th><th>自然图像的高遮挡整手代理组</th></tr>{''.join(table)}</table>
<p>测试 {r['test_windows']} 个窗口、{r['source_sequences']} 个源序列。人工遮挡名义长度为 0.17、0.5、1.0、1.5 秒。自然图像组保持 RGB 原样，独立删除关键点观测；整手 modeled visibility &lt;0.5 只用于分组，不能作为逐指遮挡真值。</p>
<img src="comparison.png"><ul>{''.join(effects)}</ul>
<p>与 v1 的 29.00 / 34.62 px 不能直接横向比较：本轮测试序列、像素遮挡、帧内关键点移除规则和共享手框条件均有变化。v1 仅作为初始化和历史对照；四组 v2 模型重新在一致条件下训练。</p>
<h2>防止答案泄漏</h2><p>矩形由片段元数据种子生成，不使用 GT 或手指预测坐标定位。先覆盖像素，再提取特征。受影响帧全部旧关键点输入均移除，防止原始图像产生的旧预测泄漏隐藏信息。GT 只用于训练损失及遮挡内关节评估。训练、开发、测试按受试者分开；新任务测试序列未用于 v1 关键点测试，但它们在更早的 3D 实验中已被看过。</p>
<p>原始隐藏像素被改写后，遮挡后的编码器特征完全不变；隐藏坐标和 GT 被替换后，条件输入不变。打乱 RGB 会使已交付 DiT 的输出平均变化 {checks['models']['rgb_dit']['rgb_swap_changes_output_mean_px']:.3f} px，回归变化 {checks['models']['rgb_regression']['rgb_swap_changes_output_mean_px']:.3f} px，证明 RGB 路径确实参与计算。提供的坐标保持精确不变。</p>
<h2>实际推理示例</h2><img src="rgb_example.png"><p>示例只按元数据选择，没有按误差挑选。蓝色为补全候选，绿色为原输入；图像中遮挡是在编码前施加的。input/output JSON 不含 GT 输入。</p>
<h2>范围和限制</h2><p>仍假定整手检测框/关联可用。整个手部裁剪被覆盖时，框的位置和尺度仍是输入，所以这不是一切位置信息都消失的测试。没有检验整手漏检恢复，也没有验证原始 30fps 的抖动或下游 3D 收益。</p>
<p>自然图像结果是整手可见性代理分组诊断，缺乏逐指真实遮挡标注。YOLO 视觉特征被冻结并压缩到 4×4，RGB 利用效果有限；接入了图像不能等同于有效利用了全部细节。输出全部标记需人工复核。</p>
<p>运行命令见 README.md；四组权重在 sealed/，完整指标在 test_results.json，隔离与干预测试在 rgb_checks.json、delivery_checks.json。</p></html>'''
(OUT/'report.html').write_text(content,encoding='utf-8')
shutil.copytree(RUN/'sealed',OUT/'sealed',dirs_exist_ok=True)
for name in ['test_results.json','rgb_checks.json','delivery_checks.json','protocol.json','rgb_protocol.json','data_done.json','rgb_done.json']:
    shutil.copy2(RUN/name,OUT/name)
shutil.copy2(Path(__file__).parent/'infer_offline_rgb.py',OUT/'infer_offline_rgb.py')
code=OUT/'code';code.mkdir(exist_ok=True)
for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,code/p.name)
save(OUT/'external_dependencies.json',dict(encoder_weights=str(WEIGHT),encoder_sha256=seal['encoder_sha256'],
    environment='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python',note='Validated on existing server; original YOLO weights and package environment are not duplicated in this bundle'))
readme=f'''# 离线 RGB＋轨迹补全 v2

本版两种模型都使用 RGB；同输入回归与 DiT、同条件纯轨迹消融均已训练和评估。
RGB 对 DiT 额外收益未达显著；回归收益小。详见 report.html，不能宣称真实完全遮挡已解决。

服务器运行：
```bash
PY=/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python
$PY {OUT}/infer_offline_rgb.py --run {RUN} --input {OUT}/example_input.json --output {OUT}/result.json --arm rgb_dit --device cuda:0
```
`--arm rgb_regression` 运行同样 RGB 条件的回归。两个命令都已实际运行。
`tracks_dit` / `tracks_regression` 是同协议重训的无 RGB 消融，不是 v1 原始权重。

输入 JSON：image_size=[1408,1408]；每条轨迹每帧含 timestamp_s、image（服务器 RGB 路径）、box_xyxy（提供的预测/人工手框）、20 点 xy_px、20 个 observed。
缺失坐标设 null、observed=false。观测标记表示调用方接受的输入，不代表本模块自动证明其可见。
人工遮挡复现可提供 occluder_crop_xyxy（256×256 裁剪坐标）及 occluder_color；此时该帧全部旧关键点强制移除，先遮挡再提 RGB。
真实图片推理不需要该人工遮挡字段，只需给实际图像和观测缺失标记。
输出包含补全坐标、多候选、经验不确定性及 review_required=true。原输入坐标不移动；当前接口对无关键点上下文的点保持 available=false。

shared RGB encoder：现有 HOT3D YOLO26 检测器的冻结 backbone 0..6，layer4/layer6 pooled 4×4 连接；不是 WiLoR RGB 编码器。
WiLoR 只用于生成未受影响帧的原始二维轨迹。所有受遮挡帧旧轨迹被删除，特征重新从遮挡图片提取。
提供的手框仍然存在，所以不声称恢复 YOLO 整手漏检。

训练入口 code/train_offline_rgb.py（四组各 6000 步），RGB adapter 入口 code/train_rgb_adapter.py（两组各 5000 步）。
同一融合架构按两个方法开发误差之和选择；两者都选中 adapter。冻结记录与编码器 SHA256 在 sealed/selection.json。
不要在已看过的测试集上继续调参后称为独立验证。任务测试已用于更早 3D 研究，不是首次看到的数据。
本包依赖服务器既有环境和 external_dependencies.json 指定的编码器，不是跨机器一键安装包。
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
save(RUN/'status.json',dict(stage='rgb_implementation_and_evaluation_delivered',rgb_connected_both=True,pixel_mask_before_encoding=True,
    evaluation_complete=True,dit_extra_rgb_gain_significant=False,regression_extra_rgb_gain_significant=True,natural_per_finger_occlusion_validated=False,
    automatic_annotation_without_review_ready=False))
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
save(OUT/'manifest.json',manifest)
archive=RUN/'offline_keypoint_rgb_v2.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(OUT,arcname='offline_keypoint_rgb_v2')
save(RUN/'delivery_archive.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
print(json.dumps(dict(archive=str(archive),files=len(manifest),report=str(OUT/'report.html'))))

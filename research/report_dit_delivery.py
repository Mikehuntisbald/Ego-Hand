"""Evidence report and representative successes/failures, never model selection."""
import os
import json
import html
from pathlib import Path
import wilor_eval_common as common
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from export_hand_labels import EDGES
from metrics_3d import EVAL_INDICES

RUN=Path(os.environ.get('HOT3D_DIT_RUN','/mnt/why/HOT3D/experiments/dit_wilor_v4'))
OUT=RUN/'delivery';OUT.mkdir(exist_ok=True)
r=json.loads((RUN/'locked_results.json').read_text())
seal=json.loads((RUN/'sealed/selection.json').read_text())
rows=json.loads((RUN/'locked_rows.json').read_text())
p=np.load(RUN/'locked_dit_predictions.npz');c=p['coarse'];gt=p['gt'];d=p['predicted']
rel=lambda x:x-x[:,5:6]
before=np.linalg.norm(rel(c)-rel(gt),axis=-1)[:,EVAL_INDICES].mean(-1)*1000
after=np.linalg.norm(rel(d)-rel(gt),axis=-1)[:,EVAL_INDICES].mean(-1)*1000
gain=before-after
order=np.argsort(gain)
selected=[('Largest regression',order[0]),('Lower quartile',order[len(order)//4]),('Median',order[len(order)//2]),('Upper quartile',order[len(order)*3//4]),('Largest gain',order[-1])]
fig,axes=plt.subplots(len(selected),2,figsize=(12,4*len(selected)))
colors=['#ef9f25','#1c9ed6','#30a66a'];labels=['Coarse','DiT','GT']
examples=[]
for ri,(label,j) in enumerate(selected):
    row=rows[int(p['sample_indices'][j])];im=cv2.cvtColor(cv2.imread(row['image']),cv2.COLOR_BGR2RGB)
    cam=common.from_json(row['camera']);box=np.asarray(row['box']);center=(box[:2]+box[2:])/2
    span=max(box[2:]-box[:2])*1.6;lo=np.maximum(center-span/2,0);hi=np.minimum(center+span/2,[im.shape[1],im.shape[0]])
    ax=axes[ri,0];ax.imshow(im)
    for pose,color,name in zip([c[j],d[j],gt[j]],colors,labels):
        uv=cam.eye_to_window(pose)
        for a,b in EDGES:ax.plot(uv[[a,b],0],uv[[a,b],1],color=color,linewidth=1.1,alpha=.9)
        ax.scatter(uv[:,0],uv[:,1],c=color,s=7,label=name)
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_aspect('equal');ax.legend(fontsize=8)
    ax.set_title(f'{label}: relative {before[j]:.1f} -> {after[j]:.1f} mm')
    ax=axes[ri,1]
    for pose,color,name in zip([c[j],d[j],gt[j]],colors,labels):
        q=(pose-pose[5])*1000
        for a,b in EDGES:ax.plot(q[[a,b],0],q[[a,b],2],color=color,linewidth=1.2)
        ax.scatter(q[:,0],q[:,2],c=color,s=10,label=name)
    ax.set_aspect('equal',adjustable='datalim');ax.invert_yaxis();ax.grid(alpha=.25)
    ax.set_xlabel('Wrist-relative camera X (mm)');ax.set_ylabel('Wrist-relative camera Z (mm)');ax.legend(fontsize=8)
    ax.set_title(f"{row['subject']} / clip {row['clip']} / frame {row['frame']}")
    examples.append(dict(selection=label,row_index=int(p['sample_indices'][j]),image=row['image'],before_mm=float(before[j]),after_mm=float(after[j])))
fig.tight_layout();fig.savefig(OUT/'examples.png',dpi=130);plt.close(fig)
(OUT/'examples.json').write_text(json.dumps(examples,indent=2))

names={'low_pose_confidence':'低姿态置信度','high_occlusion_in_view':'高遮挡且在视野内','multiple_ray_aligned_fingers':'多指沿视线重叠'}
table=[]
for kind,title in [('dit','DiT'),('regression','同条件回归')]:
    m=r['methods'][kind]
    for k,e in m['evidence'].items():
        metric='axial_relative' if k=='multiple_ray_aligned_fingers' else 'relative';g=m['groups'][k][metric]
        ci=e['ci95_change_mm'];ci_text='不足以估计' if ci is None else f'[{ci[0]:.3f}, {ci[1]:.3f}]'
        table.append(f"<tr><td>{title}</td><td>{names[k]}</td><td>{e['joints']}</td><td>{g['before_mm']:.3f} → {g['after_mm']:.3f}</td><td>{e['improvement_pct']:.2f}%</td><td>{ci_text}</td><td>{'通过' if e['passed'] else '未通过'}</td></tr>")
overall=[]
for kind,title in [('dit','DiT'),('regression','同条件回归')]:
    m=r['methods'][kind];a=m['overall'];b=a['coarse'];f=a['refined']
    overall.append(f"<tr><td>{title}</td><td>{b['mpjpe19_mm']:.3f} → {f['mpjpe19_mm']:.3f}</td><td>{b['wrist_relative_mpjpe19_mm']:.3f} → {f['wrist_relative_mpjpe19_mm']:.3f}</td><td>{100*a['correct_joints_harmed_fraction']:.2f}% / {100*m['groups']['all']['correct_relative_joints_harmed_fraction']:.2f}%</td></tr>")
status='通过预先设定的全部验收条件' if r['accepted'] else '尚未通过全部验收，不能作为已完成模型交付'
subject_table=''
for subject,sr in r['methods']['dit'].get('per_subject',{}).items():
    a=sr['coarse']['wrist_relative_mpjpe19_mm'];b=sr['refined']['wrist_relative_mpjpe19_mm']
    subject_table+=f'<tr><td>{subject}</td><td>{"未用于训练" if subject=="P0015" else "训练受试者的新序列"}</td><td>{a:.3f} → {b:.3f}</td></tr>'
comparison=[]
for k,g in r['dit_vs_regression'].items():
    metric='axial_relative' if k=='multiple_ray_aligned_fingers' else 'relative';v=g.get(metric,{})
    comparison.append(f"<li>{names.get(k,'总体')}：DiT − 回归 = {v.get('change_mm',float('nan')):.3f} mm，95% CI {v.get('ci95')}。</li>")
content=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>HOT3D DiT 验证与交付</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1100px;margin:35px auto;padding:0 24px;color:#17283b}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #ddd;padding:9px;text-align:left}}code,pre{{background:#f2f5f8;padding:4px;overflow:auto}}img{{max-width:100%}}.status{{background:#e9f2fa;padding:16px;border-radius:8px}}</style>
<h1>HOT3D：YOLO26 + WiLoR 条件 + 残差 DiT</h1><p class="status">{status}</p>
<p>冻结后一次评估：{r['sampled_frames']} 帧，{r['matched_hands']} 个匹配手实例，{len(r['source_sequences'])} 个新源序列。序列未参与本版选参，本次受试者为 {', '.join(r['subjects'])}。六个训练受试者的新序列与 P0015 的未用序列混合评估；不能据此宣称跨受试者泛化。</p>
<h2>实际实现</h2><p>YOLO26 预测手框 → 校准的鱼眼/透视裁剪 → 原有 coarse 3D 与完整 WiLoR 的 3D、局部/全局视觉特征 → 条件 DiT → 局部视线坐标系的方向门控。WiLoR detector 仅提供左右手预测，不决定主检测框。基线是原有 coarse 3D，绝非把原始 WiLoR 误差当成此处的基线。</p>
<p>DiT 使用 v-pred 扩散训练，推理采用固定零潜变量、10 步 DDIM。门控阈值 {seal['models']['dit']['operating']['threshold']}，强度 {seal['models']['dit']['operating']['strength']}，均在新测试前冻结。没有使用 GT 关节、GT 左右手、GT 可见性或受试者真实手形作为推理输入。训练损失和困难样本采样可使用训练集 GT。</p>
<h2>总体与正确点保护</h2><table><tr><th>方法</th><th>相机系 MPJPE / mm</th><th>腕相对 MPJPE / mm</th><th>原正确点被破坏：相机 / 相对</th></tr>{''.join(overall)}</table>
<p>20 个 canonical 关节输出；MPJPE 使用除腕点外的 19 点。正确点定义：原误差 ≤10 mm，增加 >1 mm 算破坏；两种坐标指标的破坏比例均要求 ≤5%。三个方向门控均为零的关节保持原预测精确不变，但这不等于保证每个正确点都不受影响。</p>
<h2>困难组验收</h2><table><tr><th>方法</th><th>组别</th><th>关节点数</th><th>误差 / mm</th><th>改善</th><th>变化的 95% CI / mm</th><th>结论</th></tr>{''.join(table)}</table>
<p>低置信度、高遮挡评估腕相对距离；沿视线重叠评估腕相对误差的视线轴分量。每组 ≥100 点、改善 ≥5%、按源序列配对 bootstrap 2000 次得到的 CI 上界 &lt;0，且总体相机系误差不得恶化超过 0.1 mm。高遮挡是整手 modeled visibility &lt;0.5 且至少 18 点投影有效，不是逐点遮挡真值。</p>
<h2>DiT 与回归对照</h2><ul>{''.join(comparison)}</ul><p>负值有利于 DiT。通过相对 coarse 的验收不代表 DiT 比回归更好；此对照保留了相同特征、网络容量与困难组训练设置。</p>
<h2>受试者分层</h2><table><tr><th>受试者</th><th>范围</th><th>腕相对 MPJPE / mm</th></tr>{subject_table}</table><p>v5 另要求 P0015 的总体腕相对误差不退化，以免混合总体掩盖留出受试者退化；这仍不是证明所有困难组跨受试者泛化。</p>
<h2>成功、中位与失败案例</h2><p>按全部匹配手的腕相对误差改善排序，展示最大退化、四分位、中位和最大改善；没有只展示成功案例。左图是输入图上的投影，右图显示腕相对 X–Z 深度平面。</p><img src="examples.png">
<h2>边界和复现</h2><p>本模型只能修正已检测到的手，不会找回漏检。评估时先对全部检测框推理，再以 IoU≥0.3 匹配 GT；不把 GT 框送入网络。YOLO26 已在 HOT3D 微调，WiLoR 上游训练配置包含 HOT3D，具体样本重叠未知，因此不能声称彻底未见数据或跨受试者泛化。未验证跨数据集、视频时序稳定性或实时帧率。</p>
<p>上一版 v3 低置信度与高遮挡改善，但沿视线组的置信区间跨零，验收失败。失败结果保留。其测试数据在本轮明确转为开发集，v4 再次因沿视线组失败并保留。v5 优化实际推理展开结果，加入方向门控，并另选未用序列；训练和门控拟合样本仍不含开发集。重复开发后的这次测试是有限证据，不能当作无限独立试验或普适保证。</p>
<p>运行目录：<code>{html.escape(str(RUN))}</code>。权重 <code>sealed/dit.pt</code>，冻结记录 <code>sealed/selection.json</code>，完整指标 <code>locked_results.json</code>，推理 API 检查 <code>sealed_api_checks.json</code>。完整命令见同目录 README.md。</p></html>'''
(OUT/'report.html').write_text(content,encoding='utf-8')
print(json.dumps(dict(report=str(OUT/'report.html'),accepted=r['accepted'])))


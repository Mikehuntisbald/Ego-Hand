"""Standalone scientific figures and an offline, filterable experiment report."""
import html,json,csv,textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RUN=Path('/mnt/why/HOT3D/experiments/dit_lowconfidence_v1')
r=json.loads((RUN/'locked_selective_temporal_test_results.json').read_text());d=json.loads((RUN/'diagnosis.json').read_text())
names={'dit':'Single-frame DiT','regression':'Single-frame regression','temporal_dit':'Causal DiT','temporal_regression':'Causal regression'}
labels={'low_pose_confidence':'Low pose confidence (relative MPJPE)','high_occlusion_in_view':'High occlusion / in view (relative MPJPE)','multiple_ray_aligned_fingers':'Multiple ray-aligned fingers (axial relative error)'}
fig,axes=plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True)
colors=['#16697a','#558b2f','#c65d15','#7b6091']
for ax,(group,label) in zip(axes,labels.items()):
    metric='axial_relative' if group=='multiple_ray_aligned_fingers' else 'relative'
    for i,(method,name) in enumerate(names.items()):
        v=r['methods'][method]['groups'][group][metric];ci=v['ci95'];change=v['change_mm']
        if ci:ax.errorbar(change,i,xerr=[[max(0,change-ci[0])],[max(0,ci[1]-change)]],fmt='o',color=colors[i],capsize=4)
        else:ax.plot(change,i,'o',color=colors[i])
    ax.axvline(0,color='#999',linewidth=1);ax.set_yticks(range(4),list(names.values()));ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('Error change vs coarse (mm); negative is better');ax.set_title(textwrap.fill(label,38),fontsize=10)
fig.savefig(RUN/'hard_group_changes.png',dpi=180);fig.savefig(RUN/'hard_group_changes.pdf');plt.close(fig)
roles=['coarse','residual','tune','test'];fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
x=np.arange(4);ax.bar(x-.18,[d['error_distribution'][k]['mean_predicted_relative_sigma_mm'] for k in roles],.35,label='Predicted per-axis sigma')
ax.bar(x+.18,[d['error_distribution'][k]['relative_rms_per_axis_mm'] for k in roles],.35,label='Actual relative RMS per axis')
ax.set_xticks(x,['Coarse train','Refiner train','P0003 tune','P0010/P0015 test']);ax.set_ylabel('mm');ax.legend();ax.grid(axis='y',alpha=.2)
fig.savefig(RUN/'uncertainty_shift.png',dpi=180);fig.savefig(RUN/'uncertainty_shift.pdf');plt.close(fig)
rows=[]
base=r['methods']['dit']['overall']['coarse']
rows.append(f'<tr><td>冻结粗模型</td><td>{base["mpjpe19_mm"]:.3f}</td><td>{base["wrist_relative_mpjpe19_mm"]:.3f}</td><td>—</td><td>—</td></tr>')
for method,name in names.items():
    a=r['methods'][method];v=a['overall'];rows.append(f'<tr><td>{name}</td><td>{v["refined"]["mpjpe19_mm"]:.3f}</td><td>{v["refined"]["wrist_relative_mpjpe19_mm"]:.3f}</td><td>{v["correct_joints_harmed_fraction"]:.2%}</td><td>{a["groups"]["all"]["correct_relative_joints_harmed_fraction"]:.2%}</td></tr>')
dist=''.join(f'<tr><td>{role}</td><td>{v["samples"]:,}</td><td>{v["wrist_relative_mpjpe19_mm"]:.2f}</td><td>{v["mean_predicted_relative_sigma_mm"]:.2f}</td><td>{v["relative_rms_per_axis_mm"]:.2f}</td></tr>' for role,v in d['error_distribution'].items())
details=''
for group,label in labels.items():
    metric='axial_relative' if group=='multiple_ray_aligned_fingers' else 'relative';group_rows=[]
    for method,name in names.items():
        g=r['methods'][method]['groups'][group];v=g[metric];ci=v['ci95'];ci_text='—' if ci is None else f'[{ci[0]:.3f}, {ci[1]:.3f}]'
        group_rows.append(f'<tr><td>{name}</td><td>{g["joints"]:,}</td><td>{v["before_mm"]:.3f}</td><td>{v["after_mm"]:.3f}</td><td>{v["improvement_pct"]:+.2f}%</td><td>{ci_text}</td></tr>')
    details+=f'<section data-group="{group}"><h3>{label}</h3><table><thead><tr><th>模型</th><th>关节点</th><th>之前 mm</th><th>之后 mm</th><th>误差下降</th><th>变化的 95% CI / mm</th></tr></thead><tbody>{"".join(group_rows)}</tbody></table></section>'
oracle=''.join(f'<tr><td>{names[k]}</td><td>{v["full_proposal_helpful_fraction"]:.1%}</td><td>{v["oracle_gt_dependent_not_a_model_result"]["mpjpe19_mm"]:.2f}</td></tr>' for k,v in d['oracle_diagnostics'].items())
verdict='本轮独立测试不支持“DiT 已解决低置信度、高遮挡和射线对齐问题”。'
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HOT3D DiT：训练与独立测试</title>
<style>body{{margin:0;background:#f4f6f8;color:#22313c;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:30px 24px}}h1{{font-size:29px}}h2{{font-size:22px;margin-top:30px}}h3{{font-size:18px}}.verdict{{background:#fff0de;padding:18px;border-radius:10px;font-weight:650}}.note{{background:#e8eff3;padding:15px;border-radius:9px}}table{{width:100%;background:white;border-collapse:collapse;margin:15px 0}}td,th{{padding:10px;border-bottom:1px solid #dbe4e9;text-align:left}}img{{max-width:100%;background:white;border-radius:9px}}a{{color:#16697a}}code{{background:#e8eef1;padding:3px 6px}}.muted{{color:#61717a}}select{{font:inherit;padding:5px}}</style>
<main><h1>HOT3D · DiT 门控 3D 残差</h1><p class="verdict">{verdict}</p>
<p>已完成 YOLO26s 检测、3D 粗姿态训练、单帧与因果时间上下文 DiT，以及同容量的直接回归对照。最终结果来自 P0010、P0015 的 {r['samples']:,} 个匹配手框；模型选择和门控校准只使用 P0003。</p>
<h2>独立测试总体指标</h2><table><thead><tr><th>模型</th><th>相机 MPJPE19 / mm</th><th>腕点相对 MPJPE19 / mm</th><th>原本准确相机点被改坏</th><th>原本准确相对点被改坏</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="muted">“原本准确”指误差 ≤10 mm，“改坏”指误差增加超过 1 mm；保护验收要求 ≤5%。所有方法在独立测试上都未通过该要求。</p>
<h2>困难子集的配对变化</h2><img src="hard_group_changes.png" alt="困难组误差变化及置信区间"><p>置信区间按原始 source sequence 聚类，4 个序列、2 位受试者；不是把相关视频帧视为独立样本。改善比例正值表示误差下降。</p>
<label>查看分组：<select id="group"><option value="all">全部</option>{''.join(f'<option value="{k}">{v}</option>' for k,v in labels.items())}</select></label>{details}
<h2>确认存在的误差与置信度偏移</h2><img src="uncertainty_shift.png" alt="预测不确定性与真实误差分布"><table><thead><tr><th>数据角色</th><th>手样本</th><th>相对 MPJPE / mm</th><th>预测每轴 sigma / mm</th><th>真实每轴 RMS / mm</th></tr></thead><tbody>{dist}</tbody></table>
<p>粗模型训练片段的相对误差为 {d['error_distribution']['coarse']['wrist_relative_mpjpe19_mm']:.2f} mm，残差训练片段为 {d['error_distribution']['residual']['wrist_relative_mpjpe19_mm']:.2f} mm，独立受试者为 {d['error_distribution']['test']['wrist_relative_mpjpe19_mm']:.2f} mm。测试预测 sigma 为 {d['error_distribution']['test']['mean_predicted_relative_sigma_mm']:.2f} mm，实际每轴 RMS 为 {d['error_distribution']['test']['relative_rms_per_axis_mm']:.2f} mm。误差分布变大而置信度未相应放宽，门控保护不能从调参集稳定迁移。</p>
<p>这支持优先修正粗模型泛化、用按受试者留出的预测训练残差，以及重新校准不确定性的路线。它不能单独证明每个退步都由这个因素造成。</p>
<h2>GT 选择门控的诊断上界</h2><div class="note">以下结果利用 GT 为每个点选择最佳修正强度，仅作问题定位，不能用于推理，不能计作模型测试成绩。</div><table><thead><tr><th>模型提案</th><th>完整提案真正有帮助的点</th><th>GT oracle 相机 MPJPE / mm</th></tr></thead><tbody>{oracle}</tbody></table>
<p>提案并非全部无用，但完整提案只有约 38%–41% 的点改善。实际门控没有可靠地区分可接受的修正；GT oracle 的上界也不意味着这些改善能从观测中识别出来。</p>
<h2>检测、遮挡与射线定义</h2><p>固定 YOLO 手框在测试上 IoU50 召回率 {r['detector']['box_recall_iou50']:.2%}；所有残差模型共用这些框，因此没有提升框召回率。低姿态置信度采用 P0003 上冻结粗模型置信度的第20百分位阈值；高遮挡要求整手建模可见比例低于0.5，且至少18/20点投影在有效视场；射线组要求至少2根手指的近端到指尖轴与视线夹角≤15°。</p>
<p>数据没有逐点遮挡真值。射线对齐是几何代理，不等同于已经确认该点不可见。单 RGB 视角的深度仍可能多解，过去帧也不能保证一定出现所需证据。</p>
<h2>训练与推理条件</h2><p>3D 粗模型使用303个片段，所有残差模型使用77个不重叠片段。粗模型预热允许训练 GT 裁剪；后续训练、选模型和测试都用预测裁剪。GT 位姿、手形、遮挡分组、左右手轨迹身份不进入推理。时间模型接收严格过去的3个上下文，轨迹从全部预测候选框关联。</p>
<p>DiT 用 v-prediction，10步 DDIM、4个固定种子提案取平均；没有 GT 选择最佳采样。门控由实际推理提案训练。最初共享根部修正造成局部保护失败，新版改成最终相机坐标逐点门控；保留这一修改和测试盲评说明。</p>
<h2>交付</h2><p>服务器：<code>/mnt/why/HOT3D/experiments/dit_lowconfidence_v1</code>。</p><p><a href="training_results.json">完整独立测试结果</a> · <a href="diagnosis.json">误差分布与 oracle 诊断</a> · <a href="proof_protocol.json">首轮实验协议</a> · <a href="test_blinding_note.json">盲评记录</a></p>
<p class="muted">当前结果针对这个数据子集、粗模型和残差实现，不是对所有 DiT 方法的否定，也不足以支持通用解决遮挡的结论。</p></main>
<script>document.getElementById('group').onchange=function(){{document.querySelectorAll('section[data-group]').forEach(s=>s.hidden=this.value!=='all'&&s.dataset.group!==this.value)}};</script></html>'''
(RUN/'training_report.html').write_text(page);print(verdict,flush=True)
summary=dict(verdict=verdict,models={k:dict(camera_mm=v['overall']['refined']['mpjpe19_mm'],relative_mm=v['overall']['refined']['wrist_relative_mpjpe19_mm'],
    low_confidence_relative_mm=v['groups']['low_pose_confidence']['relative']['after_mm'],
    occlusion_relative_mm=v['groups']['high_occlusion_in_view']['relative']['after_mm'],
    ray_axial_relative_mm=v['groups']['multiple_ray_aligned_fingers']['axial_relative']['after_mm'],
    correct_points_harmed=v['overall']['correct_joints_harmed_fraction']) for k,v in r['methods'].items()})
(RUN/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

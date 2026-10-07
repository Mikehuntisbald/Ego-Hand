"""Offline report, scientific figure and hashed model delivery for the OOF study."""
import html,json,textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oof_common import RUN,OLD,save,sha

LABELS={'oof_dit':'Subject-out DiT','oof_regression':'Subject-out regression','in_subject_dit':'In-subject DiT control','v1_dit':'Previous DiT'}
GROUPS=[('low_pose_confidence','relative','Low pose confidence: relative error'),
    ('high_occlusion_in_view','relative','High occlusion in view: relative error'),
    ('multiple_ray_aligned_fingers','axial_relative','Ray-aligned fingers: axial relative error')]

def main():
    r=json.loads((RUN/'final_results.json').read_text());audit=json.loads((RUN/'cache_audit.json').read_text())
    diagnostic=json.loads((RUN/'post_lock_diagnosis.json').read_text())['methods']['oof_dit']
    fresh=r['fresh'];methods=list(fresh['methods']);summary={}
    for method,stats in fresh['methods'].items():
        summary[method]=dict(camera_mm=stats['overall']['refined']['mpjpe19_mm'],relative_mm=stats['overall']['refined']['wrist_relative_mpjpe19_mm'],
            harm_camera=stats['overall']['correct_joints_harmed_fraction'],harm_relative=stats['groups']['all']['correct_relative_joints_harmed_fraction'],
            preservation_passed=stats['preservation_passed'],hard_groups=fresh['evidence'][method],calibration=stats['calibration'])
    positive=all(v['supported'] for v in fresh['evidence']['oof_dit'].values())
    save(RUN/'summary.json',dict(fresh_samples=fresh['samples'],fresh_sequences=len(fresh['source_sequences']),fresh_subjects=fresh['subjects'],
        positive_evidence=positive,models=summary,baseline=fresh['methods']['oof_dit']['overall']['coarse'],
        result_scope='New source sequences from previously evaluated subjects; not wholly new subjects'))
    checkpoints=[]
    for method in r['protocol']['methods']:
        p=RUN/method/'best.pt';checkpoints.append(dict(method=method,path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    for label,p in [('shared_rgb_encoder',RUN/'rgb_encoder.pt'),('frozen_coarse',OLD/'coarse_fine/best.pt'),('frozen_detector',OLD/'detector/weights/best.pt')]:
        checkpoints.append(dict(method=label,path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    save(RUN/'checkpoints.json',checkpoints)
    colors=['#2060bc','#139060','#bc7620','#777777']
    fig,axs=plt.subplots(1,3,figsize=(14,4.6))
    for ax,(group,metric,title) in zip(axs,GROUPS):
        for j,(method,color) in enumerate(zip(methods,colors)):
            g=fresh['methods'][method]['groups'][group];v=g.get(metric)
            if not v:continue
            change=v['change_mm'];ci=v.get('ci95');err=None
            if ci is not None and ci[0]<=change<=ci[1]:err=np.array([[change-ci[0]],[ci[1]-change]])
            ax.errorbar(change,j,xerr=err,fmt='o',color=color,capsize=4)
        ax.axvline(0,color='#555',ls='--',lw=1)
        ax.set_yticks(range(len(methods)),[LABELS[m] for m in methods]);ax.invert_yaxis()
        ax.set_title('\n'.join(textwrap.wrap(title,30)),fontsize=10)
        ax.set_xlabel('Refined minus coarse error (mm)\nNegative values indicate improvement')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Frozen-checkpoint evaluation on six newly downloaded source sequences\n95% bootstrap intervals clustered by source sequence; reused P0010/P0015 subjects',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.86));fig.savefig(RUN/'fresh_changes.png',dpi=160,bbox_inches='tight');fig.savefig(RUN/'fresh_changes.pdf',bbox_inches='tight');plt.close(fig)
    def table(subset):
        rows=[]
        base=subset['methods']['oof_dit']['overall']['coarse']
        rows.append(f'<tr><td>粗模型</td><td>{base["mpjpe19_mm"]:.2f}</td><td>{base["wrist_relative_mpjpe19_mm"]:.2f}</td><td>0%</td><td>0%</td></tr>')
        for m,s in subset['methods'].items():
            rows.append(f'<tr><td>{LABELS[m]}</td><td>{s["overall"]["refined"]["mpjpe19_mm"]:.2f}</td><td>{s["overall"]["refined"]["wrist_relative_mpjpe19_mm"]:.2f}</td><td>{100*s["overall"]["correct_joints_harmed_fraction"]:.2f}%</td><td>{100*s["groups"]["all"]["correct_relative_joints_harmed_fraction"]:.2f}%</td></tr>')
        return '<table><thead><tr><th>模型</th><th>相机 MPJPE / mm</th><th>腕相对 MPJPE / mm</th><th>相机准确点误伤</th><th>相对准确点误伤</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    conclusion='本轮满足预设的三组困难情况改善及可见正确点保护要求。' if positive else '本轮未同时满足预设的三组困难情况改善及正确点保护要求，不能宣称 DiT 已解决低置信度与遮挡问题。'
    distributions=''.join(f'<tr><td>{x["role"]}</td><td>{x["provenance"]}</td><td>{x["samples"]}</td><td>{x["metrics"]["mpjpe19_mm"]:.2f}</td><td>{x["metrics"]["wrist_relative_mpjpe19_mm"]:.2f}</td></tr>' for x in audit['training_distribution'])
    # Embed only results already evaluated after checkpoint lock. No web dependency.
    embedded=json.dumps(r,ensure_ascii=False).replace('</','<\\/')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>HOT3D 按受试者留出残差训练</title>
<style>body{{max-width:1150px;margin:35px auto;padding:0 22px;font:16px/1.65 system-ui;color:#263442}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{text-align:left;border-bottom:1px solid #dbe2ea;padding:9px}}th{{background:#f1f5fa}}.verdict{{padding:15px;background:#eef3fa;border-left:4px solid #2060bc}}img{{max-width:100%}}pre{{white-space:pre-wrap;font-size:13px;background:#f4f6f9;padding:16px}}select{{padding:8px;margin:8px}}</style>
<h1>HOT3D：按受试者留出预测训练残差与门控</h1><p>2026-10-03 · dit_subject_oof_v2 · 实际训练与冻结检查点评估</p>
<p class="verdict">{conclusion}</p>
<p>新增验证：{fresh['samples']} 个已匹配手部样本、6 条此前未使用的原始序列、24 个片段，P0010 / P0015。受试者在上一轮已评估，不能称作全新受试者。旧测试集明确作为开发对照；所有新增验证结果均在模型与校准设置冻结后才计算。</p>
<h2>新增序列复核</h2>{table(fresh)}
<p>误伤：原本误差 ≤10mm 的点，经修正后误差增加 &gt;1mm。相机空间、腕相对空间均须 ≤5%。所有方法共享同一检测框和粗模型；不会改变检测召回率。</p>
<p>留出 DiT 的完整提案在 {100*diagnostic['full_proposal_points_helpful_fraction']:.2f}% 的点上改善超过 1mm，在 {100*diagnostic['full_proposal_points_harmful_fraction']:.2f}% 的点上变差超过 1mm。关闭门控保护时，相机误差为 {diagnostic['ungated']['mpjpe19_mm']:.2f}mm、相对误差为 {diagnostic['ungated']['wrist_relative_mpjpe19_mm']:.2f}mm。最终平均只保留提案的 {100*diagnostic['mean_applied_point_gate']:.2f}%，因此本轮误伤下降主要来自小幅更新；并无证据表明留出训练使残差恢复更准。</p>
<label>困难分组 <select id="group"><option value="low_pose_confidence">低姿态置信度</option><option value="high_occlusion_in_view">高遮挡且在视野内</option><option value="multiple_ray_aligned_fingers">多手指沿镜头射线</option></select></label><div id="group_table"></div>
<img src="fresh_changes.png" alt="困难组误差变化及序列聚类置信区间"><p>多手指沿射线只是一项几何代理标签；HOT3D 没有逐关节点遮挡 GT。</p>
<h2>留出训练与防止泄漏</h2>
<p>训练受试者六人，每折粗模型从 COCO 预训练重新开始，仅用另外五人的粗模型训练片段。留出人的全部预测用于残差候选集。历史 3D 权重不作为折初始化。各折与最终推理均使用相同冻结 COCO RGB 特征基底，避免训练后通道空间不同。</p>
<p>25 条训练序列中，19 条只训练残差，另外 6 条只训练门控。门控输入是实际 DDIM 生成的提案，GT 只标记其收益。P0003 两条序列分别负责检查点选择和最终概率/强度校准；后者在门控检查点冻结后才访问。</p>
<p>上述序列隔离适用于本轮残差与门控的直接训练/选择。上游粗模型选择仍使用 P0003，包含这两条序列，且已有历史模型也曾用其验证；因此不把校准集称作全流程未见数据。新增复核序列未进入任何上游或下游训练与选择。</p>
<p>对照包括同容量普通残差回归，以及相同数据、RGB、训练步数、拆分和校准流程的原粗模型预测 DiT。其差别是使用见过训练受试者的粗模型预测，帮助检验留出预测本身的作用。</p>
<h2>训练误差分布</h2><table><tr><th>分区</th><th>预测来源</th><th>样本数</th><th>相机误差/mm</th><th>相对误差/mm</th></tr>{distributions}</table>
<h2>不确定性与门控校准</h2><p>不确定性缩放仅用残差训练序列，六个受试者等权。门控采用单独校准序列上的单调 Platt 概率拟合，并选择阈值/残差强度；目标为相机误差 + 相对误差 + 0.5 × 低置信度相对误差。校准要求误伤率 ≤2.5%，Wilson 上界 ≤5%；无可接受配置时精确关闭更新。相关关节点不满足独立性，因此 Wilson 上界不能作为分布无关的安全保证。</p>
<pre>{html.escape(json.dumps({m:dict(uncertainty=s['uncertainty'],calibration=s['calibration'],preservation_passed=s['preservation_passed']) for m,s in fresh['methods'].items()},ensure_ascii=False,indent=2))}</pre>
<h2>旧测试集开发对照</h2>{table(r['development'])}
<h2>推理交付验证</h2><p>真实 RGB 裁剪与标定几何输入，无 GT。三套接口输出均通过有限值、尺寸与门控范围检查。BF16 前级在不同批大小下出现最大约 2.50mm 偏移，推理接口据此采用与缓存相同的 256 批大小填充；缓存的 FP16 RGB 存储也在接口中复现。修正后粗模型和 RGB 特征与审计缓存最大差异均为零。此修正没有更换权重、调整门控或重跑最终测试。性能表中的残差耗时不包含这些前级计算。</p>
<h2>交付文件</h2><p><a href="final_results.json">完整指标 JSON</a> · <a href="protocol.json">冻结协议</a> · <a href="cache_audit.json">留出来源与实数据梯度审计</a> · <a href="checkpoints.json">检查点 SHA256</a> · <a href="post_lock_diagnosis.json">冻结后只读诊断</a> · <a href="inference_smoke.json">推理验证</a> · <a href="fresh_changes.pdf">科研图 PDF</a></p>
<pre>{html.escape(json.dumps(checkpoints,indent=2))}</pre>
<script>const results={embedded};function render(){{const group=document.getElementById('group').value,metric=group==='multiple_ray_aligned_fingers'?'axial_relative':'relative';let t='<table><tr><th>模型</th><th>关节点数</th><th>原误差/mm</th><th>修正误差/mm</th><th>改善%</th><th>Δ 95% CI /mm</th><th>通过证据标准</th></tr>';for(const [name,stats] of Object.entries(results.fresh.methods)){{const g=stats.groups[group],v=g[metric],e=results.fresh.evidence[name][group];if(!v){{t+='<tr><td>'+name+'</td><td colspan="6">无样本</td></tr>';continue;}}t+='<tr><td>'+name+'</td><td>'+g.joints+'</td><td>'+v.before_mm.toFixed(2)+'</td><td>'+v.after_mm.toFixed(2)+'</td><td>'+v.improvement_pct.toFixed(2)+'</td><td>'+(v.ci95?v.ci95.map(x=>x.toFixed(2)).join(' ~ '):'样本不足')+'</td><td>'+(e.supported?'是':'否')+'</td></tr>';}}document.getElementById('group_table').innerHTML=t+'</table>';}}document.getElementById('group').addEventListener('change',render);render();</script></html>'''
    (RUN/'report.html').write_text(page)
    print(json.dumps(dict(positive_evidence=positive,report=str(RUN/'report.html'),samples=fresh['samples'],summary=summary)),flush=True)
if __name__=='__main__':main()

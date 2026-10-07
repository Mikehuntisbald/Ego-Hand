import json,shutil,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from temporal_sampling_v6 import RUN,V4,OFFSETS
import spatial_rgb_common as s
from evaluate_offline_kp import bootstrap

def main():
    O=RUN/'delivery';O.mkdir(exist_ok=True);reports={};comparisons={}
    rows=json.loads((s.OLD/'rows.json').read_text());hard=json.loads((V4/'delivery/hardcase_review_queue.json').read_text());checks=json.loads((RUN/'sampling_checks.json').read_text())
    for kind in ['regression','dit']:
        for pattern in OFFSETS:
            arm=f'{kind}_{pattern}';assert (RUN/arm/'done.json').exists();reports[arm]=json.loads((RUN/arm/'results.json').read_text())
            dest=O/arm;dest.mkdir(exist_ok=True)
            for name in ['history.json','results.json','done.json']:shutil.copy2(RUN/arm/name,dest/name)
        proposal=np.load(RUN/f'{kind}_multiscale/predictions.npz');ids=proposal['window_indices'];mask=proposal['mask'];clusters=np.array([rows[i]['sequence'] for i in ids]);lookup={v:i for i,v in enumerate(ids)};hi=np.array([lookup[q['window_index']] for q in hard]);e=np.linalg.norm(proposal['prediction']-proposal['gt'],axis=-1)*1408
        for reference in ['uniform','wide']:
            baseline=np.load(RUN/f'{kind}_{reference}/predictions.npz')
            for k in ['window_indices','gt','mask']:assert np.array_equal(proposal[k],baseline[k]),k
            assert np.array_equal(proposal['base'][mask],baseline['base'][mask])
            delta=e-np.linalg.norm(baseline['prediction']-baseline['gt'],axis=-1)*1408
            raw_delta=(np.linalg.norm(proposal['raw']-proposal['gt'],axis=-1)-np.linalg.norm(baseline['raw']-baseline['gt'],axis=-1))*1408
            comparisons[f'{kind}_multiscale_vs_{reference}']=dict(mean_delta_px=float(delta[mask].mean()),ci95=bootstrap(delta,mask,clusters),hard_delta_px=float(delta[hi][mask[hi]].mean()),hard_ci95=bootstrap(delta[hi],mask[hi],clusters[hi]),raw_delta_px=float(raw_delta[mask].mean()),raw_ci95=bootstrap(raw_delta,mask,clusters))
    summary=dict(reports=reports,comparisons=comparisons,sampling=checks,scope='17 nominal tokens, existing 6Hz observations; nonuniform placement not higher native FPS; existing inspected test set only')
    (O/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    fig,ax=plt.subplots(figsize=(11,3.2),layout='constrained')
    for y,(name,offsets) in enumerate(OFFSETS.items()):
        t=np.array(offsets)/6;ax.hlines(y,t[0],t[-1],color='#c2cad4');ax.scatter(t,np.full(17,y),s=45,color=['#64748b','#b67624','#167e79'][y]);ax.scatter([0],[y],s=65,color='#c93d45')
    ax.set_yticks(range(3),['Uniform, +/-1.33 s','Uniform, +/-4 s','Dense near / sparse far']);ax.set_xlabel('Requested time relative to target (s)');ax.set_xticks(np.arange(-4,5));ax.grid(axis='x',alpha=.2);ax.invert_yaxis();ax.set_ylim(2.5,-.5);fig.savefig(O/'sampling.png',dpi=160);plt.close(fig)
    h='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>近密远疏采样对照</title><style>body{font:16px/1.7 system-ui;max-width:1150px;margin:36px auto;padding:0 22px;background:#f5f7fa;color:#172333}section{background:white;border-radius:12px;padding:24px;margin:22px 0}table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}.note{color:#536174}a{color:#1269aa}h1{font-size:28px}</style><h1>当前周围密、远处疏：采样对照</h1><p>3种采样 × 回归/DiT，共6组。每组17个时间位置、相同模型与原始v4起点，训练1500步、batch48、LR 2e-5；视觉主干保持冻结，风险模型及门控不变。开发集选检查点，包含第0步。</p><p><b>此轮使用已有6Hz观测，最近间隔仍约0.167秒，没有新增30Hz帧。</b>多尺度时间偏移为 0、±0.167、±0.333、±0.5、±0.667、±1、±1.667、±2.667、±4秒。使用真实时间差；超出片段或同轨迹无观测的位置保持缺失。</p><img src="sampling.png" alt="三种采样时间位置"><section><h2>固定门控后的误差与保护</h2><table><tr><th>分支</th><th>选中步数</th><th>整体 px↓</th><th>47 hard case px↓</th><th>坏点恢复数↑</th><th>好点改坏数↓</th></tr>'''
    names={'uniform':'原均匀 ±1.33s','wide':'均匀 ±4s','multiscale':'近密远疏 ±4s'}
    for arm,r in reports.items():
        g=r['gated'];h+=f"<tr><td>{r['kind']} · {names[r['pattern']]}</td><td>{r['selected_step']}</td><td>{g['mean_px']:.3f}</td><td>{r['hardcases']['mean_px']:.3f}</td><td>{g['bad_recovered_to20']}/{g['originally_bad_points']}</td><td>{g['good_harmed_to_over20']}/{g['good_points']}</td></tr>"
    h+='</table><p class="note">坏点恢复：原始&gt;20px变为≤20px；好点改坏：原始≤10px变为&gt;20px。所有分支使用相同v4回归门控，不能直接与采用其他门控的旧DiT结果对比。没有根据本表选阈值或替换默认模型。</p></section><section><h2>配对差异</h2>'
    for k,c in comparisons.items():
        h+=f"<p><b>{k}</b>：整体 {c['mean_delta_px']:+.3f}px，来源序列bootstrap 95%CI [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]；hard case {c['hard_delta_px']:+.3f}px，95%CI [{c['hard_ci95'][0]:+.3f}, {c['hard_ci95'][1]:+.3f}]。负数表示近密远疏较好。门控前提案变化 {c['raw_delta_px']:+.3f}px。</p>"
    h+='</section><section><h2>采样位置不等于实际有效帧</h2><table><tr><th>采样</th><th>平均有效帧/17</th><th>17帧齐全的窗口数</th></tr>'
    for k,v in checks['patterns'].items():h+=f"<tr><td>{names[k]}</td><td>{v['valid_frame_mean']:.2f}</td><td>{v['fully_populated_windows']}</td></tr>"
    h+='</table><p>片段约5秒，所以覆盖±4秒必然常碰到边界。输入位置数相同，但有效观测数不同，差异已如实保留；没有复制边界帧凑数，也没有跨轨迹关联。这不是严格等有效帧数量的密度实验。</p></section><section><h2>8个严重失败窗口</h2><table><tr><th>ID</th><th>回归 原均匀→近密远疏</th><th>DiT 原均匀→近密远疏</th><th>有效帧 原→新</th></tr>'
    maps={k:{q['id']:q for q in r['cases']} for k,r in reports.items()}
    for q in checks['patterns']['multiscale']['cases']:
        n=q['id'];r0=maps['regression_uniform'][n];r1=maps['regression_multiscale'][n];d0=maps['dit_uniform'][n];d1=maps['dit_multiscale'][n]
        h+=f"<tr><td>#{n}</td><td>{r0['error_px']:.2f} → {r1['error_px']:.2f}</td><td>{d0['error_px']:.2f} → {d1['error_px']:.2f}</td><td>{r0['valid_frames']} → {r1['valid_frames']}</td></tr>"
    h+='</table><p>#6杯子：近密远疏实际选到了f0（目标f120之前约4秒），远处原始线索已进入上下文。#13：仍只有f120/130/135/140/145这5帧，采样本身不能跨越已有轨迹断裂。</p><details><summary>逐案例实际采样帧</summary><pre style="white-space:pre-wrap">'+json.dumps(checks['patterns']['multiscale']['cases'],ensure_ascii=False,indent=2)+'</pre></details></section>'
    h+='''<section><h2>结论与边界</h2><p>近密远疏比单纯把所有帧均匀放疏更合适；相对原短窗口，整体收益尚不明确。回归hard case有小幅改善，DiT整体基本持平。当前默认模型保留，采样代码与检查点作为实验候选。</p><p>数据为此前已查看的测试序列，不是新的独立泛化证据。GT仅用于损失与评估；采样沿用预测轨迹，不使用GT手侧或可见性。轨迹重连、更高原生帧率和更长连续片段尚未在本轮验证。</p><a href="comparison.json">完整指标</a> · <a href="DEVELOPMENT_NOTES.txt">简明开发记录</a></section></html>'''
    (O/'report.html').write_text(h,encoding='utf-8')
    notes='近密远疏采样 v6\n\n问题1：原17帧均匀6Hz只覆盖±1.33秒。\n处理：增加近密远疏±4秒，并加同跨度均匀采样作对照。\n\n问题2：旧轨迹缓存的帧并非全都有效，不能直接靠位置重排假装补齐。\n处理：沿原预测轨迹合并已有观测，中心不变，缺失保留；验证无重复、无跨轨迹、原均匀输入逐项一致。\n\n问题3：改变时序会改变可靠性特征语义。\n处理：风险评分仍使用原均匀上下文，修正模型使用新采样，阈值冻结。\n\n问题4：约5秒片段限制±4秒采样的有效帧数。\n处理：报告每种策略实际有效帧数量，不复制边界图像。\n\n验证：6组1500步，开发集选检查点；原始RGB推理通过，34个人工确认点保持不变。\n范围：最近仍6Hz，未新增30Hz检测；未重连断轨；已有测试集只作开发诊断。\n'
    (O/'DEVELOPMENT_NOTES.txt').write_text(notes,encoding='utf-8')
    for name in ['protocol.json','sampling_checks.json','raw_smoke.json','status.json']:shutil.copy2(RUN/name,O/name)
    code=O/'code';code.mkdir(exist_ok=True)
    for name in ['temporal_sampling_v6.py','run_sampling_v6.py','infer_sampling_v6.py','report_sampling_v6.py']:shutil.copy2(Path(__file__).parent/name,code/name)
    (O/'checkpoint_locations.json').write_text(json.dumps({k:str(RUN/k/'best.pt') for k in reports},indent=2))
    (O/'manifest.json').write_text(json.dumps({str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in O.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2))
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()

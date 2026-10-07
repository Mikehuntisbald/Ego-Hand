import json,shutil,hashlib
from pathlib import Path
import numpy as np
from finetune_visual_v5 import RUN,V4
import spatial_rgb_common as s
from evaluate_offline_kp import bootstrap

def main():
    delivery=RUN/'delivery';delivery.mkdir(exist_ok=True)
    rows=json.loads((s.OLD/'rows.json').read_text());reports={};comparisons={}
    hard=json.loads((V4/'delivery/hardcase_review_queue.json').read_text());classification=json.loads((V4/'temporal_clue_audit/classification.json').read_text())
    for kind in ['regression','dit']:
        for mode in ['matched','ft']:
            arm=f'{kind}_{mode}';assert (RUN/arm/'evaluation_done.json').exists()
            reports[arm]=json.loads((RUN/arm/'results.json').read_text())
            target=delivery/arm;target.mkdir(exist_ok=True)
            for name in ['config.json','history.json','results.json','initial_parity.json','gradient_check.json','weight_update_check.json']:
                if (RUN/arm/name).exists():shutil.copy2(RUN/arm/name,target/name)
        frozen=np.load(RUN/f'{kind}_matched/predictions.npz');ft=np.load(RUN/f'{kind}_ft/predictions.npz')
        for field in ['window_indices','base','gt','mask']:assert np.array_equal(frozen[field],ft[field]),field
        ids=ft['window_indices'];clusters=np.array([rows[i]['sequence'] for i in ids]);lookup={v:j for j,v in enumerate(ids)};hi=np.array([lookup[q['window_index']] for q in hard]);mask=ft['mask']
        error=lambda p:np.linalg.norm(p-ft['gt'],axis=-1)*1408
        ef=error(ft['prediction']);ec=error(frozen['prediction']);delta=ef-ec
        raw_delta=error(ft['raw'])-error(frozen['raw'])
        baseline_error=error(ft['base']);good=mask&(baseline_error<=10);bad=mask&(baseline_error>20)
        comparisons[kind]=dict(gated_ft_minus_frozen_px=float(delta[mask].mean()),gated_ci95=bootstrap(delta,mask,clusters),raw_ft_minus_frozen_px=float(raw_delta[mask].mean()),raw_ci95=bootstrap(raw_delta,mask,clusters),hardcase_delta_px=float(delta[hi][mask[hi]].mean()),hardcase_ci95=bootstrap(delta[hi],mask[hi],clusters[hi]),good_harm_count_delta=int(((ef>20)&good).sum()-((ec>20)&good).sum()),bad_recovered_count_delta=int(((ef<=20)&bad).sum()-((ec<=20)&bad).sum()),failure_categories={})
        for cat in ['A','B']:
            case_ids=[q['id'] for q in classification['cases'] if q['category']==cat];qmap={q['id']:q for q in hard};sel=np.array([lookup[qmap[i]['window_index']] for i in case_ids])
            comparisons[kind]['failure_categories'][cat]=dict(cases=case_ids,frozen_px=float(ec[sel][mask[sel]].mean()),ft_px=float(ef[sel][mask[sel]].mean()),delta_px=float(delta[sel][mask[sel]].mean()))
        h0=json.loads((RUN/f'{kind}_matched/history.json').read_text())[0];h1=json.loads((RUN/f'{kind}_ft/history.json').read_text())[0]
        assert abs(h0['mean_px']-h1['mean_px'])<1e-5,(kind,h0['mean_px'],h1['mean_px'])
        comparisons[kind]['initial_dev_mean_difference_px']=h1['mean_px']-h0['mean_px']
    summary=dict(protocol=json.loads((RUN/'protocol.json').read_text()),matched_control_protocol=json.loads((RUN/'matched_control_protocol.json').read_text()),reports=reports,comparisons=comparisons,scope='600-step exploratory trial; last4+stem joint update, not whole-backbone unfreezing; existing inspected test subjects, not independent generalization; B category only one case')
    (delivery/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>视觉主干微调对照</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:36px auto;padding:0 24px;color:#182535;background:#f5f7fa}section{background:white;padding:24px;margin:22px 0;border-radius:12px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;text-align:left;padding:9px}h1{font-size:28px}.note{color:#536174}a{color:#1269aa}</style><h1>视觉主干参与微调：受控试验</h1><p>WiLoR最后4/32层 + 空间特征层 + 时序头联合训练，分别比较回归和DiT。每组600步、有效batch16；主干LR 1e-6，空间层1e-5，时序头2e-5。原始RGB，无人工遮挡；保持数据划分、风险模型、跟踪输入和门控不变。</p><p>冻结对照也继续训练600步，并采用相同在线视觉计算路径。开发集选择检查点（含第0步），固定v4回归门控用于所有分支。这里的DiT数字不能直接和先前采用另一套门控的DiT表比较。</p><section><h2>固定门控后的结果</h2><table><tr><th>分支</th><th>选中步数</th><th>全部误差 px↓</th><th>47 hard case px↓</th><th>坏点恢复数↑</th><th>好点改坏数↓</th></tr>'''
    labels={'regression_matched':'回归 · 冻结视觉','regression_ft':'回归 · 联合微调','dit_matched':'DiT · 冻结视觉','dit_ft':'DiT · 联合微调'}
    for arm,r in reports.items():
        g=r['gated'];html+=f"<tr><td>{labels[arm]}</td><td>{r['selected_step']}</td><td>{g['mean_px']:.3f}</td><td>{r['hardcases']['mean_px']:.3f}</td><td>{g['bad_recovered_to20']}/{g['originally_bad_points']}</td><td>{g['good_harmed_to_over20']}/{g['good_points']}</td></tr>"
    html+='</table><p class="note">坏点：原始误差&gt;20px，恢复后≤20px；好点：原始≤10px，改坏后&gt;20px。检查点只按dev_select原始提案平均误差选择；没有在这些测试数字上选模型。</p></section><section><h2>联合微调相对冻结对照</h2>'
    for kind,c in comparisons.items():
        html+=f"<p><b>{kind}</b>：全部误差变化 {c['gated_ft_minus_frozen_px']:+.3f}px，按来源序列配对bootstrap 95%CI [{c['gated_ci95'][0]:+.3f}, {c['gated_ci95'][1]:+.3f}]；hard case变化 {c['hardcase_delta_px']:+.3f}px，95%CI [{c['hardcase_ci95'][0]:+.3f}, {c['hardcase_ci95'][1]:+.3f}]。负数表示微调较好。</p>"
        html+=f"<p>门控前原始提案变化 {c['raw_ft_minus_frozen_px']:+.3f}px；坏点恢复数变化 {c['bad_recovered_count_delta']:+d}，好点改坏数变化 {c['good_harm_count_delta']:+d}。</p>"
    html+='</section><section><h2>此前8个严重失败窗口</h2><table><tr><th>ID / 分类</th><th>v4默认</th><th>回归冻结→微调</th><th>DiT冻结→微调</th></tr>'
    maps={k:{q['id']:q for q in v['cases']} for k,v in reports.items()}
    for q in classification['cases']:
        n=q['id'];html+=f"<tr><td>#{n} / {q['category']}</td><td>{maps['regression_ft'][n]['v4_px']:.2f}</td><td>{maps['regression_matched'][n]['v5_px']:.2f} → {maps['regression_ft'][n]['v5_px']:.2f}</td><td>{maps['dit_matched'][n]['v5_px']:.2f} → {maps['dit_ft'][n]['v5_px']:.2f}</td></tr>"
    html+='</table><p>A：邻帧有部分线索；B：片段内缺乏充分直接线索。B仅1例，不能外推。此轮没有修补#13上下文缺帧，也没有扩展#6的时间窗口。</p></section><section><h2>验证与边界</h2><p>冻结前28层的缓存拆分前向误差为0；最后4层均有非零梯度和实际权重更新。因旧缓存与在线BF16路径存在0.1–0.2px数值差异，增加了同在线路径的冻结对照；两者第0步开发集均值一致。</p><p>这是最后4层与空间层共同微调的短程试验，不能单独归因于主干，也不能据此判断全量解冻的上限。测试对象和hard case此前已被查看；这些结果是开发诊断，不是新的跨人泛化证明。原始v4默认模型保留。</p><a href="comparison.json">完整结果JSON</a> · <a href="DEVELOPMENT_NOTES.txt">开发记录</a></section></html>'
    (delivery/'report.html').write_text(html,encoding='utf-8')
    notes='视觉联合微调 v5 开发记录\n\n1. 原方案主干冻结：这轮解冻最后4层，空间层及两类时序头共同更新，前28层缓存保持冻结。\n2. 旧FP16缓存与在线BF16存在数值差异：追加相同在线路径的冻结对照，第0步均值一致；旧缓存控制仅保留为诊断。\n3. 防止风险模型输入漂移：风险分支继续用原始冻结编码器；推理另拷贝微调尾部提取修正特征。\n4. 保持数据/门控可比：自然RGB，无人工遮挡，600步同种子同batch；开发集选检查点，测试不选阈值。\n5. 主干是否真训练：逐层梯度非零、参数变化检查均通过。\n6. 范围：仅末4层+空间层短程联合微调，已查看测试集；未修复缺帧或增加上下文。\n'
    (delivery/'DEVELOPMENT_NOTES.txt').write_text(notes,encoding='utf-8')
    for name in ['protocol.json','matched_control_protocol.json','raw_smoke.json']:
        if (RUN/name).exists():shutil.copy2(RUN/name,delivery/name)
    code_dir=delivery/'code';code_dir.mkdir(exist_ok=True)
    for name in ['finetune_visual_v5.py','finetune_visual_matched_v5.py','run_visual_v5.py','run_visual_matched_v5.py','infer_visual_v5.py','report_visual_v5.py']:shutil.copy2(Path(__file__).parent/name,code_dir/name)
    (delivery/'checkpoint_locations.json').write_text(json.dumps({k:str(RUN/k/'best.pt') for k in reports},indent=2))
    (delivery/'manifest.json').write_text(json.dumps({str(p.relative_to(delivery)):hashlib.sha256(p.read_bytes()).hexdigest() for p in delivery.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2))
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()

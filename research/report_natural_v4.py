import json,html
import numpy as np,torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spatial_rgb_common as s
from natural_reliability import RUN
from offline_rgb_encoder import crop_image
from evaluate_offline_kp import bootstrap

def main():
    O=RUN/'delivery';O.mkdir(exist_ok=True);report=json.loads((RUN/'test_results.json').read_text());policy=report['policy']['policy']
    path=RUN/('predictions_final.npz' if (RUN/'predictions_final.npz').exists() else 'predictions.npz');p=np.load(path)
    candidates=json.loads((s.RUN/'natural_finger_audit/candidates.json').read_text());selected=json.loads((s.RUN/'natural_finger_audit/reviewed_examples.json').read_text())
    rows=json.loads((s.OLD/'rows.json').read_text());records,index=s.records_and_index();lookup={int(w):i for i,w in enumerate(p['window_indices'])}
    error=np.linalg.norm(p['prediction']-p['gt'],axis=-1)*1408;before=np.linalg.norm(p['base']-p['gt'],axis=-1)*1408
    mask=p['valid']&p['available'];hard=np.zeros_like(mask)
    reviews=[]
    for q in candidates:
        j=lookup[q['window_index']];m=mask[j];hard[j]=m;flag=(p['p_bad'][j]>=policy['threshold'])&~p['selected'][j]&m
        reviews.append(dict(**q,before_px=float(before[j,m].mean()),after_px=float(error[j,m].mean()),
            corrected_points=int((p['selected'][j]&m).sum()),high_risk_unresolved_points=int(flag.sum()),
            needs_review=True,point_flags=flag.tolist(),candidate_regression_px=float(np.linalg.norm(p['regression'][j]-p['gt'][j],axis=-1)[m].mean()*1408),
            candidate_dit_px=float(np.linalg.norm(p['dit'][j]-p['gt'][j],axis=-1)[m].mean()*1408)))
    clusters=np.array([rows[int(w)]['sequence'] for w in p['window_indices']]);ci=bootstrap(error-before,hard,clusters)
    s.save(O/'hardcase_review_queue.json',reviews);s.save(O/'hardcase_statistics.json',dict(paired_source_sequence_ci95_change_px=ci,**report['hardcases']))
    for page,start in enumerate(range(0,len(selected),4)):
        fig,axes=plt.subplots(4,4,figsize=(13,12),layout='constrained')
        for rr,q in enumerate(selected[start:start+4]):
            wi=q['window_index'];j=lookup[wi];fid=int(index['feature_ids'][wi,8]);roi=index['roi'][fid].numpy()*1408
            image=np.rot90(crop_image(cv2.imread(q['image']),roi)[:,:,::-1]);m=mask[j]
            for cc,(name,pts) in enumerate([('Original RGB',None),('WiLoR / previous pipeline',p['base'][j]),('Selective correction',p['prediction'][j]),('Raw regression candidate',p['regression'][j])]):
                ax=axes[rr,cc];ax.imshow(image);ax.axis('off');ax.set_xlim(0,255);ax.set_ylim(255,0)
                if pts is not None:
                    for xy,col,mrk in [(p['gt'][j],'#00f5c4','o'),(pts,'#ff5252','x')]:
                        uv=(xy*1408-roi[:2])/(roi[2:]-roi[:2])*256;uv=np.stack([uv[:,1],255-uv[:,0]],-1)
                        ax.scatter(uv[m,0],uv[m,1],s=12,c=col,marker=mrk,linewidths=1)
                    name+=f"\n{np.linalg.norm(pts-p['gt'][j],axis=-1)[m].mean()*1408:.1f} px"
                else:name+=f"\nID {q['id']} / {q['visual_review_category']}"
                ax.set_title(name,fontsize=8)
        fig.suptitle('Natural hard cases | No artificial masking | Reference green / prediction red',fontsize=11)
        fig.savefig(O/f'hardcases_{page}.png',dpi=140);plt.close(fig)
    a=report['all_test'];h=report['hardcases'];r=report['reliability']
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,title,labels,values in [
        (axes[0],'Reliability: identify >20px errors',['Box score','Joint risk'],[r['box']['average_precision'],r['joint']['average_precision']]),
        (axes[1],'All natural test points / px',['Before','Selective'],[a['base_mean_px'],a['mean_px']]),
        (axes[2],'47 mined hard cases / px',['Before','Selective'],[h['base_mean_px'],h['mean_px']])]:
        bars=ax.bar(labels,values,color=['#a6afbb','#2185a6']);ax.bar_label(bars,fmt='%.3f' if ax==axes[0] else '%.2f',padding=3);ax.set_ylim(0,max(values)*1.22);ax.set_title(title,fontsize=10)
    fig.suptitle('Natural reliability v4 | Reused audit, not a fresh independent benchmark',fontsize=12);fig.savefig(O/'summary.png',dpi=160);plt.close(fig)
    names={12:'玩具屋重遮挡',6:'杯子遮挡',25:'双手与罐子重叠',33:'弯曲自遮挡',37:'运动模糊',45:'边缘暗部',28:'饮料盒遮挡',36:'抓握物体'}
    tab=''.join(f"<tr><td>{names[e['id']]}</td><td>{e['base_mean_px']:.2f}</td><td>{e['mean_px']:.2f}</td><td>{e['selected_points']}</td></tr>" for e in report['examples'])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>自然预测可靠性与修正 v4</title>
<style>body{{font:16px/1.7 system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#24334b}}h1{{font-size:28px}}h2{{font-size:21px}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff2dc;padding:16px}}pre{{white-space:pre-wrap}}</style>
<h1>自然手指困难案例：可靠性识别与选择性修正</h1>
<p>输入均为原始RGB和原始前端预测，没有人工遮挡或人为坐标缺口。available只表示坐标存在；confirmed才表示人工确认、必须锁定。模型预测保留为软参考，低可靠点可被修正。</p>
<p>现成YOLO手框分数是基线。新逐点可靠性头结合手框分数、冻结WiLoR原生空间特征和前后帧一致性，学习原预测误差&gt;20 px的风险；训练真值不进入推理。修正训练使用按受试者交叉预测的风险分数。</p>
<p>全部保留序列上，识别错误点的AP：手框分数 {r['box']['average_precision']:.3f}，逐点可靠性 {r['joint']['average_precision']:.3f}。这不是可见性分类，隐藏点也可能预测准确。</p>
<img src="summary.png">
<h2>修正结果</h2><p>共同可评估的{a['points']}个原本有坐标的手指点：误差 {a['base_mean_px']:.2f}→{a['mean_px']:.2f} px；来源序列配对bootstrap变化区间 {report['mean_change_ci95_px'][0]:.2f} 至 {report['mean_change_ci95_px'][1]:.2f} px。</p>
<p>原本&gt;20 px的{a['originally_bad_points']}点中，{a['bad_recovered_to20']}点降到≤20 px；原本≤10 px的{a['good_points']}点中，{a['good_harmed_to_over20']}点被改坏到&gt;20 px（{a['harm_good_to_over20_rate']*100:.2f}%）。</p>
<p>47个hard case的共同839点：{h['base_mean_px']:.2f}→{h['mean_px']:.2f} px；324个原错误点中50点降到≤20 px，226个原本≤10 px的点中1点被改坏到&gt;20 px。hard case变化区间：{ci[0]:.2f} 至 {ci[1]:.2f} px。</p>
<table><tr><th>自然案例</th><th>修正前 / px</th><th>修正后 / px</th><th>修改点数</th></tr>{tab}</table>
<p class="note">最严重的玩具屋遮挡尚未解决，杯子等例子仍有退化。未通过自动修正规则的高风险点保留原预测和多个候选，标记needs_special_review，不能当作准确标注。平均改善不等于所有困难案例恢复成功。</p>
<h2>为何最终自动策略使用回归候选</h2>
<p>回归和DiT都已在自然预测误差上重新训练。开发集选择的策略为：原点风险≥0.35，候选位移≤手框边长10%，使用回归修正的50%。直接替换全部预测会破坏准确点；DiT仅保留为候选和对照，没有强行选它作为最终自动结果。</p>
<p>独立校准片段约束原本≤10 px的点变成&gt;20 px的比例≤1%，且总体误差与原错误点误差都改善。校准集改动比例29.7%，测试实际36.6%；不能把校准比例当作运行时硬上限。追加的候选风险方案未超过开发集基线，已拒绝，没有用hard case结果选新阈值。</p>
<h2>逐例对比</h2><img src="hardcases_0.png"><img src="hardcases_1.png">
<h2>输入与输出</h2><p>infer_natural_reliability.py接受camera、image、box_xyxy、box_confidence、timestamp_s、xy_px、available和可选confirmed。兼容旧observed字段时仅视为available，绝不自动当作人工确认。输出区分confirmed_input、corrected_prediction、completed_missing和unconfirmed_input，并提供原点风险、候选与决策原因。自动点一律需要复核。</p>
<p>训练6个受试者；P0003按clip分为模型选择与校准；P0010/P0015及47个hard case没有回灌训练或选阈值。但这些测试资料此前已查看，仍属于复用审计，不能称全新独立证据。当前误差包括所有有效手指点，不能等同于仅不可见点误差。上游WiLoR预训练重叠未知。</p>
<p><a href="test_results.json">完整结果</a> · <a href="hardcase_review_queue.json">47个案例复核清单</a> · <a href="example_input.json">输入示例</a> · <a href="example_output.json">运行结果</a> · <a href="delivery_checks.json">接口验证</a></p></html>'''
    (O/'report.html').write_text(page,encoding='utf-8')
    (O/'README.txt').write_text(f'''Natural reliability v4
Server run: {RUN}
Report: report.html. Review list: hardcase_review_queue.json.
No artificial pixel masks or coordinate removal are used in this experiment.

Run on the existing server:
/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python /mnt/why/hot3d_hand_residual/infer_natural_reliability.py --input {O}/example_input.json --output /tmp/natural_corrected.json

Each frame needs original image, camera, box_xyxy, box_confidence, timestamp_s,
20 xy_px points, and available flags. confirmed defaults to all false. Legacy
observed is interpreted as available only. Only confirmed points are immutable.
The fixture marks two points per frame confirmed to test this interface; these
are not human labels and are not used in recovery accuracy evaluation.

input_error_risk refers to the original coordinate error, not pixel visibility
or corrected-output confidence. candidate_error_scores are diagnostic scores
outside their original calibration distribution, not calibrated guarantees.
Every automatic point needs review. Large uncertain moves remain as candidates.

The frozen full WiLoR encoder/source/assets are external dependencies from
{s.common.RUN}; original HOT3D images are not bundled.
The fitted risk/correction heads and spatial stem are bundled under sealed/.
The legacy infer_spatial_rgb.py remains a previous experimental interface;
use infer_natural_reliability.py for the new trust-aware behavior.
''')

if __name__=='__main__':main()

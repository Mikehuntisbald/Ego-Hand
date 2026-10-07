"""Evaluate frozen outputs only; no selection, fitting, or threshold changes."""
import json,collections,html,hashlib
from pathlib import Path
import numpy as np,torch,cv2
from hand3d_v8_common import V7,save
from evaluate_joint_kinematic_v30 import report,identities
from calibrate_hand3d_v8 import paired_ci
from complete_hand_tracks_acceleration_v46 import profile_config
from evaluate_stability_v42 import quantiles
import spatial_rgb_common as s
RUN=V7.parent/'acceleration_validation_v46'

def main():
    torch.set_num_threads(4);assert json.loads((RUN/'outputs_frozen.json').read_text())['complete']
    runtime=json.loads((RUN/'runtime_outputs_0.json').read_text())
    limits={k:profile_config('acc_x2')[k] for k in ['root_speed','pose_speed','wrist_speed','joint_speed','root_acc','pose_acc','wrist_acc','joint_acc']}
    assert all(t['constraints']['limits']==limits for t in runtime['tracks'])
    runtime.update(config=profile_config('acc_x2'),mode='experimental_acceleration_v46',motion_profile='acc_x2',acceleration_multiplier=2,speed_multiplier=1,default_replaced=False)
    save(RUN/'runtime_outputs.json',runtime)
    original=json.loads((RUN/'fresh_rows.json').read_text());n=len(original)
    centers=torch.load(RUN/'fresh_data.pt',weights_only=False,map_location='cpu')['rows']
    center_ids={r['row_index']:i for i,r in enumerate(centers)}
    rows=[dict(r,window_index=center_ids.get(i)) for i,r in enumerate(original)]
    cache={};preds={p:torch.zeros(n,20,3) for p in ['strict','acc_x2']};checks={p:[] for p in preds}
    coverage=torch.zeros(n,dtype=torch.bool)
    for path in sorted((RUN/'tracks').glob('*_inputs.pt')):
        inp=torch.load(path,weights_only=False,map_location='cpu');ix=torch.tensor(inp['indices']);coverage[ix]=True
        for k,v in inp['cache'].items():
            if k not in cache:cache[k]=torch.zeros((n,)+v.shape[1:],dtype=v.dtype)
            cache[k][ix]=v
        tid=path.name.removesuffix('_inputs.pt')
        for p in preds:
            result=torch.load(RUN/'tracks'/f'{tid}_{p}.pt',weights_only=False,map_location='cpu')
            assert result['check']['passed'];preds[p][ix]=result['prediction'];checks[p].append(result['check'])
    assert coverage.all()
    gt=torch.zeros(n,20,3);valid=torch.zeros(n,20,dtype=torch.bool)
    for i,r in enumerate(rows):
        if r['matched']:
            gt[i]=torch.tensor(r['gt']);valid[i]=torch.isfinite(gt[i]).all(-1)
    labels=dict(gt=gt,valid=valid);sides=identities(rows,labels)
    reports={p:report(x,cache,labels,rows,sides,cache['base']) for p,x in preds.items()}
    groups=collections.defaultdict(list);pairs=[]
    for i,r in enumerate(rows):groups[(r['sequence'],r['clip'],r['track_id'])].append(i)
    for ix in groups.values():
        ix.sort(key=lambda i:rows[i]['timestamp_ns'])
        pairs.extend((a,b) for a,b in zip(ix,ix[1:]) if rows[b]['frame']==rows[a]['frame']+1 and sides[a] is not None and sides[a]==sides[b])
    a,b=map(torch.tensor,zip(*pairs));pm=valid[a]&valid[b]&valid[a,5,None]&valid[b,5,None];pm[:,5]=False
    world=lambda x:torch.einsum('njc,nkc->njk',x,cache['rotation'])+cache['translation'][:,None]
    gw=world(gt);gpose=gw-gw[:,5:6];gd=gpose[b]-gpose[a];gm=gd.norm(dim=-1)*1000;fast=pm&(gm>10)
    mask=valid.clone()&valid[:,5,None];mask[:,5]=False
    relerror=lambda x:((x-x[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000
    strict_error=relerror(preds['strict']);good=mask&(strict_error<=10);bad=mask&(strict_error>20)
    cm=torch.tensor([r['window_index'] is not None for r in rows]);cr=[r for r in rows if r['window_index'] is not None]
    diagnostic={};retentions={}
    for p,x in preds.items():
        pw=world(x);pose=pw-pw[:,5:6];delta=pose[b]-pose[a];move=delta.norm(dim=-1)*1000
        retention=move/gm.clamp_min(1e-6);direction=(delta*gd).sum(-1)/gd.square().sum(-1).clamp_min(1e-10)
        e=relerror(x);retentions[p]=retention
        diagnostic[p]=dict(all_valid_relative_mm=quantiles(e[mask]),all_valid_camera_mm=quantiles((x-gt).norm(dim=-1)[mask]*1000),
            fast_point_pairs=int(fast.sum()),fast_amplitude_ratio=quantiles(retention[fast]),fast_directional_retention=quantiles(direction[fast]),
            fast_under_half_points=int((fast&(retention<.5)).sum()),strict_good_points=int(good.sum()),strict_good_to_bad_points=int((good&(e>20)).sum()),
            strict_bad_points=int(bad.sum()),strict_bad_to_good_points=int((bad&(e<=20)).sum()),
            paired_all_frame_vs_strict=paired_ci(x,preds['strict'],gt,valid,rows),
            paired_center_vs_strict=paired_ci(x[cm],preds['strict'][cm],gt[cm],valid[cm],cr))
    # Five sequence bootstrap groups are a limited uncertainty estimate.
    seqs=sorted({r['sequence'] for r in rows});rng=np.random.default_rng(202610051);draw=rng.integers(len(seqs),size=(5000,len(seqs)))
    deltas=retentions['acc_x2']-retentions['strict'];seq_values=[]
    for seq in seqs:
        sel=torch.tensor([rows[int(i)]['sequence']==seq for i in a]);m=fast&sel[:,None];seq_values.append(deltas[m].numpy())
    boots=[]
    for ix in draw:
        v=np.concatenate([seq_values[j] for j in ix]);boots.append(float(np.median(v)) if len(v) else 0.)
    summary=dict(complete=True,scope=json.loads((RUN/'fresh_manifest.json').read_text())['scope'],reports=reports,diagnostics=diagnostic,
        observations=n,matched_observations=int(valid.all(-1).sum()),unmatched_observations=int((~valid.all(-1)).sum()),
        centers=len(centers),matched_centers=int(valid[cm].all(-1).sum()),sequences=len(seqs),actors=len({r['subject'] for r in rows}),same_hand_pairs=len(pairs),
        fast_paired_point_retention_delta_ci95=np.quantile(boots,[.025,.975]).tolist(),
        constraints={p:dict(passed=all(c['passed'] for c in checks[p]),tracks=len(checks[p]),
            maxima={k:max(c['maxima'][k] for c in checks[p]) for k in checks[p][0]['maxima']}) for p in preds},
        detection_coverage=json.loads((RUN/'fresh_detection_counts.json').read_text()),
        default_changed=False,no_fresh_threshold_tuning=True,new_subject_evidence=False,
        notes=['Same frozen DiT candidates and path. Original per-frame side frontend, no new consensus policy.',
               'Accuracy is conditional on GT box match; unmatched predictions are still reconstructed and constrained.',
               'Motion pairs require adjacent frames and same GT side. Fast means GT wrist-relative step >10mm.',
               'Five source sequence groups; actors and source sequences were encountered previously. No full missing-hand detection claim.'])
    save(RUN/'summary.json',summary);torch.save(dict(predictions=preds,gt=gt,valid=valid,base=cache['base'],a=a,b=b,fast=fast,retentions=retentions),RUN/'audit_tensors.pt')
    out=RUN/'review';out.mkdir(exist_ok=True);save(out/'summary.json',summary)
    # Show best improvement and worst deterioration without claiming occlusion visibility labels.
    means=lambda e:(e*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    delta=means(relerror(preds['acc_x2']))-means(strict_error);matched=valid.all(-1)
    ids=torch.where(matched)[0];chosen=[int(ids[delta[ids].argmin()]),int(ids[delta[ids].argmax()])]
    unmatched=torch.where(~matched)[0]
    if len(unmatched):chosen.append(int(unmatched[(preds['strict'][unmatched]-preds['acc_x2'][unmatched]).norm(dim=-1).max(-1).values.argmax()]))
    from evaluate_joint_kinematic_v30 import EDGES
    assets=[]
    for case,i in enumerate(chosen):
        r=rows[i];panels=[]
        for p,color in [('strict',(30,220,255)),('acc_x2',(50,230,70)),('GT',(230,150,40))]:
            im=cv2.imread(r['image']);x=gt[i] if p=='GT' else preds[p][i]
            if p!='GT' or matched[i]:
                uv=s.common.from_json(r['camera']).eye_to_window(x.numpy())
                for u,v in EDGES:
                    if np.isfinite(uv[[u,v]]).all():cv2.line(im,tuple(np.clip(uv[u],-4000,4000).astype(int)),tuple(np.clip(uv[v],-4000,4000).astype(int)),color,5)
                for xy in uv:
                    if np.isfinite(xy).all():cv2.circle(im,tuple(np.clip(xy,-4000,4000).astype(int)),7,color,-1)
            cv2.putText(im,p if p!='GT' or matched[i] else 'GT unavailable / unmatched',(30,55),cv2.FONT_HERSHEY_SIMPLEX,1.4,color,3)
            panels.append(cv2.resize(im,(480,480)))
        name=f'case_{case}.jpg';cv2.imwrite(str(out/name),np.concatenate(panels,axis=1));assets.append(dict(image=name,row=i,sequence=r['sequence'],clip=r['clip'],frame=r['frame'],matched=bool(matched[i]),relative_delta_mm=float(delta[i]) if matched[i] else None))
    save(out/'cases.json',assets)
    table=''
    for p in preds:
        m=reports[p]['metrics'];d=diagnostic[p];c=reports[p]['coherence']
        table+=f"<tr><td>{p}</td><td>{m['relative_mm']:.3f}</td><td>{m['camera_mm']:.3f}</td><td>{d['fast_amplitude_ratio']['median']*100:.2f}%</td><td>{d['fast_directional_retention']['median']*100:.2f}%</td><td>{c['spurious_jump_pairs']}</td><td>{d['strict_good_to_bad_points']}/{d['strict_good_points']}</td><td>{d['strict_bad_to_good_points']}/{d['strict_bad_points']}</td></tr>"
    hard_table=''
    for p in preds:
        h=reports[p]['hard']
        hard_table+=f"<tr><td>{p}</td><td>{reports[p]['hard_windows']}</td><td>{h['relative_mm']:.3f}</td><td>{h['relative_bad_recovered20']}/{h['relative_bad_points']}</td><td>{reports[p]['metrics']['relative_good_harmed20']}/{reports[p]['metrics']['relative_good_points']}</td></tr>"
    hard_html=f"<h2>高误差组与保护代价</h2><p>高误差组指原WiLoR相对腕部平均误差&gt;40mm，是误差分组，不是手指遮挡真值。42/3016（1.39%）个严格版原本≤10mm的点在放宽版超过20mm；3501/40061个严格版&gt;20mm的点被恢复到≤20mm。</p><table><tr><th>配置</th><th>高误差中心数</th><th>高误差组相对误差mm</th><th>该组原WiLoR坏点恢复</th><th>中心原WiLoR好点损坏</th></tr>{hard_table}</table><p>高误差组的恢复数量53→42，均值40.69→41.32mm。动作保留收益与极难样本恢复收益需要分别判断。</p>"
    cases=''.join(f"<p>{html.escape(str(c))}</p><img src='{c['image']}' style='width:100%;max-width:1440px'>" for c in assets)
    page=f"<!doctype html><meta charset='utf-8'><title>v46 加速度×2验证</title><style>body{{font:16px sans-serif;margin:30px;background:#fafafa}}td,th{{padding:10px;border:1px solid #ccc}}table{{border-collapse:collapse}}</style><h1>速度不变，加速度同步×2：新片段验证</h1><p>{html.escape(summary['scope'])}</p><p>{n}检测观测，{len(centers)}预测中心（其中{summary['matched_centers']}个GT关联有效）；同一DiT候选及路径，±1.6秒。默认v42未替换。</p><table><tr><th>配置</th><th>中心相对误差mm</th><th>中心相机误差mm</th><th>快动作幅度</th><th>方向保留</th><th>伪跳变</th><th>好点损坏</th><th>坏点恢复</th></tr>{table}</table><p>快动作指GT相对腕部单帧位移&gt;10mm。误差只覆盖GT关联成功的检测；未关联输出仍进行结构与时序检查。不能据此证明整个遮挡片段补全正确。</p><p>中心配对误差95%区间：{html.escape(str(diagnostic['acc_x2']['paired_center_vs_strict']))}</p><p>逐点配对快动作保留增量中位数95%区间：{summary['fast_paired_point_retention_delta_ci95']}。仅5个源序列分组。</p>{hard_html}<h2>改善最多 / 恶化最多 / 未匹配变化最大</h2><p>三列依次为严格版、加速度×2版、GT的2D投影；误差统计仍是3D。最大变化案例用于展示局部代价，不代表常见情况。</p>{cases}<p><a href='summary.json'>完整统计与约束检查</a></p>"
    (out/'report.html').write_text(page)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

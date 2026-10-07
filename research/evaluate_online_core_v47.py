"""Describe the completed joint-training core and its sealed replay tradeoffs."""
import hashlib,json,collections,html
from pathlib import Path
import numpy as np,torch,cv2
from hand3d_v8_common import V7,save
from online_parameter_model_v47 import RUN
from evaluate_joint_kinematic_v30 import report,identities,EDGES
from evaluate_stability_v42 import quantiles
from calibrate_hand3d_v8 import paired_ci
import spatial_rgb_common as s

SOURCE=V7.parent/'acceleration_validation_v46';DIAG=RUN/'diagnostic_v46';MODES=['initial','frozen_last','joint_last']
def main():
    torch.set_num_threads(4);freeze={};predictions={};seals={}
    for mode in MODES:
        seal=json.loads((DIAG/mode/'sealed.json').read_text());assert seal['complete'];seals[mode]=seal
        path=DIAG/mode/'result.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==seal['result_sha256']
        result=torch.load(path,weights_only=False,map_location='cpu');predictions[mode]=result['prediction'];freeze[mode]=seal['result_sha256']
    save(RUN/'evaluation_freeze.json',dict(all_outputs_sealed_before_labels=True,sha256=freeze,scope='Already evaluated v46 clips, diagnostic replay only'))
    rows=json.loads((SOURCE/'fresh_rows.json').read_text());windows=torch.load(SOURCE/'fresh_data.pt',weights_only=False,mmap=True)['rows']
    mapping={r['row_index']:i for i,r in enumerate(windows)};rows=[dict(r,window_index=mapping.get(i)) for i,r in enumerate(rows)]
    N=len(rows);gt=torch.zeros(N,20,3);valid=torch.zeros(N,20,dtype=torch.bool)
    for i,r in enumerate(rows):
        if r['matched']:gt[i]=torch.tensor(r['gt']);valid[i]=torch.isfinite(gt[i]).all(-1)
    labels=dict(gt=gt,valid=valid);sides=identities(rows,labels);cache=torch.load(DIAG/'cache.pt',weights_only=False,map_location='cpu')
    reports={m:report(p,cache,labels,rows,sides,cache['base']) for m,p in predictions.items()}
    groups=collections.defaultdict(list);pairs=[]
    for i,r in enumerate(rows):groups[(r['sequence'],r['clip'],r['track_id'])].append(i)
    for ix in groups.values():
        ix.sort(key=lambda i:rows[i]['timestamp_ns'])
        pairs.extend((a,b) for a,b in zip(ix,ix[1:]) if rows[b]['frame']==rows[a]['frame']+1 and sides[a] is not None and sides[a]==sides[b])
    a,b=map(torch.tensor,zip(*pairs));pm=valid[a]&valid[b]&valid[a,5,None]&valid[b,5,None];pm[:,5]=False
    world=lambda x:torch.einsum('njc,nkc->njk',x,cache['rotation'])+cache['translation'][:,None]
    gw=world(gt);gpose=gw-gw[:,5:6];gd=gpose[b]-gpose[a];gm=gd.norm(dim=-1)*1000;fast=pm&(gm>10)
    error=lambda p:((p-p[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000
    mask=valid.clone();mask[:,5]=False;base=predictions['frozen_last'];be=error(base);good=mask&(be<=10);bad=mask&(be>20)
    cm=torch.tensor([r['window_index'] is not None for r in rows]);cr=[r for r in rows if r['window_index'] is not None];extras={}
    for mode,p in predictions.items():
        pw=world(p);pose=pw-pw[:,5:6];delta=pose[b]-pose[a];move=delta.norm(dim=-1)*1000;retention=move/gm.clamp_min(1e-6)
        direction=(delta*gd).sum(-1)/gd.square().sum(-1).clamp_min(1e-10);e=error(p)
        extras[mode]=dict(fast_point_pairs=int(fast.sum()),fast_amplitude_ratio=quantiles(retention[fast]),fast_directional_retention=quantiles(direction[fast]),
            frozen_good_points=int(good.sum()),frozen_good_to_bad_points=int((good&(e>20)).sum()),frozen_bad_points=int(bad.sum()),frozen_bad_to_good_points=int((bad&(e<=20)).sum()),
            paired_center_vs_frozen=paired_ci(p[cm],base[cm],gt[cm],valid[cm],cr),paired_center_vs_initial=paired_ci(p[cm],predictions['initial'][cm],gt[cm],valid[cm],cr))
    train={}
    for mode in ['frozen','joint']:
        folder=RUN/f'dit_{mode}';done=json.loads((folder/'done.json').read_text());history=json.loads((folder/'history.json').read_text())
        assert done['complete'];train[mode]=dict(done=done,history=history,initial_parity=json.loads((folder/'initial_parity.json').read_text()),
            gradients=json.loads((folder/'gradients.json').read_text()),weight_update=json.loads((folder/'weight_update.json').read_text()),config=json.loads((folder/'config.json').read_text()))
    assert train['frozen']['config']['batch_plan_sha256']==train['joint']['config']['batch_plan_sha256']
    assert train['frozen']['config']['initial_checkpoint_sha256']==train['joint']['config']['initial_checkpoint_sha256']
    gradient=json.loads((RUN/'dit_joint/fk_gradient_path.json').read_text());assert gradient['passed']
    summary=dict(complete=True,architecture_completed=True,training=train,pure_fk_gradient_check=gradient,reports=reports,diagnostics=extras,seals=seals,
        default_changed=False,scope='2526 train centers,352 repeated dev-select centers; replay10already evaluated clips,5source sequences,2known actors. Upstream pretraining overlap unknown.',
        replay_observations=N,replay_matched=int(valid.all(-1).sum()),replay_centers=int(cm.sum()),replay_matched_centers=int(valid[cm].all(-1).sum()),
        contract='Only RGB reconstruction core is jointly trainable. YOLO/tracking, original WiLoR+IK seeds, hard candidate path and stability solver remain outside backprop.',
        decoder='Existing UmeTrack-compatible parameterized FK, not a new MANO decoder',
        replay_protocol='Same fixed whole-batch solver call and sampling layout for every model; historical v46 per-track numbers are not direct controls.',
        trained_step_diagnostics_do_not_change_dev_selection=True,new_generalization_claim=False)
    save(RUN/'summary.json',summary)
    out=RUN/'review';out.mkdir(exist_ok=True);save(out/'summary.json',summary)
    labels={'initial':'联训前初始模型','frozen_last':'冻结视觉主干＋时序头训练600步','joint_last':'末端4层＋时序头联训600步'}
    table=''
    for mode in MODES:
        r=reports[mode];m=r['metrics'];d=extras[mode];c=r['coherence']
        table+=f"<tr><td>{labels[mode]}</td><td>{m['relative_mm']:.3f}</td><td>{m['camera_mm']:.3f}</td><td>{d['fast_amplitude_ratio']['median']*100:.2f}%</td><td>{c['spurious_jump_pairs']}</td><td>{r['hard']['relative_bad_recovered20']}</td></tr>"
    def curve(key):
        histories={m:train[m]['history'] for m in train};values=[h['metrics'][key] for hs in histories.values() for h in hs];lo,hi=min(values),max(values);span=max(hi-lo,.01)
        lines=''
        for mode,color in [('frozen','#2563eb'),('joint','#d97706')]:
            pts=' '.join(f"{45+h['step']/600*490:.1f},{190-(h['metrics'][key]-lo)/span*145:.1f}" for h in histories[mode])
            lines+=f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='3'/>"
        return f"<svg viewBox='0 0 580 230' width='580'><text x='45' y='22'>{key}: {lo:.3f}–{hi:.3f} mm</text><path d='M45 40V195H540' fill='none' stroke='#aaa'/>{lines}<text x='45' y='220'>0</text><text x='510' y='220'>600步</text></svg>"
    e=error(predictions['joint_last']);means=lambda x:(x*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    change=means(e)-means(be);ix=torch.where(valid.all(-1))[0];chosen=[int(ix[change[ix].argmin()]),int(ix[change[ix].argmax()])];cases=[]
    for ci,i in enumerate(chosen):
        panels=[];r=rows[i]
        for name,p,color in [('frozen',predictions['frozen_last'],(30,220,255)),('joint',predictions['joint_last'],(50,230,70)),('GT',gt,(230,150,40))]:
            image=cv2.imread(r['image']);uv=s.common.from_json(r['camera']).eye_to_window(p[i].numpy())
            for u,v in EDGES:
                if np.isfinite(uv[[u,v]]).all():cv2.line(image,tuple(np.clip(uv[u],-4000,4000).astype(int)),tuple(np.clip(uv[v],-4000,4000).astype(int)),color,5)
            for point in uv:
                if np.isfinite(point).all():cv2.circle(image,tuple(np.clip(point,-4000,4000).astype(int)),7,color,-1)
            cv2.putText(image,name,(25,55),cv2.FONT_HERSHEY_SIMPLEX,1.4,color,3);panels.append(cv2.resize(image,(480,480)))
        filename=f'case_{ci}.jpg';cv2.imwrite(str(out/filename),np.concatenate(panels,1));cases.append(dict(image=filename,sequence=r['sequence'],clip=r['clip'],frame=r['frame'],relative_delta_mm=float(change[i])))
    save(out/'cases.json',cases);case_html=''.join(f"<p>{html.escape(str(c))}</p><img src='{c['image']}' style='width:100%;max-width:1440px'>" for c in cases)
    d=extras['joint_last'];selection={mode:train[mode]['done']['selected_step'] for mode in train}
    page=f"""<!doctype html><meta charset='utf-8'><title>v47 在线RGB三维联训</title><style>body{{font:16px sans-serif;margin:30px;max-width:1500px;background:#fafafa}}td,th{{padding:10px;border:1px solid #ccc}}table{{border-collapse:collapse}}svg{{max-width:100%}}</style>
<h1>v47：在线RGB→时序DiT→参数化FK</h1><p>视觉0–27层冻结；28–31层与末端Norm、时序头联训。保留192×1280空间特征。YOLO框/轨迹、WiLoR初值、200步IK固定；候选选择与350步稳定器在反向传播图之外。结构几何缓冲区固定，参数输出通过FK接受3D监督。</p>
<p>纯3D FK误差到视觉层梯度检查通过；末端4层均有非零梯度及实际权重更新。两组使用相同在线RGB、BF16补齐16帧、初始权重、批次计划、600步预算、3D/时序/保护损失。窗口为±1.6秒，无人工遮挡。</p>
<p>开发集选出的最佳步数：{selection}。以下另列固定600步训练权重诊断，不能用复测结果替代开发集选型。默认v42未替换。</p>
<h2>开发集训练曲线（蓝=冻结主干，橙=联训）</h2>{curve('relative_mm')}{curve('camera_mm')}
<h2>已评估v46片段复测</h2><p>{N}检测观测、{int(valid.all(-1).sum())}GT关联观测、{int(valid[cm].all(-1).sum())}有效中心；10片段、5源序列、2已见过受试者。该复测不是新的独立泛化证据。三组使用同一批处理与稳定器协议，不能直接与历史按轨迹处理的v46数字作训练收益比较。</p>
<table><tr><th>模型</th><th>中心相对误差mm</th><th>中心相机误差mm</th><th>快动作幅度保留</th><th>伪跳变</th><th>高误差组恢复点数</th></tr>{table}</table>
<p>联训相对冻结主干：好点损坏{d['frozen_good_to_bad_points']}/{d['frozen_good_points']}；坏点恢复{d['frozen_bad_to_good_points']}/{d['frozen_bad_points']}。好点≤10mm、损坏&gt;20mm；坏点&gt;20mm、恢复≤20mm。高误差组是原WiLoR相对误差&gt;40mm中心，并非遮挡真值。</p>
<p>中心配对95%区间：{html.escape(str(d['paired_center_vs_frozen']))}</p>
<h2>改善最多／恶化最多的3D输出投影</h2><p>三列：冻结主干、联训、GT。图片为2D投影，统计为3D误差。</p>{case_html}<p><a href='summary.json'>完整统计与协议</a></p>"""
    (out/'report.html').write_text(page)
    print(json.dumps(dict(complete=True,selected=selection,table={m:dict(relative=reports[m]['metrics']['relative_mm'],camera=reports[m]['metrics']['camera_mm'],fast=extras[m]['fast_amplitude_ratio']['median'],harm=extras[m]['frozen_good_to_bad_points']) for m in MODES})),flush=True)
if __name__=='__main__':main()

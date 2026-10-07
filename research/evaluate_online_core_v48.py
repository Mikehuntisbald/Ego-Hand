"""Evaluate sealed v48 outputs; report recovery, preservation and motion together."""
import collections,hashlib,html,json
import numpy as np,torch,cv2
from hand3d_v8_common import V7,save
from online_parameter_model_v47 import RUN as SOURCE
from train_online_parameter_v48 import RUN
from evaluate_joint_kinematic_v30 import report,identities,EDGES
from evaluate_stability_v42 import quantiles
from calibrate_hand3d_v8 import paired_ci
import spatial_rgb_common as s

DIAG=RUN/'diagnostic_v46';INPUT=SOURCE/'diagnostic_v46'
MODES=['initial','frozen_last','joint_last','protected_last','protected_best']
LABELS={'initial':'原准入模型','frozen_last':'冻结主干累计2400步','joint_last':'视觉联训累计2400步',
    'protected_last':'加入教师保护累计2400步','protected_best':'保护组准入权重（可能仍为原模型）'}

def main():
    torch.set_num_threads(4);predictions={};seals={}
    for mode in MODES:
        seal=json.loads((DIAG/mode/'sealed.json').read_text());assert seal['complete'];seals[mode]=seal
        path=DIAG/mode/'result.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==seal['result_sha256']
        predictions[mode]=torch.load(path,weights_only=False,map_location='cpu')['prediction']
    save(RUN/'evaluation_freeze.json',dict(all_outputs_sealed_before_labels=True,
        sha256={k:v['result_sha256'] for k,v in seals.items()},scope='Already evaluated v46 clips; diagnostic replay only'))
    rows=json.loads((V7.parent/'acceleration_validation_v46/fresh_rows.json').read_text())
    windows=torch.load(V7.parent/'acceleration_validation_v46/fresh_data.pt',weights_only=False,mmap=True)['rows']
    mapping={r['row_index']:i for i,r in enumerate(windows)};rows=[dict(r,window_index=mapping.get(i)) for i,r in enumerate(rows)]
    N=len(rows);gt=torch.zeros(N,20,3);valid=torch.zeros(N,20,dtype=torch.bool)
    for i,r in enumerate(rows):
        if r['matched']:gt[i]=torch.tensor(r['gt']);valid[i]=torch.isfinite(gt[i]).all(-1)
    labels=dict(gt=gt,valid=valid);sides=identities(rows,labels);cache=torch.load(INPUT/'cache.pt',weights_only=False,map_location='cpu')
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
    mask=valid.clone();mask[:,5]=False
    cm=torch.tensor([r['window_index'] is not None for r in rows]);cr=[r for r in rows if r['window_index'] is not None]
    extras={}
    for mode,p in predictions.items():
        pw=world(p);pose=pw-pw[:,5:6];delta=pose[b]-pose[a];move=delta.norm(dim=-1)*1000;e=error(p)
        protection={}
        for ref in ['initial','frozen_last','joint_last']:
            re=error(predictions[ref]);good=mask&(re<=10);bad=mask&(re>20)
            protection[ref]=dict(good_points=int(good.sum()),good_to_bad_points=int((good&(e>20)).sum()),
                bad_points=int(bad.sum()),bad_to_good_points=int((bad&(e<=20)).sum()),
                paired_center_CI=paired_ci(p[cm],predictions[ref][cm],gt[cm],valid[cm],cr))
        extras[mode]=dict(fast_point_pairs=int(fast.sum()),fast_amplitude_ratio=quantiles((move/gm.clamp_min(1e-6))[fast]),
            fast_directional_retention=quantiles(((delta*gd).sum(-1)/gd.square().sum(-1).clamp_min(1e-10))[fast]),protection=protection)
    training={}
    for arm in ['frozen','joint','protected']:
        folder=RUN/arm;done=json.loads((folder/'done.json').read_text());assert done['complete']
        training[arm]={key:json.loads((folder/f'{key}.json').read_text()) for key in ['done','history','config','preflight','resume_verification']}
    assert len({v['config']['batch_plan_sha256'] for v in training.values()})==1
    assert training['joint']['config']['initial_checkpoint_sha256']==training['protected']['config']['initial_checkpoint_sha256']
    summary=dict(complete=True,reports=reports,diagnostics=extras,training=training,seals=seals,default_changed=False,
        replay_observations=N,replay_matched=int(valid.all(-1).sum()),replay_centers=int(cm.sum()),replay_matched_centers=int(valid[cm].all(-1).sum()),
        scope='2526 training centers,352 reused dev-select centers;10previously evaluated clips,5source sequences,2known actors. Not fresh generalization evidence.',
        contrast='Joint/protected start from identical v47 joint600 weights; frozen continues its own matched v47 frozen600 weights. Optimizer restarted at continuation.',
        protocol='Same online RGB BF16 padded16, fixed risk and candidate evidence, batched solver, unchanged speeds and accelerationx2.',
        architecture='Online final4 ViT blocks/norm plus temporal DiT -> bounded UmeTrack-compatible parameters/FK. Detector, tracking, WiLoR+IK seeds and stability solver outside graph.',
        teacher_guard='Supervised loss only; no teacher or GT inference inputs. Selection includes protection of initial temporal model correct points.',
        diagnostic_checkpoints_do_not_override_admission=True,new_generalization_claim=False)
    save(RUN/'summary.json',summary);out=RUN/'review';out.mkdir(exist_ok=True);save(out/'summary.json',summary)
    er={m:error(p) for m,p in predictions.items()};mean=lambda x:(x*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    change=mean(er['protected_last'])-mean(er['joint_last']);eligible=torch.where(valid.all(-1)&cm)[0]
    baseerr=error(cache['base']);hard=eligible[mean(baseerr)[eligible]>40]
    tg=(er['initial']<=10)&mask;harm=(tg&(er['protected_last']>20)).sum(-1)
    selected=[('最大改善',int(eligible[change[eligible].argmin()])),('最大恶化',int(eligible[change[eligible].argmax()])),
        ('高误差组恢复最好',int(hard[mean(er['protected_last'])[hard].argmin()])),
        ('高误差组仍失败',int(hard[mean(er['protected_last'])[hard].argmax()])),
        ('教师好点损坏最多',int(eligible[harm[eligible].argmax()]))]
    cases=[]
    for ci,(name,i) in enumerate(selected):
        panels=[];r=rows[i]
        for label,p,color in [('teacher',predictions['initial'],(220,130,40)),('joint',predictions['joint_last'],(30,220,255)),
            ('protected',predictions['protected_last'],(50,230,70)),('GT',gt,(230,150,40))]:
            img=cv2.imread(r['image']);uv=s.common.from_json(r['camera']).eye_to_window(p[i].numpy())
            for u,v in EDGES:
                if np.isfinite(uv[[u,v]]).all():cv2.line(img,tuple(np.clip(uv[u],-4000,4000).astype(int)),tuple(np.clip(uv[v],-4000,4000).astype(int)),color,5)
            for pt in uv:
                if np.isfinite(pt).all():cv2.circle(img,tuple(np.clip(pt,-4000,4000).astype(int)),7,color,-1)
            cv2.putText(img,label,(25,55),cv2.FONT_HERSHEY_SIMPLEX,1.4,color,3);panels.append(cv2.resize(img,(480,480)))
        filename=f'case_{ci}.jpg';cv2.imwrite(str(out/filename),np.concatenate(panels,1))
        cases.append(dict(name=name,image=filename,sequence=r['sequence'],clip=r['clip'],frame=r['frame'],
            joint_relative_mm=float(mean(er['joint_last'])[i]),protected_relative_mm=float(mean(er['protected_last'])[i]),
            teacher_good_to_bad_points=int(harm[i]),all_points_visible_state_unknown=True))
    save(out/'cases.json',cases)
    table=''
    for mode in MODES:
        m=reports[mode]['metrics'];d=extras[mode];c=reports[mode]['coherence'];p=d['protection']['initial']
        table+=f"<tr><td>{LABELS[mode]}</td><td>{m['relative_mm']:.3f}</td><td>{m['camera_mm']:.3f}</td><td>{d['fast_amplitude_ratio']['median']*100:.1f}%</td><td>{p['good_to_bad_points']}/{p['good_points']}</td><td>{reports[mode]['hard']['relative_bad_recovered20']}</td><td>{c['spurious_jump_pairs']}</td></tr>"
    curves=''
    for key in ['relative_mm','camera_mm','camera_good_harm_rate','relative_good_harm_rate']:
        values=[h['metrics'][key] for arm in training.values() for h in arm['history']];lo,hi=min(values),max(values);span=max(hi-lo,1e-8);lines=''
        for arm,color in [('frozen','#2563eb'),('joint','#d97706'),('protected','#15803d')]:
            points=' '.join(f"{40+h['step']/1800*500:.1f},{190-(h['metrics'][key]-lo)/span*145:.1f}" for h in training[arm]['history'])
            lines+=f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='3'/>"
        curves+=f"<svg viewBox='0 0 580 230' width='580'><text x='40' y='22'>{key} ({lo:.4f}–{hi:.4f})</text><path d='M40 40V195H545' fill='none' stroke='#aaa'/>{lines}<text x='40' y='220'>600</text><text x='500' y='220'>2400步</text></svg>"
    casehtml=''.join(f"<h3>{html.escape(c['name'])}</h3><p>{html.escape(str(c))}</p><img src='{c['image']}' style='width:100%;max-width:1920px'>" for c in cases)
    selection={arm:training[arm]['done'] for arm in training}
    page=f"""<!doctype html><html lang='zh'><meta charset='utf-8'><title>v48 持续训练与教师保护</title><style>body{{font:16px sans-serif;margin:30px;max-width:1900px;background:#fafafa}}td,th{{padding:10px;border:1px solid #ccc}}table{{border-collapse:collapse}}svg{{max-width:100%}}</style>
<h1>v48：保护已经恢复的好点，再延长视觉联训</h1><p>三组分别续训1800步，累计2400步。联合训练与保护组从完全相同的v47 joint600权重开始；冻结组从自己的matched frozen600权重继续。全部真实RGB、±1.6秒、完整192×1280空间特征；仅最后4层/Norm和DiT训练。重启优化器，v48起保存完整恢复状态。</p>
<p>原保护只覆盖WiLoR本来准确的点。新增教师保护在训练标签上保护初始时序模型已准确的点，允许1mm误差裕量；推理没有教师或GT输入。保持原准入条件，并增加教师好点损坏≤1%和均值非劣检查。速度不变，加速度软/硬限制同步×2。</p>
<p>最终选型：{html.escape(str(selection))}。best.pt可能保留原准入权重，last.pt和best_trained.pt是实际续训诊断，不能偷换为默认模型。默认v42未替换。</p>
<h2>开发集（蓝=冻结，橙=联训，绿=教师保护）</h2>{curves}
<h2>已使用片段复测</h2><p>{N}检测观测、{int(valid[cm].all(-1).sum())}有效中心。10片段、5源序列、2已见过受试者；不是新的独立泛化证据。全部输出冻结后读取评估标签。</p>
<table><tr><th>模型</th><th>相对误差mm</th><th>相机误差mm</th><th>快动作幅度</th><th>教师好点损坏</th><th>高误差组恢复点</th><th>伪跳变</th></tr>{table}</table>
<p>好点≤10mm、损坏&gt;20mm；高误差组按原WiLoR误差&gt;40mm定义，不能当作手指遮挡真值。保护组相对未保护联训的配对95%区间与恢复/损坏计数：{html.escape(str(extras['protected_last']['protection']['joint_last']))}</p>
<h2>自然图像中的改善与失效</h2><p>四列：初始教师、无新增保护联训、保护联训、GT。显示3D结果投影；没有人工遮挡，可见性没有逐指真值。</p>{casehtml}<p><a href='summary.json'>完整统计、置信区间和协议</a></p></html>"""
    (out/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(complete=True,training=selection,metrics={m:reports[m]['metrics'] for m in MODES})),flush=True)

if __name__=='__main__':main()

"""Finite sealed module ablations. Never modify the released v48 implementation."""
import argparse, collections, hashlib, html, json, os, time
from pathlib import Path
import numpy as np
import torch
from hand3d_v8_common import V7, save
from parameter_candidates_v43 import select_sequence
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from complete_hand_tracks_acceleration_v46 import profile_config
from evaluate_motion_thresholds_v45 import serialize

RUN = V7.parent/'module_ablation_v49_20261007'
SOURCE = V7.parent/'online_rgb_iterative_v48/diagnostic_v46'
INPUT = V7.parent/'online_rgb_3d_v47/diagnostic_v46'
NAMES = dict(full='完整v48保护组',wilor='原WiLoR',raw_mean='候选参数均值，无整段优化',raw_sequence='时序选候选，无整段优化',
    mean_solver='参数均值＋整段优化',first_solver='固定第一个候选＋整段优化',independent_solver='逐帧选候选＋整段优化',
    no_rgb_aux='去掉选择和求解中的RGB辅助',uniform_trust='选择和求解均不使用可靠性权重',
    no_soft_motion='去掉预平滑和软时序损失，保留硬运动约束',no_hard_motion='保留软时序，去掉硬约束及其惩罚',
    no_solver_motion='求解器去掉预平滑、软硬运动约束',strict_acc='加速度限制恢复原严格值',
    joint_full='v48同起点无教师保护（未准入诊断）',frozen_full='v48冻结主干续训（起点不同，诊断）',initial_full='联训前教师',
    no_rgb_input='屏蔽网络RGB输入（依赖性诊断）',center_only='网络仅看当前帧（依赖性诊断）',
    pooled_rgb='网络空间特征取均值（依赖性诊断）')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_solver():
    """A separate immutable solver copy exposes exactly the ablated mechanisms."""
    import importlib.util
    p=RUN/'code_snapshot/ablation_solver.py'
    spec=importlib.util.spec_from_file_location('hot3d_ablation_solver',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def prepare():
    RUN.mkdir(parents=True,exist_ok=True);snap=RUN/'code_snapshot';snap.mkdir(exist_ok=True)
    src=Path(__file__).parent/'stability_trajectory_v43.py';text=src.read_text()
    edits=[('loss=data+prior+.08*temporal+100*penalty+cfg[\'rgb_weight\']*rgb',
            "loss=data+prior+cfg.get('temporal_weight',.08)*temporal+cfg.get('motion_penalty_weight',100.)*penalty+cfg['rgb_weight']*rgb"),
           ("for segment in layout['segments']:\n            ix=torch.tensor(segment", "for segment in (layout['segments'] if cfg.get('hard_restoration',True) else []):\n            ix=torch.tensor(segment"),
           ("if not check['passed']:raise RuntimeError(str(check))", "if not check['passed'] and cfg.get('hard_restoration',True):raise RuntimeError(str(check))")]
    for old,new in edits: assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new)
    path=snap/'ablation_solver.py'
    if path.exists(): assert path.read_text()==text
    else:path.write_text(text)
    hashes={}
    # Snapshot dependencies and execute via this directory to preserve historical source.
    for src in Path(__file__).parent.glob('*.py'):
        dest=snap/src.name
        if dest.exists():assert sha(dest)==sha(src),str(src)
        else:dest.write_bytes(src.read_bytes())
        hashes[src.name]=sha(src)
    save(RUN/'code_hashes.json',hashes)
    protocol=dict(variants=list(NAMES),reference='v48 protected/best.pt',
        fixed='Same 3760 detector/track observations, cached 4 DDIM10 candidates, risk/RGB evidence, decoder, 350 solver steps and original speed limits. No default change.',
        scope='Previously evaluated v46 clips, 10 clips/5 sequences/2 known actors. Diagnostic replay, no fresh generalization claim.',
        no_synthetic_occlusion=True,GT_free_inference=True,outputs_sealed_before_labels=True,
        effect='without module minus full; positive error delta supports contribution of module, sequence paired95 CI',
        intervention='Removal of hard constraints is a labeled diagnostic, never admitted or deployed; failures of original limits remain reported.',
        neural_input_tests='Inference masking/pooling is a dependence/OOD test, not a retrained architectural ablation.',
        training_controls='Use existing same-recipe regression/DiT and identical-start joint/protected separately; do not compare incompatible scopes or frozen warmstarts as clean causal evidence.')
    if (RUN/'protocol.json').exists():assert json.loads((RUN/'protocol.json').read_text())==protocol
    else:save(RUN/'protocol.json',protocol)

def seal(name,result,metadata):
    folder=RUN/name;folder.mkdir(exist_ok=True);path=folder/'result.pt'
    tmp=folder/'result.tmp';torch.save(serialize(result),tmp);os.replace(tmp,path)
    save(folder/'sealed.json',dict(complete=True,result_sha256=sha(path),**metadata))

def inference(name,device):
    prepare();dest=RUN/name
    if (dest/'sealed.json').exists():assert sha(dest/'result.pt')==json.loads((dest/'sealed.json').read_text())['result_sha256'];return
    if name in ['full','joint_full','frozen_full','initial_full']:
        mode=dict(full='protected_best',joint_full='joint_last',frozen_full='frozen_last',initial_full='initial')[name]
        src=SOURCE/mode;meta=json.loads((src/'sealed.json').read_text());assert meta['complete'] and sha(src/'result.pt')==meta['result_sha256']
        result=torch.load(src/'result.pt',weights_only=False,map_location='cpu');seal(name,result,dict(source_seal=meta,source_mode=mode));return
    start=time.time();cache={k:v.to(device) for k,v in torch.load(INPUT/'cache.pt',weights_only=False,map_location='cpu').items()}
    rows=json.loads((INPUT/'rows.json').read_text());samples=torch.load(SOURCE/'protected_best/candidates.pt',weights_only=False,map_location='cpu')
    data=torch.load(INPUT/'inputs.pt',weights_only=False,mmap=True);right=torch.load(INPUT/'right.pt',weights_only=False)
    if name=='wilor':seal(name,dict(prediction=cache['base'].cpu()),dict(parameters=False,seconds=time.time()-start));return
    decoder=ParameterTrajectoryDecoder(device)
    inference_probe=False
    if name in ['no_rgb_input','center_only','pooled_rgb']:
        samples=neural_probe(name,device,data,cache,rows);inference_probe=True
    c=dict(cache);config=profile_config('acc_x2');selector_cache=dict(c)
    if name=='no_rgb_aux':
        # Zero auxiliary residual exactly; do not affect neural RGB conditions.
        selector_cache.pop('heat_xy');config['rgb_weight']=0.
    if name=='uniform_trust':c['risk']=torch.zeros_like(c['risk']);selector_cache=dict(c)
    # Make a controlled selector copy, exposing disabled RGB and per-frame path.
    import parameter_candidates_v43 as original_selector, types
    import inspect
    text=inspect.getsource(original_selector.select_sequence)
    text=text.replace("unary+=(.15*pose_weight*rgb).mean(-1)","\n    if use_rgb:unary+=(.15*pose_weight*rgb).mean(-1)")
    text=text.replace("uv=project_fisheye624", "heat_xy=cache.get('heat_xy',torch.zeros(len(rows),20,2,device=device))\n    uv=project_fisheye624")
    text=text.replace("uv-cache['heat_xy'][:,None]","uv-heat_xy[:,None]")
    # Early return only for the deliberately independent candidate variant.
    needle="groups=collections.defaultdict(list)"
    replacement="""if independent:
        chosen=unary.argmin(1);ii=torch.arange(n);jj=torch.tensor(chosen)
        return dict(xyz=samples['xyz'][ii,jj],parameter_state=samples['states'][ii,jj],right=samples['right']),dict(selected_index=chosen.tolist(),independent_frame_argmin=True,uses_gt=False)
    groups=collections.defaultdict(list)"""
    text=text.replace(needle,replacement)
    namespace=dict(original_selector.__dict__,use_rgb=name!='no_rgb_aux',independent=name=='independent_solver');exec(text,namespace)
    obs,selection=namespace['select_sequence'](samples,selector_cache,rows)
    dirty=dict(selector_cache,gt=torch.full_like(c['base'],float('nan')),valid=torch.zeros(len(rows),20,dtype=torch.bool,device=device))
    other,poison=namespace['select_sequence'](samples,dirty,rows)
    assert poison['selected_index']==selection['selected_index'] and torch.equal(other['parameter_state'],obs['parameter_state'])
    if name in ['mean_solver','raw_mean']:
        state=samples['states'].mean(1).to(device)
        from parameter_codec_v31 import ParameterCodec
        xyz=ParameterCodec(device).decode(state[:,None],samples['right'].to(device),shared=False)[:,0]
        obs=dict(parameter_state=state.cpu(),xyz=xyz.cpu(),right=samples['right'])
    if name=='first_solver':obs=dict(parameter_state=samples['states'][:,0],xyz=samples['xyz'][:,0],right=samples['right'])
    obs['parameter_side_outlier']=obs['right']!=right[data['feature_ids'][:,8]].cpu()
    if name.startswith('raw_'):seal(name,dict(prediction=obs['xyz']),dict(parameters=True,solver=False,seconds=time.time()-start));return
    if name in ['no_soft_motion','no_solver_motion']:config.update(smoothing_s=0.,temporal_weight=0.)
    if name in ['no_hard_motion','no_solver_motion']:config.update(motion_penalty_weight=0.,hard_restoration=False)
    if name=='strict_acc':config=profile_config('strict')
    mod=load_solver();result=mod.fit_stable(decoder,c,{k:v.to(device) for k,v in obs.items()},rows,config,
        progress=lambda x:print(json.dumps(dict(variant=name,**x,seconds=time.time()-start)),flush=True))
    assert torch.isfinite(result['prediction']).all()
    world,check=mod.saved_check(decoder,result['parameters'],result['layout'],config)
    assert torch.allclose(world,result['world'],atol=1e-7,rtol=0)
    seal(name,result,dict(seconds=time.time()-start,selector_GT_poison_exact=True,selection=selection,parameter_redecode=True,
        inference_probe_only=inference_probe,original_limit_check=check,diagnostic_not_admitted=True))

def neural_probe(name,device,data,cache,rows):
    from online_parameter_model_v47 import OnlineVisual,load_head
    from parameter_codec_v31 import observation_batch
    dest=RUN/name;dest.mkdir(exist_ok=True)
    path=dest/'candidates.pt'
    if path.exists():return torch.load(path,weights_only=False,map_location='cpu')
    data={k:v.to(device) if torch.is_tensor(v) else v for k,v in data.items()}
    coarse=torch.load(INPUT/'coarse.pt',weights_only=False).to(device);right=torch.load(INPUT/'right.pt',weights_only=False).to(device)
    model,_,_=load_head('dit',device);visual=OnlineVisual(device,False)
    ck=torch.load(V7.parent/'online_rgb_iterative_v48/protected/best.pt',weights_only=False,map_location=device)
    model.load_state_dict(ck['model']);visual.load_tail(ck['visual_tail']);model.eval()
    bankpath=RUN/'protected_native_bank.pt'
    if bankpath.exists():bank=torch.load(bankpath,weights_only=False,map_location=device)
    else:
        pixels=np.load(INPUT/'pixels.npy',mmap_mode='r');bank=torch.zeros(len(right),192,1280,device=device)
        with torch.no_grad():
            for begin in range(1,len(right),128):bank[begin:begin+128]=visual.encode_pixels([pixels[i] for i in range(begin,min(begin+128,len(right)))],device)
        torch.save(bank.cpu(),bankpath)
    del visual;torch.cuda.empty_cache();pieces={k:[] for k in ['states','xyz','right']}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for begin in range(0,len(rows),8):
            ix=torch.arange(begin,min(begin+8,len(rows)),device=device)
            b=observation_batch(data,ix,cache['risk'],coarse,right,preserve_fitted=True);b['rgb_native']=bank[data['feature_ids'][ix]]
            if name=='no_rgb_input':b['rgb_native']=torch.zeros_like(b['rgb_native'])
            if name=='pooled_rgb':b['rgb_native']=b['rgb_native'].mean(2,keepdim=True).expand_as(b['rgb_native'])
            if name=='center_only':
                valid=b['rgb_valid'].clone();valid[:,:8]=False;valid[:,9:]=False;b['rgb_valid']=valid
                available=b['available'].clone();available[:,:8]=False;available[:,9:]=False;b['available']=available
            def generate(bb):
                enc=model.encode(bb);delta=model.sample_trajectory(bb,enc,10,4,202610131+begin)
                state=bb['kinematic_coarse'][None]+delta;side=enc[3]['side_logits'].argmax(-1)
                xyz=torch.stack([model.codec.decode(x,side)[:,8] for x in state])
                return state[:,:,8].float().transpose(0,1),xyz.float().transpose(0,1),side
            states,xyz,side=generate(b)
            if begin==0:
                dirty=generate(dict(b,gt=torch.full((len(ix),17,20,3),float('nan'),device=device),teacher_xyz=torch.full((len(ix),20,3),float('nan'),device=device)))
                assert torch.equal(states,dirty[0]) and torch.equal(xyz,dirty[1])
            for key,value in [('states',states),('xyz',xyz),('right',side)]:pieces[key].append(value.cpu())
            if begin%800==0:print(json.dumps(dict(probe=name,done=min(begin+8,len(rows)),total=len(rows))),flush=True)
    samples={k:torch.cat(v) for k,v in pieces.items()};torch.save(samples,path);return samples

def evaluate():
    from evaluate_joint_kinematic_v30 import report,identities,EDGES
    from evaluate_stability_v42 import quantiles
    from calibrate_hand3d_v8 import paired_ci
    import cv2,spatial_rgb_common as s
    predictions={};seals={}
    for name in NAMES:
        meta=json.loads((RUN/name/'sealed.json').read_text());assert meta['complete'] and meta['result_sha256']==sha(RUN/name/'result.pt')
        seals[name]=meta;predictions[name]=torch.load(RUN/name/'result.pt',weights_only=False,map_location='cpu')['prediction']
    save(RUN/'freeze.json',dict(complete=True,all_outputs_sealed_before_labels=True,sha256={n:m['result_sha256'] for n,m in seals.items()}))
    rows=json.loads((V7.parent/'acceleration_validation_v46/fresh_rows.json').read_text())
    windows=torch.load(V7.parent/'acceleration_validation_v46/fresh_data.pt',weights_only=False,mmap=True)['rows'];mapping={r['row_index']:i for i,r in enumerate(windows)}
    rows=[dict(r,window_index=mapping.get(i)) for i,r in enumerate(rows)];gt=torch.zeros(len(rows),20,3);valid=torch.zeros(len(rows),20,dtype=torch.bool)
    for i,r in enumerate(rows):
        if r['matched']:gt[i]=torch.tensor(r['gt']);valid[i]=torch.isfinite(gt[i]).all(-1)
    cache=torch.load(INPUT/'cache.pt',weights_only=False,map_location='cpu');labels=dict(gt=gt,valid=valid);sides=identities(rows,labels)
    cm=torch.tensor([r['window_index'] is not None for r in rows]);cr=[r for r in rows if r['window_index'] is not None];mask=valid.clone();mask[:,5]=False
    error=lambda p:((p-p[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[(r['sequence'],r['clip'],r['track_id'])].append(i)
    pairs=[]
    for ix in groups.values():
        ix.sort(key=lambda i:rows[i]['timestamp_ns']);pairs.extend((a,b) for a,b in zip(ix,ix[1:]) if rows[b]['frame']==rows[a]['frame']+1 and sides[a] is not None and sides[a]==sides[b])
    a,b=map(torch.tensor,zip(*pairs));pm=valid[a]&valid[b]&valid[a,5,None]&valid[b,5,None];pm[:,5]=False
    world=lambda p:torch.einsum('njc,nkc->njk',p,cache['rotation'])+cache['translation'][:,None]
    gw=world(gt);gpose=gw-gw[:,5:6];gd=gpose[b]-gpose[a];gm=gd.norm(dim=-1)*1000;fast=pm&(gm>10)
    base=predictions['full'];be=error(base);good=mask&(be<=10);bad=mask&(be>20);entries={};cases=[]
    review=RUN/'review';review.mkdir(exist_ok=True)
    mean=lambda e:(e*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    for name,p in predictions.items():
        r=report(p,cache,labels,rows,sides,cache['base']);e=error(p);pw=world(p);pose=pw-pw[:,5:6];d=pose[b]-pose[a]
        entries[name]=dict(report=r,paired_vs_full=paired_ci(p[cm],base[cm],gt[cm],valid[cm],cr),
            full_good_points=int(good.sum()),full_good_to_bad=int((good&(e>20)).sum()),full_bad_points=int(bad.sum()),full_bad_to_good=int((bad&(e<=20)).sum()),
            fast_point_pairs=int(fast.sum()),fast_amplitude_ratio=quantiles((d.norm(dim=-1)*1000/gm.clamp_min(1e-6))[fast]),
            fast_directional_retention=quantiles(((d*gd).sum(-1)/gd.square().sum(-1).clamp_min(1e-10))[fast]),seal=seals[name])
        if name!='full':
            delta=mean(e)-mean(be);ix=torch.where(cm&valid.all(-1))[0]
            for kind,index in [('removed_worse',int(ix[delta[ix].argmax()])),('removed_better',int(ix[delta[ix].argmin()]))]:
                rr=rows[index];panels=[]
                for label,q,color in [('full',base,(50,220,70)),(name,p,(30,150,255)),('GT',gt,(230,180,30))]:
                    img=cv2.imread(rr['image']);uv=s.common.from_json(rr['camera']).eye_to_window(q[index].numpy())
                    for u,v in EDGES:
                        if np.isfinite(uv[[u,v]]).all():cv2.line(img,tuple(np.clip(uv[u],-4000,4000).astype(int)),tuple(np.clip(uv[v],-4000,4000).astype(int)),color,4)
                    cv2.putText(img,label,(20,45),cv2.FONT_HERSHEY_SIMPLEX,.9,color,2);panels.append(cv2.resize(img,(480,480)))
                fname=f'{name}_{kind}.jpg';cv2.imwrite(str(review/fname),np.concatenate(panels,1))
                cases.append(dict(variant=name,kind=kind,image=fname,sequence=rr['sequence'],frame=rr['frame'],full_mm=float(mean(be)[index]),ablated_mm=float(mean(e)[index]),finger_visibility_not_GT=True))
    # Historical matched training results retain their own scope, never merge rows.
    historical={}
    for kind in ['regression_side_consistent','dit_side_consistent']:
        path=V7.parent/'matched_stability_v43'/kind/'summary.json'
        if path.exists():historical[kind]=dict(source=str(path),sha256=sha(path),result=json.loads(path.read_text()))
    summary=dict(complete=True,entries=entries,historical_matched_v43=historical,protocol=json.loads((RUN/'protocol.json').read_text()),
        observations=len(rows),matched_observations=int(valid.all(-1).sum()),centers=int(cm.sum()),matched_centers=int(valid[cm].all(-1).sum()),
        sequences=len({r['sequence'] for r in rows}),no_default_changed=True,training_ablation_complete=False)
    save(RUN/'summary.json',summary);save(review/'summary.json',summary);save(review/'cases.json',cases)
    table=''
    fm=entries['full']['report']['metrics']
    for name,x in entries.items():
        m=x['report']['metrics'];c=x['report']['coherence'];ci=x['paired_vs_full']['relative']['ci95_delta_mm'];hard=x['report']['hard'];s=x['seal'].get('original_limit_check',{})
        table+=f"<tr><td>{NAMES[name]}</td><td>{m['relative_mm']:.3f}</td><td>{m['camera_mm']:.3f}</td><td>{m['relative_mm']-fm['relative_mm']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]</td><td>{hard['relative_bad_recovered20']}</td><td>{m['relative_good_harmed20']}/{m['relative_good_points']}</td><td>{c['spurious_jump_pairs']}</td><td>{c['bone_extreme_frames']}</td><td>{x['fast_amplitude_ratio']['median']*100:.1f}%</td><td>{s.get('passed','—')}</td></tr>"
    panels=''.join(f"<details><summary>{NAMES[c['variant']]} · {c['kind']} · {c['full_mm']:.1f} → {c['ablated_mm']:.1f} mm</summary><img src='{c['image']}' width='1440' style='max-width:100%'></details>" for c in cases)
    page=f"""<!doctype html><html lang='zh'><meta charset='utf-8'><title>HOT3D v49 模块消融</title><style>body{{font:16px sans-serif;margin:28px;background:#fafafa}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ccc}}details{{margin:15px 0}}</style><h1>模块消融：逐项去掉，观察贡献与代价</h1>
<p>固定 v48 protected/best，原生RGB、4候选/10去噪步、350优化步；原检测、跟踪、骨架、速度上限不变。去硬约束是明确标记的诊断，不能部署。屏蔽RGB/时间/空间是分布外依赖测试，不能当作对应重训后的收益。</p>
<p>{len(rows)}检测观测、{int(valid[cm].all(-1).sum())}匹配中心、5序列、10已使用片段、2已见演员。重复诊断，非新泛化。所有输出封存后读取标签。中心误差/CI与全观测快动作统计分别报告。高误差组并非真实逐指遮挡标签。</p>
<table><tr><th>消融</th><th>相对mm</th><th>相机mm</th><th>相对完整版本差值及序列配对95%CI</th><th>高误差组恢复点</th><th>原WiLoR相对好点损坏</th><th>伪跳变对</th><th>骨长异常帧</th><th>快动作幅度</th><th>原运动限制检查</th></tr>{table}</table>
<p>差值正且CI完全大于0：在这些片段上支持该模块改善误差；CI跨0则证据不足。误差降低也需同时看损坏、运动和骨长，不能自动认定可替换默认。</p><h2>自然案例：完整结果 / 消融结果 / GT</h2>{panels}<p><a href='summary.json'>完整统计及历史匹配DiT/回归对照</a></p></html>"""
    (review/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(complete=True,metrics={n:x['report']['metrics']['relative_mm'] for n,x in entries.items()})),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=list(NAMES));p.add_argument('--prepare',action='store_true');p.add_argument('--evaluate',action='store_true');p.add_argument('--device',default='cuda:3');a=p.parse_args();torch.set_num_threads(4)
    if a.prepare:prepare()
    elif a.evaluate:evaluate()
    else:inference(a.variant,a.device)

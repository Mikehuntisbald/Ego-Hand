"""Score all sealed native-head predictions; paired clusters and natural cases."""
import hashlib,html,json,sys,tarfile,time
from pathlib import Path
import cv2,numpy as np,torch
from scipy.optimize import linear_sum_assignment
R=Path('/mnt/why/native_mano_multiseq_v3_20261019');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019')
sys.path.insert(0,str(PUBLIC/'src'));sys.path.insert(0,str(PUBLIC/'vendor/Ego-Hand/research'))
import wilor_eval_common as common
from geometry import camera,transform,load_frames
from hand_tracking_toolkit.dataset import decode_hand_pose
from hand_tracking_toolkit.hand_models.umetrack_hand_model import from_json,forward_kinematics
MAPPING=common.MAPPING;WRIST=5
EDGES=[(5,6),(6,7),(7,0),(5,8),(8,9),(9,10),(10,1),(5,11),(11,12),(12,13),(13,2),(5,14),(14,15),(15,16),(16,3),(5,17),(17,18),(18,19),(19,4)]
ARMS=['WiLoR','regression','linear_dit','cosine_dit','cosine_protected','cosine_shared','cosine_protected_displacement','cosine_RGB_off','rollout_dit','rollout_protected','rollout_dit_displacement']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(n=int(x.size),mean=float(x.mean()),RMSE=float(np.sqrt(np.mean(x*x))),P95=float(np.quantile(x,.95)),max=float(x.max())) if x.size else dict(n=0)
def iou(a,b):
    p=np.maximum(a[:2],b[:2]);q=np.minimum(a[2:],b[2:]);intersection=np.maximum(q-p,0).prod()
    return intersection/max(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-intersection,1e-9)
def protection(base,error,valid):
    good=valid&(base<=10);hard=valid&(base>20)
    return dict(valid_points=int(valid.sum()),base_good10=int(good.sum()),good_harmed_over1mm=int((good&(error-base>1)).sum()),good_to_over20=int((good&(error>20)).sum()),base_hard20=int(hard.sum()),recovered_to20=int((hard&(error<=20)).sum()),improved_over1mm=int((valid&(base-error>1)).sum()),worsened_over1mm=int((valid&(error-base>1)).sum()))
def clustered_ci(values,clusters):
    groups=sorted(set(clusters));sums=np.array([sum(v['sum'] for v,c in zip(values,clusters) if c==g) for g in groups]);counts=np.array([sum(v['n'] for v,c in zip(values,clusters) if c==g) for g in groups])
    rng=np.random.default_rng(190019);draw=rng.integers(len(groups),size=(10000,len(groups)));means=sums[draw].sum(1)/np.maximum(counts[draw].sum(1),1)
    return dict(clusters=len(groups),point_weighted_delta_mm=float(sums.sum()/max(counts.sum(),1)),CI95_mm=np.quantile(means,[.025,.975]).tolist(),bootstrap_draws=10000,units='candidate minus WiLoR mm; negative is better',limitation='only 3 subjects / 6 recordings; exploratory clustered uncertainty')

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    protocol=json.loads((R/'protocol.json').read_text());frozen=R/'receipts/predictions_frozen.json';seal=json.loads(frozen.read_text());assert seal['complete'] and sha(R/'protocol.json')==seal['protocol_sha256']
    for name,h in seal['seals'].items():
        path=R/name;assert sha(path)==h
        for file,v in json.loads(path.read_text())['files'].items():assert sha(path.parent/file)==v
    extra_path=R/'receipts/rollout_predictions_frozen.json';extra=json.loads(extra_path.read_text());assert extra['complete']
    assert extra['original_prediction_manifest_sha256']==sha(frozen)
    for name,h in extra['seals'].items():
        path=R/name;assert sha(path)==h
        for file,v in json.loads(path.read_text())['files'].items():assert sha(path.parent/file)==v
    out=R/'evaluation';out.mkdir(exist_ok=False);(out/'images').mkdir()
    (out/'label_open_receipt.json').write_text(json.dumps(dict(open_unix=time.time(),predictions_frozen_sha256=sha(frozen),new_predictions_complete_before_labels=True,additional_rollout_prediction_manifest_sha256=sha(extra_path),historical_project_exposure=protocol['historical_exposure']),indent=2))
    all_errors={a:[] for a in ARMS};all_relative={a:[] for a in ARMS};all_valid=[];all_good_relative=[];per_clip=[];cards=[]
    motion={a:dict(velocity=[],acceleration=[],relative_velocity=[],relative_acceleration=[],wrist_velocity=[],fast_pred=[],fast_GT=[],bone_cv=[]) for a in ARMS}
    count=dict(frames=0,GT_hand_frames=0,predictions=0,matched_hand_frames=0,unmatched_predictions=0,side_correct=0,side_total=0,matched_track_GT_side_changes=0)
    delta_values={a:dict(camera=[],relative=[]) for a in ARMS if a!='WiLoR'};sequences=[];subjects=[];errors_saved={}
    for clip in protocol['clips']:
        view=Path(clip['view']);rows=json.loads((view/'outputs/wilor_clip2102/observations.json').read_text())['frames'];frames=load_frames(view/'data/clip-002102');N=len(rows)
        predictions={a:np.load(view/('predictions_rollout' if a.startswith('rollout_') else 'predictions')/f'{a}.npz')['joints_camera21'][:,MAPPING] for a in ARMS}
        assert all(np.isfinite(v).all() for v in predictions.values())
        GT=np.full((N,20,3),np.nan);world_GT=np.full_like(GT,np.nan);GTside=np.full(N,-1);matched=np.zeros(N,bool);poses=[]
        with tarfile.open(clip['archive']) as tar:
            shape=from_json(json.load(tar.extractfile('__hand_shapes.json__'))['umetrack'])
            for fi,frame in enumerate(frames):
                pose=transform(json.load(tar.extractfile(f'{frame["id"]}.cameras.json'))['214-1']['T_world_from_camera']);poses.append(pose)
                labels=json.load(tar.extractfile(f'{frame["id"]}.hands.json'));decoded=decode_hand_pose(labels);targets=[]
                for side,hand in decoded.items():
                    if hand.umetrack is None:continue
                    box=np.array(labels[side.value]['boxes_amodal']['214-1']);box[[0,2]]=np.clip(box[[0,2]],0,1408);box[[1,3]]=np.clip(box[[1,3]],0,1408)
                    if np.maximum(box[2:]-box[:2],0).prod()<64:continue
                    with torch.no_grad():xyz=forward_kinematics(hand.umetrack,shape,requires_mesh=False)[0].numpy()
                    targets.append(dict(side=side.value,world=xyz,box=box))
                count['GT_hand_frames']+=len(targets);ids=[i for i,r in enumerate(rows) if r['frame_index']==fi]
                if not ids or not targets:continue
                overlap=np.array([[iou(np.array(rows[i]['box_xyxy']),g['box']) for g in targets] for i in ids]);a,b=linear_sum_assignment(1-overlap)
                for x,y in zip(a,b):
                    if overlap[x,y]<.1:continue
                    i=ids[x];target=targets[y];GT[i]=(target['world']-pose[:3,3])@pose[:3,:3];world_GT[i]=target['world'];matched[i]=True;GTside[i]=int(target['side']=='right')
                    count['side_total']+=1;count['side_correct']+=int(rows[i]['right']==GTside[i])
        poses=np.array(poses);valid=matched[:,None]&np.isfinite(GT).all(-1);relative_valid=valid.copy();relative_valid[:,WRIST]=False
        errors={a:np.linalg.norm(x-GT,axis=-1)*1000 for a,x in predictions.items()}
        rel=lambda x:x-x[:,WRIST:WRIST+1]
        relative={a:np.linalg.norm(rel(x)-rel(GT),axis=-1)*1000 for a,x in predictions.items()}
        report=dict(clip=clip['clip'],sequence=clip['sequence'],subject=clip['subject'],predictions=N,matched=int(matched.sum()),metrics={})
        for arm in ARMS:
            all_errors[arm].append(errors[arm]);all_relative[arm].append(relative[arm]);report['metrics'][arm]=dict(camera_mm=stats(errors[arm][valid]),relative_without_wrist_mm=stats(relative[arm][relative_valid]),wrist_mm=stats(errors[arm][matched,WRIST]))
            if arm!='WiLoR':
                for key,array,ref,mask in [('camera',errors[arm],errors['WiLoR'],valid),('relative',relative[arm],relative['WiLoR'],relative_valid)]:delta_values[arm][key].append(dict(sum=float((array-ref)[mask].sum()),n=int(mask.sum())))
        all_valid.append(valid);all_good_relative.append(relative_valid);sequences.append(clip['sequence']);subjects.append(clip['subject'])
        count['frames']+=len(frames);count['predictions']+=N;count['matched_hand_frames']+=int(matched.sum());count['unmatched_predictions']+=int((~matched).sum())
        frame_indices=np.array([r['frame_index'] for r in rows]);rotation=poses[frame_indices,:3,:3];translation=poses[frame_indices,:3,3]
        for tid in sorted({r['track_id'] for r in rows}):
            ids=np.array([i for i,r in enumerate(rows) if r['track_id']==tid and matched[i]]);ids=ids[np.argsort([rows[i]['timestamp_ns'] for i in ids])]
            if len(ids)<3:continue
            stamp=np.array([rows[i]['timestamp_ns'] for i in ids],np.int64);dt=np.diff(stamp)/1e9;pair=(dt>0)&(dt<.1);triple=pair[1:]&pair[:-1]
            count['matched_track_GT_side_changes']+=int((GTside[ids[1:]][pair]!=GTside[ids[:-1]][pair]).sum())
            gt=world_GT[ids];vg=np.diff(gt,axis=0)/(60*dt[:,None,None]);ag=np.diff(vg,axis=0)/(60*(dt[1:]+dt[:-1])[:,None,None]/2)
            rgt=rel(gt);rvg=np.diff(rgt,axis=0)/(60*dt[:,None,None]);rag=np.diff(rvg,axis=0)/(60*(dt[1:]+dt[:-1])[:,None,None]/2)
            gt_speed=np.linalg.norm(vg,axis=-1);fast=(gt_speed>=np.quantile(gt_speed[pair],.9))&pair[:,None]
            for arm,cam in predictions.items():
                p=np.einsum('nij,nkj->nki',rotation[ids],cam[ids])+translation[ids,None];vp=np.diff(p,axis=0)/(60*dt[:,None,None]);ap=np.diff(vp,axis=0)/(60*(dt[1:]+dt[:-1])[:,None,None]/2)
                rp=rel(p);rvp=np.diff(rp,axis=0)/(60*dt[:,None,None]);rap=np.diff(rvp,axis=0)/(60*(dt[1:]+dt[:-1])[:,None,None]/2);m=motion[arm]
                m['velocity'].extend((np.linalg.norm(vp-vg,axis=-1)*1000)[pair].ravel());m['acceleration'].extend((np.linalg.norm(ap-ag,axis=-1)*1000)[triple].ravel())
                m['relative_velocity'].extend((np.linalg.norm(rvp-rvg,axis=-1)*1000)[pair,:][:,np.arange(20)!=WRIST].ravel());m['relative_acceleration'].extend((np.linalg.norm(rap-rag,axis=-1)*1000)[triple,:][:,np.arange(20)!=WRIST].ravel())
                m['wrist_velocity'].extend((np.linalg.norm(vp-vg,axis=-1)*1000)[pair,WRIST]);m['fast_pred'].extend(np.linalg.norm(vp,axis=-1)[fast]);m['fast_GT'].extend(gt_speed[fast])
                lengths=np.stack([np.linalg.norm(cam[ids,a]-cam[ids,b],axis=-1) for a,b in EDGES],-1);m['bone_cv'].extend(lengths.std(0)/np.maximum(lengths.mean(0),1e-6))
        candidate='rollout_dit_displacement';per=np.nanmean(relative[candidate]-relative['WiLoR'],axis=1);hard=np.nanmean(relative[candidate],axis=1)
        chosen=[]
        for category,order in [('改善',np.argsort(per)),('损伤',np.argsort(per)[::-1]),('仍困难',np.argsort(hard)[::-1])]:
            for i in order:
                if matched[i] and i not in chosen:chosen.append(int(i));break
            else:continue
            row=rows[i];fi=row['frame_index'];native=camera(frames[fi]['cameras']['214-1']['calibration']);photo=cv2.imread(row['image']);box=np.array(row['box_xyxy']);center=(box[:2]+box[2:])/2;size=max(box[2:]-box[:2])*1.35
            x0=max(0,int(center[0]-size/2));x1=min(photo.shape[1],int(center[0]+size/2));y0=max(0,int(center[1]-size/2));y1=min(photo.shape[0],int(center[1]+size/2));panels=[]
            for name,arm,color in [('RGB',None,None),('WiLoR','WiLoR',(255,150,0)),('native DiT guard',candidate,(180,0,255))]:
                image=photo.copy()
                if arm:
                    uv=native.eye_to_window(predictions[arm][i]);truth=native.eye_to_window(GT[i])
                    for a,b in EDGES:cv2.line(image,tuple(np.rint(uv[a]).astype(int)),tuple(np.rint(uv[b]).astype(int)),color,2,cv2.LINE_AA)
                    for pt in truth:cv2.circle(image,tuple(np.rint(pt).astype(int)),3,(0,255,0),-1)
                crop=image[y0:y1,x0:x1];scale=480/max(crop.shape[:2]);crop=cv2.resize(crop,None,fx=scale,fy=scale);panel=np.full((540,500,3),28,np.uint8);panel[50:50+crop.shape[0],:crop.shape[1]]=crop;cv2.putText(panel,name,(10,32),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);panels.append(panel)
            path=out/'images'/f'{clip["clip"]}_{category}_{i}.jpg';cv2.imwrite(str(path),np.concatenate(panels,1));caption=f'{clip["sequence"]} clip {clip["clip"]} frame {fi} {category}：腕对齐 {np.nanmean(relative["WiLoR"][i]):.1f} → {np.nanmean(relative[candidate][i]):.1f} mm'
            cards.append(f'<figure><img src="images/{path.name}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        per_clip.append(report);errors_saved[f'clip{clip["clip"]}_GT_camera20']=GT;errors_saved[f'clip{clip["clip"]}_valid']=valid
        print(json.dumps(dict(clip=clip['clip'],matched=int(matched.sum()),relative={a:report['metrics'][a]['relative_without_wrist_mm']['mean'] for a in ARMS})),flush=True)
    valid=np.concatenate(all_valid);relative_valid=np.concatenate(all_good_relative);errors={a:np.concatenate(v) for a,v in all_errors.items()};relative={a:np.concatenate(v) for a,v in all_relative.items()}
    metrics={};guards={};temporal={};uncertainty={}
    for arm in ARMS:
        metrics[arm]=dict(camera_mm=stats(errors[arm][valid]),relative_without_wrist_mm=stats(relative[arm][relative_valid]))
        guards[arm]=dict(camera=protection(errors['WiLoR'],errors[arm],valid),relative_without_wrist=protection(relative['WiLoR'],relative[arm],relative_valid))
        m=motion[arm];temporal[arm]={k+'_mm':stats(m[k]) for k in ['velocity','acceleration','relative_velocity','relative_acceleration']};temporal[arm].update(wrist_error_jumps_over10mm=int((np.array(m['wrist_velocity'])>10).sum()),wrist_pairs=len(m['wrist_velocity']),fast_motion_amplitude_ratio=float(np.mean(m['fast_pred'])/max(np.mean(m['fast_GT']),1e-12)),predicted_bone_length_CV_mean=float(np.mean(m['bone_cv'])))
        if arm!='WiLoR':uncertainty[arm]={key:dict(recording_cluster=clustered_ci(value,sequences),subject_cluster=clustered_ci(value,subjects)) for key,value in delta_values[arm].items()}
        errors_saved[f'{arm}_error_camera_mm']=errors[arm];errors_saved[f'{arm}_error_relative_mm']=relative[arm]
    # P0003 influenced checkpoint selection on a different recording. Report
    # the two completely unseen native-head subjects separately as well.
    strict_ids=[i for i,c in enumerate(protocol['clips']) if c['subject'] in ['P0010','P0015']]
    strict_valid=np.concatenate([all_valid[i] for i in strict_ids]);strict_relative_valid=np.concatenate([all_good_relative[i] for i in strict_ids])
    strict_errors={a:np.concatenate([all_errors[a][i] for i in strict_ids]) for a in ARMS}
    strict_relative={a:np.concatenate([all_relative[a][i] for i in strict_ids]) for a in ARMS}
    strict={}
    for arm in ARMS:
        strict[arm]=dict(camera_mm=stats(strict_errors[arm][strict_valid]),relative_without_wrist_mm=stats(strict_relative[arm][strict_relative_valid]),
            relative_preservation=protection(strict_relative['WiLoR'],strict_relative[arm],strict_relative_valid))
        if arm!='WiLoR':
            strict[arm]['relative_paired_uncertainty']={kind:clustered_ci([delta_values[arm]['relative'][i] for i in strict_ids],[cluster[i] for i in strict_ids]) for kind,cluster in [('recording',sequences),('subject',subjects)]}
    summary=dict(complete=True,counts=count,metrics=metrics,preservation=guards,temporal=temporal,paired_uncertainty=uncertainty,per_clip=per_clip,
        strictly_unseen_native_subjects=dict(subjects=['P0010','P0015'],recordings=len(strict_ids),metrics=strict),evaluator_sha256=sha(Path(__file__)),
        scope=protocol['scope'],historical_exposure=protocol['historical_exposure'],reference_camera_trajectory_used=True,hand_GT_inference=False,predictions_sealed_before_label_read=True,
        full_end_to_end_evaluation=False,production_admitted=False,GT_geometry='WiLoR MANO21 mapped to HOT3D UmeTrack20; cross-model landmark differences remain',relative_excludes_zero_wrist=True,
        motion='predicted track/time only; world motion using supplied camera; subtract GT motion; do not hide GT side switches or predicted jumps; no crossing >.1s',
        limitations='six recordings but only three held-out native-training subjects; P0003 participated in native model selection on other recordings; no independent full-pipeline/production claim')
    (out/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));np.savez_compressed(out/'labels_and_errors.npz',**errors_saved)
    rows=[]
    for arm,m in metrics.items():
        p=guards[arm]['relative_without_wrist'];rows.append(f'<tr><td>{arm}</td><td>{m["camera_mm"]["mean"]:.2f}</td><td>{m["relative_without_wrist_mm"]["mean"]:.2f}</td><td>{m["relative_without_wrist_mm"]["P95"]:.2f}</td><td>{p["recovered_to20"]}/{p["base_hard20"]}</td><td>{p["good_harmed_over1mm"]}/{p["base_good10"]}</td><td>{p["good_to_over20"]}</td></tr>')
    document=f'''<!doctype html><meta charset="utf-8"><title>原生MANO DiT：六序列与实际采样训练对照</title><style>body{{font:16px system-ui;max-width:1500px;margin:32px auto;background:#14191e;color:#eee;padding:0 24px}}td,th{{border:1px solid #64717a;padding:10px}}table{{border-collapse:collapse;margin:24px 0}}img{{width:100%}}pre{{white-space:pre-wrap}}</style><h1>原生MANO DiT：三人员、六段录制与实际采样训练对照</h1><p>冻结v1/v2/v4权重、采样和阈值；全部预测封存后评分，没有在本轮标签上训练。900帧；{count['predictions']}个预测、{count['matched_hand_frames']}个匹配手帧。P0003/P0010/P0015未参与原生头训练，P0003另一录制用于过权重选择；这些公共验证录制在历史项目中已存在，不能称整个项目的新鲜测试。</p><p>本轮只判断补全模块：各方案共用已提供的相机轨迹、RF框和预测轨迹，不含预测headpose、物理深度和BA。全部使用自然RGB，无人工遮挡。腕对齐统计排除恒为零的腕点；不能与旧版含腕均值直接混比。</p><table><tr><th>方案</th><th>相机均值mm</th><th>腕对齐均值mm</th><th>腕对齐P95</th><th>困难点恢复</th><th>原准确点损伤&gt;1mm</th><th>原准确点变&gt;20mm</th></tr>{''.join(rows)}</table><p>RGB_off仅固定权重敏感性诊断；共享噪声仅采样相关性对照。保护版只在MANO参数中插值，不做XYZ点裁剪；没有GT选候选。完整配对序列/人员置信区间、覆盖和时序统计在下方。六序列/三人员的区间仍属探索性证据。另在完整统计中单列P0010/P0015四录制：两人均未参与原生头训练或权重选择，仍只有两个人，不能把大量帧当作大量独立人员。</p><h2>自然改善、损伤与未解决案例</h2><p>左RGB，中WiLoR，右实际采样训练DiT＋参数保护。绿色为GT20，只在封存后用于评分和叠图。</p>{''.join(cards)}<details><summary>完整统计与边界</summary><pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre></details>'''
    (out/'report.html').write_text(document);print(json.dumps(dict(complete=True,counts=count,metrics=metrics,preservation=guards['rollout_dit_displacement'],uncertainty=uncertainty['rollout_dit_displacement']),indent=2))

if __name__=='__main__':main()

"""Open replay labels only after every complete-model output is sealed."""
import collections,html,json
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from prepare_fair_v54 import ROOT,E,save,sha
from infer_fair_v54 import MODES
import wilor_eval_common as common
from hand3d_data_v7 import camera_pose
from build_surgical_pose_cache_v51 import MAPPING
from evaluate_joint_kinematic_v30 import EDGES

def iou(a,b):
    a=np.asarray(a).reshape(-1,4);b=np.asarray(b).reshape(-1,4);lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:]);inter=np.maximum(hi-lo,0).prod(-1)
    return inter/np.maximum(np.maximum(a[:,2:]-a[:,:2],0).prod(-1)[:,None]+np.maximum(b[:,2:]-b[:,:2],0).prod(-1)[None]-inter,1e-9)
def ci(rows,mode,ref,field='success'):
    groups=sorted({r['group'] for r in rows});values=[]
    for group in groups:
        rs=[r for r in rows if r['group']==group];num=den=0
        for r in rs:
            valid=r['valid'];a=r['modes'][mode][field];b=r['modes'][ref][field]
            if field=='success':num+=int(a.sum())-int(b.sum());den+=int(valid.sum())
            else:
                both=valid&np.isfinite(a)&np.isfinite(b);num+=float((a[both]-b[both]).sum());den+=int(both.sum())
        values.append([num,den])
    v=np.asarray(values,dtype=float);rng=np.random.default_rng(2026100754);ix=rng.integers(0,len(v),(10000,len(v)));s=v[ix].sum(1);delta=s[:,0]/np.maximum(s[:,1],1)
    return dict(groups=len(groups),paired_pooled_delta_CI95=np.quantile(delta,[.025,.975]).tolist(),pooled_delta=float(v[:,0].sum()/max(v[:,1].sum(),1)),denominator=int(v[:,1].sum()),bootstrap='10000 whole-source paired resamples')
def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());clips=json.loads((ROOT/'replay_clips.json').read_text());tags=[x['id'] for x in clips]+['glove']+protocol['qualitative_tags'];outputs={};seals={}
    for mode in MODES:
        outputs[mode]={}
        for tag in tags:
            folder=ROOT/'inference'/mode/tag;seal=json.loads((folder/'freeze.json').read_text());assert seal['complete'] and seal['GT_free'] and sha(folder/'prediction.json')==seal['prediction_sha256']
            outputs[mode][tag]=json.loads((folder/'prediction.json').read_text());seals[f'{mode}/{tag}']=seal['prediction_sha256']
    save(ROOT/'evaluation_freeze.json',dict(all_inference_outputs_sealed_before_test_labels=True,sha256=seals,protocol_sha256=sha(ROOT/'protocol.json')))
    # Only now read the existing native annotations and surgical point labels.
    labels={};annotations={}
    for clip in clips:
        frames=json.loads(Path(clip['input']).read_text())['frames']
        for f in frames:
            rel=Path(f['image']).relative_to(E.parent/'export/images');sp,seq,clip_dir=rel.parts[:3];p=E.parent/'export/annotations'/sp/seq/(clip_dir+'.jsonl')
            if p not in annotations:annotations[p]=[json.loads(x) for x in p.read_text().splitlines()]
            index=int(Path(f['image']).stem);hands=annotations[p][index]['hands'];gt=[];uvs=[];valid=[];boxes=[];sides=[]
            for h in hands:
                x=np.asarray(h['xyz_camera_m'],float);v=np.isfinite(x).all(-1);uv=common.from_json(f['camera']).eye_to_window(np.nan_to_num(x));finite=v&np.isfinite(uv).all(-1)
                if finite.sum()<2:continue
                box=np.r_[uv[finite].min(0),uv[finite].max(0)];gt.append(x);uvs.append(uv);valid.append(v);boxes.append(box);sides.append(h['side'])
            labels[(clip['id'],f['image'])]=dict(frame=f,group=clip['sequence'],tag=clip['id'],xyz=np.asarray(gt).reshape(-1,20,3),uv=np.asarray(uvs).reshape(-1,20,2),valid=np.asarray(valid).reshape(-1,20),boxes=np.asarray(boxes).reshape(-1,4),sides=sides,domain='native')
    surgical={r['id']:r for r in map(json.loads,(E.parent/'domain_data_v51/surgical_hands/selected_records.jsonl').read_text().splitlines())}
    for f in json.loads((ROOT/'glove_replay_rgb.json').read_text())['frames']:
        row=surgical[f['id']];kp=np.array([np.asarray(h['keypoints']).reshape(21,3)[MAPPING] for h in row['hands']]);scale=f['image_transform']['scale'];offset=np.asarray(f['image_transform']['offset'])
        labels[('glove',f['image'])]=dict(frame=f,group=f['group'],tag='glove',uv=kp[:,:,:2]*scale+offset,valid=kp[:,:,2]>0,visible=kp[:,:,2]==2,boxes=np.asarray([h['bbox'] for h in row['hands']])*scale+np.tile(offset,2),domain='glove')
    rows=[];tracks_stats={m:dict(tracks=0,short_tracks=0,constraint_failures=0,raw_xyz_fallback=0,max_bone_CV=0.) for m in MODES};prediction_frames={m:{} for m in MODES}
    for mode in MODES:
        for tag,res in outputs[mode].items():
            tracks_stats[mode]['constraint_failures']+=int(not res['constraint_checks_passed']);tracks_stats[mode]['raw_xyz_fallback']+=res.get('raw_xyz_fallback_frames',0)
            for tr in res['tracks']:
                tracks_stats[mode]['tracks']+=1;tracks_stats[mode]['short_tracks']+=int(len(tr['frames'])<=5)
                xs=np.array([f['candidate_xyz_camera_m'] for f in tr['frames']])
                lengths=np.array([np.linalg.norm(xs[:,a]-xs[:,b],axis=-1) for a,b in EDGES]);cv=lengths.std(1)/np.maximum(lengths.mean(1),1e-8)
                tracks_stats[mode]['max_bone_CV']=max(tracks_stats[mode]['max_bone_CV'],float(cv.max()))
                for f in tr['frames']:prediction_frames[mode].setdefault((tag,f['image']),[]).append(dict(f,track_id=tr['id']))
    for key,label in labels.items():
        valid=label['valid'].copy()
        if label['domain']=='native':valid&=np.isfinite(label['xyz'][:,5]).all(-1)[:,None];valid[:,5]=False
        row=dict(label,valid=valid,modes={})
        for mode in MODES:
            candidates=prediction_frames[mode].get(key,[]);boxes=np.asarray([f['box_xyxy'] for f in candidates]).reshape(-1,4);ov=iou(label['boxes'],boxes)
            error=np.full(valid.shape,np.inf);camera=np.full(valid.shape,np.inf);xyz=np.full((*valid.shape,3),np.nan);matched=np.zeros(len(valid),bool);ids=[None]*len(valid);uv=np.full((*valid.shape,2),np.nan)
            if ov.size:
                a,b=linear_sum_assignment(-((ov>=.5).astype(float)+ov*.001))
                for gi,pj in zip(a,b):
                    if ov[gi,pj]<.5:continue
                    pred=np.asarray(candidates[pj]['candidate_xyz_camera_m']);matched[gi]=True;xyz[gi]=pred;ids[gi]=candidates[pj]['track_id'];uv[gi]=common.from_json(label['frame']['camera']).eye_to_window(pred)
                    if label['domain']=='native':
                        gt=label['xyz'][gi];camera[gi]=np.linalg.norm(pred-gt,axis=-1)*1000;error[gi]=np.linalg.norm((pred-pred[5])-(gt-gt[5]),axis=-1)*1000
                    else:
                        size=max(np.mean(label['boxes'][gi,2:]-label['boxes'][gi,:2]),1);error[gi]=np.linalg.norm(uv[gi]-label['uv'][gi],axis=-1)/size
            threshold=20 if label['domain']=='native' else .1
            row['modes'][mode]=dict(error=error,camera_error=camera,success=valid&(error<=threshold),matched=matched,xyz=xyz,uv=uv,ids=ids,predictions=len(candidates))
        rows.append(row)
    result={}
    for domain in ['native','glove']:
        rr=[r for r in rows if r['domain']==domain];summary={}
        for mode in MODES:
            points=sum(int(r['valid'].sum()) for r in rr);correct=sum(int(r['modes'][mode]['success'].sum()) for r in rr);total_hands=sum(len(r['valid']) for r in rr);matched=sum(int(r['modes'][mode]['matched'].sum()) for r in rr)
            entry=dict(images=len(rr),source_groups=len({r['group'] for r in rr}),hands=total_hands,matched_hands=matched,points=points,correct_points=correct,complete_pipeline_PCK=correct/max(points,1),comparisons={})
            for ref in ['yolo_v43','yolo_v48','rf_v53']:
                harm=recovery=good=bad=common_n=0;rel_sum=cam_sum=0.
                for r in rr:
                    v=r['valid'];base=r['modes'][ref]['error'];new=r['modes'][mode]['error'];good_mask=v&(base<=(10 if domain=='native' else .1));bad_mask=v&(base>(20 if domain=='native' else .1));good+=int(good_mask.sum());bad+=int(bad_mask.sum());harm+=int((good_mask&(new>(20 if domain=='native' else .1))).sum());recovery+=int((bad_mask&(new<=(20 if domain=='native' else .1))).sum())
                    common_mask=v&np.isfinite(base)&np.isfinite(new);common_n+=int(common_mask.sum());rel_sum+=float(new[common_mask].sum());cam_sum+=float(r['modes'][mode]['camera_error'][common_mask].sum()) if domain=='native' else 0
                comp=dict(old_good_points=good,old_good_damaged=harm,old_bad_points=bad,recovered_points=recovery,all_point_PCK_difference=ci(rr,mode,ref),common_matched_points=common_n,common_matched_error=rel_sum/max(common_n,1),common_matched_error_difference=ci(rr,mode,ref,'error'))
                if domain=='native':comp.update(common_matched_camera_mm=cam_sum/max(common_n,1),common_matched_camera_difference=ci(rr,mode,ref,'camera_error'))
                entry['comparisons'][ref]=comp
            if domain=='glove':
                entry['visible_points']=sum(int((r['valid']&r['visible']).sum()) for r in rr);entry['visible_correct']=sum(int((r['modes'][mode]['success']&r['visible']).sum()) for r in rr);entry['occluded_points']=sum(int((r['valid']&~r['visible']).sum()) for r in rr);entry['occluded_correct']=sum(int((r['modes'][mode]['success']&~r['visible']).sum()) for r in rr)
            summary[mode]=entry
        result[domain]=summary
    # Native pose movement in world coordinates; GT defines fast pairs after
    # predictions freeze. Missing/fragmented pairs are counted separately.
    motion={mode:dict(fast_GT_pairs=0,covered_fast_pairs=0,amplitude=[],spurious_jump_points=0,GT_identity_switches=0,GT_ID_fragments=0) for mode in MODES}
    clips_rows=collections.defaultdict(list)
    for r in rows:
        if r['domain']=='native':clips_rows[r['tag']].append(r)
    for rs in clips_rows.values():
        rs.sort(key=lambda r:r['frame']['timestamp_s'])
        for a,b in zip(rs,rs[1:]):
            dt=b['frame']['timestamp_s']-a['frame']['timestamp_s']
            if dt<=0 or dt>.1:continue
            for side in set(a['sides'])&set(b['sides']):
                ai=a['sides'].index(side);bi=b['sides'].index(side);v=a['valid'][ai]&b['valid'][bi]
                R1,T1=camera_pose(a['frame']['camera']);R2,T2=camera_pose(b['frame']['camera'])
                ga=a['xyz'][ai]@np.asarray(R1).T;gb=b['xyz'][bi]@np.asarray(R2).T;gd=(gb-gb[5])-(ga-ga[5]);gm=np.linalg.norm(gd,axis=-1)*1000;fast=v&(gm>10)
                for mode in MODES:
                    stats=motion[mode];stats['fast_GT_pairs']+=int(fast.sum());pa=a['modes'][mode];pb=b['modes'][mode];id1=pa['ids'][ai];id2=pb['ids'][bi]
                    if id1 is None or id2 is None:continue
                    if id1!=id2:stats['GT_ID_fragments']+=1;continue
                    x=pa['xyz'][ai]@np.asarray(R1).T;y=pb['xyz'][bi]@np.asarray(R2).T;move=np.linalg.norm((y-y[5])-(x-x[5]),axis=-1)*1000
                    stats['covered_fast_pairs']+=int(fast.sum());stats['amplitude']+=list((move/np.maximum(gm,1e-6))[fast]);stats['spurious_jump_points']+=int((v&(gm<=20)&(move>50)).sum())
            for mode in MODES:
                left={tid:a['sides'][i] for i,tid in enumerate(a['modes'][mode]['ids']) if tid is not None};right={tid:b['sides'][i] for i,tid in enumerate(b['modes'][mode]['ids']) if tid is not None}
                motion[mode]['GT_identity_switches']+=sum(left[k]!=right[k] for k in set(left)&set(right))
    for mode,stats in motion.items():
        values=stats.pop('amplitude');stats['fast_motion_amplitude_median']=float(np.median(values)) if values else None;stats['fast_motion_amplitude_mean']=float(np.mean(values)) if values else None;stats.update(tracks_stats[mode])
    training={arm:json.loads((ROOT/arm/'core_r1/dit_joint/done.json').read_text()) for arm in ['yolo_condition_control','rf_condition_adapt']}
    result.update(complete=True,protocol=protocol,training=training,motion=motion,scope=protocol['replay_scope'],all_outputs_frozen_before_test_labels=True,full_pipeline_replacement_admitted=False,default_changed=False)
    out=ROOT/'review';out.mkdir(exist_ok=True);save(ROOT/'summary.json',result);save(out/'statistics.json',result)
    display=['yolo_v43','yolo_v48','rf_v48','rf_v53','rf_v54','rf_v54_side_locked','rf_v54_best','yolo_v54_control'];tables=''
    for domain in ['native','glove']:
        body=''
        for mode in display:
            v=result[domain][mode];c=v['comparisons']['yolo_v48'];body+=f"<tr><td>{mode}</td><td>{v['matched_hands']}/{v['hands']}</td><td>{v['complete_pipeline_PCK']*100:.2f}%</td><td>{c['recovered_points']}/{c['old_bad_points']}</td><td>{c['old_good_damaged']}/{c['old_good_points']}</td><td>{html.escape(str(c['all_point_PCK_difference']['paired_pooled_delta_CI95']))}</td></tr>"
        tables+=f"<h2>{domain}</h2><table><tr><th>路径</th><th>手覆盖</th><th>完整PCK</th><th>相对v48恢复</th><th>原好点损伤</th><th>源组配对CI</th></tr>{body}</table>"
    changes=[(float(r['modes']['rf_v54']['success'].sum()-r['modes']['yolo_v48']['success'].sum()),i) for i,r in enumerate(rows)];chosen=sorted(changes)[:4]+sorted(changes,reverse=True)[:4];gallery=[]
    colors=[(255,140,30),(20,230,255),(30,220,50),(200,80,230)]
    for n,(change,i) in enumerate(chosen):
        r=rows[i];panels=[]
        for mode,color in zip(['yolo_v48','rf_v53','rf_v54','GT'],colors):
            image=cv2.imread(r['frame']['image']);points=r['uv'] if mode=='GT' else r['modes'][mode]['uv']
            for hand in points:
                for x,y in EDGES:
                    if np.isfinite(hand[[x,y]]).all():cv2.line(image,tuple(np.clip(hand[x],-4000,4000).astype(int)),tuple(np.clip(hand[y],-4000,4000).astype(int)),color,4)
            cv2.putText(image,mode,(20,55),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3);panels.append(cv2.resize(image,(420,420)))
        filename=f'case_{n:02d}.jpg';cv2.imwrite(str(out/filename),np.concatenate(panels,1));gallery.append(f"<p>{html.escape(r['group'])}；恢复净变化 {change:.0f} 点</p><img src='{filename}'>")
    nail_frames=json.loads((ROOT/'replay/nail.json').read_text())['frames']
    for index in [30,60,90]:
        frame=nail_frames[min(index,len(nail_frames)-1)];panels=[]
        for mode,color in zip(['yolo_v48','rf_v53','rf_v54','rf_v54_side_locked'],colors):
            image=cv2.imread(frame['image'])
            for candidate in prediction_frames[mode].get(('nail',frame['image']),[]):
                uv=common.from_json(frame['camera']).eye_to_window(np.asarray(candidate['candidate_xyz_camera_m']))
                for a,b in EDGES:
                    if np.isfinite(uv[[a,b]]).all():cv2.line(image,tuple(np.clip(uv[a],-4000,4000).astype(int)),tuple(np.clip(uv[b],-4000,4000).astype(int)),color,4)
                cv2.putText(image,str(candidate['track_id']),tuple(np.clip(uv[5],0,1300).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
            cv2.putText(image,mode,(20,55),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3);panels.append(cv2.resize(image,(420,420)))
        filename=f'nail_{index:03d}.jpg';cv2.imwrite(str(out/filename),np.concatenate(panels,1));gallery.append(f"<p>自然护理帧 {index}：无GT身份/3D，只作定性复核。</p><img src='{filename}'>")
    page=f"""<!doctype html><html lang='zh'><meta charset='utf-8'><title>v54 完整模型公平对照</title><style>body{{font:16px sans-serif;margin:24px;max-width:1800px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:9px}}img{{max-width:100%}}pre{{white-space:pre-wrap}}</style><h1>v54：RF-DETR接入后，完整模型公平对照</h1><p>{html.escape(protocol['replay_scope'])}</p><p>同GPU、RGB、时间戳、相机、关联器、±1.6秒、4候选/10步DDIM、FK和速度/加速度×2。v43指权重在共同运行配置下的表现。漏手计入PCK失败，三维均值只在双方共同匹配的点上比较。手套只有2D标注。</p><p>两组从同一v51权重各训160步；最后160步用于预算匹配诊断，best另列。没有通过完整流程准入，默认未换。手侧锁定是预测侧一致性试验，不能修正错误的初始手侧。</p>{tables}<h2>时序、空间与轨迹</h2><pre>{html.escape(json.dumps(motion,indent=2,ensure_ascii=False))}</pre><h2>自然RGB改善与损伤</h2>{''.join(gallery)}<p><a href='statistics.json'>完整计数、共同匹配误差、CI和训练协议</a></p></html>"""
    (out/'report.html').write_text(page,encoding='utf-8');print(json.dumps(dict(complete=True,report=str(out/'report.html'),default_changed=False)),flush=True)
if __name__=='__main__':main()

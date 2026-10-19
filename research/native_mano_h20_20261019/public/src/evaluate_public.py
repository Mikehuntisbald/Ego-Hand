"""Post-freeze HOT3D20 scoring and natural RGB overlays. No inference mutation."""
import argparse, json, sys, tarfile, hashlib, time, html
from pathlib import Path
import numpy as np
import torch
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
import os
ROOT=Path(os.environ.get('PUBLIC_HEADHAND_ROOT','/mnt/why/public_headhand_h20_20261019'))
sys.path.insert(0,str(ROOT/'vendor/Ego-Hand/research'))
import wilor_eval_common as common
from geometry import camera, transform, load_frames
from hand_tracking_toolkit.dataset import decode_hand_pose
from hand_tracking_toolkit.hand_models.umetrack_hand_model import from_json,forward_kinematics
MAPPING=common.MAPPING

def stats(values):
    a=np.asarray(values);a=a[np.isfinite(a)]
    return dict(n=int(len(a)),mean=float(a.mean()),RMSE=float(np.sqrt(np.mean(a*a))),P95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else dict(n=0)

def iou(a,b):
    p=np.maximum(a[:2],b[:2]);q=np.minimum(a[2:],b[2:]);inter=np.maximum(q-p,0).prod()
    return inter/max(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter,1e-9)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suffix',default='');parser.add_argument('--head',default='head_clip2102');args=parser.parse_args()
    frozen=ROOT/f'receipts/predictions_frozen_before_GT{args.suffix}.json';assert frozen.exists()
    manifest=json.loads(frozen.read_text())
    for rel,sha in manifest['seals'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha
    for name in [f'mano_baseline_clip2102{args.suffix}',f'mano_dit_clip2102{args.suffix}']:
        folder=ROOT/'outputs'/name
        for f,sha in json.loads((folder/'seal.json').read_text())['files'].items():assert hashlib.sha256((folder/f).read_bytes()).hexdigest()==sha
    output=ROOT/f'outputs/review_clip2102{args.suffix}';output.mkdir(exist_ok=False);(output/'images').mkdir()
    (output/'label_open_receipt.json').write_text(json.dumps(dict(open_unix=time.time(),frozen_prediction_manifest_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),inference_complete_before_labels=True)))
    frames=load_frames(ROOT/'data/clip-002102')
    coarse_rows=json.loads((ROOT/'outputs/wilor_clip2102/observations.json').read_text())['frames']
    coarse=np.load(ROOT/'outputs/wilor_clip2102/observations.npz');N=len(coarse_rows)
    arms={k:np.load(ROOT/f'outputs/{v}{args.suffix}/parameters.npz') for k,v in dict(BA='mano_baseline_clip2102',DiT_BA='mano_dit_clip2102').items()}
    predictions={'WiLoR':coarse['xyz_camera'][:,MAPPING],**{k:z['joints_camera'][:,MAPPING] for k,z in arms.items()}}
    head=json.loads((ROOT/f'outputs/{args.head}/headpose.json').read_text())['frames']
    GT=np.full((N,20,3),np.nan);gt_world=np.full_like(GT,np.nan);matched=np.zeros(N,bool);gt_count=0;matches=[];reference_head=[]
    hand_sides=[]
    with tarfile.open(ROOT/'assets/clip-002102-full.tar') as archive:
        shape=from_json(json.load(archive.extractfile('__hand_shapes.json__'))['umetrack'])
        for fi,frame in enumerate(frames):
            calibrations=json.load(archive.extractfile(f'{frame["id"]}.cameras.json'))
            gt_pose=transform(calibrations['214-1']['T_world_from_camera']);reference_head.append(gt_pose)
            hands=json.load(archive.extractfile(f'{frame["id"]}.hands.json'));poses=decode_hand_pose(hands)
            targets=[]
            for side,pose in poses.items():
                if pose.umetrack is None:continue
                with torch.no_grad():world=forward_kinematics(pose.umetrack,shape,requires_mesh=False)[0].numpy()
                box=np.array(hands[side.value]['boxes_amodal']['214-1'],float);box[[0,2]]=np.clip(box[[0,2]],0,1408);box[[1,3]]=np.clip(box[[1,3]],0,1408)
                if np.prod(np.maximum(box[2:]-box[:2],0))<64:continue
                targets.append(dict(side=side.value,world=world,box=box))
            gt_count+=len(targets);ids=[i for i,r in enumerate(coarse_rows) if r['frame_index']==fi]
            if not ids or not targets:continue
            scores=np.array([[iou(np.array(coarse_rows[i]['box_xyxy']),t['box']) for t in targets] for i in ids])
            a,b=linear_sum_assignment(1-scores)
            for x,y in zip(a,b):
                if scores[x,y]<.1:continue
                i=ids[x];target=targets[y];world=target['world'];gt_world[i]=world
                GT[i]=(world-gt_pose[:3,3])@gt_pose[:3,:3];matched[i]=True
                matches.append(dict(observation=i,frame=fi,track=coarse_rows[i]['track_id'],side=target['side'],IoU=float(scores[x,y])))
                hand_sides.append((bool(coarse['right'][i]),target['side']=='right'))
    assert matched.any()
    gtuv=np.stack([camera(frames[r['frame_index']]['cameras']['214-1']['calibration']).eye_to_window(xyz) if m else np.full((20,2),np.nan) for r,xyz,m in zip(coarse_rows,GT,matched)])
    visible=matched[:,None]&np.isfinite(gtuv).all(-1)&(GT[:,:,2]>.01)&(gtuv[:,:,0]>=0)&(gtuv[:,:,0]<1408)&(gtuv[:,:,1]>=0)&(gtuv[:,:,1]<1408)
    valid=matched[:,None]&np.isfinite(GT).all(-1)
    head_gt=np.array(reference_head);head_pred=np.array([h['T_world_from_rgb'] for h in head]);gauge=head_gt[0]@np.linalg.inv(head_pred[0]);aligned=gauge@head_pred
    head_pos=np.linalg.norm(aligned[:,:3,3]-head_gt[:,:3,3],axis=-1)*1000
    head_rot=Rotation.from_matrix(np.transpose(head_gt[:,:3,:3],(0,2,1))@aligned[:,:3,:3]).magnitude()*180/np.pi
    errors={};relative_errors={};report={};temporal={}
    for name,xyz in predictions.items():
        error=np.linalg.norm(xyz-GT,axis=-1)*1000;relative=np.linalg.norm((xyz-xyz[:,5:6])-(GT-GT[:,5:6]),axis=-1)*1000
        errors[name]=error;relative_errors[name]=relative
        predworld=np.array([x@head_pred[r['frame_index'],:3,:3].T+head_pred[r['frame_index'],:3,3] for x,r in zip(xyz,coarse_rows)])
        predworld=predworld@gauge[:3,:3].T+gauge[:3,3]
        we=np.linalg.norm(predworld-gt_world,axis=-1)*1000
        report[name]=dict(camera_mapped20_mm=stats(error[valid]),world_firstpose_SE3_mapped20_mm=stats(we[valid]),relative_mapped20_mm=stats(relative[valid]),
            wrist_mm=stats(error[:,5][matched]),visible_camera_mm=stats(error[visible]),PCK20_percent=float((error[valid]<=20).mean()*100))
        velocity=[];acceleration=[];fast_pred=[];fast_gt=[];pair_errors=[];bone_cv=[]
        for tid in sorted({r['track_id'] for r in coarse_rows}):
            ids=[i for i,r in enumerate(coarse_rows) if r['track_id']==tid and matched[i]];ids=sorted(ids,key=lambda i:coarse_rows[i]['timestamp_ns'])
            if len(ids)<3:continue
            t=np.array([coarse_rows[i]['timestamp_ns'] for i in ids],np.int64);dt=np.diff(t)/1e9
            p=xyz[ids];g=GT[ids];vp=np.diff(p,axis=0)/(60*dt[:,None,None]);vg=np.diff(g,axis=0)/(60*dt[:,None,None]);good=dt<.1
            ev=np.linalg.norm(vp-vg,axis=-1)*1000;velocity.extend(ev[good].ravel());pair_errors.extend(ev[good,5])
            ap=np.diff(vp,axis=0)/(60*(dt[1:]+dt[:-1])[:,None,None]/2);ag=np.diff(vg,axis=0)/(60*(dt[1:]+dt[:-1])[:,None,None]/2)
            acceleration.extend((np.linalg.norm(ap-ag,axis=-1)*1000)[good[1:]&good[:-1]].ravel())
            motion=np.linalg.norm(vg,axis=-1);fast=motion>=np.quantile(motion[good],.9);fast&=good[:,None]
            fast_pred.extend(np.linalg.norm(vp,axis=-1)[fast]);fast_gt.extend(motion[fast])
            # Canonical20 links exclude the unrepresented MANO thumb-MCP point.
            edges=[(5,6),(6,7),(7,0),(5,8),(8,9),(9,10),(10,1),(5,11),(11,12),(12,13),(13,2),(5,14),(14,15),(15,16),(16,3),(5,17),(17,18),(18,19),(19,4)]
            lengths=np.stack([np.linalg.norm(p[:,a]-p[:,b],axis=-1) for a,b in edges],-1);bone_cv.extend((lengths.std(0)/np.maximum(lengths.mean(0),1e-6)).tolist())
        temporal[name]=dict(error_velocity_mm_at_60Hz=stats(velocity),error_acceleration_mm_at_60Hz2=stats(acceleration),
            wrist_error_jumps_over10mm=int((np.array(pair_errors)>10).sum()),wrist_pairs=len(pair_errors),
            fast_motion_amplitude_ratio=float(np.mean(fast_pred)/np.mean(fast_gt)),bone_length_CV_mean=float(np.mean(bone_cv)))
    base=errors['BA'];dit=errors['DiT_BA'];rbase=relative_errors['BA'];rdit=relative_errors['DiT_BA']
    def protection(a,b):
        good=valid&(a<=10);hard=valid&(a>20)
        return dict(valid_points=int(valid.sum()),base_good10=int(good.sum()),good_harmed_over1mm=int((good&(b-a>1)).sum()),
                    good_to_over20mm=int((good&(b>20)).sum()),base_hard20=int(hard.sum()),recovered_to20=int((hard&(b<=20)).sum()),
                    improved_over1mm=int((valid&(a-b>1)).sum()),worsened_over1mm=int((valid&(b-a>1)).sum()))
    summary=dict(complete=True,dataset='HOT3D Aria clip2102, public diagnostic pilot',private_5090_reproduction=False,full_private_gt9_evaluation=False,
        inference_frozen_before_GT=True,frames=len(frames),predicted_hands=N,GT_visible_hands=gt_count,matched_hand_frames=int(matched.sum()),
        match_rule='clipped amodal box IoU Hungarian >=0.1, evaluation only',unmatched_predictions=int((~matched).sum()),
        hand_side_correct=sum(a==b for a,b in hand_sides),hand_side_total=len(hand_sides),metrics=report,temporal=temporal,
        DiT_vs_BA_protection_camera=protection(base,dit),DiT_vs_BA_protection_relative=protection(rbase,rdit),
        head=dict(position_firstpose_SE3_mm=stats(head_pos),rotation_firstpose_SE3_degrees=stats(head_rot),reference='provided HOT3D reference poses; not independent optical mocap; no scale or whole-trajectory alignment'),
        sequence_CI=None,sequence_CI_reason='only one sequence; sequence-level CI not estimable',training_overlap='not established; treat as diagnostic replay, not held-out generalization',
        evaluation_geometry='MANO21 predictions mapped to HOT3D UmeTrack20 landmarks; cross-model landmark differences remain; not private MANO21/GT17 protocol',
        development_reuse=bool(args.suffix) or os.environ.get('PUBLIC_DEVELOPMENT_REPLAY')=='1',head_source=args.head)
    (output/'metrics.json').write_text(json.dumps(summary,indent=2));(output/'associations.json').write_text(json.dumps(matches,indent=2))
    # Rank natural failures and improvements after freeze. No synthetic occlusion.
    ranked=np.nanmean(dit-base,axis=1);hard=np.nanmean(dit,axis=1)
    chosen=[]
    for order in [np.argsort(ranked)[::-1],np.argsort(ranked),np.argsort(hard)[::-1]]:
        count=0
        for i in order:
            if not matched[i]:continue
            fi=coarse_rows[i]['frame_index']
            if fi not in chosen:chosen.append(fi);count+=1
            if count>=4:break
    edges21=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
    cards=[]
    for fi in chosen:
        raw=cv2.imread(frames[fi]['images']['214-1']);panels=[]
        for name in ['WiLoR','BA','DiT_BA']:
            image=raw.copy();native=camera(frames[fi]['cameras']['214-1']['calibration'])
            for i,r in enumerate(coarse_rows):
                if r['frame_index']!=fi:continue
                j=coarse['xyz_camera'][i] if name=='WiLoR' else arms[name]['joints_camera'][i]
                uv=native.eye_to_window(j);color=(0,200,255) if name=='WiLoR' else ((255,120,0) if name=='BA' else (180,0,255))
                for a,b in edges21:
                    if np.isfinite(uv[[a,b]]).all():cv2.line(image,tuple(np.rint(uv[a]).astype(int)),tuple(np.rint(uv[b]).astype(int)),color,3,cv2.LINE_AA)
                for point in gtuv[i][visible[i]]:cv2.circle(image,tuple(np.rint(point).astype(int)),5,(0,255,0),-1)
            image=cv2.resize(image,(512,512));cv2.putText(image,name,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);panels.append(image)
        jpg=output/'images'/f'frame_{fi:06}.jpg';cv2.imwrite(str(jpg),np.concatenate(panels,axis=1))
        ids=[i for i,r in enumerate(coarse_rows) if r['frame_index']==fi and matched[i]]
        caption=f'frame {fi}: BA {np.nanmean(base[ids]):.1f} mm; DiT+BA {np.nanmean(dit[ids]):.1f} mm; green=GT20'
        cards.append(f'<figure><img src="images/{jpg.name}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    table=''.join(f'<tr><td>{name}</td><td>{m["camera_mapped20_mm"]["mean"]:.2f}</td><td>{m["camera_mapped20_mm"]["P95"]:.2f}</td><td>{m["relative_mapped20_mm"]["mean"]:.2f}</td><td>{m["wrist_mm"]["mean"]:.2f}</td></tr>' for name,m in report.items())
    document=f'''<!doctype html><meta charset="utf-8"><title>H20公开链路与DiT对照</title><style>body{{font:16px system-ui;max-width:1560px;margin:32px auto;background:#14191e;color:#eee}}table{{border-collapse:collapse}}td,th{{padding:12px;border:1px solid #777}}img{{max-width:100%}}figure{{margin:30px 0}}pre{{white-space:pre-wrap}}</style><h1>H20公开链路与DiT对照</h1><p>真实运行：RF → WiLoR → 双目头部轨迹 / LightESS+UniDepth → MANO BA；DiT为公开v48 UmeTrack20提案，经显式MANO拟合后进入同一BA。不是私有5090源码或九段指标复现。</p><p>单个公开片段，150帧，{matched.sum()}个匹配手帧；训练重叠未确认，作为诊断，不能称留出泛化。全部输出封存后才读GT。评估MANO21→HOT3D20，不能与私有21/17点协议混比。</p><table><tr><th>版本</th><th>相机MPJPE mm</th><th>P95 mm</th><th>腕对齐 mm</th><th>腕误差 mm</th></tr>{table}</table><p>绿色点为GT20；彩色骨架为预测。自然RGB案例，无人工遮挡。</p>{''.join(cards)}<details><summary>完整统计与边界</summary><pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre></details>'''
    (output/'report.html').write_text(document)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

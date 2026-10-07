"""Freeze unused clips, export them and cache prediction-only 3D/RGB observations.
Labels are attached after box tracking and are never used for inference inputs.
"""
import json,time,hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
import numpy as np,torch,cv2
from scipy.optimize import linear_sum_assignment
from hand3d_v8_common import RUN,V7,save
from ultralytics import YOLO
from download_rgb_subset import download
from export_hand_labels import export_clip
from hand3d_data_v7 import camera_pose
from dit_v3_inference import ObservationEncoder,prepare_observation
from compare_detectors import iou
from offline_rgb_encoder import crop_roi
from temporal_sampling_v6 import OFFSETS
from spatial_rgb_model import SpatialHead
import spatial_rgb_common as s
ROOT=s.common.ROOT

def freeze():
    dest=RUN/'fresh_manifest.json'
    if dest.exists():return json.loads(dest.read_text())
    inventory=json.loads((RUN/'fresh_inventory.json').read_text());lookup={r['sequence']:r for r in inventory['targets']}
    seqs=['P0010_0ecbf39f','P0010_e481fd15','P0010_fd1891a8','P0015_42b8b389','P0015_b0c5102b','P0015_e7458eb3']
    base=json.loads((ROOT/'subset_manifest.json').read_text());records=[]
    for seq in seqs:
        unused=lookup[seq]['unused'];assert len(unused)>=2
        clips=[unused[len(unused)//3],unused[2*len(unused)//3]];assert len(set(clips))==2
        records.append(dict(split='hand3d_v8_fresh',subject=seq.split('_')[0],sequence=seq,clips=clips,previously_used=lookup[seq]['used']))
    obj={k:base[k] for k in ['official_repo','revision','mirror','stream']};obj.update(sequences=records,created_unix=time.time(),frames_stride=5,scope='Unused clips of previously encountered subjects/source sequences; upstream WiLoR pretraining overlap unknown',selection='IDs frozen without metrics or visual case selection; never used to fit head, gate, threshold or checkpoint',inventory_sha256=hashlib.sha256((RUN/'fresh_inventory.json').read_bytes()).hexdigest())
    save(dest,obj);save(RUN/'fresh_manifest_receipt.json',dict(sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),clips=sum(len(r['clips']) for r in records),sequences=len(records)))
    return obj

def prepare():
    torch.set_num_threads(4);cv2.setNumThreads(0);device='cuda:3';manifest=freeze();started=time.time()
    jobs=[(r['split'],r['subject'],r['sequence'],cid) for r in manifest['sequences'] for cid in r['clips']]
    tree={r['path']:r for r in json.loads((ROOT/'provenance/train_aria_tree.json').read_text())}
    if not (RUN/'fresh_download_done.json').exists():
        receipts=[]
        with ThreadPoolExecutor(max_workers=12) as pool:
            fs=[pool.submit(download,j,manifest,tree) for j in jobs]
            for f in as_completed(fs):
                receipts.append(f.result());save(RUN/'fresh_download_status.json',dict(done=len(receipts),total=len(jobs)));print(json.dumps(dict(stage='download',done=len(receipts),total=len(jobs))),flush=True)
        with ProcessPoolExecutor(max_workers=4) as pool:exports=list(pool.map(export_clip,jobs))
        save(RUN/'fresh_download_done.json',dict(complete=True,receipts=receipts,exports=exports))
    frames=[]
    for split,subject,seq,cid in jobs:
        path=ROOT/'export/annotations'/split/seq/f'clip-{cid:06d}.jsonl'
        frames.extend(json.loads(line) for i,line in enumerate(path.read_text().splitlines()) if i%5==0)
    if not (RUN/'fresh_detections.json').exists():
        detector=YOLO(str(ROOT/'experiments/dit_lowconfidence_v1/detector/weights/best.pt'));side=YOLO(str(ROOT/'experiments/detector_compare_wilor_20261003/wilor_detector.pt'));detections=[]
        for start in range(0,len(frames),32):
            images=[str(ROOT/'export'/f['image']) for f in frames[start:start+32]]
            boxes=detector.predict(images,imgsz=960,device='3',batch=32,conf=.01,max_det=10,verbose=False)
            sides=side.predict(images,imgsz=960,device='3',batch=32,conf=.001,max_det=100,verbose=False)
            for b,p in zip(boxes,sides):detections.append(dict(boxes=b.boxes.xyxy.cpu().tolist(),scores=b.boxes.conf.cpu().tolist(),side=dict(boxes=p.boxes.xyxy.cpu().tolist(),scores=p.boxes.conf.cpu().tolist(),classes=p.boxes.cls.cpu().tolist())))
        save(RUN/'fresh_detections.json',detections);del detector,side;torch.cuda.empty_cache()
    detections=json.loads((RUN/'fresh_detections.json').read_text());rows=[];tracks={};next_id=0;prev_clip=None
    for fi,(frame,pred) in enumerate(zip(frames,detections)):
        group=(frame['sequence'],frame['clip']);now=frame['timestamp_ns']*1e-9
        if group!=prev_clip:tracks={};prev_clip=group
        active=[k for k,v in tracks.items() if now-v[-1]['time']<=.51];old=[]
        for tid in active:
            tr=tracks[tid];box=np.array(tr[-1]['box'],float)
            if len(tr)>1:box+=np.clip((box-np.array(tr[-2]['box']))/max(tr[-1]['time']-tr[-2]['time'],.01)*(now-tr[-1]['time']),-120,120)
            old.append(box)
        association={}
        if pred['boxes'] and active:
            overlap=iou(pred['boxes'],old);ii,jj=linear_sum_assignment(-overlap);association={int(i):active[int(j)] for i,j in zip(ii,jj) if overlap[i,j]>.1}
        for pi,box in enumerate(pred['boxes']):
            if pi not in association:association[pi]=next_id;tracks[next_id]=[];next_id+=1
            tid=association[pi];tracks[tid].append(dict(time=now,box=box))
            # No annotation fields participate in this prediction-only track.
            rows.append(dict(index=len(rows),frame_index=fi,sequence=frame['sequence'],subject=frame['subject'],clip=frame['clip'],frame=frame['frame'],timestamp_ns=frame['timestamp_ns'],image=str(ROOT/'export'/frame['image']),camera=frame['camera'],box=box,score=pred['scores'][pi],side_predictions=pred['side'],track_id=tid))
    # GT associations are attached after the complete independent box tracking.
    counts=dict(eligible_hands=0,matched=0,proposals=len(rows),frames=len(frames));frame_rows=defaultdict(list)
    for r in rows:frame_rows[r['frame_index']].append(r)
    for fi,frame in enumerate(frames):
        eligible=[h for h in frame['hands'] if h['box_amodal_xyxy'] is not None and (h['modeled_hand_visible_fraction'] or 0)>0 and h['xyz_camera_m'][5][2]>.05]
        now=frame_rows[fi];overlap=iou([r['box'] for r in now],[np.clip(h['box_amodal_xyxy'],0,1408) for h in eligible]);matches={}
        if now and eligible:
            ii,jj=linear_sum_assignment(-overlap);matches={int(i):int(j) for i,j in zip(ii,jj) if overlap[i,j]>=.3}
        counts['eligible_hands']+=len(eligible);counts['matched']+=len(matches)
        for pi,r in enumerate(now):
            h=eligible[matches[pi]] if pi in matches else None;r.update(matched=h is not None,gt=h['xyz_camera_m'] if h else None,projection_valid=h['keypoint_projection_valid'] if h else None,visible_fraction=h['modeled_hand_visible_fraction'] if h else None)
    save(RUN/'fresh_rows.json',rows);save(RUN/'fresh_detection_counts.json',counts)
    folder=RUN/'fresh_observations';folder.mkdir(exist_ok=True);encoder=ObservationEncoder(device);probe=SpatialHead().to(device).eval();probe.load_state_dict(torch.load(s.RUN/'sealed/rgb_probe.pt',weights_only=False,map_location=device)['model'])
    projection=torch.load(ROOT/'experiments/natural_reliability_v4/sealed/risk_projection.pt',weights_only=False).to(device)
    def prep(row):
        image=cv2.imread(row['image']);native=prepare_observation(image,row['box'],row['camera'],row['side_predictions']);roi=crop_roi(row['box']);crops,pos,tr,focal,*_=s.prepare({k:row[k] for k in ['image','camera','clip']},roi,[[0,0,0,0]])
        return native,crops[0],pos,tr,focal,roi/1408
    canonical=np.stack(np.meshgrid(np.arange(12)*16+37.5,np.arange(16)*16+5.5),-1).reshape(192,2)
    with ThreadPoolExecutor(max_workers=8) as pool,torch.inference_mode():
        for start in range(0,len(rows),128):
            end=min(start+128,len(rows));dest=folder/f'{start:06d}.pt'
            if dest.exists():continue
            prepared=list(pool.map(prep,rows[start:end]));fields=defaultdict(list)
            for k in range(0,len(prepared),16):
                part=prepared[k:k+16];n=len(part);part+=part[-1:]*(16-n);native=[p[0] for p in part]
                output=encoder.features(*[[p[key] for p in native] for key in ['wilor_crop','rotation','focal','right','side_conf','side_iou','reprojection_error','crop_fallback']]);fields['xyz'].append(output['wilor'][:n].cpu())
                inp=s.common.input_tensor([p[1] for p in part],[1]*16,rotation=1).to(device)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    feature=encoder.wilor.backbone(inp[:,:,:,32:-32])[-1].flatten(2).transpose(1,2);rgb=probe.features(feature).flatten(2).transpose(1,2);risk_rgb=feature.float()@projection.float()
                fields['rgb'].append(rgb[:n].half().cpu());fields['dense'].append(feature[:n].half().cpu());fields['risk_rgb'].append(risk_rgb[:n].half().cpu())
            chunk={key:torch.cat(v) for key,v in fields.items()};chunk.update(start=start,end=end,positions=torch.tensor(np.stack([p[2] for p in prepared])),roi=torch.tensor(np.stack([p[5] for p in prepared])))
            rays=[]
            for p in prepared:
                z=np.c_[(canonical-127.5)/p[4],np.ones(192)]@p[3].T;z/=np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-6);rays.append(z)
            chunk['rays_camera']=torch.tensor(np.stack(rays),dtype=torch.float32);torch.save(chunk,dest);print(json.dumps(dict(stage='encode',done=end,total=len(rows),seconds=time.time()-started)),flush=True)
    assemble(rows,folder)

def assemble(rows,folder):
    n=len(rows)+1;xyz=torch.zeros(n,20,3);rgb=torch.zeros(n,192,128,dtype=torch.float16);riskrgb=torch.zeros(n,192,32,dtype=torch.float16);dense=torch.zeros(n,192,1280,dtype=torch.float16);positions=torch.zeros(n,192,2);roi=torch.zeros(n,4);rays_camera=torch.zeros(n,192,3);last=0
    for path in sorted(folder.glob('*.pt')):
        c=torch.load(path,weights_only=False);assert c['start']==last;last=c['end'];a,z=c['start']+1,c['end']+1
        for dest,key in [(xyz,'xyz'),(rgb,'rgb'),(riskrgb,'risk_rgb'),(dense,'dense'),(positions,'positions'),(roi,'roi'),(rays_camera,'rays_camera')]:dest[a:z]=c[key]
    assert last==len(rows);rotation=torch.eye(3)[None].repeat(n,1,1);translation=torch.zeros(n,3);scores=torch.zeros(n);gt_bank=torch.zeros(n,20,3);valid_bank=torch.zeros(n,20,dtype=torch.bool);uv_bank=torch.zeros(n,20,2);uv_valid_bank=torch.zeros(n,20,dtype=torch.bool);xy_bank=torch.zeros(n,20,2);obs_bank=torch.zeros(n,20,dtype=torch.bool)
    groups=defaultdict(dict)
    for fid,r in enumerate(rows,1):
        R,t=camera_pose(r['camera']);rotation[fid]=torch.tensor(R);translation[fid]=torch.tensor(t);scores[fid]=r['score'];groups[(r['sequence'],r['clip'],r['track_id'])][r['frame']]=fid
        cam=s.common.from_json(r['camera']);uv=cam.eye_to_window(xyz[fid].numpy())/1408;xy_bank[fid]=torch.tensor(np.nan_to_num(uv));obs_bank[fid]=torch.tensor(np.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<1).all(-1))
        if r['matched']:
            gt_bank[fid]=torch.tensor(r['gt']);valid_bank[fid]=torch.isfinite(gt_bank[fid]).all(-1);g_uv=cam.eye_to_window(np.asarray(r['gt']))/1408;uv_bank[fid]=torch.tensor(np.nan_to_num(g_uv));uv_valid_bank[fid]=torch.tensor(r['projection_valid'])&torch.tensor(np.isfinite(g_uv).all(-1))
    fids=[];dts=[];metadata=[]
    for center,r in enumerate(rows,1):
        group=groups[(r['sequence'],r['clip'],r['track_id'])]
        # Same benchmark center inclusion as training; no target values used to associate frames.
        if not r['matched'] or uv_valid_bank[center].sum()<10:continue
        if sum(0<r['frame']-f<=40 for f in group)<2 or sum(0<f-r['frame']<=40 for f in group)<2:continue
        row=[];dt=[]
        for offset in OFFSETS['multiscale']:
            fid=group.get(r['frame']+5*offset,0);row.append(fid);dt.append((rows[fid-1]['timestamp_ns']-r['timestamp_ns'])*1e-9 if fid else offset/6)
        fids.append(row);dts.append(dt);metadata.append(dict(sequence=r['sequence'],subject=r['subject'],clip=r['clip'],frame=r['frame'],image=r['image'],track_id=r['track_id'],source='hand3d_v8_fresh',row_index=center-1))
    fids=torch.tensor(fids);center=fids[:,8];available=torch.isfinite(xyz).all(-1);available[0]=False;world=torch.einsum('njc,nkc->njk',xyz,rotation)+translation[:,None];rays_world=torch.einsum('nsc,nkc->nsk',rays_camera,rotation)
    data=dict(feature_ids=fids,dt=torch.tensor(dts,dtype=torch.float32),xy=xy_bank[fids],observed_2d=obs_bank[fids],world=world,xyz_camera_bank=xyz,available_bank=available,rotation=rotation,translation=translation,rays_world=rays_world,rgb_bank=rgb,positions_bank=positions,roi=roi,scores=scores,risk_rgb_bank=riskrgb,gt=gt_bank[center],valid=valid_bank[center],gt_uv=uv_bank[center],uv_valid=uv_valid_bank[center],roles=['fresh']*len(metadata),subjects=[r['subject'] for r in metadata],rows=metadata)
    torch.save(data,RUN/'fresh_data.pt');torch.save(dense,RUN/'fresh_dense.pt');save(RUN/'fresh_ready.json',dict(complete=True,windows=len(metadata),observations=len(rows),sequences=len({r['sequence'] for r in metadata}),clips=len({r['clip'] for r in metadata}),manifest_sha256=hashlib.sha256((RUN/'fresh_manifest.json').read_bytes()).hexdigest(),metrics_read=False));print((RUN/'fresh_ready.json').read_text(),flush=True)

if __name__=='__main__':prepare()

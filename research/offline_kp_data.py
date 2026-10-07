"""Prediction-tracked bidirectional windows for offline keypoint annotation.

No GT associations, handedness, visibility or poses enter tracking/model inputs.
GT is attached afterwards solely for training losses and benchmark evaluation.
"""
import json
from pathlib import Path
from collections import defaultdict
import wilor_eval_common as common
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from compare_detectors import iou

RUN=common.ROOT/'experiments/offline_keypoint_diffusion_v1'
WINDOW=17
CHAINS=[[6,7,0],[8,9,10,1],[11,12,13,2],[14,15,16,3],[17,18,19,4]]

def save(path,obj):
    p=path.with_suffix('.partial');p.write_text(json.dumps(obj,indent=2));p.replace(path)

def build():
    RUN.mkdir(exist_ok=True)
    if (RUN/'data_done.json').exists():return
    sources=[('dit_wilor_v5',{'P0001','P0002','P0009','P0011','P0012','P0014'},'train'),
             ('dit_wilor_v3',{'P0003'},'development'),('dit_wilor_v4',{'P0010','P0015'},'test')]
    protocol=dict(task='Offline bidirectional 2D finger-keypoint gap completion for annotation',
        inputs='Predicted 2D tracks, observed masks, timestamps; no RGB/hidden coordinates/GT inputs',
        observation_source='Frozen WiLoR 2D projections inside YOLO-predicted boxes; YOLO currently has no trained finger-keypoint head',
        sources=[dict(run=s,subjects=sorted(p),role=r) for s,p,r in sources],window=WINDOW,nominal_sample_hz=6,
        training_subjects_disjoint_from_development_and_test=True,
        evidence_scope='New fixed task split using previously opened 3D study data; NOT untouched new-data evidence; upstream WiLoR HOT3D overlap unknown',
        missing_protocol='Random finger-chain gaps of 1/3/6/9 sampled frames; 20% all-keypoint gaps. Masked coordinates removed in the whole gap, not just central frame.',
        limitations=['Controlled coordinate-observation removal, not a natural pixel-occlusion benchmark',
            'Current hand detection/association remains available; this version does not recover an entirely missing hand track',
            'No per-finger natural visibility labels; valid projection is not visibility',
            'Bidirectional offline processing is intentional; no real-time or causal claim'],
        baselines=['previous available observation','nearest observation','bidirectional linear interpolation','same-input same-capacity regression'],
        primary_metrics=['hidden-joint pixel error','PCK10/PCK20','visible-input exact preservation','per-gap errors','source-sequence paired bootstrap'],
        acceptance='Report improvement over linear interpolation with paired sequence CI; separately compare regression; never call synthetic dropout real occlusion recovery',seed=202610034)
    save(RUN/'protocol.json',protocol)
    windows=[];meta=[];counts={}
    for source,subjects,role in sources:
        folder=common.ROOT/'experiments'/source
        rows=json.loads((folder/'locked_rows.json').read_text());frames=json.loads((folder/'locked_frames.json').read_text())
        arrays={k:[] for k in ['wilor_2d','geometry','transform']};last=0
        for p in sorted((folder/'locked_observations').glob('*.pt')):
            chunk=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
            assert chunk['start']==last;last=chunk['end']
            for k in arrays:arrays[k].append(chunk[k].float())
        assert last==len(rows)
        arrays={k:torch.cat(v).numpy() for k,v in arrays.items()}
        frame_rows=defaultdict(list)
        for i,r in enumerate(rows):
            if r['subject'] in subjects:frame_rows[r['frame_index']].append(i)
        clips=defaultdict(list)
        for fi,f in enumerate(frames):
            if f['subject'] in subjects:clips[(f['sequence'],f['clip'])].append(fi)
        for (sequence,clip),fis in sorted(clips.items()):
            fis.sort(key=lambda i:frames[i]['timestamp_ns']);tracks={};next_id=0
            for slot,fi in enumerate(fis):
                f=frames[fi];now=f['timestamp_ns']*1e-9;ids=frame_rows[fi]
                boxes=[rows[i]['box'] for i in ids];active=[k for k,t in tracks.items() if now-t[-1]['time']<=.51]
                associations={}
                if ids and active:
                    old=[]
                    for tid in active:
                        tr=tracks[tid];box=np.array(tr[-1]['box'],float)
                        if len(tr)>1:
                            dt=tr[-1]['time']-tr[-2]['time']
                            box+=np.clip((box-np.array(tr[-2]['box']))/max(dt,.01)*(now-tr[-1]['time']),-120,120)
                        old.append(box)
                    overlap=iou(boxes,old);ii,jj=linear_sum_assignment(-overlap)
                    associations={int(i):active[int(j)] for i,j in zip(ii,jj) if overlap[i,j]>.1}
                for j,idx in enumerate(ids):
                    if j not in associations:associations[j]=next_id;tracks[next_id]=[];next_id+=1
                    r=rows[idx];cam=common.from_json(r['camera']);xy=arrays['wilor_2d'][idx]
                    ray=np.column_stack([xy*256/arrays['geometry'][idx,0],np.ones(20)])@arrays['transform'][idx].T
                    uv=cam.eye_to_window(ray).astype(np.float32)
                    obs_valid=np.isfinite(uv).all(-1)&(uv[:,0]>=0)&(uv[:,0]<1408)&(uv[:,1]>=0)&(uv[:,1]<1408)
                    target=np.zeros((20,2),np.float32);target_valid=np.zeros(20,bool)
                    if r['matched']:
                        target=cam.eye_to_window(np.asarray(r['gt'])).astype(np.float32)
                        target_valid=np.asarray(r['projection_valid'],bool)&np.isfinite(target).all(-1)
                    tracks[associations[j]].append(dict(slot=slot,time=now,box=r['box'],xy=np.nan_to_num(uv)/1408,
                        observed=obs_valid,gt=np.nan_to_num(target)/1408,valid=target_valid,row_index=idx,matched=r['matched']))
            for tid,track in tracks.items():
                by_slot={r['slot']:r for r in track}
                for center in track:
                    if not center['matched'] or center['valid'].sum()<10:continue
                    left=sum(t['slot']<center['slot'] and center['slot']-t['slot']<=8 for t in track)
                    right=sum(t['slot']>center['slot'] and t['slot']-center['slot']<=8 for t in track)
                    if left<2 or right<2:continue
                    xy=np.zeros((WINDOW,20,2),np.float32);observed=np.zeros((WINDOW,20),bool)
                    dt=np.arange(-8,9,dtype=np.float32)/6
                    for wi,slot in enumerate(range(center['slot']-8,center['slot']+9)):
                        t=by_slot.get(slot)
                        if t is None:continue
                        xy[wi]=t['xy'];observed[wi]=t['observed'];dt[wi]=t['time']-center['time']
                    windows.append(dict(xy=xy,observed=observed,dt=dt,gt=center['gt'],valid=center['valid']))
                    meta.append(dict(role=role,subject=frames[fis[0]]['subject'],sequence=sequence,clip=clip,
                        track_id=tid,source=source,row_index=center['row_index'],frame=rows[center['row_index']]['frame'],
                        image=rows[center['row_index']]['image']))
        counts[role]=sum(r['role']==role for r in meta)
        print(json.dumps(dict(role=role,windows=counts[role],all_proposals_tracked=True)),flush=True)
    tensors={k:torch.from_numpy(np.stack([w[k] for w in windows])) for k in windows[0]}
    torch.save(tensors,RUN/'windows.pt');save(RUN/'rows.json',meta)
    save(RUN/'data_done.json',dict(complete=True,windows=len(windows),counts=counts,gt_used_for_inputs=False))

if __name__=='__main__':build()

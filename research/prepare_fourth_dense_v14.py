"""Freeze the fourth unused-clip batch before any new evaluation metrics."""
import json,time,hashlib,collections
import torch
from hand3d_v8_common import V7,save
import prepare_dense_sampling_v13 as worker
RUN=V7.parent/'fourth_dense_v14'
SPARSE=[-120,-80,-50,-30,-20,-15,-10,-5,0,5,10,15,20,30,50,80,120]
DENSE=[-120,-80,-50,-30,-10,-3,-2,-1,0,1,2,3,10,30,50,80,120]

def freeze():
    RUN.mkdir(exist_ok=True);inventory=json.loads((RUN/'expanded_inventory.json').read_text());previous=[json.loads((V7.parent/folder/'fresh_manifest.json').read_text()) for folder in ['offline_hand3d_v8','offline_hand3d_v9','offline_hand3d_v10_native']];sequences=[];targets=[]
    for subject in ['P0010','P0015']:
        eligible=[r for r in inventory['rows'] if r['subject']==subject and len(r['unused'])>=2];assert len(eligible)>=3
        targets.extend(eligible[:3])
    for r in targets:
        available=r['unused'];used={c for m in previous for row in m['sequences'] if row['sequence']==r['sequence'] for c in row['clips']};unused=[c for c in available if c not in used];assert len(unused)>=2
        clips=[unused[len(unused)//4],unused[3*len(unused)//4]];assert len(set(clips))==2
        sequences.append(dict(split='hand3d_fourth_dense_v14',subject=r['subject'],sequence=r['sequence'],clips=clips))
    base=previous[0];manifest={k:base[k] for k in ['official_repo','revision','mirror','stream']};manifest.update(sequences=sequences,created_unix=time.time(),frames_stride=1,center_stride=5,scope='Fourth12unused clips; all previous3batches excluded; subjects/source sequences reused, no new-subject claim',selection='Clip IDs frozen before final checkpoint/gate selection; metrics unopened',sparse_offsets_frames=SPARSE,dense_offsets_frames=DENSE)
    path=RUN/'fresh_manifest.json'
    if path.exists():assert json.loads(path.read_text())['sequences']==sequences
    else:save(path,manifest)
    save(RUN/'manifest_receipt.json',dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),clips=12,sequences=6,all_previous3batches_excluded=True))

def schedules():
    original=torch.load(RUN/'fresh_data.pt',weights_only=False,mmap=True);records=json.loads((RUN/'fresh_rows.json').read_text());groups=collections.defaultdict(dict)
    for fid,r in enumerate(records,1):groups[(r['sequence'],int(r['clip']),int(r['track_id']))][int(r['frame'])]=fid
    n=len(records)+1;xy=torch.zeros(n,20,2);observed=torch.zeros(n,20,dtype=torch.bool)
    import numpy as np,spatial_rgb_common as s
    for fid,r in enumerate(records,1):
        p=s.common.from_json(r['camera']).eye_to_window(original['xyz_camera_bank'][fid].numpy())/1408;xy[fid]=torch.tensor(np.nan_to_num(p));observed[fid]=torch.tensor(np.isfinite(p).all(-1)&(p>=0).all(-1)&(p<1).all(-1))
    center=original['feature_ids'][:,8];details={}
    for label,offsets in [('sparse',SPARSE),('dense',DENSE)]:
        fids=torch.zeros(len(center),17,dtype=torch.long);dt=torch.tensor(offsets,dtype=torch.float32)[None].repeat(len(center),1)/30
        for k,fid in enumerate(center):
            r=records[int(fid)-1];group=groups[(r['sequence'],int(r['clip']),int(r['track_id']))]
            for slot,off in enumerate(offsets):
                f=group.get(int(r['frame'])+off,0);fids[k,slot]=f
                if f:dt[k,slot]=(records[f-1]['timestamp_ns']-r['timestamp_ns'])*1e-9
        assert torch.equal(fids[:,8],center)
        data={**original,'feature_ids':fids,'dt':dt,'xy':xy[fids],'observed_2d':observed[fids]};torch.save(data,RUN/f'{label}_data.pt');near=dt[:,[7,9]].abs()[fids[:,[7,9]]>0];details[label]=dict(offsets_frames=offsets,near_interval_median_s=float(near.median()),observed_fraction=float((fids>0).float().mean()))
    save(RUN/'prepared.json',dict(complete=True,windows=len(center),observations=len(records),schedules=details,metrics_opened=False,scope='Observation preparation only. GT attached after prediction-only tracking; no model/policy chosen or metric computed on this batch.'))
    print((RUN/'prepared.json').read_text(),flush=True)

def main():
    torch.set_num_threads(4);freeze();worker.RUN=RUN;worker.prepare_observations();schedules()

if __name__=='__main__':main()

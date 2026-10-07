"""Same old centers, new prediction-only tracks, sparse/dense time controls.

Center XYZ/RGB/ROI/camera/XY/targets stay identical. New track assignment uses
box IoU only; no label, side, visibility, or target participates in association.
Unmatched centers are retained with missing context, never dropped for accuracy.
"""
import json,time,collections
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,load,save
from hand3d_rollout_v8 import project_fisheye624
from hand3d_visual_v8 import dense_bank
from hand3d_trajectory_data_v9 import RUN as V9,targets
import spatial_rgb_common as s
RUN=V7.parent/'dense_sampling_v13';OUT=V7.parent/'aligned_density_v13'
SPARSE=[-120,-80,-50,-30,-20,-15,-10,-5,0,5,10,15,20,30,50,80,120]
DENSE=[-120,-80,-50,-30,-10,-3,-2,-1,0,1,2,3,10,30,50,80,120]

def box_iou(box,boxes):
    b=np.asarray(box,float);p=np.asarray(boxes,float)
    lo=np.maximum(b[:2],p[:,:2]);hi=np.minimum(b[2:],p[:,2:]);intersection=np.maximum(hi-lo,0).prod(-1)
    return intersection/np.maximum(np.maximum(b[2:]-b[:2],0).prod()+np.maximum(p[:,2:]-p[:,:2],0).prod(-1)-intersection,1e-9)

def main():
    torch.set_num_threads(4);started=time.time();OUT.mkdir(exist_ok=True)
    assert json.loads((RUN/'dense_ready.json').read_text())['complete'];old=load('cpu');records,_=s.records_and_index();new=torch.load(RUN/'data.pt',weights_only=False,mmap=True);new_records=json.loads((RUN/'fresh_rows.json').read_text());new_extra=torch.load(RUN/'trajectory_targets.pt',weights_only=False,mmap=True);old_extra=torch.load(V9/'trajectory_targets.pt',weights_only=False,mmap=True)
    ids=torch.tensor([i for i,r in enumerate(old['roles']) if r in ['train','dev_select','dev_calibrate']]);old_fids=old['feature_ids'][ids,8];unique=torch.unique(old_fids,sorted=True);n=len(new['world']);mapping=torch.zeros(len(old['world']),dtype=torch.long);mapping[unique]=torch.arange(n,n+len(unique));center=mapping[old_fids]
    fields=['world','xyz_camera_bank','available_bank','rotation','translation','rays_world','rgb_bank','positions_bank','roi','scores','risk_rgb_bank']
    common={k:torch.cat([new[k],old[k][unique]]) for k in fields}
    common.update(gt=old['gt'][ids].clone(),valid=old['valid'][ids].clone(),gt_uv=old['gt_uv'][ids].clone(),uv_valid=old['uv_valid'][ids].clone(),roles=[old['roles'][int(i)] for i in ids],subjects=[old['subjects'][int(i)] for i in ids],rows=[dict(old['rows'][int(i)]) for i in ids],source_window_indices=ids)
    extra={k:torch.cat([new_extra[k],old_extra[k][unique]]) for k in new_extra};torch.save(extra,OUT/'trajectory_targets.pt')
    params=torch.zeros(len(common['world']),16)
    for fid,r in enumerate(new_records,1):
        cam=s.common.from_json(r['camera']);params[fid]=torch.tensor(list(cam.f)+list(cam.c)+list(cam.distort))
    for fid in unique:
        cam=s.common.from_json(records[int(fid)-1]['camera']);params[int(mapping[fid])]=torch.tensor(list(cam.f)+list(cam.c)+list(cam.distort))
    uv=torch.zeros(len(common['world']),20,2);observed=torch.zeros(len(common['world']),20,dtype=torch.bool)
    for start in range(1,len(uv),1024):
        end=start+1024;p=project_fisheye624(common['xyz_camera_bank'][start:end],params[start:end])/1408;uv[start:end]=torch.nan_to_num(p);observed[start:end]=torch.isfinite(p).all(-1)&(p>=0).all(-1)&(p<1).all(-1)&common['available_bank'][start:end]
    candidates=collections.defaultdict(list);tracks=collections.defaultdict(dict)
    for fid,r in enumerate(new_records,1):
        candidates[(r['sequence'],int(r['clip']),int(r['frame']))].append(fid);tracks[(r['sequence'],int(r['clip']),int(r['track_id']))][int(r['frame'])]=fid
    association=[]
    for k,oldfid in enumerate(old_fids):
        r=records[int(oldfid)-1];possible=candidates[(r['sequence'],int(r['clip']),int(r['frame']))];chosen=None;quality=0.
        if possible:
            overlap=box_iou(r['box'],[new_records[fid-1]['box'] for fid in possible]);j=int(overlap.argmax());quality=float(overlap[j])
            if quality>=.5:chosen=possible[j]
        association.append(dict(source_window_index=int(ids[k]),new_center_detection=chosen,box_iou=quality,matched=chosen is not None))
    results={}
    for arm,offsets in [('sparse',SPARSE),('dense',DENSE)]:
        fids=torch.zeros(len(ids),17,dtype=torch.long);dt=torch.tensor(offsets,dtype=torch.float32)[None].expand(len(ids),-1).clone()/30
        for k,(oldfid,a) in enumerate(zip(old_fids,association)):
            if a['matched']:
                r=new_records[a['new_center_detection']-1];group=tracks[(r['sequence'],int(r['clip']),int(r['track_id']))]
                for slot,offset in enumerate(offsets):
                    fid=group.get(int(r['frame'])+offset,0);fids[k,slot]=fid
                    if fid:dt[k,slot]=(new_records[fid-1]['timestamp_ns']-r['timestamp_ns'])*1e-9
            fids[k,8]=center[k];dt[k,8]=0
        xy=uv[fids];obs=observed[fids];xy[:,8]=old['xy'][ids,8];obs[:,8]=old['observed_2d'][ids,8]
        data={**common,'feature_ids':fids,'dt':dt,'xy':xy,'observed_2d':obs};torch.save(data,OUT/f'{arm}_data.pt')
        for key in ['xyz_camera_bank','rgb_bank','roi','positions_bank','rotation','translation','scores','risk_rgb_bank']:
            assert torch.equal(data[key][fids[:,8]],old[key][old_fids]),key
        assert torch.equal(xy[:,8],old['xy'][ids,8]) and torch.equal(obs[:,8],old['observed_2d'][ids,8]);assert torch.equal(data['gt'],old['gt'][ids]) and torch.equal(data['valid'],old['valid'][ids])
        g,v,_,_=targets(data,extra,torch.arange(len(ids)));assert float((g[:,8]-data['gt']).abs().max())<2e-6
        interval=dt[:,[7,9]].abs()[(fids[:,[7,9]]>0)];results[arm]=dict(context_offsets_frames=offsets,valid_context_fraction=float((fids>0).float().mean()),near_interval_median_s=float(interval.median()),near_interval_p90_s=float(interval.quantile(.9)),camera_target_max_difference_m=float((g[:,8]-data['gt']).abs().max()))
    # A single common native bank. Original centers are copied exactly, all new
    # context features came from original pixels and predicted ROIs.
    bank=torch.load(RUN/'fresh_dense.pt',weights_only=False,mmap=True);old_bank=dense_bank('cpu',len(old['world']));joined=torch.cat([bank,old_bank[unique]])
    assert torch.equal(joined[center],old_bank[old_fids]);torch.save(joined,OUT/'native_bank.pt');torch.save(params,OUT/'camera_params.pt')
    save(OUT/'center_association.json',association)
    save(OUT/'ready.json',dict(complete=True,windows=len(ids),roles=dict(collections.Counter(common['roles'])),center_observations_and_targets_exact=True,center_native_rgb_exact=True,matched_centers=sum(a['matched'] for a in association),missing_context_centers=sum(not a['matched'] for a in association),association='Predicted box IoU>=0.5 only; labels/side/visibility never used; unmatched old centers retained',controls=results,seconds=time.time()-started,scope='Original train/development centers only. New sparse/dense observations share detector/tracks. Outer4slots on each side keep original times; closest3slots become1/30,2/30,3/30s. Shared visual training is not full OOF.'))
    print((OUT/'ready.json').read_text(),flush=True)

if __name__=='__main__':main()

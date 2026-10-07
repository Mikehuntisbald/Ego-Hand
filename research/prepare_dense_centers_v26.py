"""Train real context observations as current targets; development fixed.

Frame choices use predicted tracks/time only. GT assigns supervised targets,
not inputs, identity votes, or missing slots. No artificial occlusion.
"""
import collections,json,time
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,save
from hand3d_rollout_v8 import project_fisheye624
import spatial_rgb_common as s

def main():
    torch.set_num_threads(4)
    root=V7.parent/'dense_centers_v26';root.mkdir(exist_ok=True)
    assert not (root/'ready.json').exists()
    started=time.time();source=V7.parent/'context_data_v17/control'
    data=torch.load(source/'dense_data.pt',weights_only=False,mmap=True)
    teacher=torch.load(source/'trajectory_labels.pt',weights_only=False,mmap=True)
    records=json.loads((V7.parent/'dense_sampling_v13/fresh_rows.json').read_text())
    train_subjects={subject for subject,role in zip(data['subjects'],data['roles']) if role=='train'}
    old_records,_=s.records_and_index();old_data=torch.load(V7/'data.pt',weights_only=False,mmap=True)
    old_boxes=collections.defaultdict(list)
    for k,(role,window) in enumerate(zip(data['roles'],data['source_window_indices'])):
        if role=='train':
            rec=old_records[int(old_data['feature_ids'][int(window),8])-1]
            old_boxes[rec['image']].append(np.array(rec['box'],float))
    def duplicates_old(rec):
        box=np.array(rec['box'],float)
        for other in old_boxes[rec['image']]:
            intersection=np.maximum(np.minimum(box[2:],other[2:])-np.maximum(box[:2],other[:2]),0).prod()
            union=np.maximum(box[2:]-box[:2],0).prod()+np.maximum(other[2:]-other[:2],0).prod()-intersection
            if intersection/max(union,1e-9)>=.9:return True
        return False
    groups=collections.defaultdict(list)
    for fid,rec in enumerate(records,1):
        groups[(rec['sequence'],rec['clip'],rec['track_id'])].append((int(rec['timestamp_ns']),fid))
    for key,values in groups.items():groups[key]=sorted(values)
    offsets=np.array([-120,-80,-50,-30,-10,-3,-2,-1,0,1,2,3,10,30,50,80,120])/30.
    centers=[];fids=[];times=[];duplicates=0
    for fid,rec in enumerate(records,1):
        if rec['subject'] not in train_subjects or not rec['matched'] or not torch.isfinite(torch.tensor(rec['gt'])).all():continue
        if duplicates_old(rec):duplicates+=1;continue
        values=groups[(rec['sequence'],rec['clip'],rec['track_id'])]
        stamp=np.array([v[0] for v in values],np.int64);ids=np.array([v[1] for v in values])
        target=int(rec['timestamp_ns'])+np.rint(offsets*1e9).astype(np.int64)
        distance=np.abs(target[:,None]-stamp[None]);nearest=distance.argmin(1)
        slot=np.where(distance[np.arange(17),nearest]<=20_000_000,ids[nearest],0)
        slot[8]=fid
        dt=np.where(slot>0,(stamp[nearest]-int(rec['timestamp_ns']))/1e9,offsets)
        dt[8]=0.
        centers.append(fid);fids.append(slot);times.append(dt)
    assert centers
    feature_ids=torch.tensor(np.array(fids),dtype=torch.long)
    dt=torch.tensor(np.array(times),dtype=torch.float32)
    assert set(records[i-1]['subject'] for i in centers)<=train_subjects
    # Teacher banks are filled only after the input windows have been chosen.
    annotation_cache={};n=len(data['world'])
    gt_world=torch.zeros(2,n,20,3);gt_valid=torch.zeros(2,n,20,dtype=torch.bool)
    gt_uv=torch.zeros(2,n,20,2);gt_uv_valid=torch.zeros_like(gt_valid)
    label_side={}
    needed=set(feature_ids.flatten().tolist())-{0}
    for fid in sorted(needed):
        rec=records[fid-1]
        sp,sequence,clip=Path(rec['image']).relative_to(s.common.ROOT/'export/images').parts[:3]
        path=s.common.ROOT/'export/annotations'/sp/sequence/(clip+'.jsonl')
        if path not in annotation_cache:annotation_cache[path]=[json.loads(line) for line in path.read_text().splitlines()]
        hands=annotation_cache[path][rec['frame']]['hands']
        for hand in hands:
            side=0 if hand['side']=='left' else 1
            xyz=torch.tensor(hand['xyz_camera_m'],dtype=torch.float32)
            assert xyz.shape==(20,3)
            valid=torch.isfinite(xyz).all(-1);safe=torch.nan_to_num(xyz)
            gt_world[side,fid]=safe@data['rotation'][fid].T+data['translation'][fid]
            gt_valid[side,fid]=valid
            uv=s.common.from_json(rec['camera']).eye_to_window(safe.numpy())/1408
            gt_uv[side,fid]=torch.tensor(np.nan_to_num(uv))
            roi=data['roi'][fid].numpy();inside=(uv>=roi[:2]).all(-1)&(uv<roi[2:]).all(-1)
            gt_uv_valid[side,fid]=torch.tensor(hand['keypoint_projection_valid'])&torch.tensor(np.isfinite(uv).all(-1)&inside)&valid
            if rec['matched'] and torch.isfinite(torch.tensor(rec['gt'])).all() and (xyz-torch.tensor(rec['gt'])).abs().max()<2e-6:
                assert fid not in label_side,'Ambiguous teacher identity'
                label_side[fid]=side
    assert all(i in label_side for i in centers),'Center teacher cannot be matched; do not guess a side'
    side=torch.tensor([label_side[i] for i in centers])
    center=torch.tensor(centers)
    world=gt_world[side[:,None],feature_ids]
    valid=gt_valid[side[:,None],feature_ids]&(feature_ids>0)[...,None]
    labels=torch.einsum('btjc,bck->btjk',world-data['translation'][center][:,None,None],data['rotation'][center])
    labels=torch.where(valid[...,None],labels,0.)
    uv=gt_uv[side[:,None],feature_ids];uv_valid=gt_uv_valid[side[:,None],feature_ids]&valid
    expected=torch.stack([torch.tensor(records[i-1]['gt']) for i in centers])
    assert (labels[:,8]-expected).abs().max()<2e-6 and valid[:,8].all()
    params=torch.load(V7.parent/'aligned_density_v13/camera_params.pt',weights_only=False,mmap=True)
    image_uv=project_fisheye624(data['xyz_camera_bank'],params)/1408
    observed=torch.isfinite(image_uv).all(-1)&(image_uv>=0).all(-1)&(image_uv<1).all(-1)&data['available_bank']
    before_side=torch.load(V7.parent/'aligned_density_v13/dense_data.pt',weights_only=False,mmap=True)['xyz_camera_bank']
    original_n=len(data['roles']);new=dict(data)
    for key,value in dict(feature_ids=feature_ids,dt=dt,gt=labels[:,8],valid=valid[:,8],gt_uv=uv[:,8],uv_valid=uv_valid[:,8],
        xy=torch.nan_to_num(image_uv)[feature_ids],observed_2d=observed[feature_ids],
        source_window_indices=torch.full((len(center),),-1,dtype=torch.long),original_base_for_evaluation=before_side[center]).items():
        new[key]=torch.cat([data[key],value])
    new['roles']=list(data['roles'])+['train']*len(center)
    new['subjects']=list(data['subjects'])+[records[i-1]['subject'] for i in centers]
    new['rows']=list(data['rows'])+[dict(role='train',subject=records[i-1]['subject'],sequence=records[i-1]['sequence'],clip=records[i-1]['clip'],track_id=records[i-1]['track_id'],source='natural_dense_training_v26',row_index=i,frame=records[i-1]['frame'],image=records[i-1]['image']) for i in centers]
    extra={k:torch.cat([teacher[k],value]) for k,value in dict(gt=labels,valid=valid,gt_uv=uv,uv_valid=uv_valid).items()}
    for key in ['feature_ids','dt','gt','valid','gt_uv','uv_valid','xy','observed_2d','source_window_indices','original_base_for_evaluation']:
        assert torch.equal(new[key][:original_n],data[key]),key
    for key in teacher:assert torch.equal(extra[key][:original_n],teacher[key]),key
    torch.save(new,root/'dense_data.pt');torch.save(extra,root/'trajectory_labels.pt')
    (root/'native_bank.pt').symlink_to(source/'native_bank.pt')
    save(root/'ready.json',dict(complete=True,old_windows=original_n,additional_training_centers=len(center),duplicates_removed=duplicates,
        roles={role:new['roles'].count(role) for role in set(new['roles'])},training_subjects=sorted(train_subjects),original_rows_and_labels_exact=True,
        actual_time_tolerance_s=.02,history_slots=17,frame_choice='Predictedtrack/timestamps only; no GT repaired or reassociated inputs',
        supervision='GT only assigns same centerhand across17frames. Unmatched/misassociated context inputs retained. OldcurrentGT/RGB/XYZ and all development windows unchanged.',
        scope='Denseframes already used as contexts, now currenttraining targets. No newimages/newsubjects/independent evidence. No fifth/retainedfailures in training. Natural only; no artificial occlusion.',seconds=time.time()-started))
    print((root/'ready.json').read_text(),flush=True)

if __name__=='__main__':main()

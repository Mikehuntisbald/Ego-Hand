"""Freeze observation-only short windows and natural RGB pixels for v47."""
import collections,hashlib,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np,torch
from hand3d_v8_common import V7,save
from parameter_codec_v31 import OUT
from prepare_joint_mano_v28 import records_bank
from temporal_window_v44 import ObservationSampler,SHORT_OFFSETS,window_statistics
from hand3d_rollout_v8 import project_fisheye624
from joint_mano_model_v29 import six_to_rotation,rotation_to_six
from offline_rgb_encoder import crop_roi
import spatial_rgb_common as s

RUN=V7.parent/'online_rgb_3d_v47'

def main():
    torch.set_num_threads(4);cv2.setNumThreads(0);RUN.mkdir(exist_ok=True);start=time.time()
    if (RUN/'ready.json').exists():return
    root=V7.parent;data=torch.load(root/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
    primary,full=records_bank(data);sampler=ObservationSampler(primary,full);centers=data['feature_ids'][:,8].tolist()
    fs=[];ds=[];unsupported=[]
    for wi,fid in enumerate(centers):
        try:f,dt=sampler.sample([fid])
        except (AssertionError,KeyError,ValueError):
            f=torch.zeros(1,17,dtype=torch.long);f[0,8]=fid;dt=torch.tensor(SHORT_OFFSETS)[None].float()/30;dt[0,8]=0;unsupported.append(wi)
        fs.append(f);ds.append(dt)
    f=torch.cat(fs);dt=torch.cat(ds);assert torch.equal(f[:,8],data['feature_ids'][:,8])
    params=torch.load(root/'aligned_density_v13/camera_params.pt',weights_only=False,mmap=True)
    xy=project_fisheye624(data['xyz_camera_bank'],params)/1408
    observed=torch.isfinite(xy).all(-1)&(xy>=0).all(-1)&(xy<1).all(-1)&data['available_bank']
    infer={k:v for k,v in data.items() if k not in ['gt','valid','gt_uv','uv_valid','original_base_for_evaluation']}
    infer.update(feature_ids=f,dt=dt,xy=torch.nan_to_num(xy)[f],observed_2d=observed[f])
    infer['xy'][:,8]=data['xy'][:,8];infer['observed_2d'][:,8]=data['observed_2d'][:,8]
    # Keep metadata whitelist separate from annotation fields in bank records.
    metadata=[]
    for r in full:metadata.append({k:r[k] for k in ['image','camera','clip','box']})
    save(RUN/'records.json',metadata);torch.save(infer,RUN/'inputs.pt')
    train_dev=torch.tensor([i for i,r in enumerate(data['roles']) if r in ['train','dev_select']])
    needed=torch.unique(f[train_dev]);needed=needed[needed>0].tolist();n=len(data['world'])
    # Original alias crops used the historical normalized ROI; dense contexts
    # used exact crop_roi(box). Preserve both recipes without an extra roundtrip.
    old_records,index=s.records_and_index();old=torch.load(V7/'data.pt',weights_only=False,mmap=True)
    rois={fid:crop_roi(full[fid-1]['box']) for fid in range(1,len(primary)+1)}
    for wi,source in enumerate(data['source_window_indices']):
        oldfid=int(old['feature_ids'][int(source),8]);rois[int(data['feature_ids'][wi,8])]=index['roi'][oldfid].numpy()*1408
    assert all(fid in rois for fid in needed)
    pixels_path=RUN/'pixels.npy';done_path=RUN/'pixels_done.json'
    if not done_path.exists():
        pixels=np.lib.format.open_memmap(pixels_path,mode='w+',dtype=np.uint8,shape=(n,256,256,3))
        def prep(fid):
            r=metadata[fid-1];images,pos,*_=s.prepare(r,rois[fid],[[0,0,0,0]])
            assert np.max(np.abs(pos-data['positions_bank'][fid].numpy()))<2e-5
            return fid,images[0]
        with ThreadPoolExecutor(max_workers=8) as pool:
            for begin in range(0,len(needed),256):
                for fid,image in pool.map(prep,needed[begin:begin+256]):pixels[fid]=image
                if begin%1024==0:print(json.dumps(dict(stage='pixels',done=min(begin+256,len(needed)),total=len(needed),seconds=time.time()-start)),flush=True)
        pixels.flush();del pixels;save(done_path,dict(complete=True,observations=len(needed),natural_rgb_only=True,no_feature_cache=True))
    # Labels are a separate file and are never returned by the input builder.
    source=torch.load(OUT/'refined_targets.pt',weights_only=False,mmap=True);oldf=data['feature_ids'];N=len(data['world'])
    state_world=torch.zeros(N,2,63);state_ok=torch.zeros(N,2,dtype=torch.bool)
    for wi in range(len(data['roles'])):
        R=data['rotation'][oldf[wi,8]];T=data['translation'][oldf[wi,8]];side=int(source['right'][wi])
        for slot in torch.where(source['mask'][wi])[0].tolist():
            fid=int(oldf[wi,slot])
            if not fid or state_ok[fid,side]:continue
            x=source['state'][wi,slot].flatten().clone();x[:3]=(R@(x[:3]*.1)+T)/.1;x[3:9]=rotation_to_six(R@six_to_rotation(x[3:9]))
            state_world[fid,side]=x;state_ok[fid,side]=True
    gt_world=torch.zeros(N,2,20,3);valid_world=torch.zeros(N,2,20,dtype=torch.bool);annotations={}
    for fid in torch.unique(f).tolist():
        if not fid:continue
        r=full[fid-1];sp,seq,clip=Path(r['image']).relative_to(s.common.ROOT/'export/images').parts[:3]
        path=s.common.ROOT/'export/annotations'/sp/seq/(clip+'.jsonl')
        if path not in annotations:annotations[path]=[json.loads(x) for x in path.read_text().splitlines()]
        for hand in annotations[path][r['frame']]['hands']:
            side=int(hand['side']=='right');x=torch.tensor(hand['xyz_camera_m']);ok=torch.isfinite(x).all(-1)
            gt_world[fid,side]=torch.einsum('jc,kc->jk',torch.nan_to_num(x),data['rotation'][fid])+data['translation'][fid]
            valid_world[fid,side]=ok
    sides=source['right'];state=state_world[f,sides[:,None]].clone();mask=state_ok[f,sides[:,None]]&(f>0)
    flat=state.flatten(2);R=data['rotation'][f[:,8]];T=data['translation'][f[:,8]]
    flat[:,:,:3]=torch.einsum('btj,bjk->btk',flat[:,:,:3]*.1-T[:,None],R)/.1
    flat[:,:,3:9]=rotation_to_six(R.transpose(-1,-2)[:,None]@six_to_rotation(flat[:,:,3:9]))
    state=flat.reshape(-1,17,21,3);state[:,8]=source['state'][:,8];mask[:,8]=source['mask'][:,8]
    gt=torch.einsum('btjc,bck->btjk',gt_world[f,sides[:,None]]-T[:,None,None],R)
    valid=valid_world[f,sides[:,None]]&(f>0)[:,:,None];gt=torch.where(valid[:,:, :,None],gt,0.)
    assert (gt[:,8]-data['gt']).abs().max()<2e-6
    torch.save(dict(state=state,mask=mask,right=sides,gt=gt,valid=valid),RUN/'targets.pt')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'records.json',RUN/'targets.pt']}
    save(RUN/'ready.json',dict(complete=True,train_windows=data['roles'].count('train'),dev_select_windows=data['roles'].count('dev_select'),
        parameter_target_coverage=float(mask[f>0].float().mean()),missing_context_retained_centers=unsupported,
        input_window_statistics=window_statistics(f,dt),center_targets_exact=True,natural_only=True,
        gt_not_in_inputs=True,calibration_roles_not_used_for_training=True,hashes=hashes,seconds=time.time()-start))
    print((RUN/'ready.json').read_text(),flush=True)

if __name__=='__main__':main()

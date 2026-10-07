import json
import numpy as np,torch
from hand3d_v8_common import V7,save
from hand3d_data_v7 import load,batch
import spatial_rgb_common as s
RUN=V7.parent/'offline_hand3d_v9';RUN.mkdir(exist_ok=True)

def prepare():
    records,_=s.records_and_index();data=load('cpu');n=len(data['world']);gt=torch.zeros(n,20,3);valid=torch.zeros(n,20,dtype=torch.bool);uv=torch.zeros(n,20,2);uv_valid=torch.zeros(n,20,dtype=torch.bool)
    for fid,r in enumerate(records,1):
        if r['matched']:
            gt[fid]=torch.tensor(r['gt']);valid[fid]=torch.isfinite(gt[fid]).all(-1);p=s.common.from_json(r['camera']).eye_to_window(np.asarray(r['gt']))/1408;uv[fid]=torch.tensor(np.nan_to_num(p));uv_valid[fid]=torch.tensor(r['projection_valid'])&torch.tensor(np.isfinite(p).all(-1))
    world=torch.einsum('njc,nkc->njk',gt,data['rotation'])+data['translation'][:,None]
    torch.save(dict(gt_world=world,valid=valid,gt_uv=uv,uv_valid=uv_valid),RUN/'trajectory_targets.pt')
    center=data['feature_ids'][:,8];reconstructed=torch.einsum('njc,nck->njk',world[center]-data['translation'][center,None],data['rotation'][center]);delta=float((reconstructed-data['gt']).abs().max());assert delta<2e-6
    save(RUN/'data_checks.json',dict(complete=True,center_target_max_m=delta,targets='Entire17frame20joint current-camera XYZ aligned with original predicted tracks; training supervision only',inference_inputs='Same original RGB/XYZ/timestamps/camera and risk as v8; no GT changes',scope='v8 fresh12clip metrics already inspected; unavailable for a new independent claim; another manifest will be frozen'))

def targets(data,extra,ids):
    f=data['feature_ids'][ids];center=f[:,8];gt=torch.einsum('btjc,bck->btjk',extra['gt_world'][f]-data['translation'][center,None,None],data['rotation'][center]);valid=extra['valid'][f]&(f>0)[...,None]
    gt=torch.where(valid[...,None],gt,0.)
    return gt,valid,extra['gt_uv'][f],extra['uv_valid'][f]&(f>0)[...,None]

def center_only(b):
    # This is an ablation, not training with manufactured pixel occlusion.
    b={k:v.clone() for k,v in b.items()};keep=torch.zeros_like(b['rgb_valid']);keep[:,8]=True
    for k in ['xyz','xy','rgb','positions','camera_origin','roi','risk_rgb']:
        x=b[k];b[k]=x*keep.reshape(*keep.shape,*([1]*(x.ndim-2)))
    b['available']&=keep[...,None];b['observed_2d']&=keep[...,None];b['rgb_valid']&=keep;b['scores']*=keep
    return b

if __name__=='__main__':prepare()

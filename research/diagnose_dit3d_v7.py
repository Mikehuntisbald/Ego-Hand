"""Development-only diagnostic; not a checkpoint/threshold selection run."""
import json
import numpy as np,torch
from hand3d_data_v7 import RUN as BASE,load
from hand3d_temporal_v7 import WRIST
from train_hand3d_v7 import metrics
P=BASE.parent/'offline_hand3d_v7_protected'
def main():
    torch.set_num_threads(4);d=load('cpu');c=np.load(P/'rgb_dit/calibration.npz');ids=torch.from_numpy(c['indices']);pred=torch.from_numpy(c['prediction']);base=d['xyz_camera_bank'][d['feature_ids'][ids,8]];gt=d['gt'][ids];valid=d['valid'][ids]
    dr=pred[:,WRIST:WRIST+1]-base[:,WRIST:WRIST+1];dp=(pred-pred[:,WRIST:WRIST+1])-(base-base[:,WRIST:WRIST+1]);ray=base[:,WRIST]/base[:,WRIST].norm(dim=-1,keepdim=True)
    axial=(dr[:,0]*ray).sum(-1,keepdim=True)*ray;tangential=dr[:,0]-axial
    outputs={'identity':base,'root_only':base+dr,'pose_only_fixed_root':base+dp,'both':pred,'root_ray_component_only':base+axial[:,None],'root_transverse_only':base+tangential[:,None]}
    result=dict(scope='dev_calibrate; existing locked checkpoint; diagnostic only, no fitting/selection',metrics={k:metrics(v,base,gt,valid) for k,v in outputs.items()},root_shift_mm=dict(mean=float(dr.norm(dim=-1).mean()*1000),p95=float(torch.quantile(dr.norm(dim=-1),.95)*1000),mean_ray_axis=float(axial.norm(dim=-1).mean()*1000),mean_transverse=float(tangential.norm(dim=-1).mean()*1000)))
    dest=BASE.parent/'offline_hand3d_v7_bounded/dit_optimization_diagnosis.json';dest.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()

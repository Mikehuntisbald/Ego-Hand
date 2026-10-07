import json,numpy as np,torch
from hand3d_v8_common import V7,load,save,camera_bank
from hand3d_rollout_v8 import project_fisheye624

def main():
    torch.set_num_threads(4);data=load('cpu');center=data['feature_ids'][:,8];xyz=data['xyz_camera_bank'][center];params=camera_bank()[center];physical=project_fisheye624(xyz,params)/1408
    observed=data['observed_2d'][:,8]&torch.isfinite(physical).all(-1);error=(data['xy'][:,8]-physical).norm(dim=-1)*1408;groups={}
    for role in ['train','dev_select','dev_calibrate','test']:
        keep=torch.tensor(np.array(data['roles'])==role);values=error[keep][observed[keep]];groups[role]=dict(points=len(values),mean_px=float(values.mean()),median_px=float(values.median()),p95_px=float(values.quantile(.95)))
    obj=dict(groups=groups,source='Original XY from WiLoR normalized2D output mapped through crop rays; newly prepared XY comes from reconstructed XYZ camera projection',GT_used=False,scope='Input-lineage diagnostic only. Both density arms preserve old center XY exactly and share new context XY; this difference cannot explain dense-minus-sparse by itself.')
    save(V7.parent/'aligned_density_v13/center_xy_source_audit.json',obj);print(json.dumps(obj,indent=2))

if __name__=='__main__':main()

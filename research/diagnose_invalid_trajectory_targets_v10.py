import json
import numpy as np,torch
from hand3d_v8_common import V7,load,batch,save
from hand3d_trajectory_data_v9 import RUN as V9,targets
from hand3d_trajectory_v9 import TrajectoryHand3D
RUN=V7.parent/'offline_hand3d_v10_native';RUN.mkdir(exist_ok=True)

@torch.no_grad()
def main():
    torch.set_num_threads(4);device='cuda:0';data=load(device);extra={k:v.to(device) for k,v in torch.load(V9/'trajectory_targets.pt',weights_only=False).items()};ids=torch.tensor(np.where(np.asarray(data['roles'])=='train')[0][:8],device=device);prob=torch.load(V7/'risk_probabilities.pt',weights_only=False)['train_oof'].to(device);b=batch(data,ids,prob);gt,valid,uv,uvvalid=targets(data,extra,ids);changed=gt.clone();unknown=(~valid).all(-1);changed[unknown]+=100
    ck=torch.load(V9/'rgb_dit/best.pt',weights_only=False,map_location=device);model=TrajectoryHand3D('dit').to(device).eval();model.load_state_dict(ck['model']);values=[]
    for target in [gt,changed]:
        torch.manual_seed(202610107)
        with torch.autocast('cuda',dtype=torch.bfloat16):value=model.loss(b,target,valid,uv,uvvalid)
        values.append(float(value))
    result=dict(loss_original=values[0],loss_invalid_placeholders_perturbed=values[1],absolute_difference=abs(values[0]-values[1]),invalid_full_frames=int(unknown.sum()),scope='Training diagnostic only; valid annotations and every inference condition were identical; invalid target values should not affect training')
    save(RUN/'invalid_target_diagnosis.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

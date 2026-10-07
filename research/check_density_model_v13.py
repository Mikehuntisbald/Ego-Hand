import json,torch,numpy as np
from hand3d_v8_common import V7,save
from hand3d_trajectory_data_v9 import targets
from train_aligned_density_v13 import DATA,INITIAL,make
from density_model_v13 import DensityTrajectoryHand3D,POLICY
from native_projection_policy_v11 import apply

def main():
    torch.set_num_threads(4);device='cuda:3';data_raw=torch.load(DATA/'dense_data.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in data_raw.items()};bank=torch.load(DATA/'native_bank.pt',weights_only=False,mmap=True).to(device);extra={k:v.to(device) for k,v in torch.load(DATA/'trajectory_targets.pt',weights_only=False,mmap=True).items()};ids=torch.tensor(np.where(np.array(data['roles'])=='train')[0][:2],device=device);prob=torch.zeros(len(data['roles']),20,2,device=device);b=make(data,bank,ids,prob);gt,valid,uv,uv_valid=targets(data,extra,ids)
    ck=torch.load(INITIAL/'rgb_dit/best.pt',weights_only=False,map_location=device);model=DensityTrajectoryHand3D('dit',True).to(device).eval();model.load_state_dict(ck['model'])
    torch.manual_seed(202610114)
    with torch.autocast('cuda',dtype=torch.bfloat16):value=model.rollout_loss(b,gt,valid,uv,uv_valid,202610114)
    assert torch.isfinite(value);value.backward();norms=dict(native_projection=float(model.rgb_project.weight.grad.norm()),localization=float(model.localization.project[1].weight.grad.norm()));assert all(v>0 and np.isfinite(v) for v in norms.values());model.zero_grad(set_to_none=True)
    with torch.no_grad():
        changed=gt.clone();changed[~valid]+=999;torch.manual_seed(202610114)
        with torch.autocast('cuda',dtype=torch.bfloat16):after=model.rollout_loss(b,changed,valid,uv,uv_valid,202610114)
        difference=float((value.detach()-after).abs());assert difference==0,difference
        with torch.autocast('cuda',dtype=torch.bfloat16):
            p=model.predict_rollout(b,seed=202610114);poisoned=model.predict_rollout({**b,'gt':gt*99,'visibility':torch.zeros(2,20,device=device),'gt_side':['left','left']},seed=202610114)
        delta=float((p['xyz_camera_m']-poisoned['xyz_camera_m']).abs().max());assert delta==0
        confirmed=torch.zeros(2,20,device=device,dtype=torch.bool);confirmed[:,[0,5,12]]=True
        pred=apply(p['xyz_camera_m'].float(),b['base'],POLICY,confirmed);assert torch.equal(pred[confirmed],b['base'][confirmed]);dc=(pred-b['base']).norm(dim=-1).max();dr=((pred-pred[:,5:6])-(b['base']-b['base'][:,5:6])).norm(dim=-1).max();assert dc<.009501 and dr<.009501
    near=b['dt'][:,[7,9]].abs()[b['rgb_valid'][:,[7,9]]];assert float(near.median())<.04
    save(DATA/'model_preflight.json',dict(passed=True,gradient_norms=norms,invalid_target_loss_difference=difference,GT_poison_prediction_max_m=delta,output_shape=list(p['xyz_camera_m'].shape),confirmed_exact=True,nearest_real_dt_median_s=float(near.median()),max_camera_m=float(dc),max_relative_m=float(dr),scope='Matched-center engineering preflight, not recovery evidence'))
    print((DATA/'model_preflight.json').read_text(),flush=True)

if __name__=='__main__':main()

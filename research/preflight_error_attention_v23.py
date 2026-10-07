import json,numpy as np,torch
from hand3d_v8_common import V7,save
from hand3d_data_v7 import batch
from recovery_model_v15 import RecoveryHand3D
from error_weighted_attention_v23 import ErrorWeightedHand3D

def main():
    torch.set_num_threads(4);device='cuda:3';source=V7.parent/'context_data_v17/control';root=V7.parent/'error_attention_v23';root.mkdir(exist_ok=True)
    raw=torch.load(source/'dense_data.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw.items()};bank=torch.load(source/'native_bank.pt',weights_only=False,mmap=True)
    ids=torch.tensor(np.where(np.asarray(data['roles'])=='dev_select')[0][:2],device=device);risk=torch.load(source/'risk_dense/risk_probabilities.pt',weights_only=False)['joint'].to(device)
    b=batch(data,ids,risk);b['rgb_native']=bank[data['feature_ids'][ids].cpu()].to(device);b['temporal_error_risk']=torch.load(V7.parent/'temporal_reliability_v22/risk_probabilities.pt',weights_only=False)['joint'][ids.cpu()].to(device)
    policy=json.loads((V7.parent/'side_native_v16/fifth_seal.json').read_text())['policy'];ck=torch.load(V7.parent/'side_native_v16/consensus/uniform_adaptive/best.pt',weights_only=False,map_location=device)
    base=RecoveryHand3D(policy).to(device).eval();base.load_state_dict(ck['model']);model=ErrorWeightedHand3D(policy,strength=0.).to(device).eval();model.load_state_dict(ck['model'])
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        p=base.predict_rollout(b,seed=202610114)['xyz_camera_m'];q=model.predict_rollout(b,seed=202610114)['xyz_camera_m']
    assert torch.equal(p,q)
    model.error_attention_strength=1.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        q=model.predict_rollout(b,seed=202610114)['xyz_camera_m'];other=model.predict_rollout(dict(b,gt=torch.full_like(b['xyz'],9999.),valid=torch.zeros_like(b['available'])),seed=202610114)['xyz_camera_m']
    assert torch.equal(q,other) and q.shape==(2,20,3) and torch.isfinite(q).all()
    labels=torch.load(source/'trajectory_labels.pt',weights_only=False,mmap=True);target=tuple(labels[k][ids.cpu()].to(device) for k in ['gt','valid','gt_uv','uv_valid'])
    model.train();model.localization.eval()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.rollout_loss(b,*target,202610117)
    assert torch.isfinite(loss);loss.backward();gradient=float(model.rgb_project.weight.grad.norm());assert gradient>0
    save(root/'preflight.json',dict(passed=True,disabled_exact_sampler_parity=True,active_gt_poison_exact=True,output_shape=list(q.shape),active_max_xyz_change_mm=float((q-p).abs().max()*1000),native_projection_gradient=gradient,scope='Engineering only; frozen log-trust bias not yet accuracy evidence; no preflightupdates carried to training'))
    print((root/'preflight.json').read_text())

if __name__=='__main__':main()

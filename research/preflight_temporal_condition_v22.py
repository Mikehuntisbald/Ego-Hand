import json,numpy as np,torch
from hand3d_v8_common import V7,save
from hand3d_data_v7 import batch
from recovery_model_v15 import RecoveryHand3D
from temporal_risk_condition_v22 import TemporalRiskHand3D

def main():
    torch.set_num_threads(4);device='cuda:3';source=V7.parent/'context_data_v17/control';root=V7.parent/'temporal_condition_v22';root.mkdir(exist_ok=True)
    raw=torch.load(source/'dense_data.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw.items()}
    bank=torch.load(source/'native_bank.pt',weights_only=False,mmap=True)
    ids=torch.tensor(np.where(np.asarray(data['roles'])=='dev_select')[0][:2],device=device)
    risk=torch.load(source/'risk_dense/risk_probabilities.pt',weights_only=False)['joint'].to(device)
    b=batch(data,ids,risk);b['rgb_native']=bank[data['feature_ids'][ids].cpu()].to(device)
    b['temporal_error_risk']=torch.load(V7.parent/'temporal_reliability_v22/risk_probabilities.pt',weights_only=False)['joint'][ids.cpu()].to(device)
    policy=json.loads((V7.parent/'side_native_v16/fifth_seal.json').read_text())['policy']
    ck=torch.load(V7.parent/'side_native_v16/consensus/uniform_adaptive/best.pt',weights_only=False,map_location=device)
    base=RecoveryHand3D(policy).to(device).eval();base.load_state_dict(ck['model'])
    model=TemporalRiskHand3D(policy).to(device).eval();missing=model.load_state_dict(ck['model'],strict=False);assert not missing.unexpected_keys and all(k.startswith('temporal_risk_project.') for k in missing.missing_keys)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        p=base.predict_rollout(b,seed=202610114)['xyz_camera_m'];q=model.predict_rollout(b,seed=202610114)['xyz_camera_m']
    assert q.shape==(2,20,3) and torch.equal(p,q) and torch.isfinite(q).all()
    target=torch.load(source/'trajectory_labels.pt',weights_only=False,mmap=True);labels=tuple(target[k][ids.cpu()].to(device) for k in ['gt','valid','gt_uv','uv_valid'])
    opt=torch.optim.AdamW(model.parameters(),lr=5e-5);grad=[]
    for step in range(3):
        model.train();model.localization.eval();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.rollout_loss(b,*labels,202610117+step)
        assert torch.isfinite(loss);loss.backward();norm=float(model.temporal_risk_project[-1].weight.grad.norm());assert norm>0;grad.append(norm);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    save(root/'preflight.json',dict(passed=True,initial_sampler_max_delta_mm=0.,output_shape=list(q.shape),new_condition_gradients=grad,scope='Engineering check only; preflightupdates discarded; currentgates unchanged; no accuracy conclusion'))
    print((root/'preflight.json').read_text())

if __name__=='__main__':main()

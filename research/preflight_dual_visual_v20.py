"""Verify zero-gain parity, physical coordinates and actual new-branch learning."""
import json
import numpy as np
import torch
from hand3d_v8_common import V7, save
from hand3d_data_v7 import batch
from recovery_model_v15 import RecoveryHand3D
from dual_visual_condition_v20 import DualVisualHand3D

def main():
    torch.set_num_threads(4)
    device = 'cuda:3'
    root = V7.parent/'dual_visual_v20'
    root.mkdir(exist_ok=True)
    source = V7.parent/'context_data_v17/control'
    original = torch.load(source/'dense_data.pt',weights_only=False,mmap=True)
    data = {k:v.to(device) if torch.is_tensor(v) else v for k,v in original.items()}
    canonical = torch.load(V7.parent/'canonical_rgb_v19/dense_data.pt',weights_only=False,mmap=True)
    bank = torch.load(source/'native_bank.pt',weights_only=False,mmap=True)
    extra_bank = torch.load(V7.parent/'canonical_rgb_v19/native_bank.pt',weights_only=False,mmap=True)
    ids = torch.tensor(np.where(np.asarray(data['roles'])=='dev_select')[0][:2],device=device)
    fids = data['feature_ids'][ids]
    risk = torch.load(source/'risk_dense/risk_probabilities.pt',weights_only=False)['joint'].to(device)
    b = batch(data,ids,risk)
    b['rgb_native'] = bank[fids.cpu()].to(device)
    b['extra_rgb_native'] = extra_bank[fids.cpu()].to(device)
    b['extra_positions'] = canonical['positions_bank'][fids.cpu()].to(device)
    b['extra_rays'] = torch.einsum('btsc,bck->btsk',canonical['rays_world'][fids.cpu()].to(device),data['rotation'][fids[:,8]])
    control_rays = torch.einsum('btsc,bck->btsk',data['rays_world'][fids],data['rotation'][fids[:,8]])
    assert torch.equal(control_rays,b['rays'])
    policy = json.loads((V7.parent/'side_native_v16/fifth_seal.json').read_text())['policy']
    checkpoint = torch.load(V7.parent/'side_native_v16/consensus/uniform_adaptive/best.pt',weights_only=False,map_location=device)
    torch.manual_seed(202610117)
    base = RecoveryHand3D(policy).to(device).eval()
    base.load_state_dict(checkpoint['model'])
    extra = DualVisualHand3D(policy).to(device).eval()
    mismatch = extra.load_state_dict(checkpoint['model'],strict=False)
    assert not mismatch.unexpected_keys and all(k.startswith('extra_') for k in mismatch.missing_keys)
    extra.initialize_extra()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        p = base.predict_rollout(b,seed=202610114)['xyz_camera_m']
        q = extra.predict_rollout(b,seed=202610114)['xyz_camera_m']
    assert p.shape==(2,20,3) and torch.isfinite(q).all()
    delta = float((p-q).abs().max()*1000)
    assert delta==0, delta
    labels = torch.load(source/'trajectory_labels.pt',weights_only=False,mmap=True)
    target = tuple(labels[k][ids.cpu()].to(device) for k in ['gt','valid','gt_uv','uv_valid'])
    optimizer = torch.optim.AdamW(extra.parameters(),lr=5e-6)
    gradients = []
    for step in range(3):
        extra.train()
        extra.localization.eval()
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(202610117+step)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss = extra.rollout_loss(b,*target,202610117+step)
        assert torch.isfinite(loss)
        loss.backward()
        gain = float(extra.extra_gain.grad.norm())
        project = float(extra.extra_rgb_project.weight.grad.norm())
        attention = float(extra.extra_attention[0].in_proj_weight.grad.norm())
        gradients.append(dict(gain=gain,projection=project,attention=attention))
        assert np.isfinite([gain,project,attention]).all() and gain>0
        if step>0:
            assert project>0 and attention>0
        torch.nn.utils.clip_grad_norm_(extra.parameters(),1.)
        optimizer.step()
    save(root/'preflight.json',dict(passed=True,output_shape=list(q.shape),initial_sampler_max_delta_mm=delta,
        control_world_to_center_rays_exact=True,new_branch_gradients=gradients,
        scope='Engineering parity and actualgradient only; 3preflightupdates discarded; no dev model selection/accuracy claim'))
    print((root/'preflight.json').read_text())

if __name__=='__main__':
    main()

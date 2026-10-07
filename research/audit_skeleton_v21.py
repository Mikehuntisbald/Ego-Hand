"""Check mean-diffusion hand length collapse on development selection only."""
import json
import numpy as np
import torch
from hand3d_v8_common import V7, save, metrics
from hand3d_data_v7 import batch
from hand3d_temporal_v7 import EDGES
from density_model_v13 import DensityTrajectoryHand3D
from adaptive_projection_v14 import apply

def lengths(x):
    return torch.stack([(x[...,v,:]-x[...,u,:]).norm(dim=-1) for u,v in EDGES],-1)

def observed_lengths(b):
    value = lengths(b['xyz'])
    valid = torch.stack([b['available'][...,u]&b['available'][...,v] for u,v in EDGES],-1)
    valid[:,8] = False
    value = value.masked_fill(~valid,float('nan'))
    median = value.nanmedian(1).values
    return torch.where(torch.isfinite(median),median,lengths(b['base']))

def project(proposal, reference, strength):
    points = [proposal[:,j] for j in range(20)]
    for k,(u,v) in enumerate(EDGES):
        direction = proposal[:,v]-proposal[:,u]
        current = direction.norm(dim=-1).clamp_min(1e-6)
        desired = (1-strength)*current + strength*reference[:,k]
        points[v] = points[u] + direction*(desired/current)[:,None]
    return torch.stack(points,1)

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    device = 'cuda:3'
    root = V7.parent/'skeleton_diagnostic_v21'
    root.mkdir(exist_ok=True)
    source = V7.parent/'side_data_v16/consensus'
    raw_data = torch.load(source/'dense_data.pt',weights_only=False,mmap=True)
    data = {k:v.to(device) if torch.is_tensor(v) else v for k,v in raw_data.items()}
    bank = torch.load(source/'native_bank.pt',weights_only=False,mmap=True).to(device)
    risk = torch.load(source/'risk_dense/risk_probabilities.pt',weights_only=False)['joint'].to(device)
    ids = torch.tensor(np.where(np.asarray(data['roles'])=='dev_select')[0],device=device)
    checkpoint = torch.load(V7.parent/'side_native_v16/consensus/uniform_adaptive/best.pt',weights_only=False,map_location=device)
    model = DensityTrajectoryHand3D('dit',True).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    policy = json.loads((V7.parent/'side_native_v16/fifth_seal.json').read_text())['policy']
    proposals, references = [], []
    for start in range(0,len(ids),8):
        ix = ids[start:start+8]
        b = batch(data,ix,risk)
        b['rgb_native'] = bank[data['feature_ids'][ix]]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result = model.predict_rollout(b,seed=202610114+start)
        proposals.append(result['xyz_camera_m'].float())
        references.append(observed_lengths(b))
    proposal, reference = torch.cat(proposals), torch.cat(references)
    base, gt, valid = data['original_base_for_evaluation'][ids],data['gt'][ids],data['valid'][ids]
    edge_valid = torch.stack([valid[:,u]&valid[:,v] for u,v in EDGES],-1)
    target_lengths = lengths(gt)
    bone_diagnostic = {key: float((value-target_lengths).abs()[edge_valid].mean()*1000)
        for key,value in [('original_wilor',lengths(base)),('mean_diffusion',lengths(proposal)),('history_reference',reference)]}
    b = batch(data,ids,risk)
    variants = {}
    nonwrist = valid.clone();nonwrist[:,5] = False
    hard = ((((base-base[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000*nonwrist).sum(-1)/nonwrist.sum(-1).clamp_min(1))>40
    for strength in [0.,.25,.5,1.]:
        candidate = project(proposal,reference,strength) if strength else proposal
        pred = apply(candidate,b,policy)
        variants[str(strength)] = dict(final=metrics(pred,base,gt,valid),raw=metrics(candidate,base,gt,valid),
            hard=metrics(pred[hard],base[hard],gt[hard],valid[hard]))
    save(root/'dev_select_diagnostic.json',dict(complete=True,bone_length_mae_mm=bone_diagnostic,variants=variants,
        scope='Dev_select only. Hypothesis diagnostic, not adopted. Reference uses predicted historical/future hand lengths, excluding current; no GT in reference/projection. No new test/old failures opened. No guarantee biomechanical plausibility or occlusion recovery.'))
    print(json.dumps(dict(bones=bone_diagnostic,variants={k:dict(relative=v['final']['relative_mm'],camera=v['final']['camera_mm'],hard_relative=v['hard']['relative_mm'],hard_recovery=v['hard']['relative_bad_recovered20']) for k,v in variants.items()}),indent=2))

if __name__=='__main__':
    main()

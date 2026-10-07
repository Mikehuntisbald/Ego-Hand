"""Compare historical error probabilities against current-risk broadcast."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from hand3d_v8_common import V7,save
from hand3d_data_v7 import batch

def auroc(y,p):
    order=p.argsort();scores=p[order];label=y[order].double()
    _,counts=torch.unique_consecutive(scores,return_counts=True)
    ends=counts.cumsum(0).double();starts=ends-counts.double()+1
    ranks=((starts+ends)/2).repeat_interleave(counts)
    positive=label.sum();negative=len(label)-positive
    return float(((ranks*label).sum()-positive*(positive+1)/2)/(positive*negative))

def main():
    torch.set_num_threads(4)
    source=V7.parent/'context_data_v17/control';root=V7.parent/'temporal_reliability_v22'
    data=torch.load(source/'dense_data.pt',weights_only=False,mmap=True)
    target=torch.load(source/'trajectory_labels.pt',weights_only=False,mmap=True)
    xyz=torch.cat([batch(data,torch.arange(i,min(i+64,len(data['roles']))))['xyz'] for i in range(0,len(data['roles']),64)])
    available=data['available_bank'][data['feature_ids']]&(data['feature_ids']>0)[...,None]
    valid=target['valid']&available
    assert (target['valid'].any(-1) <= target['valid'][:,:,5]).all(), 'Partial teacher wrist validity requires separate relative mask'
    gt=target['gt']
    camera=(xyz-gt).norm(dim=-1)
    relative=((xyz-xyz[:,:,5:6])-(gt-gt[:,:,5:6])).norm(dim=-1)
    labels=torch.stack([camera>.02,relative>.02],-1).float()
    original=torch.load(source/'risk_dense/risk_probabilities.pt',weights_only=False)
    temporal=torch.load(root/'risk_probabilities.pt',weights_only=False)
    reports={}
    for role in ['train','dev_select','dev_calibrate']:
        ids=torch.tensor(np.where(np.asarray(data['roles'])==role)[0])
        base=original['train_oof' if role=='train' else 'joint'][ids,None].expand(-1,17,-1,-1)
        new=temporal['train_oof' if role=='train' else 'joint'][ids]
        historical=valid[ids].clone();historical[:,8]=False;historical[:,:,5]=False
        reports[role]={}
        for name,probability in [('current_broadcast',base),('perframe',new)]:
            reports[role][name]={}
            for c,key in enumerate(['camera','relative']):
                p=probability[:,:,:,c][historical].clamp(1e-6,1-1e-6)
                y=labels[ids,:,:,c][historical]
                reports[role][name][key]=dict(points=len(y),positive_fraction=float(y.mean()),
                    bce=float(F.binary_cross_entropy(p,y)),brier=float((p-y).square().mean()),
                    auroc=auroc(y,p))
    save(root/'quality_diagnostic.json',dict(complete=True,roles=reports,
        scope='Historical nonwrist finite-labeled predictions only. Train riskheads exclude ownsubject. Dev_select check, dev_calibrate temperature already fitted, so calibration is not independent. Error probability is not anatomical visibility. These are reliability diagnostics, not 3Drecovery results.'))
    print(json.dumps(reports['dev_select'],indent=2))

if __name__=='__main__':main()

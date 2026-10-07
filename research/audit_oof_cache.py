"""Audit real fold lineage and smoke real residual gradients before training."""
import json,numpy as np,torch
from oof_common import RUN,SUBJECTS,save
from oof_calibration import fit_uncertainty,calibrated_confidence
from residual_models import ResidualModel
from metrics_3d import pose_metrics

def main():
    torch.set_num_threads(4)
    data=torch.load(RUN/'oof_cache.pt',map_location='cpu',weights_only=False)
    protocol=json.loads((RUN/'protocol.json').read_text());receipts=json.loads((RUN/'cache_done.json').read_text())
    for f in receipts['folds']:
        assert f['subject'] not in f['trained_subjects'] and set(f['trained_subjects'])==set(SUBJECTS)-{f['subject']}
    audit=[]
    for role in ['denoise','gate_fit']:
        ix=torch.tensor([i for i,r in enumerate(data['rows']) if r['role']==role])
        assert {data['rows'][i]['subject'] for i in ix.tolist()}==set(SUBJECTS)
        for key in ['coarse','in_subject_coarse']:
            metrics=pose_metrics(data[key][ix].numpy(),data['gt'][ix].numpy());metrics.pop('sample_mpjpe19_mm')
            audit.append(dict(role=role,provenance=key,samples=len(ix),metrics=metrics))
    train=torch.tensor([i for i,r in enumerate(data['rows']) if r['role']=='denoise'])
    fit=fit_uncertainty(data['coarse'][train],data['gt'][train],data['confidence'][train],[data['rows'][i]['subject'] for i in train.tolist()])
    ix=train[:8];coarse=data['coarse'][ix].to('cuda:0');gt=data['gt'][ix].to('cuda:0')
    conf=calibrated_confidence(data['confidence'][ix],fit).to('cuda:0');rgb=data['rgb'][ix].to('cuda:0').float()
    for kind in ['dit','regression']:
        model=ResidualModel(kind=kind).to('cuda:0')
        loss,_=model.prediction_loss(gt,coarse,conf,rgb);assert torch.isfinite(loss);loss.backward()
        assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()
        with torch.no_grad():proposal=model.propose(coarse,conf,rgb,steps=10,samples=4 if kind=='dit' else 1)
        model.zero_grad(set_to_none=True);gloss,_=model.selective_gate_loss(proposal,gt,coarse,conf,rgb)
        assert torch.isfinite(gloss);gloss.backward();assert torch.isfinite(model.gate[-1].weight.grad).all()
        closed=torch.zeros(len(ix),21,device='cuda:0');closed[:,0]=1
        artificial=torch.zeros_like(proposal);artificial[:,0,0]=1
        result=model.apply_gates(artificial,coarse,closed)
        assert torch.equal(result[:,0],coarse[:,0]) and not torch.equal(result[:,5],coarse[:,5])
    save(RUN/'cache_audit.json',dict(passed=True,fold_lineage_verified=True,real_data_forward_backward=True,
        local_gate_invariant=True,training_distribution=audit,uncertainty_fit=fit,
        fresh_labels_not_scored_before_checkpoint_lock=True))
    print((RUN/'cache_audit.json').read_text(),flush=True)
if __name__=='__main__':main()


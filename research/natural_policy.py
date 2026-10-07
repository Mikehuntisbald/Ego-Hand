"""Selective correction: reliability triggers a candidate, not an unconditional replacement."""
import torch

def apply_policy(base,candidates,p_bad,available,confirmed,roi,policy,candidate_risk=None):
    assert not (confirmed&~available).any(),'Confirmed point requires a coordinate'
    arm=policy.get('arm','regression');proposal=candidates[arm]
    size=(roi[:,2:]-roi[:,:2]).mean(-1).clamp_min(.01)
    delta=(proposal-base).norm(dim=-1)/size[:,None]
    selected=(p_bad>=policy.get('threshold',1.1))&(delta<=policy.get('max_delta_roi',.5))&~confirmed
    if policy.get('risk_margin') is not None:
        assert candidate_risk is not None,'Candidate reliability must be evaluated'
        selected=selected&((p_bad-candidate_risk[arm])>=policy['risk_margin'])&(candidate_risk[arm]<=policy['proposal_risk_max'])
    if policy.get('agreement_roi') is not None:
        disagreement=(candidates['dit']-candidates['regression']).norm(dim=-1)/size[:,None]
        selected=selected&(disagreement<=policy['agreement_roi'])
    if policy.get('review_only'):selected=torch.zeros_like(selected)
    selected=selected|(~available&~confirmed)
    corrected=base+policy.get('blend',1.)*(proposal-base)
    corrected=torch.where(available[...,None],corrected,proposal)
    result=torch.where(selected[...,None],corrected,base)
    result=torch.where(confirmed[...,None],base,result)
    return result,selected

"""Whole-hand scalar bound: preserve sampled root/pose cancellation and both errors."""
import torch
from hand3d_temporal_v7 import WRIST

def apply(pred,base,policy,confirmed=None):
    delta=(pred-base).float()*policy.get('strength',1.)
    if confirmed is not None:delta=torch.where(confirmed[...,None],0.,delta)
    cap=policy.get('cap_m')
    if cap is not None:
        camera=delta.norm(dim=-1).amax(-1)
        relative=(delta-delta[:,WRIST:WRIST+1]).norm(dim=-1).amax(-1)
        scale=(cap/torch.maximum(camera,relative).clamp_min(1e-9)).clamp_max(1.)
        delta=delta*scale[:,None,None]
    output=base.float()+delta
    if confirmed is not None:output=torch.where(confirmed[...,None],base,output)
    return output

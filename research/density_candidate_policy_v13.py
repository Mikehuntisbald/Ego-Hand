"""Whole-hand large proposals require joint baseline risk and temporal support.

These are fallible, development-calibrated decisions, never visibility truth.
Rejected proposals keep the simultaneous camera/relative9.5mm projection.
"""
import torch
from native_projection_policy_v11 import apply
from density_model_v13 import POLICY

def temporal_reference(b):
    exists=b['available'];dt=b['dt'];B,T,J=exists.shape;index=torch.arange(T,device=dt.device)[None,:,None].expand(B,T,J)
    left=torch.where(exists&(dt[:,:,None]<0),index,-1).amax(1);right=torch.where(exists&(dt[:,:,None]>0),index,T).amin(1);li=left.clamp_min(0);ri=right.clamp_max(T-1);rows=torch.arange(B,device=dt.device)[:,None];joints=torch.arange(J,device=dt.device)[None]
    x=b['xyz'][rows,li,joints];y=b['xyz'][rows,ri,joints];a=dt.gather(1,li);z=dt.gather(1,ri);weight=(-a/(z-a).clamp_min(.001)).clamp(0,1);both=(left>=0)&(right<T)&(a.abs()<=.2)&(z.abs()<=.2)
    reference=x*(1-weight[...,None])+y*weight[...,None];return reference,both

def choose(raw,b,policy):
    base=b['base'];candidate=base+policy['alpha']*(raw-base);pc=b['risk_camera'].amin(-1);mask=torch.ones(20,device=base.device,dtype=torch.bool);mask[5]=False;pr=b['risk_relative'][:,mask].amin(-1)
    eligible=(pc>=policy['camera_risk_min'])&(pr>=policy['relative_risk_min'])&~b['confirmed'].any(-1)&(candidate[:,:,2]>0).all(-1)
    temporal=policy.get('temporal_gain_min_mm')
    if temporal is not None:
        ref,valid=temporal_reference(b);m=valid[:,mask].float();before=(base-ref).norm(dim=-1)+.75*((base-base[:,5:6])-(ref-ref[:,5:6])).norm(dim=-1);after=(candidate-ref).norm(dim=-1)+.75*((candidate-candidate[:,5:6])-(ref-ref[:,5:6])).norm(dim=-1)
        improvement=((before[:,mask]-after[:,mask])*m).sum(-1)/m.sum(-1).clamp_min(1)*1000;eligible&=(m.sum(-1)>=15)&valid[:,5]&(improvement>=temporal)
    fallback=apply(raw,base,POLICY,b['confirmed']);return torch.where(eligible[:,None,None],candidate,fallback),eligible

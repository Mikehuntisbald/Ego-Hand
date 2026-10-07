"""Coupled camera/relative displacement projection, preserving proposal direction."""
import torch
from hand3d_temporal_v7 import WRIST

def ball(x,r):return x/(x.norm(dim=-1,keepdim=True)/max(float(r),1e-9)).clamp_min(1)

def apply_feasible(candidate,b,policy):
    base=b['base'];root_delta=candidate[:,WRIST]-base[:,WRIST]
    pose_delta=(candidate-candidate[:,WRIST:WRIST+1])-(base-base[:,WRIST:WRIST+1])
    root_on=(b['risk_camera'][:,WRIST]>=policy['threshold'])&~b['confirmed'][:,WRIST]&(candidate[:,:,2]>0).all(1)
    pose_on=(b['risk_relative']>=policy['threshold'])&~b['confirmed']&(candidate[:,:,2]>0);pose_on[:,WRIST]=False
    if policy.get('review_only'):root_on.zero_();pose_on.zero_()
    desired=policy['blend']*(root_delta[:,None]*root_on[:,None,None]+pose_delta*pose_on[...,None])
    dr=ball(policy['blend']*root_delta,policy['root_limit_m'])*root_on[:,None]
    # A locked non-wrist point requires zero displacement to remain feasible.
    dr=torch.where(b['confirmed'].any(1)[:,None],ball(dr,min(policy['root_limit_m'],policy['pose_limit_m'])),dr)
    center=dr[:,None].expand_as(desired);x=desired;p=torch.zeros_like(x);q=torch.zeros_like(x)
    camera_radius=.0099;relative_radius=policy['pose_limit_m']
    assert max(policy['root_limit_m'],relative_radius)<=.00990001
    # Dykstra projection onto |delta_j|<=9.9mm AND |delta_j-delta_wrist|<=r.
    for _ in range(32):
        a=x+p;y=ball(a,camera_radius);p=a-y;a=y+q;x=center+ball(a-center,relative_radius);q=a-x
    converged=(x.norm(dim=-1)<=camera_radius+1e-7)&((x-center).norm(dim=-1)<=relative_radius+1e-7)
    x=torch.where(converged[...,None],x,center)
    x[:,WRIST]=dr;x=torch.where(b['confirmed'][...,None],0.,x)
    out=base+x
    assert float(x.norm(dim=-1).max())<=.0100001
    assert float((x-x[:,WRIST:WRIST+1]).norm(dim=-1).max())<=.0100001
    return out,dict(root=root_on,pose=pose_on)

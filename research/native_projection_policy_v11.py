"""Joint camera/relative safe projection for native3D proposals.

Both displacement bounds are enforced simultaneously. The root is selected
once, every joint references that same root, and confirmed XYZ is copied exactly.
This provides the <=10mm to >20mm protection guarantee at cap<=9.5mm.
"""
import torch

def ball(x,r):
    return x*(r/x.norm(dim=-1,keepdim=True).clamp_min(1e-9)).clamp_max(1.)

def apply(raw,base,policy,confirmed=None):
    cap=float(policy['cap_m']);root_cap=float(policy.get('root_cap_m',cap))
    assert 0<root_cap<=cap<=.009500001
    if confirmed is None:confirmed=torch.zeros(base.shape[:2],device=base.device,dtype=torch.bool)
    desired=(raw-base).float()*policy['strength'];root=ball(desired[:,5],root_cap)
    root=torch.where(confirmed[:,5,None],0.,root);center=root[:,None]
    desired_relative=desired-desired[:,5:6];weight=policy['relative_weight']
    target=(desired+weight*(desired_relative+center))/(1+weight)
    x=target;p=torch.zeros_like(x);q=torch.zeros_like(x)
    # Projection on intersection of two balls, using Dykstra corrections.
    for _ in range(32):
        a=x+p;y=ball(a,cap);p=a-y;a=y+q;x=center+ball(a-center,cap);q=a-x
    good=(x.norm(dim=-1)<=cap+1e-7)&((x-center).norm(dim=-1)<=cap+1e-7)
    x=torch.where(good[...,None],x,center.expand_as(x));x[:,5]=root
    x=torch.where(confirmed[...,None],0.,x);output=base.float()+x
    return torch.where(confirmed[...,None],base,output)

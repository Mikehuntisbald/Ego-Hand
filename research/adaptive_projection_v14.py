"""Joint root/finger projection with error-risk-dependent displacement radii.

Absolute and wrist-relative constraints are solved together. Large radii rely
on estimated baseline error and must pass development/frozen harm checks.
"""
import torch

def ball(x,r):return x*(r[...,None]/x.norm(dim=-1,keepdim=True).clamp_min(1e-9)).clamp_max(1.)

def apply(raw,b,policy):
    base=b['base'];large=float(policy['large_cap_m']);small=.0095
    rc=torch.where(b['risk_camera']>=policy['camera_risk_threshold'],large,small)
    rr=torch.where(b['risk_relative']>=policy['relative_risk_threshold'],large,small)
    # Confirmed XYZ is fixed. Its relative displacement still constrains the
    # shared root. Partial-lock inputs remain subject to separate validation.
    rc=torch.where(b['confirmed'],0.,rc);rr=torch.where(b['confirmed'],small,rr)
    desired=(raw-base).float()*policy['strength'];root_cap=torch.minimum(rc[:,5],(rc+rr).amin(-1));root=ball(desired[:,5],root_cap)
    center=root[:,None];relative=desired-desired[:,5:6];weight=float(policy.get('relative_weight',4.));x=(desired+weight*(relative+center))/(1+weight)
    p=torch.zeros_like(x);q=torch.zeros_like(x)
    for _ in range(32):
        a=x+p;y=ball(a,rc);p=a-y;a=y+q;x=center+ball(a-center,rr);q=a-x
    good=(x.norm(dim=-1)<=rc+1e-7)&((x-center).norm(dim=-1)<=rr+1e-7)
    # A guaranteed feasible point on the segment between the two ball centers.
    fallback=root[:,None]*(rc/(rc+rr).clamp_min(1e-9))[...,None]
    x=torch.where(good[...,None],x,fallback);x[:,5]=root;x=torch.where(b['confirmed'][...,None],0.,x)
    output=base.float()+x;return torch.where(b['confirmed'][...,None],base,output)

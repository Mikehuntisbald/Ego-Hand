"""Observation-only sparse trajectory anchors evaluated at dense query times."""
import torch

def trajectory_reference(observation,query_dt,horizon=.5):
    xyz=observation['xyz'].float();time=observation['dt'].float();dt=time[:,None]-query_dt[:,:,None]
    exists=observation['available'].all(-1)&observation['rgb_valid'];mask=exists[:,None]&(dt.abs()<=horizon);mask[:,:,8]=False
    weights=torch.exp(-dt.abs()/(horizon/2))*mask.float()*observation['scores'][:,None].float().clamp_min(.05)
    def fit(w):
        total=w.sum(-1);t=(w*dt).sum(-1);tt=(w*dt.square()).sum(-1);y=torch.einsum('bqt,btjc->bqjc',w,xyz);ty=torch.einsum('bqt,btjc->bqjc',w*dt,xyz);det=total*tt-t.square()
        intercept=(tt[:,:,None,None]*y-t[:,:,None,None]*ty)/det.clamp_min(1e-8)[:,:,None,None];slope=(total[:,:,None,None]*ty-t[:,:,None,None]*y)/det.clamp_min(1e-8)[:,:,None,None]
        mean=y/total.clamp_min(1e-8)[:,:,None,None];good=det>1e-8;return torch.where(good[:,:,None,None],intercept,mean),torch.where(good[:,:,None,None],slope,0.)
    current,slope=fit(weights)
    for _ in range(4):
        fitted=current[:,:,None]+dt[:,:,:,None,None]*slope[:,:,None];error=(xyz[:,None]-fitted).norm(dim=-1).mean(-1)
        scale=error.masked_fill(~mask,float('nan')).nanmedian(-1).values.nan_to_num(.05).clamp_min(.01);factor=(1.5*scale[:,:,None]/error.clamp_min(1e-8)).clamp_max(1.)
        current,slope=fit(weights*factor)
    return current,mask.sum(-1)>=2

def replace_with_anchor(b,reference_observation,blend):
    target,usable=trajectory_reference(reference_observation,b['dt']);original=b['xyz'];anchor=torch.where(usable[:,:,None,None],original+blend*(target-original),original)
    anchor=torch.where(b['available'][...,None],anchor,0.)
    return {**b,'observation_xyz':original,'xyz':anchor}

"""Observation-only robust whole-hand trajectory fits, excluding current frame."""
import torch

def reference(b,horizon=1.,robust=True):
    xyz=b['xyz'].float();dt=b['dt'].float();observed=b['available'].all(-1)&b['rgb_valid'];mask=observed&(dt.abs()<=horizon)&(dt.abs()>1e-5)
    weight=torch.exp(-dt.abs()/max(horizon/2,.05))*mask.float()*b['scores'].float().clamp_min(.05)
    def fit(w):
        total=w.sum(-1);t=(w*dt).sum(-1);tt=(w*dt.square()).sum(-1);y=(w[:,:,None,None]*xyz).sum(1);ty=(w[:,:,None,None]*dt[:,:,None,None]*xyz).sum(1);det=total*tt-t.square()
        intercept=(tt[:,None,None]*y-t[:,None,None]*ty)/det.clamp_min(1e-8)[:,None,None];slope=(total[:,None,None]*ty-t[:,None,None]*y)/det.clamp_min(1e-8)[:,None,None]
        mean=y/total.clamp_min(1e-8)[:,None,None];intercept=torch.where((det>1e-8)[:,None,None],intercept,mean);slope=torch.where((det>1e-8)[:,None,None],slope,0.)
        return intercept,slope
    current,slope=fit(weight)
    if robust:
        for _ in range(4):
            error=(xyz-(current[:,None]+dt[:,:,None,None]*slope[:,None])).norm(dim=-1).mean(-1)
            clipped=error.masked_fill(~mask,float('nan'));scale=clipped.nanmedian(-1).values.nan_to_num(.05).clamp_min(.01);factor=(1.5*scale[:,None]/error.clamp_min(1e-8)).clamp_max(1.)
            current,slope=fit(weight*factor)
    usable=mask.sum(-1)>=2
    return torch.where(usable[:,None,None],current,b['base']),usable

"""Deployment-only diffusion sampling options, selected on development data."""
import torch


def propose(model,b,policy='independent',steps=10,samples=2,generator=None):
    if model.kind=='regression' or policy=='independent':
        return model.propose(b,steps=steps,samples=samples,generator=generator)
    encoded=model.encode(b);n=len(b['coarse']);device=b['coarse'].device
    if policy=='zero':noises=[torch.zeros(n,20,3,device=device)]
    elif policy=='antithetic':
        assert samples%2==0
        noises=[]
        for _ in range(samples//2):
            z=torch.randn(n,20,3,device=device,generator=generator);noises.extend([z,-z])
    else:raise ValueError(policy)
    proposals=[]
    for x in noises:
        schedule=torch.linspace(99,0,steps,device=device).round().long()
        for j,idx in enumerate(schedule):
            _,clean,eps=model.denoise(x,idx.expand(n),encoded);clean=clean.clamp(-6,6)
            if j+1<len(schedule):
                a=model.alpha[schedule[j+1]];x=a.sqrt()*clean+(1-a).sqrt()*eps
            else:x=clean
        proposals.append(x)
    return torch.stack(proposals).mean(0)

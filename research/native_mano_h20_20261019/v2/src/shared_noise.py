"""Sampling-only control: common Gaussian draws across temporal centers.

This reduces Monte Carlo inconsistency. It is NOT a jointly denoised full
trajectory, a new training method, or a guarantee of temporal correctness.
"""
import torch
from native_dit import NativeMANODiT

class SharedNoiseDiT(NativeMANODiT):
    @torch.no_grad()
    def sample(self,b,draws=4,steps=10,seed=190019,zero_rgb=False):
        if self.kind=='regression':return super().sample(b,draws,steps,seed,zero_rgb)
        encoded=self.encode(b,zero_rgb);B=len(b['rgb']);device=b['rgb'].device
        repeated=tuple(x[None].expand(draws,*x.shape).flatten(0,1) for x in encoded)
        generator=torch.Generator(device=device).manual_seed(seed)
        # Each center has the same marginal N(0,I). Only cross-center
        # correlation changes, remaining invariant to inference batch size.
        x=torch.randn(draws,1,17,3,device=device,generator=generator).expand(-1,B,-1,-1).reshape(draws*B,17,3)*self.output_mask
        schedule=torch.linspace(99,0,steps,device=device).round().long()
        for i,t in enumerate(schedule):
            v=self.net(x,t.expand(draws*B),repeated);a=self.alpha[t]
            clean=(a.sqrt()*x-(1-a).sqrt()*v).clamp(-1,1)*self.output_mask
            eps=(1-a).sqrt()*x+a.sqrt()*v
            if i+1==len(schedule):x=clean
            else:
                nxt=self.alpha[schedule[i+1]];x=nxt.sqrt()*clean+(1-nxt).sqrt()*eps
        return x.reshape(draws,B,17,3)

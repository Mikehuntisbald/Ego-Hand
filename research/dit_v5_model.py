"""Diffusion proposal with separate gates along the local viewing axes."""
import torch
from torch import nn
from dit_v3_model import VisualResidual as BaseResidual


class VisualResidual(BaseResidual):
    def __init__(self,kind='dit',width=256,depth=6):
        super().__init__(kind,width,depth)
        self.coherent_gate=False
        self.gate[-1]=nn.Linear(width//2,3)
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-3)

    def activate_gate(self,features):
        logits=self.gate(features)
        if self.coherent_gate:logits=logits.float().mean(1,keepdim=True).expand(-1,20,-1)
        return logits.sigmoid().float()

    def gate_values(self,proposal,b):
        h=self.features(proposal,torch.zeros(len(proposal),device=proposal.device,dtype=torch.long),self.encode(b))
        c=b['coarse']@b['transform'];w=b['wilor']@b['transform']
        other=torch.cat([proposal,(c-c[:,5:6])/.1,b['confidence'][:,1:,None],
            b['confidence'][:,:1,None].expand(-1,20,-1),proposal.norm(dim=-1,keepdim=True),
            ((w-c)/.1).norm(dim=-1,keepdim=True).clamp_max(10)],-1)
        return self.activate_gate(torch.cat([h,other],-1))

    def apply(self,proposal,b,gates):
        if gates.ndim==2:gates=gates[...,None]
        return b['coarse']+(proposal.float()*.05*gates)@b['transform'].transpose(-1,-2)

    def rollout(self,b,steps=10):
        encoded=self.encode(b);n=len(b['coarse']);device=b['coarse'].device
        x=torch.zeros(n,20,3,device=device)
        if self.kind=='regression':
            return self.head(self.features(x,torch.zeros(n,device=device,dtype=torch.long),encoded)).clamp(-6,6)
        schedule=torch.linspace(99,0,steps,device=device).round().long()
        for j,t in enumerate(schedule):
            _,clean,eps=self.denoise(x,t.expand(n),encoded);clean=clean.clamp(-6,6)
            if j+1<len(schedule):
                a=self.alpha[schedule[j+1]];x=a.sqrt()*clean+(1-a).sqrt()*eps
            else:x=clean
        return x

    def rollout_loss(self,b):
        proposal=self.rollout(b)
        p=b['coarse']+self.native_delta(proposal,b);g=b['gt'];c=b['coarse']
        rel=(p-p[:,5:6])-(g-g[:,5:6]);ray=g/g.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        camera=(p-g).norm(dim=-1);relative=rel.norm(dim=-1);axial=(rel*ray).sum(-1).abs()
        mask=b['hard_ray'].float();mask[:,5]=0
        before=(c-g).norm(dim=-1);rb=((c-c[:,5:6])-(g-g[:,5:6])).norm(dim=-1)
        pc=(before<=.01).float();pr=(rb<=.01).float();pr[:,5]=0
        protect=((camera-before-.0005).clamp_min(0)*pc).sum()/pc.sum().clamp_min(1)/.003
        protect+=((relative-rb-.0005).clamp_min(0)*pr).sum()/pr.sum().clamp_min(1)/.003
        diffusion,_=super().prediction_loss(b)
        loss=camera.mean()/.03+relative.mean()/.03+2*(axial*mask).sum()/mask.sum().clamp_min(1)/.03+.3*protect+.1*diffusion
        return loss,dict(camera_mm=float(camera.detach().mean()*1000),relative_mm=float(relative.detach().mean()*1000),axial_mm=float((axial.detach()*mask).sum()/mask.sum().clamp_min(1)*1000))

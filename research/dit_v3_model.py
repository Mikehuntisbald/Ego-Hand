"""Visual/geometry-conditioned point-residual DiT and identical-capacity regressor."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from pose_residual_dit import AdaLNBlock,time_embedding
from export_hand_labels import EDGES


class VisualResidual(nn.Module):
    def __init__(self,kind='dit',width=256,depth=6):
        super().__init__();self.kind=kind;self.width=width
        self.global_projection=nn.Sequential(nn.LayerNorm(1280),nn.Linear(1280,width))
        self.local_projection=nn.Sequential(nn.LayerNorm(1280),nn.Linear(1280,width))
        self.numeric=nn.Sequential(nn.Linear(34,width),nn.SiLU(),nn.Linear(width,width))
        self.noisy=nn.Linear(3,width)
        self.position=nn.Parameter(torch.randn(1,20,width)*.02)
        self.image_position=nn.Parameter(torch.randn(1,48,width)*.02)
        self.time=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList(AdaLNBlock(width,8) for _ in range(depth))
        self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,3)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.gate=nn.Sequential(nn.Linear(width+10,width//2),nn.SiLU(),nn.Linear(width//2,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-3)
        t=torch.linspace(0,1,101,dtype=torch.float64)
        ac=torch.cos((t+.008)/1.008*math.pi/2).square();ac/=ac[0].clone()
        beta=(1-ac[1:]/ac[:-1]).clamp(.0001,.999)
        self.register_buffer('alpha',(1-beta).cumprod(0).float())

    def encode(self,b):
        c=b['coarse']@b['transform'];w=b['wilor']@b['transform']
        confidence=b['confidence'];meta=b['geometry'].clone()
        meta[:,0]/=500;meta[:,4]=meta[:,4].clamp(0,10)/10
        numeric=torch.cat([(c-c[:,5:6])/.1,(w-w[:,5:6])/.1,(w-c)/.1,
                           c[:,5:6].expand(-1,20,-1),w[:,5:6].expand(-1,20,-1),
                           confidence[:,1:,None],confidence[:,:1,None].expand(-1,20,-1),
                           b['wilor_2d'],meta[:,None].expand(-1,20,-1),b['shape'][:,None].expand(-1,20,-1)],-1).clamp(-10,10)
        assert numeric.shape[-1]==34
        joint=self.numeric(numeric)+self.local_projection(b['local'].float())+self.position
        image=self.global_projection(b['global'].float())+self.image_position
        return joint,image,joint.mean(1)+image.mean(1)

    def features(self,x,t,encoded):
        joint,image,condition=encoded
        h=joint+self.noisy(x)
        condition=condition+self.time(time_embedding(t,self.width))
        for block in self.blocks:h=block(h,image,condition)
        return self.norm(h)

    def denoise(self,x,t,encoded):
        v=self.head(self.features(x,t,encoded));a=self.alpha[t][:,None,None]
        clean=a.sqrt()*x-(1-a).sqrt()*v
        eps=(1-a).sqrt()*x+a.sqrt()*v
        return v,clean,eps

    def native_delta(self,proposal,b):
        return (proposal.float()*.05)@b['transform'].transpose(-1,-2)

    def prediction_loss(self,b):
        target=((b['gt']-b['coarse'])@b['transform'])/.05
        encoded=self.encode(b)
        if self.kind=='dit':
            t=torch.randint(100,(len(target),),device=target.device);a=self.alpha[t][:,None,None]
            noise=torch.randn_like(target);x=a.sqrt()*target+(1-a).sqrt()*noise
            v,clean,_=self.denoise(x,t,encoded)
            desired=a.sqrt()*noise-(1-a).sqrt()*target
            per_joint=(v-desired).square().mean(-1)
            valid=(self.alpha[t]>.1).float()[:,None]
        else:
            t=torch.zeros(len(target),device=target.device,dtype=torch.long)
            clean=self.head(self.features(torch.zeros_like(target),t,encoded))
            per_joint=(clean-target).square().mean(-1);valid=torch.ones_like(per_joint[:,:1])
        before=(b['coarse']-b['gt']).norm(dim=-1)
        rel_before=((b['coarse']-b['coarse'][:,5:6])-(b['gt']-b['gt'][:,5:6])).norm(dim=-1)
        weights=1+2*(rel_before>.02).float()+(before>.05).float()
        if 'hard_ray' in b:weights=weights+4*b['hard_ray'].float()
        diffusion=(per_joint*weights).sum()/weights.sum()
        reconstruction=(F.smooth_l1_loss(clean,target,reduction='none').mean(-1)*valid).sum()/(valid.sum().clamp_min(1)*20)
        full=b['coarse']+self.native_delta(clean,b)
        after=(full-b['gt']).norm(dim=-1)
        rel_after=((full-full[:,5:6])-(b['gt']-b['gt'][:,5:6])).norm(dim=-1)
        precise=(before<=.01).float()*valid;precise_rel=(rel_before<=.01).float()*valid;precise_rel[:,5]=0
        protection=((after-before-.0005).clamp_min(0)*precise).sum()/precise.sum().clamp_min(1)/.01
        protection_rel=((rel_after-rel_before-.0005).clamp_min(0)*precise_rel).sum()/precise_rel.sum().clamp_min(1)/.01
        u,v=zip(*EDGES)
        bones=(full[:,u]-full[:,v]).norm(dim=-1);gt_bones=(b['gt'][:,u]-b['gt'][:,v]).norm(dim=-1)
        bone=(F.smooth_l1_loss(bones/.03,gt_bones/.03,reduction='none')*valid).sum()/(valid.sum().clamp_min(1)*len(u))
        loss=diffusion+.2*reconstruction+.1*bone+.1*(protection+protection_rel)
        if 'hard_ray' in b:
            ray=b['gt']/b['gt'].norm(dim=-1,keepdim=True).clamp_min(1e-8)
            rel_error=(full-full[:,5:6])-(b['gt']-b['gt'][:,5:6])
            axial=(rel_error*ray).sum(-1)/.03
            weight=b['hard_ray'].float()*valid
            loss=loss+.25*(F.smooth_l1_loss(axial,torch.zeros_like(axial),reduction='none')*weight).sum()/weight.sum().clamp_min(1)
        return loss,dict(diffusion=float(diffusion.detach()),reconstruction=float(reconstruction.detach()),protection=float(protection.detach()))

    def propose(self,b,steps=10,samples=2,generator=None):
        encoded=self.encode(b);n=len(b['coarse'])
        if self.kind=='regression':
            x=torch.zeros(n,20,3,device=b['coarse'].device)
            t=torch.zeros(n,device=x.device,dtype=torch.long)
            return self.head(self.features(x,t,encoded)).clamp(-6,6)
        proposals=[]
        for _ in range(samples):
            x=torch.randn((n,20,3),device=b['coarse'].device,generator=generator)
            schedule=torch.linspace(99,0,steps,device=x.device).round().long()
            for j,idx in enumerate(schedule):
                _,clean,eps=self.denoise(x,idx.expand(n),encoded);clean=clean.clamp(-6,6)
                if j+1<len(schedule):
                    a=self.alpha[schedule[j+1]];x=a.sqrt()*clean+(1-a).sqrt()*eps
                else:x=clean
            proposals.append(x)
        return torch.stack(proposals).mean(0)

    def gate_values(self,proposal,b):
        h=self.features(proposal,torch.zeros(len(proposal),device=proposal.device,dtype=torch.long),self.encode(b))
        c=b['coarse']@b['transform'];w=b['wilor']@b['transform']
        other=torch.cat([proposal,(c-c[:,5:6])/.1,b['confidence'][:,1:,None],
                         b['confidence'][:,:1,None].expand(-1,20,-1),proposal.norm(dim=-1,keepdim=True),
                         ((w-c)/.1).norm(dim=-1,keepdim=True).clamp_max(10)],-1)
        return self.gate(torch.cat([h,other],-1)).squeeze(-1).sigmoid()

    def apply(self,proposal,b,gates):
        return b['coarse']+gates[...,None]*self.native_delta(proposal,b)

    def gate_loss(self,proposal,b):
        proposal=proposal.detach();delta=self.native_delta(proposal,b)
        desired=b['gt']-b['coarse']
        optimal=(desired*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-8)
        optimal=optimal.clamp(0,1).detach()
        gate=self.gate_values(proposal,b)
        pred=self.apply(proposal,b,gate)
        before=(b['coarse']-b['gt']).norm(dim=-1);after=(pred-b['gt']).norm(dim=-1)
        rb=((b['coarse']-b['coarse'][:,5:6])-(b['gt']-b['gt'][:,5:6])).norm(dim=-1)
        ra=((pred-pred[:,5:6])-(b['gt']-b['gt'][:,5:6])).norm(dim=-1)
        pc=(before<=.01).float();pr=(rb<=.01).float();pr[:,5]=0
        protect=((after-before-.0005).clamp_min(0)*pc).sum()/pc.sum().clamp_min(1)/.003
        protect_rel=((ra-rb-.0005).clamp_min(0)*pr).sum()/pr.sum().clamp_min(1)/.003
        loss=after.mean()/.03+.7*ra.mean()/.03+.2*F.mse_loss(gate,optimal)+3*protect+3*protect_rel
        return loss,dict(camera_mm=float(after.detach().mean()*1000),relative_mm=float(ra.detach().mean()*1000),protection=float(protect.detach()))

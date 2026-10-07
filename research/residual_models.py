"""Actual training refiners: v-prediction diffusion and matched direct-regression control."""
import math,torch
from torch import nn
from torch.nn import functional as F
from pose_residual_dit import AdaLNBlock,time_embedding

class ResidualModel(nn.Module):
    def __init__(self,kind='dit',width=192,depth=4):
        super().__init__();self.kind=kind;self.width=width
        self.noisy=nn.Linear(3,width);self.coarse=nn.Linear(4,width);self.rgb=nn.Linear(128,width);self.rgbpos=nn.Linear(2,width)
        self.jointpos=nn.Parameter(torch.randn(1,21,width)*.02)
        self.time=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList(AdaLNBlock(width,6) for _ in range(depth))
        self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,3)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.gate=nn.Sequential(nn.Linear(width+6,width),nn.SiLU(),nn.Linear(width,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-4)
        u=torch.arange(101,dtype=torch.float64)/100;ac=torch.cos((u+.008)/1.008*math.pi/2).square();ac/=ac[0].clone()
        beta=(1-ac[1:]/ac[:-1]).clamp(.0001,.999);self.register_buffer('alpha',(1-beta).cumprod(0).float())
        scale=torch.full((1,21,1),.03);scale[:,:1]=.1;self.register_buffer('scale',scale)
        mask=torch.ones(1,21,1);mask[:,6]=0;self.register_buffer('mask',mask)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,8),torch.linspace(-1,1,8),indexing='ij')
        self.register_buffer('xy',torch.stack([xx,yy],-1).reshape(1,64,2))
    def pack(self,xyz):return torch.cat([xyz[:,5:6],xyz-xyz[:,5:6]],1)/self.scale
    def unpack(self,p):return p[:,:1]*.1+p[:,1:]*.03
    def features(self,x,t,coarse,confidence,rgb):
        c=self.pack(coarse);emb=self.coarse(torch.cat([c,confidence[...,None]],-1))
        image=self.rgb(rgb)+self.rgbpos(self.xy)
        cond=self.time(time_embedding(t,self.width))+emb.mean(1)+image.mean(1)
        h=self.noisy(x)+emb+self.jointpos
        for block in self.blocks:h=block(h,image,cond)
        return self.norm(h)
    def denoise(self,x,t,coarse,confidence,rgb):
        h=self.features(x,t,coarse,confidence,rgb);v=self.head(h)*self.mask
        a=self.alpha[t][:,None,None]
        clean=(a.sqrt()*x-(1-a).sqrt()*v)*self.mask
        eps=(1-a).sqrt()*x+a.sqrt()*v
        return v,clean,eps
    def prediction_loss(self,gt,coarse,confidence,rgb):
        target=(self.pack(gt)-self.pack(coarse))*self.mask
        if self.kind=='regression':
            t=torch.zeros(len(gt),device=gt.device,dtype=torch.long)
            h=self.features(torch.zeros_like(target),t,coarse,confidence,rgb)
            clean=self.head(h)*self.mask
            loss=((clean-target).square()*self.mask).sum()/(len(gt)*20*3)
        else:
            t=torch.randint(100,(len(gt),),device=gt.device);a=self.alpha[t][:,None,None];noise=torch.randn_like(target)*self.mask
            x=a.sqrt()*target+(1-a).sqrt()*noise
            v,clean,_=self.denoise(x,t,coarse,confidence,rgb)
            desired=a.sqrt()*noise-(1-a).sqrt()*target
            loss=((v-desired).square()*self.mask).sum()/(len(gt)*20*3)
        # Low-noise reconstruction is a label loss, not gate supervision.
        if self.kind=='dit':valid=(self.alpha[t]>.1).float()[:,None,None]
        else:valid=torch.ones(len(gt),1,1,device=gt.device)
        rec=((clean-target).abs()*self.mask*valid).sum()/(valid.sum().clamp_min(1)*20*3)
        return loss+.05*rec,dict(prediction=float(loss.detach()),reconstruction=float(rec.detach()))
    def propose(self,coarse,confidence,rgb,steps=10,generator=None,samples=1):
        if self.kind=='regression':
            x=torch.zeros(len(coarse),21,3,device=coarse.device);t=torch.zeros(len(coarse),device=x.device,dtype=torch.long)
            return self.head(self.features(x,t,coarse,confidence,rgb))*self.mask
        proposals=[]
        for _ in range(samples):
            x=torch.randn((len(coarse),21,3),device=coarse.device,generator=generator)*self.mask
            schedule=torch.linspace(99,0,steps,device=x.device).round().long()
            for i,idx in enumerate(schedule):
                t=idx.expand(len(x));_,clean,eps=self.denoise(x,t,coarse,confidence,rgb)
                if i+1<len(schedule):a=self.alpha[schedule[i+1]];x=a.sqrt()*clean+(1-a).sqrt()*eps
                else:x=clean
            proposals.append(x)
        return torch.stack(proposals).mean(0)
    def gates(self,proposal,coarse,confidence,rgb):
        t=torch.zeros(len(coarse),device=coarse.device,dtype=torch.long)
        h=self.features(proposal,t,coarse,confidence,rgb)
        logits=self.gate(torch.cat([h,self.pack(coarse),proposal],-1)).squeeze(-1)
        return logits,logits.sigmoid()*self.mask.squeeze(-1)
    def apply_gates(self,proposal,coarse,gates):
        # Root correction is part of the proposal, but never bypasses a joint's
        # local gate. Wrist uses the dedicated root gate.
        point_gates=gates[:,1:].clone();point_gates[:,5]=gates[:,0]
        return coarse+point_gates[...,None]*self.unpack(proposal)
    def selective_gate_loss(self,proposal,gt,coarse,confidence,rgb):
        proposal=proposal.detach();logits,g=self.gates(proposal,coarse,confidence,rgb)
        before=(coarse-gt).norm(dim=-1);full=coarse+self.unpack(proposal)
        after_full=(full-gt).norm(dim=-1)
        beneficial=(after_full+.001<before).float()
        targets=torch.cat([beneficial[:,5:6],beneficial],1)
        bce=(F.binary_cross_entropy_with_logits(logits,targets,reduction='none')*self.mask.squeeze(-1)).sum()/(len(gt)*20)
        refined=self.apply_gates(proposal,coarse,g)
        after=(refined-gt).norm(dim=-1)
        before_rel=((coarse-coarse[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)
        after_rel=((refined-refined[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)
        precise=before<=.010;precise_rel=(before_rel<=.010);precise_rel[:,5]=False
        harm=(after-before-.0002).clamp_min(0)
        harm_rel=(after_rel-before_rel-.0002).clamp_min(0)
        protect=(harm*precise).sum()/precise.sum().clamp_min(1)/.003
        protect_rel=(harm_rel*precise_rel).sum()/precise_rel.sum().clamp_min(1)/.003
        loss=after.mean()/.03+.5*after_rel.mean()/.03+.1*bce+3*protect+2*protect_rel
        return loss,dict(camera_mm=float(after.detach().mean()*1000),relative_mm=float(after_rel.detach().mean()*1000),
                         protect=float(protect.detach()),protect_relative=float(protect_rel.detach()),bce=float(bce.detach()))
    def gate_loss(self,proposal,gt,coarse,confidence,rgb):
        # proposal must come from actual inference, never q(GT,t).
        proposal=proposal.detach();target=(self.pack(gt)-self.pack(coarse))*self.mask
        before=(target*self.scale).norm(dim=-1);after=((target-proposal)*self.scale).norm(dim=-1)
        beneficial=(after+.001<before).float();logits,g=self.gates(proposal,coarse,confidence,rgb)
        mask=self.mask.squeeze(-1)
        bce=(F.binary_cross_entropy_with_logits(logits,beneficial,reduction='none')*mask).sum()/(len(gt)*20)
        refined=self.unpack(self.pack(coarse)+g[...,None]*proposal)
        pose=(refined-gt).norm(dim=-1).mean()/.03
        protection=((refined-gt).norm(dim=-1)-(coarse-gt).norm(dim=-1)).clamp_min(0).mean()/.03
        loss=.1*bce+pose+.5*protection
        return loss,dict(bce=float(bce.detach()),pose=float(pose.detach()),protection=float(protection.detach()))

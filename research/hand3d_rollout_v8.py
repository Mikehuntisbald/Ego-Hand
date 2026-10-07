"""Differentiable deployed sampling and coherent camera-space proposal trust."""
import torch
from torch import nn
from torch.nn import functional as F
from hand3d_temporal_v7 import TemporalHand3D,WRIST,pack,unpack,EDGES

def project_fisheye624(xyz,params):
    xyz=xyz.float();params=params.float();r=xyz[...,:2].square().sum(-1).clamp_min(1e-16).sqrt()
    angle=torch.atan2(r,xyz[...,2]);p=xyz[...,:2]*(angle/r)[...,None];r2=p.square().sum(-1).clamp_max(torch.pi**2)
    k=params[:,None,4:];power=r2;radial=torch.ones_like(r2)
    for j in range(6):radial=radial+k[:,:,j]*power;power=power*r2
    x=p[...,0]*radial;y=p[...,1]*radial;xx=x.square();yy=y.square();xy=x*y;rr=xx+yy
    xo=x+2*k[:,:,7]*xy+k[:,:,6]*(rr+2*xx)+k[:,:,8]*rr+k[:,:,9]*rr.square()
    yo=y+2*k[:,:,6]*xy+k[:,:,7]*(rr+2*yy)+k[:,:,10]*rr+k[:,:,11]*rr.square()
    return torch.stack([xo,yo],-1)*params[:,None,:2]+params[:,None,2:4]

class RolloutHand3D(TemporalHand3D):
    def __init__(self,*args,cap_m=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.cap_m=cap_m
        self.coherent_trust=nn.Sequential(nn.Linear(self.width+126,128),nn.SiLU(),nn.Linear(128,1))
        nn.init.zeros_(self.coherent_trust[-1].weight);nn.init.constant_(self.coherent_trust[-1].bias,-2)

    def sample_delta(self,b,encoded,steps=10,draws=4,seed=202610081):
        B=len(b['base']);device=b['base'].device;gen=torch.Generator(device=device).manual_seed(seed)
        if self.kind=='regression':
            return self.net(torch.zeros(B,21,3,device=device),torch.zeros(B,device=device,dtype=torch.long),encoded)[None]
        joint,memory,c,heat=encoded
        def repeat(x):return x[None].expand(draws,*x.shape).flatten(0,1)
        enc=(repeat(joint),repeat(memory),repeat(c),heat)
        x=torch.randn(draws*B,21,3,device=device,generator=gen)*self.token_mask
        schedule=torch.linspace(99,0,steps,device=device).round().long()
        for j,t in enumerate(schedule):
            v=self.net(x,t.expand(draws*B),enc);a=self.alpha[t]
            clean=(a.sqrt()*x-(1-a).sqrt()*v).clamp(-8,8)*self.token_mask
            eps=(1-a).sqrt()*x+a.sqrt()*v
            x=clean if j+1==len(schedule) else self.alpha[schedule[j+1]].sqrt()*clean+(1-self.alpha[schedule[j+1]]).sqrt()*eps
        return x.reshape(draws,B,21,3)

    def trust(self,proposal,b,encoded):
        # One coefficient for the full camera-space residual preserves root/pose cancellation.
        delta=(proposal-b['base'])/.05;pose=(proposal-proposal[:,WRIST:WRIST+1])-(b['base']-b['base'][:,WRIST:WRIST+1])
        risk=torch.stack([b['risk_camera'].mean(1),b['risk_relative'].mean(1),b['risk_camera'].amin(1),b['risk_relative'].amin(1),b['risk_camera'][:,WRIST],b['scores'][:,8]],-1)
        features=torch.cat([encoded[0].mean(1),delta.flatten(1), (pose/.05).flatten(1),risk],-1)
        return self.coherent_trust(features).sigmoid()

    def proposal(self,b,steps=10,draws=4,seed=202610081):
        encoded=self.encode(b);delta=self.sample_delta(b,encoded,steps,draws,seed)
        poses=unpack(pack(b['base'])[None]+delta);raw=poses.mean(0);confidence=self.trust(raw,b,encoded)
        xyz=b['base']+confidence[...,None]*(raw-b['base'])
        xyz=torch.where(b['confirmed'][...,None],b['base'],xyz)
        if self.cap_m is not None:
            from bounded_policy_v8 import apply
            xyz=apply(xyz,b['base'],dict(cap_m=self.cap_m,strength=1.),b['confirmed'])
        return dict(xyz_camera_m=xyz,raw_xyz_camera_m=raw,std_m=poses.std(0,unbiased=False),trust=confidence,draws=poses),encoded

    @torch.no_grad()
    def predict(self,b,steps=10,samples=4,seed=202610081):return self.proposal(b,steps,samples,seed)[0]

    def loss(self,b,gt,valid,gt_uv,uv_valid,camera_params,seed,mode='rollout'):
        if mode=='rollout' or self.kind=='regression':
            output,encoded=self.proposal(b,10,4,seed)
        elif mode=='one_step':
            encoded=self.encode(b);base=b['base'];residual=(pack(gt)-pack(base))*self.token_mask
            gen=torch.Generator(device=gt.device).manual_seed(seed)
            t=torch.randint(100,(len(gt),),device=gt.device,generator=gen);a=self.alpha[t][:,None,None]
            noise=torch.randn(residual.shape,device=gt.device,generator=gen)*self.token_mask
            x=a.sqrt()*residual+(1-a).sqrt()*noise;v=self.net(x,t,encoded)
            delta=(a.sqrt()*x-(1-a).sqrt()*v).clamp(-8,8)*self.token_mask
            raw=unpack(pack(base)+delta);confidence=self.trust(raw,b,encoded)
            pred=base+confidence[...,None]*(raw-base);pred=torch.where(b['confirmed'][...,None],base,pred)
            output=dict(raw_xyz_camera_m=raw,xyz_camera_m=pred,trust=confidence)
        else:raise ValueError(mode)
        raw=output['raw_xyz_camera_m'];pred=output['xyz_camera_m'];base=b['base']
        canonical=valid.clone();canonical[:,WRIST]=False;mask=canonical.float();count=mask.sum().clamp_min(1)
        def errors(x):
            return (x-gt).norm(dim=-1),((x-x[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])).norm(dim=-1)
        be,br=errors(base);re,rr=errors(raw);pe,pr=errors(pred)
        good_camera=canonical&(be<=.01);good_relative=canonical&(br<=.01)
        def protect(e,r):
            return ((e-be-.001).clamp_min(0)/.005*good_camera).sum()/good_camera.sum().clamp_min(1)+((r-br-.001).clamp_min(0)/.005*good_relative).sum()/good_relative.sum().clamp_min(1)
        raw_fit=((re/.03+.75*rr/.03)*mask).sum()/count
        final_fit=((pe/.03+.75*pr/.03)*mask).sum()/count
        raw_protect=protect(re,rr);final_protect=protect(pe,pr)
        # Training-only target alpha is assessed on the complete, actually sampled proposal.
        with torch.no_grad():
            strengths=torch.tensor([0,.05,.1,.2,.35,.5,.75,1.],device=gt.device)
            xyz=base[None]+strengths[:,None,None,None]*(raw.detach()-base)[None]
            ce=(xyz-gt[None]).norm(dim=-1);rel=((xyz-xyz[:,:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])[None]).norm(dim=-1)
            score=((ce+.75*rel)*mask[None]).sum(-1)/mask.sum(-1).clamp_min(1)[None]
            harm_c=((ce>.02)&good_camera[None]).any(-1);harm_r=((rel>.02)&good_relative[None]).any(-1)
            score=score.masked_fill(harm_c|harm_r,float('inf'));best=score.argmin(0);target=strengths[best,None]
        gate_loss=F.smooth_l1_loss(output['trust'],target,beta=.1)
        # Diffusion prior regularizes the deployed sampler; GT is confined to this training target.
        residual=(pack(gt)-pack(base))*self.token_mask;tv=torch.cat([valid[:,WRIST:WRIST+1],valid],1).float()*self.token_mask[...,0]
        t=torch.randint(100,(len(gt),),device=gt.device);a=self.alpha[t][:,None,None];noise=torch.randn_like(residual)*self.token_mask
        if self.kind=='dit':
            v=self.net(a.sqrt()*residual+(1-a).sqrt()*noise,t,encoded);desired=a.sqrt()*noise-(1-a).sqrt()*residual
            denoise=((v-desired).square().sum(-1)*tv).sum()/tv.sum().clamp_min(1)
        else:denoise=raw.sum()*0
        uv=project_fisheye624(pred,camera_params)/1408
        reproj=(F.smooth_l1_loss((uv-gt_uv)*50,torch.zeros_like(uv),reduction='none',beta=.5).sum(-1)*uv_valid).sum()/uv_valid.sum().clamp_min(1)
        bone=raw.sum()*0
        for u,v in EDGES:
            m=(valid[:,u]&valid[:,v]).float();de=((pred[:,u]-pred[:,v]).norm(dim=-1)-(gt[:,u]-gt[:,v]).norm(dim=-1)).abs()/.01
            bone+=(de*m).sum()/m.sum().clamp_min(1)/len(EDGES)
        positive=((.05-pred[...,2]).clamp_min(0)/.03*valid).sum()/valid.sum().clamp_min(1)
        loss=.45*raw_fit+final_fit+.3*raw_protect+2*final_protect+.5*gate_loss+.1*denoise+.05*reproj+.025*bone+.1*positive
        if self.use_rgb:loss+=.025*self.visual_head.loss(encoded[3],gt_uv,uv_valid,b['positions'][:,8],b['roi'][:,8])
        return loss,dict(raw_fit=float(raw_fit.detach()),final_fit=float(final_fit.detach()),protect=float(final_protect.detach()),gate=float(output['trust'].detach().mean()))

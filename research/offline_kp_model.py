"""Offline 2D keypoint inpainting; only coordinates, masks and times are inputs."""
import math
import torch
from torch import nn
from pose_residual_dit import AdaLNBlock,time_embedding

WINDOW=17
CHAINS=[[6,7,0],[8,9,10,1],[11,12,13,2],[14,15,16,3],[17,18,19,4]]

def condition(xy,observed,dt):
    """Erase hidden values BEFORE all normalization/interpolation/encoding."""
    observed=observed.bool()&torch.isfinite(xy).all(-1)
    xy=torch.where(observed[...,None],xy,torch.zeros_like(xy))
    b,s,j,_=xy.shape;idx=torch.arange(s,device=xy.device)[None,:,None].expand(b,s,j)
    left=torch.where(observed&(dt[:,:,None]<=0),idx,torch.full_like(idx,-1)).amax(1)
    right=torch.where(observed&(dt[:,:,None]>=0),idx,torch.full_like(idx,s)).amin(1)
    lp=left.clamp_min(0);rp=right.clamp_max(s-1)
    rows=torch.arange(b,device=xy.device)[:,None];joints=torch.arange(j,device=xy.device)[None]
    a=xy[rows,lp,joints];z=xy[rows,rp,joints]
    ta=dt.gather(1,lp);tz=dt.gather(1,rp)
    left_ok=left>=0;right_ok=right<s;both=left_ok&right_ok
    any_ok=left_ok|right_ok
    weight=(-ta/(tz-ta).clamp_min(1e-6)).clamp(0,1)
    linear=a*(1-weight[...,None])+z*weight[...,None]
    nearest=torch.where((left_ok&(~right_ok|(ta.abs()<=tz.abs())))[...,None],a,z)
    linear=torch.where(both[...,None],linear,nearest)
    linear=torch.where(any_ok[...,None],linear,torch.full_like(linear,.5))
    previous=torch.where(left_ok[...,None],a,nearest)
    previous=torch.where(any_ok[...,None],previous,torch.full_like(previous,.5))
    missing=~observed[:,s//2]
    return dict(xy=xy,observed=observed,dt=dt,linear=linear,nearest=nearest,previous=previous,
                missing=missing,has_context=any_ok,both_sides=both)

def artificial_gap(xy,observed,dt,lengths,fingers):
    # fingers: Bx5 booleans; all five also masks the wrist (whole observation gap).
    b,s,j,_=xy.shape;selected=torch.zeros(b,j,device=xy.device,dtype=torch.bool)
    for fi,chain in enumerate(CHAINS):selected[:,chain]=fingers[:,fi,None]
    selected[:,5]=fingers.all(-1)
    start=s//2-(lengths-1)//2
    span=(torch.arange(s,device=xy.device)[None]>=start[:,None])&(torch.arange(s,device=xy.device)[None]<(start+lengths)[:,None])
    hidden=span[:,:,None]&selected[:,None]
    return condition(xy,observed&~hidden,dt),selected

class KeypointCompleter(nn.Module):
    def __init__(self,kind='dit',width=128,depth=3):
        super().__init__();self.kind=kind;self.width=width
        self.local=nn.Sequential(nn.Linear(WINDOW*4+6,width),nn.SiLU(),nn.Linear(width,width))
        self.frame=nn.Sequential(nn.Linear(61,width),nn.SiLU(),nn.Linear(width,width))
        self.position=nn.Parameter(torch.randn(1,20,width)*.02)
        self.noisy=nn.Linear(2,width);self.time=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList(AdaLNBlock(width,4) for _ in range(depth))
        self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,2)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        t=torch.linspace(0,1,101,dtype=torch.float64);ac=torch.cos((t+.008)/1.008*math.pi/2).square();ac/=ac[0].clone()
        beta=(1-ac[1:]/ac[:-1]).clamp(.0001,.999);self.register_buffer('alpha',(1-beta).cumprod(0).float())

    def encode(self,b):
        xy=b['xy'];mask=b['observed'];base=b['linear'];dt=b['dt']
        relative=torch.where(mask[...,None],(xy-base[:,None])/.1,torch.zeros_like(xy)).clamp(-10,10)
        per=torch.cat([relative,mask[...,None].float(),dt[:,:,None,None].expand(-1,-1,20,1)],-1).permute(0,2,1,3).flatten(2)
        center=base.mean(1,keepdim=True)
        extras=torch.cat([(base-center)/.1,center.expand(-1,20,-1),b['missing'][...,None].float(),b['both_sides'][...,None].float()],-1)
        joint=self.local(torch.cat([per,extras],-1))+self.position
        frame_xy=torch.where(mask[...,None],(xy-center[:,None])/.1,torch.zeros_like(xy)).clamp(-10,10)
        frame=self.frame(torch.cat([frame_xy.flatten(2),mask.float(),dt[...,None]],-1))
        return joint,frame,joint.mean(1)+frame.mean(1)

    def net(self,x,t,encoded):
        joint,frame,c=encoded;h=joint+self.noisy(x);c=c+self.time(time_embedding(t,self.width))
        for block in self.blocks:h=block(h,frame,c)
        return self.head(self.norm(h))

    def loss(self,b,gt,valid):
        hidden=b['missing']&valid;target=torch.where(hidden[...,None],(gt-b['linear'])/.1,0.)
        encoded=self.encode(b)
        if self.kind=='dit':
            t=torch.randint(100,(len(gt),),device=gt.device);a=self.alpha[t][:,None,None]
            noise=torch.randn_like(target)*b['missing'][...,None]
            x=a.sqrt()*target+(1-a).sqrt()*noise
            v=self.net(x,t,encoded);desired=a.sqrt()*noise-(1-a).sqrt()*target
            clean=a.sqrt()*x-(1-a).sqrt()*v
            err=(v-desired).square().sum(-1)
            rec=((clean-target).square().sum(-1)*(.1+self.alpha[t][:,None]))
            loss=((err+.2*rec)*hidden).sum()/hidden.sum().clamp_min(1)
        else:
            clean=self.net(torch.zeros_like(target),torch.zeros(len(gt),device=gt.device,dtype=torch.long),encoded)
            loss=((clean-target).square().sum(-1)*hidden).sum()/hidden.sum().clamp_min(1)
        return loss

    @torch.inference_mode()
    def predict(self,b,steps=10,samples=4,seed=20261003):
        encoded=self.encode(b);n=len(b['xy']);device=b['xy'].device;draws=[]
        generator=torch.Generator(device=device).manual_seed(seed)
        if self.kind=='regression':samples=1
        for si in range(samples):
            if self.kind=='regression':
                x=self.net(torch.zeros(n,20,2,device=device),torch.zeros(n,device=device,dtype=torch.long),encoded)
            else:
                x=torch.randn(n,20,2,device=device,generator=generator)*b['missing'][...,None]
                schedule=torch.linspace(99,0,steps,device=device).round().long()
                for k,t in enumerate(schedule):
                    v=self.net(x,t.expand(n),encoded);a=self.alpha[t]
                    clean=(a.sqrt()*x-(1-a).sqrt()*v).clamp(-4,4)
                    eps=(1-a).sqrt()*x+a.sqrt()*v
                    x=clean if k+1==len(schedule) else self.alpha[schedule[k+1]].sqrt()*clean+(1-self.alpha[schedule[k+1]]).sqrt()*eps
                    x=x*b['missing'][...,None]
            prediction=b['linear']+x.float()*.1
            prediction=torch.where(b['missing'][...,None],prediction,b['xy'][:,WINDOW//2])
            draws.append(prediction)
        draws=torch.stack(draws)
        mean=torch.where(b['missing'][...,None],draws.mean(0),b['xy'][:,WINDOW//2])
        return dict(xy=mean,samples=draws,std=draws.std(0,unbiased=False),has_context=b['has_context'])

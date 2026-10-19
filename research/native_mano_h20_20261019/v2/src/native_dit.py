"""RGB-conditioned MANO Lie-residual DiT with explicit temporal attention."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def time_embed(t,width):
    f=torch.exp(-math.log(10000)*torch.arange(width//2,device=t.device)/(width//2-1))
    a=t.float()[:,None]*f[None];return torch.cat([a.sin(),a.cos()],-1)

class DenoiseBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.norms=nn.ModuleList([nn.LayerNorm(width,elementwise_affine=False) for _ in range(3)])
        self.self_attn=nn.MultiheadAttention(width,heads,batch_first=True)
        self.cross_attn=nn.MultiheadAttention(width,heads,batch_first=True)
        self.mlp=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
        self.condition=nn.Sequential(nn.SiLU(),nn.Linear(width,width*9))
        nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)
    def forward(self,h,memory,condition,padding):
        s1,b1,g1,s2,b2,g2,s3,b3,g3=self.condition(condition).chunk(9,-1)
        q=self.norms[0](h)*(1+s1[:,None])+b1[:,None]
        h=h+g1[:,None]*self.self_attn(q,q,q,need_weights=False)[0]
        q=self.norms[1](h)*(1+s2[:,None])+b2[:,None]
        h=h+g2[:,None]*self.cross_attn(q,memory,memory,key_padding_mask=padding,need_weights=False)[0]
        q=self.norms[2](h)*(1+s3[:,None])+b3[:,None]
        return h+g3[:,None]*self.mlp(q)

class NativeMANODiT(nn.Module):
    def __init__(self,kind='dit',width=128,blocks=4,heads=4,schedule='linear'):
        super().__init__();self.kind=kind;self.width=width;self.schedule=schedule
        self.rgb_stem=nn.Sequential(nn.LayerNorm(1280),nn.Linear(1280,width))
        self.patch_pos=nn.Parameter(torch.randn(1,1,192,width)*.02)
        self.parameter_pos=nn.Parameter(torch.randn(1,1,17,width)*.02)
        self.coarse=nn.Linear(6,width);self.noisy=nn.Linear(3,width)
        self.frame_info=nn.Sequential(nn.Linear(5,width),nn.SiLU(),nn.Linear(width,width))
        self.spatial=nn.MultiheadAttention(width,heads,batch_first=True)
        self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,heads,width*4,dropout=0.,batch_first=True,norm_first=True),2,enable_nested_tensor=False)
        self.time=nn.Sequential(nn.Linear(width,width*4),nn.SiLU(),nn.Linear(width*4,width))
        self.blocks=nn.ModuleList([DenoiseBlock(width,heads) for _ in range(blocks)])
        self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,3)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        if schedule=='linear':alpha=torch.cumprod(1-torch.linspace(.0001,.02,100),0)
        elif schedule=='cosine':
            # v-prediction needs a terminal training distribution compatible
            # with pure Gaussian sampling, unlike the old alpha[-1] = .364.
            grid=torch.linspace(0,1,101,dtype=torch.float64)
            curve=torch.cos((grid+.008)/1.008*math.pi/2).square()
            alpha=(curve[1:]/curve[0]).clamp_min(1e-8).float()
        else:raise ValueError(schedule)
        self.register_buffer('alpha',alpha)
        mask=torch.ones(17,3);mask[0]=0;self.register_buffer('output_mask',mask)

    def encode(self,b,zero_rgb=False):
        rgb=b['rgb'].float();B,T,S,_=rgb.shape
        rgb=self.rgb_stem(torch.zeros_like(rgb) if zero_rgb else rgb)+self.patch_pos
        # Coarse rotation columns and root vector are aligned into the center camera.
        coarse=b['coarse_tokens'];h=self.coarse(coarse)+self.parameter_pos
        info=torch.stack([b['dt'],b['dt'].abs(),b['scores'],b['right_window'].float(),b['valid'].float()],-1)
        h=h+self.frame_info(info)[:,:,None]
        q=h.flatten(0,1);r=rgb.flatten(0,1)
        h=(q+self.spatial(q,r,r,need_weights=False)[0]).reshape(B,T,17,self.width)
        padding=~b['valid'].bool()
        temporal=h.permute(0,2,1,3).reshape(B*17,T,self.width)
        temporal=self.temporal(temporal,src_key_padding_mask=padding[:,None].expand(-1,17,-1).reshape(B*17,T))
        h=temporal.reshape(B,17,T,self.width).permute(0,2,1,3)
        memory=torch.cat([h.flatten(1,2),rgb.flatten(1,2)],1)
        memory_padding=torch.cat([padding[:,:,None].expand(-1,-1,17).flatten(1),padding[:,:,None].expand(-1,-1,192).flatten(1)],1)
        context=(h.mean(2)*b['valid'][:,:,None]).sum(1)/b['valid'].sum(1,keepdim=True).clamp_min(1)
        return h[:,8],memory,context,memory_padding

    def net(self,x,t,encoded):
        center,memory,context,padding=encoded
        h=center+self.noisy(x);c=context+self.time(time_embed(t,self.width))
        for block in self.blocks:h=block(h,memory,c,padding)
        return self.head(self.norm(h))*self.output_mask

    def training_prediction(self,b,target,generator):
        encoded=self.encode(b);B=len(target)
        if self.kind=='regression':
            clean=self.net(torch.zeros_like(target),torch.zeros(B,device=target.device,dtype=torch.long),encoded)
            return clean,clean.sum()*0
        t=torch.randint(0,100,(B,),device=target.device,generator=generator)
        noise=torch.randn(target.shape,device=target.device,generator=generator)*self.output_mask
        alpha=self.alpha[t,None,None]
        x=alpha.sqrt()*target+(1-alpha).sqrt()*noise
        v=self.net(x,t,encoded);true_v=alpha.sqrt()*noise-(1-alpha).sqrt()*target
        clean=alpha.sqrt()*x-(1-alpha).sqrt()*v
        loss=((v-true_v).square()*self.output_mask).sum()/max(1,B*48)
        return clean,loss

    @torch.no_grad()
    def sample(self,b,draws=4,steps=10,seed=190019,zero_rgb=False):
        encoded=self.encode(b,zero_rgb);B=len(b['rgb']);device=b['rgb'].device
        if self.kind=='regression':return self.net(torch.zeros(B,17,3,device=device),torch.zeros(B,device=device,dtype=torch.long),encoded)[None]
        repeated=tuple(x[None].expand(draws,*x.shape).flatten(0,1) for x in encoded)
        g=torch.Generator(device=device).manual_seed(seed)
        x=torch.randn(draws*B,17,3,device=device,generator=g)*self.output_mask
        schedule=torch.linspace(99,0,steps,device=device).round().long()
        for i,t in enumerate(schedule):
            v=self.net(x,t.expand(draws*B),repeated);a=self.alpha[t]
            clean=(a.sqrt()*x-(1-a).sqrt()*v).clamp(-1,1)*self.output_mask
            eps=(1-a).sqrt()*x+a.sqrt()*v
            if i+1==len(schedule):x=clean
            else:
                nxt=self.alpha[schedule[i+1]];x=nxt.sqrt()*clean+(1-nxt).sqrt()*eps
        return x.reshape(draws,B,17,3)

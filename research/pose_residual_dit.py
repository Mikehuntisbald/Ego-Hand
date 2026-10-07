"""Untrained 3D residual DiT prototype; not a complete trained pose pipeline.

Inputs: predicted coarse camera XYZ in meters, predicted confidences, RGB feature
tokens. GT and the subject's GT hand shape never enter inference conditioning.
Separate root and wrist-relative corrections prevent translation from masking
finger-pose error. Adaptation of the AdaLN-Zero conditioning idea from DiT.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

def time_embedding(t, width):
    frequencies=torch.exp(-math.log(10000)*torch.arange(width//2,device=t.device)/max(1,width//2))
    angle=t[:,None].float()*frequencies[None]
    return torch.cat([angle.cos(),angle.sin()],dim=-1)

def modulate(x, shift, scale):return x*(1+scale[:,None])+shift[:,None]

class AdaLNBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__()
        self.norms=nn.ModuleList(nn.LayerNorm(width,elementwise_affine=False) for _ in range(3))
        self.self_attention=nn.MultiheadAttention(width,heads,batch_first=True)
        self.cross_attention=nn.MultiheadAttention(width,heads,batch_first=True)
        self.mlp=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
        self.condition=nn.Sequential(nn.SiLU(),nn.Linear(width,9*width))
        nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)
    def forward(self,x,rgb,condition,rgb_mask=None):
        s1,c1,g1,s2,c2,g2,s3,c3,g3=self.condition(condition).chunk(9,dim=-1)
        h=modulate(self.norms[0](x),s1,c1)
        x=x+g1[:,None]*self.self_attention(h,h,h,need_weights=False)[0]
        h=modulate(self.norms[1](x),s2,c2)
        x=x+g2[:,None]*self.cross_attention(h,rgb,rgb,need_weights=False,key_padding_mask=rgb_mask)[0]
        h=modulate(self.norms[2](x),s3,c3)
        return x+g3[:,None]*self.mlp(h)

class PoseResidualDiT(nn.Module):
    def __init__(self,rgb_dim=256,width=192,depth=4,heads=6,joints=20,wrist=5,
                 diffusion_steps=100,root_scale_m=.10,relative_scale_m=.03):
        super().__init__()
        assert width%2==0 and width%heads==0 and 0<=wrist<joints
        self.width=width;self.joints=joints;self.wrist=wrist;self.steps=diffusion_steps
        self.root_scale=root_scale_m;self.relative_scale=relative_scale_m
        self.noisy_projection=nn.Linear(3,width)
        self.coarse_projection=nn.Linear(4,width)
        self.joint_embedding=nn.Parameter(torch.randn(1,joints+1,width)*.02)
        self.rgb_projection=nn.Linear(rgb_dim,width)
        self.rgb_position_projection=nn.Linear(2,width)
        self.time_mlp=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList(AdaLNBlock(width,heads) for _ in range(depth))
        self.output_norm=nn.LayerNorm(width)
        self.noise_head=nn.Linear(width,3)
        self.gate_head=nn.Sequential(nn.Linear(width+6,width//2),nn.SiLU(),nn.Linear(width//2,1))
        nn.init.zeros_(self.noise_head.weight);nn.init.zeros_(self.noise_head.bias)
        nn.init.zeros_(self.gate_head[-1].weight);nn.init.constant_(self.gate_head[-1].bias,-4)
        # Cosine cumulative alpha schedule; training and DDIM use the same buffer.
        s=.008;u=torch.arange(diffusion_steps+1,dtype=torch.float64)/diffusion_steps
        cumulative=torch.cos((u+s)/(1+s)*math.pi/2).square();cumulative/=cumulative[0].clone()
        beta=(1-cumulative[1:]/cumulative[:-1]).clamp(.0001,.999)
        self.register_buffer('alpha_bar',(1-beta).cumprod(0).float())
        mask=torch.ones(1,joints+1,1);mask[:,wrist+1]=0
        self.register_buffer('token_mask',mask)

    def pack(self,xyz):
        assert xyz.shape[-2:]==(self.joints,3)
        root=xyz[:,self.wrist:self.wrist+1]
        return torch.cat([root/self.root_scale,(xyz-root)/self.relative_scale],dim=1)

    def unpack(self,packed):
        return packed[:,:1]*self.root_scale+packed[:,1:]*self.relative_scale

    def forward(self,noisy,t,coarse_xyz,coarse_confidence,rgb_tokens,rgb_xy=None):
        coarse=self.pack(coarse_xyz)
        assert coarse_confidence.shape==coarse_xyz.shape[:2]
        confidence=torch.cat([coarse_confidence.mean(dim=1,keepdim=True),coarse_confidence],dim=1)
        embedded=self.coarse_projection(torch.cat([coarse,confidence[...,None]],dim=-1))
        if rgb_xy is None:
            side=math.isqrt(rgb_tokens.shape[1])
            assert side*side==rgb_tokens.shape[1],'Pass normalized RGB token XY for non-square feature grids'
            axis=torch.linspace(-1,1,side,device=rgb_tokens.device,dtype=rgb_tokens.dtype)
            yy,xx=torch.meshgrid(axis,axis,indexing='ij')
            rgb_xy=torch.stack([xx,yy],dim=-1).reshape(1,-1,2)
        rgb=self.rgb_projection(rgb_tokens)+self.rgb_position_projection(rgb_xy)
        condition=self.time_mlp(time_embedding(t,self.width))+rgb.mean(dim=1)+embedded.mean(dim=1)
        h=self.noisy_projection(noisy)+embedded+self.joint_embedding
        for block in self.blocks:h=block(h,rgb,condition)
        h=self.output_norm(h)
        eps=self.noise_head(h)*self.token_mask
        a=self.alpha_bar[t][:,None,None]
        clean=(noisy-(1-a).sqrt()*eps)/a.sqrt()
        clean=clean*self.token_mask
        logits=self.gate_head(torch.cat([h,coarse,clean],dim=-1)).squeeze(-1)
        return eps,clean,logits

    def loss(self,gt_xyz,coarse_xyz,coarse_confidence,rgb_tokens,valid_joints=None,
             gate_weight=.05,pose_weight=.2,improvement_margin_m=.001,rgb_xy=None):
        # Baseline remains frozen during the first residual training stage.
        coarse_xyz=coarse_xyz.detach();coarse_confidence=coarse_confidence.detach();rgb_tokens=rgb_tokens.detach()
        b=gt_xyz.shape[0];t=torch.randint(self.steps,(b,),device=gt_xyz.device)
        residual=(self.pack(gt_xyz)-self.pack(coarse_xyz))*self.token_mask
        noise=torch.randn_like(residual)*self.token_mask
        a=self.alpha_bar[t][:,None,None];noisy=a.sqrt()*residual+(1-a).sqrt()*noise
        eps,clean,logits=self(noisy,t,coarse_xyz,coarse_confidence,rgb_tokens,rgb_xy)
        if valid_joints is None:valid_joints=torch.ones((b,self.joints),device=gt_xyz.device,dtype=torch.bool)
        token_valid=torch.cat([valid_joints[:,self.wrist:self.wrist+1],valid_joints],dim=1)[...,None]*self.token_mask
        count=token_valid.sum().clamp_min(1)
        diffusion=((eps-noise).square()*token_valid).sum()/(3*count)
        scale=torch.full((1,self.joints+1,1),self.relative_scale,device=gt_xyz.device);scale[:,:1]=self.root_scale
        before=(residual*scale).norm(dim=-1)
        after=((residual-clean.detach())*scale).norm(dim=-1)
        gate_target=(after+improvement_margin_m<before).to(logits.dtype)
        # At nearly pure noise, x0 estimates are numerically large and are not
        # proposals that will reach the inference gate. Learn gate/pose only at
        # usable signal levels; epsilon denoising still learns at every step.
        low_noise=(a[:,0,0]>.1).to(gt_xyz.dtype)
        gate_valid=token_valid.squeeze(-1)*low_noise[:,None]
        gate=(F.binary_cross_entropy_with_logits(logits,gate_target,reduction='none')*gate_valid).sum()/gate_valid.sum().clamp_min(1)
        gated=logits.sigmoid()[...,None]*clean*self.token_mask
        refined=self.unpack(self.pack(coarse_xyz)+gated)
        pose_valid=valid_joints*low_noise[:,None]
        pose=((refined-gt_xyz).norm(dim=-1)*pose_valid).sum()/pose_valid.sum().clamp_min(1)/self.relative_scale
        total=diffusion+gate_weight*gate+pose_weight*pose
        return dict(loss=total,diffusion=diffusion,gate=gate,pose=pose,
                    beneficial_correction_fraction=(gate_target*token_valid.squeeze(-1)).sum()/count,
                    refined_xyz_m=refined,gate_target=gate_target)

    @torch.no_grad()
    def sample(self,coarse_xyz,coarse_confidence,rgb_tokens,sampling_steps=10,residual_strength=1.,generator=None,rgb_xy=None):
        if residual_strength==0:
            return dict(xyz_camera_m=coarse_xyz.clone(),gates=torch.zeros(coarse_xyz.shape[0],self.joints+1,device=coarse_xyz.device))
        assert 1<=sampling_steps<=self.steps
        x=torch.randn((coarse_xyz.shape[0],self.joints+1,3),device=coarse_xyz.device,dtype=coarse_xyz.dtype,generator=generator)*self.token_mask
        schedule=torch.linspace(self.steps-1,0,sampling_steps,device=x.device).round().long()
        for i,index in enumerate(schedule):
            t=index.expand(x.shape[0]);eps,clean,_=self(x,t,coarse_xyz,coarse_confidence,rgb_tokens,rgb_xy)
            if i+1<len(schedule):
                a_next=self.alpha_bar[schedule[i+1]]
                x=a_next.sqrt()*clean+(1-a_next).sqrt()*eps
            else:x=clean
        # Confidence gate sees the final proposal at t=0, matching its training
        # interpretation; do not reuse detection confidence as pose accuracy.
        t=torch.zeros(x.shape[0],device=x.device,dtype=torch.long)
        _,_,logits=self(self.alpha_bar[0].sqrt()*x,t,coarse_xyz,coarse_confidence,rgb_tokens,rgb_xy)
        gates=logits.sigmoid()*self.token_mask.squeeze(-1)
        refined=self.unpack(self.pack(coarse_xyz)+residual_strength*gates[...,None]*x)
        return dict(xyz_camera_m=refined,gates=gates,delta_packed=x)

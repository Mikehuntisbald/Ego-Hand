"""Natural offline 3D completion: aligned XYZ + full spatial RGB, root/pose split."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from pose_residual_dit import AdaLNBlock,time_embedding
from spatial_rgb_model import SpatialHead
WRIST=5
CHAINS=[[5,6,7,0],[5,8,9,10,1],[5,11,12,13,2],[5,14,15,16,3],[5,17,18,19,4]]
EDGES=[(a,b) for chain in CHAINS for a,b in zip(chain[:-1],chain[1:])]

def pack(xyz):
    root=xyz[...,WRIST:WRIST+1,:]
    return torch.cat([root/.1,(xyz-root)/.03],dim=-2)

def unpack(x):return x[...,:1,:]*.1+x[...,1:,:]*.03

class TemporalHand3D(nn.Module):
    def __init__(self,kind='regression',use_rgb=True,width=192,depth=4):
        super().__init__();self.kind=kind;self.use_rgb=use_rgb;self.width=width
        self.local=nn.Sequential(nn.Linear(17*6+8,width),nn.SiLU(),nn.Linear(width,width))
        self.frame=nn.Sequential(nn.Linear(20*5+4,width),nn.SiLU(),nn.Linear(width,width))
        self.base_embed=nn.Sequential(nn.Linear(5,width),nn.SiLU(),nn.Linear(width,width))
        self.joint_position=nn.Parameter(torch.randn(1,21,width)*.02)
        self.risk_embed=nn.Sequential(nn.Linear(3,width),nn.SiLU(),nn.Linear(width,width))
        self.visual_head=SpatialHead();self.visual_head.project=nn.Identity();self.visual_head.spatial=nn.Identity()
        self.rgb_norm=nn.LayerNorm(128);self.rgb_project=nn.Linear(128,width)
        self.rgb_position=nn.Sequential(nn.Linear(9,width),nn.SiLU(),nn.Linear(width,width))
        self.visual_motion=nn.Sequential(nn.Linear(17*4,width),nn.SiLU(),nn.Linear(width,width))
        self.visual_attention=nn.MultiheadAttention(width,6,batch_first=True)
        self.fusion=nn.Linear(2*width,width)
        self.noisy=nn.Linear(3,width);self.time=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList(AdaLNBlock(width,6) for _ in range(depth));self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,3)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        t=torch.linspace(0,1,101,dtype=torch.float64);ac=torch.cos((t+.008)/1.008*math.pi/2).square();ac/=ac[0].clone();beta=(1-ac[1:]/ac[:-1]).clamp(.0001,.999)
        self.register_buffer('alpha',(1-beta).cumprod(0).float())
        mask=torch.ones(1,21,1);mask[:,WRIST+1]=0;self.register_buffer('token_mask',mask)

    def encode(self,b):
        xyz=b['xyz'];exists=b['available'];base=b['base'];dt=b['dt'];B=len(xyz)
        bp=pack(base);tp=pack(xyz);mask=torch.cat([exists[:,:,WRIST:WRIST+1],exists],dim=2)
        score=b['scores'][:,:,None].expand(-1,-1,21);risk=torch.cat([b['risk_camera'][:,WRIST:WRIST+1],b['risk_relative']],1)
        delta=(tp-bp[:,None]).clamp(-10,10)*mask[...,None]
        per=torch.cat([delta,mask[...,None].float(),score[...,None],dt[:,:,None,None].expand(-1,-1,21,1)],-1).permute(0,2,1,3).flatten(2)
        conf=torch.cat([b['confirmed'][:,WRIST:WRIST+1],b['confirmed']],1)
        center_mask=mask[:,8]
        extras=torch.cat([bp.clamp(-10,10),center_mask[...,None].float(),conf[...,None].float(),risk[...,None],b['roi'][:,8,None,:2].expand(-1,21,-1)],-1)
        joint=self.local(torch.cat([per,extras],-1))+self.joint_position
        center=base[:,WRIST:WRIST+1]
        rel=(xyz-center[:,None])/.1;rel=rel.clamp(-10,10)*exists[...,None]
        frame=self.frame(torch.cat([rel.flatten(2),exists.float(),b['scores'][:,:,None].expand(-1,-1,20),dt[...,None],b['camera_origin']/.1],-1))
        joint=joint+self.base_embed(torch.cat([bp.clamp(-10,10),risk[...,None],center_mask[...,None].float()],-1))
        relative_risk=torch.cat([b['risk_camera'][:,WRIST:WRIST+1],b['risk_relative']],1)
        camera_risk=torch.cat([b['risk_camera'][:,WRIST:WRIST+1],b['risk_camera']],1)
        joint=joint+self.risk_embed(torch.stack([camera_risk,relative_risk,center_mask.float()],-1))
        current_heat=None
        if self.use_rgb:
            B,T,S,C=b['rgb'].shape;x=b['rgb'].float().reshape(B*T,S,C).transpose(1,2).reshape(B*T,C,16,12)
            heat=self.visual_head.decode(x,b['positions'].reshape(B*T,S,2),b['roi'].reshape(B*T,4))
            heat={k:v.reshape(B,T,*v.shape[1:]) for k,v in heat.items()};current_heat={k:v[:,8] for k,v in heat.items()}
            meta=torch.cat([b['positions'],b['rays'],b['camera_origin'][:,:,None].expand(-1,-1,192,-1),dt[:,:,None,None].expand(-1,-1,192,1)],-1)
            vision=(self.rgb_project(self.rgb_norm(b['rgb'].float()))+self.rgb_position(meta))*b['rgb_valid'][:,:,None,None]
            per_joint=torch.einsum('btjs,btsc->btjc',heat['probability'],vision)
            entropy=-(heat['probability']*heat['probability'].clamp_min(1e-8).log()).sum(-1)/math.log(192)
            motion=torch.cat([((heat['xy']-b['xy'][:,8,None])/.1).clamp(-10,10),entropy[...,None],b['rgb_valid'][:,:,None,None].expand(-1,-1,20,1)],-1)
            motion=motion*b['rgb_valid'][:,:,None,None]
            q=per_joint[:,8]+self.visual_motion(motion.permute(0,2,1,3).flatten(2))
            q=torch.cat([q.mean(1,keepdim=True),q],1)+self.joint_position
            padding=(~b['rgb_valid'])[:,:,None].expand(-1,-1,192).flatten(1);cells=vision.flatten(1,2)
            assert b['rgb_valid'].any(1).all()
            visual=q+self.visual_attention(q,cells,cells,key_padding_mask=padding,need_weights=False)[0]
            joint=self.fusion(torch.cat([joint,visual],-1))
            memory=torch.cat([frame,per_joint.flatten(1,2)],1)
        else:memory=frame
        return joint,memory,joint.mean(1)+frame.mean(1),current_heat

    def net(self,x,t,encoded):
        joint,memory,c,_=encoded;h=joint+self.noisy(x);c=c+self.time(time_embedding(t,self.width))
        for block in self.blocks:h=block(h,memory,c)
        return self.head(self.norm(h))*self.token_mask

    def loss(self,b,gt,valid,gt_uv,uv_valid):
        tv=torch.cat([valid[:,WRIST:WRIST+1],valid],1).float()*self.token_mask[...,0]
        residual=(pack(gt)-pack(b['base']))*self.token_mask
        camera_error=(gt-b['base']).norm(dim=-1)
        relative_error=((gt-gt[:,WRIST:WRIST+1])-(b['base']-b['base'][:,WRIST:WRIST+1])).norm(dim=-1)
        weights=tv*(1+3*torch.cat([(camera_error[:,WRIST:WRIST+1]>.02),(relative_error>.02)],1).float())
        encoded=self.encode(b)
        if self.kind=='dit':
            t=torch.randint(100,(len(gt),),device=gt.device);a=self.alpha[t][:,None,None];noise=torch.randn_like(residual)*self.token_mask
            x=a.sqrt()*residual+(1-a).sqrt()*noise;v=self.net(x,t,encoded)
            desired=a.sqrt()*noise-(1-a).sqrt()*residual;clean=a.sqrt()*x-(1-a).sqrt()*v
            error=(v-desired).square().sum(-1)+.3*(clean-residual).square().sum(-1)*(.1+self.alpha[t][:,None])
            usable=(self.alpha[t]>.05).float()
        else:
            clean=self.net(torch.zeros_like(residual),torch.zeros(len(gt),device=gt.device,dtype=torch.long),encoded)
            error=F.smooth_l1_loss(clean,residual,reduction='none',beta=.5).sum(-1);usable=torch.ones(len(gt),device=gt.device)
        prediction=unpack(pack(b['base'])+clean)
        camloss=F.smooth_l1_loss((prediction-gt)/.03,torch.zeros_like(gt),reduction='none',beta=.5).sum(-1)
        pose=(prediction-prediction[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])
        relloss=F.smooth_l1_loss(pose/.03,torch.zeros_like(pose),reduction='none',beta=.5).sum(-1)
        geom_mask=valid.float()*usable[:,None]
        geometry=((camloss+.5*relloss)*geom_mask).sum()/geom_mask.sum().clamp_min(1)
        bone_loss=prediction.sum()*0
        for u,v in EDGES:
            m=valid[:,u]&valid[:,v];m=m.float()*usable
            bone=((prediction[:,u]-prediction[:,v]).norm(dim=-1)-(gt[:,u]-gt[:,v]).norm(dim=-1)).abs()/.01
            bone_loss=bone_loss+(bone*m).sum()/m.sum().clamp_min(1)/len(EDGES)
        loss=(error*weights).sum()/weights.sum().clamp_min(1)+.3*geometry+.05*bone_loss
        if self.use_rgb:loss=loss+.025*self.visual_head.loss(encoded[3],gt_uv,uv_valid,b['positions'][:,8],b['roi'][:,8])
        return loss

    @torch.no_grad()
    def predict(self,b,steps=10,samples=4,seed=202610071):
        encoded=self.encode(b);B=len(b['base']);device=b['base'].device;gen=torch.Generator(device=device).manual_seed(seed);draws=[]
        for si in range(samples if self.kind=='dit' else 1):
            if self.kind=='regression':delta=self.net(torch.zeros(B,21,3,device=device),torch.zeros(B,dtype=torch.long,device=device),encoded)
            else:
                x=torch.randn(B,21,3,device=device,generator=gen)*self.token_mask;schedule=torch.linspace(99,0,steps,device=device).round().long()
                for k,t in enumerate(schedule):
                    v=self.net(x,t.expand(B),encoded);a=self.alpha[t];clean=(a.sqrt()*x-(1-a).sqrt()*v).clamp(-8,8)*self.token_mask;eps=(1-a).sqrt()*x+a.sqrt()*v
                    x=clean if k+1==len(schedule) else self.alpha[schedule[k+1]].sqrt()*clean+(1-self.alpha[schedule[k+1]]).sqrt()*eps
                delta=x
            xyz=unpack(pack(b['base'])+delta);xyz=torch.where(b['confirmed'][...,None],b['base'],xyz);draws.append(xyz)
        draw=torch.stack(draws);return dict(xyz_camera_m=draw.mean(0),std_m=draw.std(0,unbiased=False),draws=draw)

"""Bidirectional RGB-conditioned diffusion/regression of an entire3D trajectory."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from hand3d_temporal_v7 import pack,unpack,WRIST,EDGES
from pose_residual_dit import AdaLNBlock,time_embedding
from spatial_rgb_model import SpatialHead

class TrajectoryHand3D(nn.Module):
    def __init__(self,kind='dit',width=192,depth=4):
        super().__init__();self.kind=kind;self.width=width
        self.observation=nn.Sequential(nn.Linear(17,width),nn.SiLU(),nn.Linear(width,width))
        self.joint_position=nn.Parameter(torch.randn(1,1,21,width)*.02)
        self.timestamp=nn.Sequential(nn.Linear(1,width),nn.SiLU(),nn.Linear(width,width))
        self.visual_head=SpatialHead();self.visual_head.project=nn.Identity();self.visual_head.spatial=nn.Identity()
        self.rgb_norm=nn.LayerNorm(128);self.rgb_project=nn.Linear(128,width);self.rgb_position=nn.Sequential(nn.Linear(9,width),nn.SiLU(),nn.Linear(width,width))
        self.spatial_attention=nn.MultiheadAttention(width,6,batch_first=True);self.fusion=nn.Linear(width*2,width)
        self.noisy=nn.Linear(3,width);self.time=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width));self.blocks=nn.ModuleList([AdaLNBlock(width,6) for _ in range(depth)]);self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,3)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        t=torch.linspace(0,1,101,dtype=torch.float64);ac=torch.cos((t+.008)/1.008*math.pi/2).square();ac/=ac[0].clone();beta=(1-ac[1:]/ac[:-1]).clamp(.0001,.999);self.register_buffer('alpha',(1-beta).cumprod(0).float())
        mask=torch.ones(1,1,21,1);mask[:,:,WRIST+1]=0;self.register_buffer('token_mask',mask)

    def encode(self,b):
        B,T,S,C=b['rgb'].shape;xyz=b['xyz'];exists=b['available'];bp=pack(b['base'])[:,None];tp=pack(xyz);available=torch.cat([exists[:,:,WRIST:WRIST+1],exists],2)
        uv=torch.cat([b['xy'].mean(2,keepdim=True),b['xy']],2);uvvalid=torch.cat([b['observed_2d'][:,:,WRIST:WRIST+1],b['observed_2d']],2)
        pc=torch.cat([b['risk_camera'][:,WRIST:WRIST+1],b['risk_camera']],1)[:,None].expand(-1,T,-1);pr=torch.cat([b['risk_camera'][:,WRIST:WRIST+1],b['risk_relative']],1)[:,None].expand(-1,T,-1)
        features=torch.cat([((tp-bp).clamp(-10,10)*available[...,None]),bp.clamp(-10,10).expand(-1,T,-1,-1),available[...,None].float(),b['scores'][:,:,None,None].expand(-1,-1,21,1),b['dt'][:,:,None,None].expand(-1,-1,21,1),b['camera_origin'][:,:,None].expand(-1,-1,21,-1)/.1,pc[...,None],pr[...,None],uv,uvvalid[...,None].float()],-1)
        h=self.observation(features)+self.joint_position+self.timestamp(b['dt'][:,:,None,None])
        x=b['rgb'].float().reshape(B*T,S,C).transpose(1,2).reshape(B*T,C,16,12);heat=self.visual_head.decode(x,b['positions'].reshape(B*T,S,2),b['roi'].reshape(B*T,4));heat={k:v.reshape(B,T,*v.shape[1:]) for k,v in heat.items()}
        meta=torch.cat([b['positions'],b['rays'],b['camera_origin'][:,:,None].expand(-1,-1,192,-1),b['dt'][:,:,None,None].expand(-1,-1,192,1)],-1)
        vision_rgb=b.get('rgb_native',b['rgb'])
        vision=(self.rgb_project(self.rgb_norm(vision_rgb.float()))+self.rgb_position(meta))*b['rgb_valid'][:,:,None,None]
        local=torch.einsum('btjs,btsc->btjc',heat['probability'],vision);query=torch.cat([local.mean(2,keepdim=True),local],2)+self.joint_position
        cells=vision.reshape(B*T,S,self.width);q=query.reshape(B*T,21,self.width);spatial=q+self.spatial_attention(q,cells,cells,need_weights=False)[0]
        h=self.fusion(torch.cat([h,spatial.reshape(B,T,21,self.width)],-1));memory=torch.cat([h.flatten(1,2)*b['rgb_valid'][:,:,None,None].expand(-1,-1,21,1).flatten(1,2),vision.flatten(1,2)],1)
        return h.flatten(1,2),memory,h[:,8].mean(1),heat

    def net(self,x,t,encoded):
        joint,memory,c,_=encoded;h=joint+self.noisy(x.flatten(1,2));c=c+self.time(time_embedding(t,self.width))
        for block in self.blocks:h=block(h,memory,c)
        return self.head(self.norm(h)).reshape(len(x),17,21,3)*self.token_mask

    def sample_trajectory(self,b,encoded,steps=10,draws=4,seed=202610093):
        B=len(b['base']);device=b['base'].device;gen=torch.Generator(device=device).manual_seed(seed)
        if self.kind=='regression':return self.net(torch.zeros(B,17,21,3,device=device),torch.zeros(B,device=device,dtype=torch.long),encoded)[None]
        joint,memory,c,heat=encoded
        def repeat(x):return x[None].expand(draws,*x.shape).flatten(0,1)
        enc=repeat(joint),repeat(memory),repeat(c),heat;x=torch.randn(draws*B,17,21,3,device=device,generator=gen)*self.token_mask;schedule=torch.linspace(99,0,steps,device=device).round().long()
        for j,t in enumerate(schedule):
            v=self.net(x,t.expand(draws*B),enc);a=self.alpha[t];clean=(a.sqrt()*x-(1-a).sqrt()*v).clamp(-8,8)*self.token_mask;eps=(1-a).sqrt()*x+a.sqrt()*v
            x=clean if j+1==len(schedule) else self.alpha[schedule[j+1]].sqrt()*clean+(1-self.alpha[schedule[j+1]]).sqrt()*eps
        return x.reshape(draws,B,17,21,3)

    def rollout_loss(self,b,gt,valid,gt_uv,uv_valid,seed):
        from bounded_policy_v8 import apply
        encoded=self.encode(b);delta=self.sample_trajectory(b,encoded,10,4,seed);draw=unpack(pack(b['xyz'])[None]+delta);raw=draw.mean(0);pred=apply(raw[:,8],b['base'],dict(cap_m=.0095,strength=.5),b['confirmed'])
        canonical=valid.clone();canonical[:,:,WRIST]=False;m=canonical.float()
        def errors(x,g):return (x-g).norm(dim=-1),((x-x[...,WRIST:WRIST+1,:])-(g-g[...,WRIST:WRIST+1,:])).norm(dim=-1)
        ce,re=errors(raw,gt);pe,pr=errors(pred,gt[:,8]);be,br=errors(b['base'],gt[:,8]);cm=m[:,8]
        fit=((ce/.03+.75*re/.03)*m).sum()/m.sum().clamp_min(1);final=((pe/.03+.75*pr/.03)*cm).sum()/cm.sum().clamp_min(1)
        goodc=canonical[:,8]&(be<=.01);goodr=canonical[:,8]&(br<=.01);protect=((pe-be-.001).clamp_min(0)/.005*goodc).sum()/goodc.sum().clamp_min(1)+((pr-br-.001).clamp_min(0)/.005*goodr).sum()/goodr.sum().clamp_min(1)
        rawprotect=((ce[:,8]-be-.001).clamp_min(0)/.005*goodc).sum()/goodc.sum().clamp_min(1)+((re[:,8]-br-.001).clamp_min(0)/.005*goodr).sum()/goodr.sum().clamp_min(1)
        dt=(b['dt'][:,1:]-b['dt'][:,:-1]).clamp_min(.05);diff=(raw[:,1:]-raw[:,:-1])-(gt[:,1:]-gt[:,:-1]);vm=(valid[:,1:]&valid[:,:-1]).float();velocity=(F.smooth_l1_loss(diff/(dt[:,:,None,None]*.1),torch.zeros_like(diff),reduction='none',beta=.5).sum(-1)*vm).sum()/vm.sum().clamp_min(1)
        # Prior loss is auxiliary; main geometry/velocity/protection use actual samples.
        prior=self.loss(b,gt,valid,gt_uv,uv_valid)
        return .4*fit+final+2*protect+.1*velocity+.1*prior+getattr(self,'raw_protection_weight',0.)*rawprotect

    @torch.no_grad()
    def predict_rollout(self,b,steps=10,samples=4,seed=202610093):
        encoded=self.encode(b);delta=self.sample_trajectory(b,encoded,steps,samples,seed);full=unpack(pack(b['xyz'])[None]+delta);draws=full[:,:,8];mean=torch.where(b['confirmed'][...,None],b['base'],draws.mean(0))
        return dict(xyz_camera_m=mean,raw_xyz_camera_m=mean,std_m=draws.std(0,unbiased=False),draws=draws,trajectory_xyz_camera_m=full.mean(0),trust=torch.ones(len(b['base']),1,device=b['base'].device))

    def loss(self,b,gt,valid,gt_uv,uv_valid):
        pose_valid=valid&valid[:,:,WRIST:WRIST+1]
        tv=torch.cat([valid[:,:,WRIST:WRIST+1],pose_valid],2).float()*self.token_mask[...,0]
        # Unlabeled placeholders are not clean poses. In particular, absent
        # slots must not inject the arbitrary world origin into noisy targets.
        residual=(pack(gt)-pack(b['xyz']))*self.token_mask*tv[...,None];encoded=self.encode(b)
        if self.kind=='dit':
            t=torch.randint(100,(len(gt),),device=gt.device);a=self.alpha[t][:,None,None,None];noise=torch.randn_like(residual)*self.token_mask
            noise_scale=torch.where(tv[...,None]>0,(1-a).sqrt(),torch.ones_like(a))
            x=a.sqrt()*residual+noise_scale*noise;v=self.net(x,t,encoded);clean=a.sqrt()*x-(1-a).sqrt()*v;desired=a.sqrt()*noise-(1-a).sqrt()*residual;error=(v-desired).square().sum(-1);usable=(self.alpha[t]>.05).float()[:,None,None]
        else:
            clean=self.net(torch.zeros_like(residual),torch.zeros(len(gt),device=gt.device,dtype=torch.long),encoded);error=F.smooth_l1_loss(clean,residual,reduction='none',beta=.5).sum(-1);usable=torch.ones(len(gt),1,1,device=gt.device)
        prediction=unpack(pack(b['xyz'])+clean);pose=(prediction-prediction[:,:,WRIST:WRIST+1])-(gt-gt[:,:,WRIST:WRIST+1]);camera=(prediction-gt)/.03
        geom=(F.smooth_l1_loss(camera,torch.zeros_like(camera),reduction='none',beta=.5).sum(-1)+.75*F.smooth_l1_loss(pose/.03,torch.zeros_like(pose),reduction='none',beta=.5).sum(-1));m=valid.float()*usable
        geometry=(geom*m).sum()/m.sum().clamp_min(1)
        # Consecutive nonuniform timestamps supervise actual3D velocities, not a
        # hand-crafted constant-pose interpolation or artificial occlusion.
        dt=(b['dt'][:,1:]-b['dt'][:,:-1]).clamp_min(.05);diff=(prediction[:,1:]-prediction[:,:-1])-(gt[:,1:]-gt[:,:-1]);vm=(valid[:,1:]&valid[:,:-1]).float()*usable
        velocity=(F.smooth_l1_loss(diff/(dt[:,:,None,None]*.1),torch.zeros_like(diff),reduction='none',beta=.5).sum(-1)*vm).sum()/vm.sum().clamp_min(1)
        bone=prediction.sum()*0
        for u,v in EDGES:
            bm=(valid[:,:,u]&valid[:,:,v]).float()*usable[...,0];de=((prediction[:,:,u]-prediction[:,:,v]).norm(dim=-1)-(gt[:,:,u]-gt[:,:,v]).norm(dim=-1)).abs()/.01;bone+=(de*bm).sum()/bm.sum().clamp_min(1)/len(EDGES)
        heat={k:v.flatten(0,1) for k,v in encoded[3].items()};aux=self.visual_head.loss(heat,gt_uv.flatten(0,1),uv_valid.flatten(0,1),b['positions'].flatten(0,1),b['roi'].flatten(0,1))
        return (error*tv).sum()/tv.sum().clamp_min(1)+.3*geometry+.1*velocity+.05*bone+.025*aux

    @torch.no_grad()
    def predict(self,b,steps=10,samples=4,seed=202610091):
        encoded=self.encode(b);B=len(b['base']);device=b['base'].device;gen=torch.Generator(device=device).manual_seed(seed);draws=[];trajectory=[]
        for si in range(samples if self.kind=='dit' else 1):
            if self.kind=='regression':delta=self.net(torch.zeros(B,17,21,3,device=device),torch.zeros(B,device=device,dtype=torch.long),encoded)
            else:
                x=torch.randn(B,17,21,3,device=device,generator=gen)*self.token_mask;schedule=torch.linspace(99,0,steps,device=device).round().long()
                for j,t in enumerate(schedule):
                    v=self.net(x,t.expand(B),encoded);a=self.alpha[t];clean=(a.sqrt()*x-(1-a).sqrt()*v).clamp(-8,8)*self.token_mask;eps=(1-a).sqrt()*x+a.sqrt()*v;x=clean if j+1==len(schedule) else self.alpha[schedule[j+1]].sqrt()*clean+(1-self.alpha[schedule[j+1]]).sqrt()*eps
                delta=x
            full=unpack(pack(b['xyz'])+delta);center=torch.where(b['confirmed'][...,None],b['base'],full[:,8]);draws.append(center);trajectory.append(full)
        draws=torch.stack(draws);mean=torch.where(b['confirmed'][...,None],b['base'],draws.mean(0));return dict(xyz_camera_m=mean,raw_xyz_camera_m=mean,std_m=draws.std(0,unbiased=False),draws=draws,trajectory_xyz_camera_m=torch.stack(trajectory).mean(0),trust=torch.ones(B,1,device=device))

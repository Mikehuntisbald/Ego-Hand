import torch
from torch.nn import functional as F
from hand3d_temporal_v7 import pack,unpack,WRIST,EDGES
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
    dt=(b['dt'][:,1:]-b['dt'][:,:-1]).clamp_min(.001);diff=(prediction[:,1:]-prediction[:,:-1])-(gt[:,1:]-gt[:,:-1]);vm=(valid[:,1:]&valid[:,:-1]).float()*usable
    velocity=(F.smooth_l1_loss(diff/(dt[:,:,None,None]*.1),torch.zeros_like(diff),reduction='none',beta=.5).sum(-1)*vm).sum()/vm.sum().clamp_min(1)
    bone=prediction.sum()*0
    for u,v in EDGES:
        bm=(valid[:,:,u]&valid[:,:,v]).float()*usable[...,0];de=((prediction[:,:,u]-prediction[:,:,v]).norm(dim=-1)-(gt[:,:,u]-gt[:,:,v]).norm(dim=-1)).abs()/.01;bone+=(de*bm).sum()/bm.sum().clamp_min(1)/len(EDGES)
    heat={k:v.flatten(0,1) for k,v in encoded[3].items()};aux=self.visual_head.loss(heat,gt_uv.flatten(0,1),uv_valid.flatten(0,1),b['positions'].flatten(0,1),b['roi'].flatten(0,1))
    return (error*tv).sum()/tv.sum().clamp_min(1)+.3*geometry+.1*velocity+.05*bone+.025*aux

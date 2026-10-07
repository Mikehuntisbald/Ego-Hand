"""Match the verified adaptive deployment policy during actual sampler training."""
import torch
from torch.nn import functional as F
from density_model_v13 import DensityTrajectoryHand3D,POLICY
from hand3d_temporal_v7 import pack,unpack,WRIST
from native_projection_policy_v11 import apply as conservative
from adaptive_projection_v14 import apply

class RecoveryHand3D(DensityTrajectoryHand3D):
    def __init__(self,policy,adaptive=True):
        super().__init__('dit',True);self.recovery_policy=policy;self.adaptive_training=adaptive

    def rollout_loss(self,b,gt,valid,gt_uv,uv_valid,seed):
        encoded=self.encode(b);delta=self.sample_trajectory(b,encoded,10,4,seed)
        raw=unpack(pack(b['xyz'])[None]+delta).mean(0)
        pred=apply(raw[:,8],b,self.recovery_policy) if self.adaptive_training else conservative(raw[:,8],b['base'],POLICY,b['confirmed'])
        canonical=valid.clone();canonical[:,:,WRIST]=False;m=canonical.float();cm=m[:,8]
        def errors(x,g):return (x-g).norm(dim=-1),((x-x[...,WRIST:WRIST+1,:])-(g-g[...,WRIST:WRIST+1,:])).norm(dim=-1)
        ce,re=errors(raw,gt);pe,pr=errors(pred,gt[:,8]);be,br=errors(b['base'],gt[:,8])
        fit=((ce/.03+.75*re/.03)*m).sum()/m.sum().clamp_min(1)
        final=((pe/.03+.75*pr/.03)*cm).sum()/cm.sum().clamp_min(1)
        goodc=canonical[:,8]&(be<=.01);goodr=canonical[:,8]&(br<=.01)
        protect=((pe-be-.001).clamp_min(0)/.005*goodc).sum()/goodc.sum().clamp_min(1)+((pr-br-.001).clamp_min(0)/.005*goodr).sum()/goodr.sum().clamp_min(1)
        rawprotect=((ce[:,8]-be-.001).clamp_min(0)/.005*goodc).sum()/goodc.sum().clamp_min(1)+((re[:,8]-br-.001).clamp_min(0)/.005*goodr).sum()/goodr.sum().clamp_min(1)
        dt=b['dt'][:,1:]-b['dt'][:,:-1];vm=(valid[:,1:]&valid[:,:-1]&(dt>0)[:,:,None]).float()
        diff=(raw[:,1:]-raw[:,:-1])-(gt[:,1:]-gt[:,:-1])
        velocity=(F.smooth_l1_loss(diff/(dt.clamp_min(.001)[:,:,None,None]*.1),torch.zeros_like(diff),reduction='none',beta=.5).sum(-1)*vm).sum()/vm.sum().clamp_min(1)
        prior=self.loss_without_velocity_floor(b,gt,valid,gt_uv,uv_valid)
        return .4*fit+final+2*protect+.1*velocity+.1*prior+.3*rawprotect

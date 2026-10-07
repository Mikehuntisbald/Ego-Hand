"""Deploy-sampling3D loss with actual dense timestamps (not a50ms floor)."""
import torch
from torch.nn import functional as F
from hand3d_native_v10 import NativeTrajectoryHand3D
from hand3d_temporal_v7 import pack,unpack,WRIST
from native_projection_policy_v11 import apply
POLICY=dict(strength=1.,cap_m=.0095,root_cap_m=.0095,relative_weight=4.)

class DensityTrajectoryHand3D(NativeTrajectoryHand3D):
    def rollout_loss(self,b,gt,valid,gt_uv,uv_valid,seed):
        encoded=self.encode(b);delta=self.sample_trajectory(b,encoded,10,4,seed);raw=unpack(pack(b['xyz'])[None]+delta).mean(0);pred=apply(raw[:,8],b['base'],POLICY,b['confirmed'])
        canonical=valid.clone();canonical[:,:,WRIST]=False;m=canonical.float();cm=m[:,8]
        def error(x,g):return (x-g).norm(dim=-1),((x-x[...,WRIST:WRIST+1,:])-(g-g[...,WRIST:WRIST+1,:])).norm(dim=-1)
        ce,re=error(raw,gt);pe,pr=error(pred,gt[:,8]);be,br=error(b['base'],gt[:,8])
        fit=((ce/.03+.75*re/.03)*m).sum()/m.sum().clamp_min(1);final=((pe/.03+.75*pr/.03)*cm).sum()/cm.sum().clamp_min(1)
        goodc=canonical[:,8]&(be<=.01);goodr=canonical[:,8]&(br<=.01)
        protect=((pe-be-.001).clamp_min(0)/.005*goodc).sum()/goodc.sum().clamp_min(1)+((pr-br-.001).clamp_min(0)/.005*goodr).sum()/goodr.sum().clamp_min(1)
        rawprotect=((ce[:,8]-be-.001).clamp_min(0)/.005*goodc).sum()/goodc.sum().clamp_min(1)+((re[:,8]-br-.001).clamp_min(0)/.005*goodr).sum()/goodr.sum().clamp_min(1)
        dt=b['dt'][:,1:]-b['dt'][:,:-1];vm=(valid[:,1:]&valid[:,:-1]&(dt>0)[:,:,None]).float()
        # True intervals include33ms. Only numerical protection at1ms remains.
        diff=(raw[:,1:]-raw[:,:-1])-(gt[:,1:]-gt[:,:-1]);velocity=(F.smooth_l1_loss(diff/(dt.clamp_min(.001)[:,:,None,None]*.1),torch.zeros_like(diff),reduction='none',beta=.5).sum(-1)*vm).sum()/vm.sum().clamp_min(1)
        # The prior snapshot also uses original timestamps and a1ms numerical
        # guard; its repaired target masking and other loss terms stay unchanged.
        prior=self.loss_without_velocity_floor(b,gt,valid,gt_uv,uv_valid)
        return .4*fit+final+2*protect+.1*velocity+.1*prior+.3*rawprotect

    def loss_without_velocity_floor(self,b,gt,valid,gt_uv,uv_valid):
        # Overriding the original velocity helper through copying its loss keeps
        # the repaired target-mask and all original inputs unchanged.
        from density_prior_loss_v13 import loss
        return loss(self,b,gt,valid,gt_uv,uv_valid)

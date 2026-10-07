"""Joint-supervised whole-hand proposal judge with native RGB queries.

Inference receives observations/proposals only. XYZ root and fingers still use
one accepted strength for the whole hand; point predictions are auxiliary.
"""
import torch
from torch import nn
from hand3d_rollout_v8 import project_fisheye624


def visual_queries(b, proposal, params, strength):
    candidate=b['base']+strength*(proposal-b['base'])
    uv=project_fisheye624(candidate,params)/1408
    candidate_valid=torch.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<1).all(-1)&(candidate[...,2]>0)
    def local(t,xy,valid):
        size=(b['roi'][:,t,2:]-b['roi'][:,t,:2]).mean(-1).clamp_min(.01)
        d=((b['positions'][:,t,None]-torch.nan_to_num(xy)[:,:,None])/size[:,None,None,None]).square().sum(-1)
        weights=(-d/(2*(1/12)**2)).softmax(-1)
        return torch.einsum('bjs,bsc->bjc',weights,b['rgb_native'][:,t].float())*(valid&b['rgb_valid'][:,t,None])[...,None]
    z=b['rgb_native'].float();available=b['rgb_valid'].float()
    context=(z.mean(2)*available[...,None]).sum(1)/available.sum(1,keepdim=True).clamp_min(1)
    return torch.stack([
        local(8,b['xy'][:,8],b['observed_2d'][:,8]),
        local(8,uv,candidate_valid),
        local(5,b['xy'][:,5],b['observed_2d'][:,5]),
        local(11,b['xy'][:,11],b['observed_2d'][:,11]),
        z[:,8].mean(1)[:,None].expand(-1,20,-1),
        context[:,None].expand(-1,20,-1)],2)


def point_labels(base,pred,gt,valid):
    be=(base-gt).norm(dim=-1)
    pe=(pred-gt).norm(dim=-1)
    br=((base-base[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)
    pr=((pred-pred[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)
    mask=valid.clone();mask[:,5]=False
    errors=torch.stack([be,br,pe,pr],-1)
    # Log targets limit domination by very large errors. These never enter inference.
    values=torch.log1p(errors.clamp_max(.5)/.03)
    harm_c=(be<=.01)&(pe>.02);harm_r=(br<=.01)&(pr>.02)
    useful=(br-pr>.001)&(be-pe>-.001)&~harm_c&~harm_r
    return torch.cat([values,torch.stack([harm_c,harm_r,useful],-1).float()],-1),mask


class JointProposalCritic(nn.Module):
    def __init__(self,dim,native=True):
        super().__init__();self.native=native
        self.register_buffer('mean',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim))
        self.geometry=nn.Sequential(nn.Linear(dim,128),nn.GELU())
        if native:
            self.visual=nn.Sequential(nn.LayerNorm(1280),nn.Linear(1280,64),nn.GELU())
            self.fuse=nn.Linear(6*64,128)
        self.joint=nn.Parameter(torch.randn(1,20,128)*.02)
        layer=nn.TransformerEncoderLayer(128,4,256,dropout=.15,batch_first=True,norm_first=True,activation='gelu')
        self.context=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.point=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,7))
        self.hand=nn.Sequential(nn.Linear(384,96),nn.GELU(),nn.Dropout(.15),nn.Linear(96,5))
    def forward(self,x,visual=None,return_points=False):
        h=self.geometry(((x-self.mean)/self.scale.clamp_min(.02)).clamp(-10,10))+self.joint
        if self.native:
            assert visual is not None
            h=h+self.fuse(self.visual(visual.float()).flatten(2))
        h=self.context(h);hand=self.hand(torch.cat([h.mean(1),h.amax(1),h[:,5]],-1))
        return (hand,self.point(h)) if return_points else hand

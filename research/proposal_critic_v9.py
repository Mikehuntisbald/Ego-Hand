"""Assess an entire3D proposal; root and fingers are never gated independently."""
import torch
from torch import nn
from hand3d_data_v7 import risk_features
from hand3d_rollout_v8 import project_fisheye624
from hand3d_temporal_v7 import WRIST

def features(b,proposal,std,params,strength):
    candidate=b['base']+strength*(proposal-b['base']);other=dict(b);other['base']=candidate;other['xyz']=b['xyz'].clone();other['xyz'][:,8]=candidate
    uv=project_fisheye624(candidate,params)/1408;obs=torch.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<1).all(-1)&(candidate[...,2]>0)
    other['xy']=b['xy'].clone();other['observed_2d']=b['observed_2d'].clone();other['xy'][:,8]=torch.nan_to_num(uv)*obs[...,None];other['observed_2d'][:,8]=obs
    old=risk_features(b);new=risk_features(other);delta=(candidate-b['base'])/.05;relative=delta-delta[:,WRIST:WRIST+1]
    alpha=torch.full_like(b['risk_camera'][...,None],strength)
    return torch.cat([old,new,delta,relative,std/.03,b['risk_camera'][...,None],b['risk_relative'][...,None],alpha],-1).float()

class ProposalCritic(nn.Module):
    def __init__(self,dim):
        super().__init__();self.register_buffer('mean',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim));self.local=nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU());self.head=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,5))
    def forward(self,x):
        h=self.local(((x-self.mean)/self.scale.clamp_min(.02)).clamp(-10,10));return self.head(torch.cat([h.mean(1),h.amax(1)],-1))

def labels(base,pred,gt,valid):
    mask=valid.clone();mask[:,WRIST]=False
    def error(x):return (x-gt).norm(dim=-1),((x-x[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])).norm(dim=-1)
    be,br=error(base);pe,pr=error(pred);cg=((be-pe)*mask).sum(-1)/mask.sum(-1).clamp_min(1);rg=((br-pr)*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    hc=(mask&(be<=.01)&(pe>.02)).any(-1);hr=(mask&(br<=.01)&(pr>.02)).any(-1);useful=(cg+.75*rg>.0005)&(rg>.0002)&~hc&~hr
    return torch.stack([(cg/.03).clamp(-5,5),(rg/.03).clamp(-5,5),hc.float(),hr.float(),useful.float()],-1)

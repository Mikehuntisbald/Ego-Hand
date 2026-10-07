"""Coordinates are soft predictions. Only explicitly confirmed points are locked."""
import torch
from torch import nn
from spatial_temporal_model import SpatialTemporalCompleter
from offline_kp_model import condition

def natural_batch(data,ids,probability,confirmed=None):
    b=condition(data['xy'][ids],data['observed'][ids],data['dt'][ids]);f=data['feature_ids'][ids]
    if confirmed is None:confirmed=torch.zeros(len(ids),20,dtype=torch.bool,device=ids.device)
    assert not (confirmed&~b['observed'][:,8]).any(),'Cannot confirm an absent coordinate'
    b.update(rgb=data['bank'][f,0],positions=data['positions_bank'][f],roi=data['roi'][f],rgb_valid=f>0,
        p_bad=probability[ids],confirmed=confirmed,missing=~confirmed)
    return b

class NaturalCorrector(SpatialTemporalCompleter):
    def __init__(self,kind='dit'):
        super().__init__(kind,True)
        self.reliability=nn.Sequential(nn.Linear(2,128),nn.SiLU(),nn.Linear(128,128))
        nn.init.zeros_(self.reliability[-1].weight);nn.init.zeros_(self.reliability[-1].bias)
    def encode(self,b):
        joint,frame,c=super().encode(b)
        reliability=self.reliability(torch.stack([b['p_bad'],b['observed'][:,8].float()],-1))
        return joint+reliability,frame,c+reliability.mean(1)
    def loss(self,b,gt,valid):
        eligible=valid&~b['confirmed'];target=torch.where(eligible[...,None],(gt-b['linear'])/.1,0.)
        weights=eligible.float()*(1+3*((gt-b['linear']).norm(dim=-1)*1408>20))
        encoded=self.encode(b)
        if self.kind=='dit':
            t=torch.randint(100,(len(gt),),device=gt.device);a=self.alpha[t][:,None,None]
            noise=torch.randn_like(target)*b['missing'][...,None];x=a.sqrt()*target+(1-a).sqrt()*noise
            v=self.net(x,t,encoded);desired=a.sqrt()*noise-(1-a).sqrt()*target
            clean=a.sqrt()*x-(1-a).sqrt()*v
            errors=(v-desired).square().sum(-1)+.2*(clean-target).square().sum(-1)*(.1+self.alpha[t][:,None])
        else:
            clean=self.net(torch.zeros_like(target),torch.zeros(len(gt),device=gt.device,dtype=torch.long),encoded)
            errors=(clean-target).square().sum(-1)
        out=self.visual(b);current={k:v[:,8] for k,v in out.items()}
        return (errors*weights).sum()/weights.sum().clamp_min(1)+.05*self.visual_head.loss(current,gt,valid,b['positions'][:,8],b['roi'][:,8])

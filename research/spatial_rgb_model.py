"""192 spatial cells throughout visual localization, with heatmap supervision."""
import torch
from torch import nn
from torch.nn import functional as F

class SpatialHead(nn.Module):
    def __init__(self,use_rgb=True):
        super().__init__();self.use_rgb=use_rgb
        self.project=nn.Sequential(nn.LayerNorm(1280),nn.Linear(1280,128),nn.GELU())
        self.spatial=nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.GroupNorm(8,128),nn.GELU(),nn.Dropout2d(.1),nn.Conv2d(128,128,3,padding=1),nn.GELU())
        self.heatmap=nn.Conv2d(128,20,1)
        self.prior=nn.Parameter(torch.zeros(1,20,16,12))
        self.offset=nn.Conv2d(128,40,1)
        nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)

    def features(self,rgb):
        x=self.project(rgb.float() if self.use_rgb else torch.zeros_like(rgb,dtype=torch.float32))
        x=x.transpose(1,2).reshape(-1,128,16,12)
        return x+self.spatial(x)

    def decode(self,x,positions,roi):
        logits=(self.heatmap(x)+self.prior).flatten(2).float()
        probability=logits.softmax(-1)
        offset=self.offset(x).reshape(-1,20,2,192).transpose(-1,-2).float().tanh()
        size=(roi[:,2:]-roi[:,:2])[:,None,None].clamp_min(1e-4)
        xy=(probability[...,None]*(positions[:,None]+offset*size*.15)).sum(2)
        return dict(xy=xy,logits=logits,probability=probability,features=x.flatten(2).transpose(1,2))

    def forward(self,rgb,positions,roi):return self.decode(self.features(rgb),positions,roi)

    def loss(self,out,gt,valid,positions,roi):
        size=(roi[:,2:]-roi[:,:2]).mean(-1).clamp_min(.01)
        dist=((positions[:,None]-gt[:,:,None])/size[:,None,None,None]).square().sum(-1)
        target=(-dist/(2*(1.2/12)**2)).softmax(-1)
        ce=-(target*out['logits'].log_softmax(-1)).sum(-1)
        coordinate=F.smooth_l1_loss((out['xy']-gt)/size[:,None,None]*10,torch.zeros_like(gt),reduction='none',beta=.5).sum(-1)
        return ((ce+4*coordinate)*valid).sum()/valid.sum().clamp_min(1)

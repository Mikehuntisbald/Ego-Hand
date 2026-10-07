"""Causal RGB/coarse-pose contexts, shared equally by diffusion and regression."""
import torch
from torch import nn
from residual_models import ResidualModel
from pose_residual_dit import time_embedding

class TemporalResidualModel(ResidualModel):
    def __init__(self,kind='dit'):
        super().__init__(kind=kind)
        self.history_pose=nn.Sequential(nn.Linear(84,self.width),nn.SiLU(),nn.Linear(self.width,self.width))
        self.history_time=nn.Linear(9,self.width)
        self.context=None
    def set_context(self,context):
        self.context=context
        assert context['valid'][:,0].all()
        assert (context['dt']<=.00001).all(),'Future frames are not allowed'
    def features(self,x,t,coarse,confidence,rgb):
        assert self.context is not None
        context=self.context;b,s=context['valid'].shape
        c=self.pack(coarse);emb=self.coarse(torch.cat([c,confidence[...,None]],-1))
        past=self.pack(context['pose'].reshape(b*s,20,3)).reshape(b,s,63)
        poseemb=self.history_pose(torch.cat([past,context['confidence']],-1))
        dt=context['dt'];freq=torch.tensor([1,2,4,8],device=dt.device)*torch.pi
        angle=dt[...,None]*freq
        timeemb=self.history_time(torch.cat([dt[...,None],angle.sin(),angle.cos()],-1))
        image=self.rgb(context['rgb'])+self.rgbpos(self.xy)[:,None]+poseemb[:,:,None]+timeemb[:,:,None]
        valid=context['valid'];weights=valid[:,:,None,None]
        pooled=(image*weights).sum((1,2))/(valid.sum(1)[:,None]*64)
        condition=self.time(time_embedding(t,self.width))+emb.mean(1)+pooled
        h=self.noisy(x)+emb+self.jointpos
        image=image.reshape(b,s*64,self.width);mask=(~valid)[:,:,None].expand(-1,-1,64).reshape(b,-1)
        for block in self.blocks:h=block(h,image,condition,rgb_mask=mask)
        return self.norm(h)

def attach_context(model,data,ids,device):
    if isinstance(model,TemporalResidualModel):
        model.set_context({key:value[ids].to(device).float() if key!='valid' else value[ids].to(device)
                           for key,value in data['temporal'].items()})

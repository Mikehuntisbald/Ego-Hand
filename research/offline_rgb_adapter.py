"""RGB residual adapter around a frozen temporal predictor; zero gain is exact identity."""
import torch
from torch import nn
from torch.nn import functional as F
from offline_rgb_model import RGBKeypointCompleter
from offline_kp_model import KeypointCompleter

class RGBAdapterCompleter(RGBKeypointCompleter):
    def __init__(self,kind='dit',use_rgb=True):
        super().__init__(kind,use_rgb)
        self.visual_attention=nn.MultiheadAttention(128,4,batch_first=True)
        self.visual_dropout=nn.Dropout(.15)
        self.visual_gain=nn.Parameter(torch.zeros(()));self.local_gain=nn.Parameter(torch.zeros(()))
    def encode(self,b):
        joint,frame,c=KeypointCompleter.encode(self,b)
        box=self.box(b['roi'])*b['rgb_valid'][...,None];frame=frame+box;joint=joint+box[:,8,None];c=c+box.mean(1)
        if not self.use_rgb:return joint,frame,c
        roi=b['roi'];rgb=b['rgb'].float();valid=b['rgb_valid']
        locations=roi[:,:,:2,None].transpose(-1,-2)+self.grid[None,None]*(roi[:,:,2:]-roi[:,:,:2])[:,:,None]
        pos=torch.cat([locations,b['dt'][:,:,None,None].expand(-1,-1,16,1)],-1)
        vision=self.visual_dropout((self.visual(rgb)+self.visual_position(pos))*valid[:,:,None,None])
        v=vision.flatten(1,2);extra=self.visual_attention(joint,v,v,need_weights=False)[0]
        q=(b['linear']-roi[:,8,None,:2])/(roi[:,8,None,2:]-roi[:,8,None,:2]).clamp_min(1e-4)
        current=rgb[:,8].transpose(1,2).reshape(-1,512,4,4)
        local=F.grid_sample(current,(q*2-1)[:,:,None],mode='bilinear',padding_mode='zeros',align_corners=False)[:,:,:,0].transpose(1,2)
        a=self.visual_gain.tanh();d=self.local_gain.tanh()
        joint=joint+a*extra+d*self.local_visual(local)*valid[:,8,None,None]
        return joint,frame,c+a*vision.mean((1,2))

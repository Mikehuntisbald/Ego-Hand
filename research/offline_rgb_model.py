"""Identical visual-temporal conditioning for direct regression and diffusion."""
import torch
from torch import nn
from torch.nn import functional as F
from offline_kp_model import KeypointCompleter

class RGBKeypointCompleter(KeypointCompleter):
    def __init__(self,kind='dit',use_rgb=True):
        super().__init__(kind,width=128,depth=3);self.use_rgb=use_rgb
        self.box=nn.Sequential(nn.Linear(4,128),nn.SiLU(),nn.Linear(128,128))
        self.visual=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,128))
        self.visual_position=nn.Sequential(nn.Linear(3,128),nn.SiLU(),nn.Linear(128,128))
        self.local_visual=nn.Linear(512,128)
        gy,gx=torch.meshgrid((torch.arange(4)+.5)/4,(torch.arange(4)+.5)/4,indexing='ij')
        self.register_buffer('grid',torch.stack([gx,gy],-1).reshape(16,2))

    def encode(self,b):
        joint,frame,condition=super().encode(b)
        box=self.box(b['roi'])*b['rgb_valid'][...,None];frame=frame+box
        joint=joint+box[:,8,None];condition=condition+box.mean(1)
        if not self.use_rgb:return joint,frame,condition
        rgb=b['rgb'].float();roi=b['roi'];valid=b['rgb_valid']
        locations=roi[:,:,:2,None].transpose(-1,-2)+self.grid[None,None]*(roi[:,:,2:]-roi[:,:,:2])[:,:,None]
        position=torch.cat([locations,b['dt'][:,:,None,None].expand(-1,-1,16,1)],-1)
        vision=(self.visual(rgb)+self.visual_position(position))*valid[:,:,None,None]
        current=rgb[:,8].transpose(1,2).reshape(-1,512,4,4)
        q=(b['linear']-roi[:,8,None,:2])/(roi[:,8,None,2:]-roi[:,8,None,:2]).clamp_min(1e-4)
        sampled=F.grid_sample(current,(q*2-1)[:,:,None],mode='bilinear',padding_mode='zeros',align_corners=False)[:,:,:,0].transpose(1,2)
        joint=joint+self.local_visual(sampled)*valid[:,8,None,None]
        return joint,torch.cat([frame,vision.flatten(1,2)],1),condition+vision.mean((1,2))

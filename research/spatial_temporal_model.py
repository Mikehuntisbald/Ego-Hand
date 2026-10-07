"""Dense RGB-driven localization and bidirectional coordinate completion."""
import torch
from torch import nn
from offline_kp_model import KeypointCompleter
from spatial_rgb_model import SpatialHead

class SpatialTemporalCompleter(KeypointCompleter):
    def __init__(self,kind='dit',use_rgb=True):
        super().__init__(kind,width=128,depth=3);self.use_rgb=use_rgb
        self.box=nn.Sequential(nn.Linear(4,128),nn.SiLU(),nn.Linear(128,128))
        self.visual_head=SpatialHead()
        # Feature stem is already applied in the cache; retain decoder only.
        self.visual_head.project=nn.Identity();self.visual_head.spatial=nn.Identity()
        self.visual_norm=nn.LayerNorm(128)
        self.visual_position=nn.Sequential(nn.Linear(3,128),nn.SiLU(),nn.Linear(128,128))
        self.visual_motion=nn.Sequential(nn.Linear(17*4,128),nn.SiLU(),nn.Linear(128,128))
        self.visual_attention=nn.MultiheadAttention(128,4,batch_first=True)
        self.fuse=nn.Linear(256,128)
        with torch.no_grad():
            self.fuse.weight.zero_();self.fuse.weight[:,:128]=torch.eye(128)
            self.fuse.weight[:,128:]=.25*torch.eye(128);self.fuse.bias.zero_()
        self.rgb_dropout=nn.Dropout(.1)

    def visual(self,b):
        B,T,S,C=b['rgb'].shape
        x=b['rgb'].float().reshape(B*T,S,C).transpose(1,2).reshape(B*T,C,16,12)
        o=self.visual_head.decode(x,b['positions'].reshape(B*T,S,2),b['roi'].reshape(B*T,4))
        return {k:v.reshape(B,T,*v.shape[1:]) for k,v in o.items()}

    def encode(self,b):
        joint,frame,c=super().encode(b)
        box=self.box(b['roi'])*b['rgb_valid'][...,None]
        joint=joint+box[:,8,None];frame=frame+box;c=c+box.mean(1)
        if not self.use_rgb:return joint,frame,c
        o=self.visual(b);valid=b['rgb_valid']
        pos=torch.cat([b['positions'],b['dt'][:,:,None,None].expand(-1,-1,192,1)],-1)
        vision=self.rgb_dropout(self.visual_norm(b['rgb'].float())+self.visual_position(pos))
        vision=vision*valid[:,:,None,None]
        per_joint=torch.einsum('btjs,btsc->btjc',o['probability'],vision)
        entropy=-(o['probability']*o['probability'].clamp_min(1e-8).log()).sum(-1)/5.2575
        relative=(o['xy']-b['linear'][:,None])/.1
        motion=torch.cat([relative,entropy[...,None],valid[:,:,None,None].expand(-1,-1,20,1)],-1)
        motion=motion*valid[:,:,None,None]
        q=per_joint[:,8]+self.visual_motion(motion.permute(0,2,1,3).flatten(2))+self.position
        cells=vision.flatten(1,2);padding=(~valid)[:,:,None].expand(-1,-1,192).flatten(1)
        # A supplied central hand ROI is required; hence at least one RGB frame.
        assert valid.any(1).all()
        attention=self.visual_attention(q,cells,cells,key_padding_mask=padding,need_weights=False)[0]
        visual_joint=q+attention
        joint=self.fuse(torch.cat([joint,visual_joint],-1))
        # Joint heatmap summaries keep all 17 frames available to each denoising step.
        memory=torch.cat([frame,per_joint.flatten(1,2)],1)
        return joint,memory,c+visual_joint.mean(1)

    def loss(self,b,gt,valid):
        loss=super().loss(b,gt,valid)
        if self.use_rgb:
            o=self.visual(b);current={k:v[:,8] for k,v in o.items()}
            loss=loss+.15*self.visual_head.loss(current,gt,valid,b['positions'][:,8],b['roi'][:,8])
        return loss

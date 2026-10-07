"""Separate native1280 semantic conditions from128-channel2D localization stem."""
import torch
from torch import nn
from hand3d_trajectory_v9 import TrajectoryHand3D
from spatial_rgb_model import SpatialHead
from pose_residual_dit import modulate,time_embedding
import spatial_rgb_common as s

class NativeTrajectoryHand3D(TrajectoryHand3D):
    def __init__(self,kind='dit',native=True):
        super().__init__(kind);self.native=native;self.raw_protection_weight=.3
        self.localization=SpatialHead();self.visual_head.project=nn.Identity();self.visual_head.spatial=nn.Identity()
        if native:self.rgb_norm=nn.LayerNorm(1280);self.rgb_project=nn.Linear(1280,self.width)
    def initialize_visual(self,device):
        ck=torch.load(s.RUN/'sealed/rgb_probe.pt',weights_only=False,map_location=device)['model'];self.localization.load_state_dict(ck)
    def encode(self,b):
        B,T,S,C=b['rgb_native'].shape;z=self.localization.features(b['rgb_native'].reshape(B*T,S,C));bb={**b,'rgb':z.flatten(2).transpose(1,2).reshape(B,T,S,128)}
        if not self.native:bb.pop('rgb_native')
        encoded=super().encode(bb);encoded[3]['observable_time_valid']=b['rgb_valid'];return encoded

    def net(self,x,t,encoded):
        joint,memory,c,heat=encoded;h=joint+self.noisy(x.flatten(1,2));c=c+self.time(time_embedding(t,self.width));valid=heat['observable_time_valid']
        if len(valid)!=len(x):valid=valid.repeat(len(x)//len(valid),1)
        joint_padding=(~valid)[:,:,None].expand(-1,-1,21).flatten(1);spatial_padding=(~valid)[:,:,None].expand(-1,-1,192).flatten(1);memory_padding=torch.cat([joint_padding,spatial_padding],1)
        assert valid.any(1).all()
        for block in self.blocks:
            s1,c1,g1,s2,c2,g2,s3,c3,g3=block.condition(c).chunk(9,-1)
            q=modulate(block.norms[0](h),s1,c1);h=h+g1[:,None]*block.self_attention(q,q,q,key_padding_mask=joint_padding,need_weights=False)[0]
            q=modulate(block.norms[1](h),s2,c2);h=h+g2[:,None]*block.cross_attention(q,memory,memory,key_padding_mask=memory_padding,need_weights=False)[0]
            q=modulate(block.norms[2](h),s3,c3);h=h+g3[:,None]*block.mlp(q)
        return self.head(self.norm(h)).reshape(len(x),17,21,3)*self.token_mask

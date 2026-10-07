"""34 scalar parameter queries associated with actual anatomical evidence.

Changing XYZ output into packed pose parameters also changes token semantics.
Root/global/finger/shape queries must attend to corresponding observation
joints, rather than inheriting the arbitrary 21x3 storage layout as semantics.
"""
import torch
from torch import nn
from torch.nn import functional as F
from parameter_temporal_model_v31 import ParameterTemporalHand
from hand3d_native_v10 import NativeTrajectoryHand3D
from pose_residual_dit import modulate, time_embedding


class SemanticParameterHand(ParameterTemporalHand):
    def __init__(self,kind='regression',device='cpu'):
        super().__init__(kind,device)
        # 3 root, 6 rotation, 20 joint angles, 5 static shape coefficients.
        # Encoded observation slots are [camera root, relative joint0..19].
        # Therefore actual landmark j lives in slot j+1, not slot j.
        mapping=[0]*9+[7,7,8,1,9,9,10,11,12,12,13,14,15,15,16,17,18,18,19,20]+[0]*5
        self.register_buffer('parameter_joint_index',torch.tensor(mapping,dtype=torch.long))
        self.semantic_position=nn.Parameter(torch.randn(1,1,34,self.width)*.02)
        self.scalar_noisy=nn.Linear(1,self.width)
        self.semantic_head=nn.Linear(self.width,1)
        nn.init.zeros_(self.semantic_head.weight);nn.init.zeros_(self.semantic_head.bias)
        for parameter in self.head.parameters():parameter.requires_grad_(False)
        for parameter in self.noisy.parameters():parameter.requires_grad_(False)
        self.parameter_position.requires_grad_(False)

    def encode(self,b):
        joint,memory,c,heat=NativeTrajectoryHand3D.encode(self,b)
        B=len(c);valid=b['rgb_valid'].float();h=joint.reshape(B,17,21,self.width).mean(2)
        context=c+(h*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        heat['side_logits']=self.side_head(context)+F.one_hot(b['predicted_right'].long(),2).float()*4
        return joint,memory,c,heat

    def net(self,x,t,encoded):
        joint,memory,c,heat=encoded;B,T=x.shape[:2]
        source=joint.reshape(B,T,21,self.width)
        h=source[:,:,self.parameter_joint_index]+self.semantic_position
        scalar=x.flatten(2)[:,:,:34,None]
        h=(h+self.scalar_noisy(scalar)).flatten(1,2)
        c=c+self.time(time_embedding(t,self.width));valid=heat['observable_time_valid']
        if len(valid)!=B:valid=valid.repeat(B//len(valid),1)
        query_padding=(~valid)[:,:,None].expand(-1,-1,34).flatten(1)
        joint_padding=(~valid)[:,:,None].expand(-1,-1,21).flatten(1)
        spatial_padding=(~valid)[:,:,None].expand(-1,-1,192).flatten(1)
        memory_padding=torch.cat([joint_padding,spatial_padding],1)
        for block in self.blocks:
            s1,c1,g1,s2,c2,g2,s3,c3,g3=block.condition(c).chunk(9,-1)
            q=modulate(block.norms[0](h),s1,c1)
            h=h+g1[:,None]*block.self_attention(q,q,q,key_padding_mask=query_padding,need_weights=False)[0]
            q=modulate(block.norms[1](h),s2,c2)
            h=h+g2[:,None]*block.cross_attention(q,memory,memory,key_padding_mask=memory_padding,need_weights=False)[0]
            q=modulate(block.norms[2](h),s3,c3)
            h=h+g3[:,None]*block.mlp(q)
        # The 21x3 tensor is only a storage carrier, not the attention topology.
        delta=self.semantic_head(self.norm(h)).reshape(B,T,34)
        return F.pad(delta,(0,29)).reshape(B,T,21,3)*self.token_mask

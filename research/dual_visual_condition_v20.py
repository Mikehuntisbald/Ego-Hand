"""Preserve existing RGB localization; add separately positioned semantic RGB."""
import torch
from torch import nn
from recovery_model_v15 import RecoveryHand3D
from pose_residual_dit import modulate, time_embedding

class DualVisualHand3D(RecoveryHand3D):
    def __init__(self, policy):
        super().__init__(policy, True)
        self.extra_rgb_norm = nn.LayerNorm(1280)
        self.extra_rgb_project = nn.Linear(1280, self.width)
        self.extra_attention = nn.ModuleList([
            nn.MultiheadAttention(self.width, 6, batch_first=True) for _ in self.blocks])
        self.extra_gain = nn.Parameter(torch.zeros(len(self.blocks)))

    def initialize_extra(self):
        self.extra_rgb_norm.load_state_dict(self.rgb_norm.state_dict())
        self.extra_rgb_project.load_state_dict(self.rgb_project.state_dict())
        for extra, block in zip(self.extra_attention, self.blocks):
            extra.load_state_dict(block.cross_attention.state_dict())

    def encode(self, b):
        joint, memory, condition, heat = super().encode(b)
        metadata = torch.cat([b['extra_positions'], b['extra_rays'],
            b['camera_origin'][:,:,None].expand(-1,-1,192,-1),
            b['dt'][:,:,None,None].expand(-1,-1,192,1)], -1)
        vision = (self.extra_rgb_project(self.extra_rgb_norm(b['extra_rgb_native'].float()))
            + self.rgb_position(metadata)) * b['rgb_valid'][:,:,None,None]
        heat = dict(heat, extra_memory=vision.flatten(1,2))
        return joint, memory, condition, heat

    def net(self, x, t, encoded):
        joint, memory, condition, heat = encoded
        h = joint + self.noisy(x.flatten(1,2))
        condition = condition + self.time(time_embedding(t,self.width))
        valid = heat['observable_time_valid']
        extra = heat['extra_memory']
        if len(valid) != len(x):
            repeats = len(x)//len(valid)
            valid = valid.repeat(repeats,1)
            extra = extra.repeat(repeats,1,1)
        joint_padding = (~valid)[:,:,None].expand(-1,-1,21).flatten(1)
        spatial_padding = (~valid)[:,:,None].expand(-1,-1,192).flatten(1)
        memory_padding = torch.cat([joint_padding,spatial_padding],1)
        assert valid.any(1).all()
        for i, block in enumerate(self.blocks):
            s1,c1,g1,s2,c2,g2,s3,c3,g3 = block.condition(condition).chunk(9,-1)
            q = modulate(block.norms[0](h),s1,c1)
            h = h + g1[:,None]*block.self_attention(q,q,q,key_padding_mask=joint_padding,need_weights=False)[0]
            q = modulate(block.norms[1](h),s2,c2)
            h = h + g2[:,None]*block.cross_attention(q,memory,memory,key_padding_mask=memory_padding,need_weights=False)[0]
            h = h + self.extra_gain[i].tanh()*self.extra_attention[i](q,extra,extra,
                key_padding_mask=spatial_padding,need_weights=False)[0]
            q = modulate(block.norms[2](h),s3,c3)
            h = h + g3[:,None]*block.mlp(q)
        return self.head(self.norm(h)).reshape(len(x),17,21,3)*self.token_mask

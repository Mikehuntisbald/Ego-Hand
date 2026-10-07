"""Reduce unreliable observation-token attention while retaining all RGB cells."""
import torch
from recovery_model_v15 import RecoveryHand3D
from pose_residual_dit import modulate,time_embedding

class ErrorWeightedHand3D(RecoveryHand3D):
    def __init__(self,policy,strength=1.):
        super().__init__(policy,True)
        self.error_attention_strength=strength

    def encode(self,b):
        joint,memory,condition,heat=super().encode(b)
        probability=b['temporal_error_risk']
        # Root error and wrist-relative finger error are distinct quantities.
        risk=torch.cat([probability[:,:,5:6,0],probability[:,:,:,1]],-1)
        available=torch.cat([b['available'][:,:,5:6],b['available']],2)
        trust=torch.where(available,1-risk,torch.zeros_like(risk)).clamp(.05,1.)
        return joint,memory,condition,dict(heat,observation_log_trust=trust.log())

    def net(self,x,t,encoded):
        joint,memory,condition,heat=encoded
        h=joint+self.noisy(x.flatten(1,2))
        condition=condition+self.time(time_embedding(t,self.width))
        valid=heat['observable_time_valid'];trust=heat['observation_log_trust']
        if len(valid)!=len(x):
            repeat=len(x)//len(valid);valid=valid.repeat(repeat,1);trust=trust.repeat(repeat,1,1)
        joint_padding=(~valid)[:,:,None].expand(-1,-1,21).flatten(1)
        spatial_padding=(~valid)[:,:,None].expand(-1,-1,192).flatten(1)
        memory_padding=torch.cat([joint_padding,spatial_padding],1)
        assert valid.any(1).all()
        # Unchanged self-attention models the latent trajectory. Confidence
        # changes only fixed observation memories, including their fused RGB;
        # the full separate spatial RGB memory remains unweighted.
        bias=torch.cat([trust.flatten(1)*self.error_attention_strength,
            torch.zeros_like(spatial_padding,dtype=trust.dtype)],1)
        bias=bias.masked_fill(memory_padding,float('-inf'))
        for block in self.blocks:
            s1,c1,g1,s2,c2,g2,s3,c3,g3=block.condition(condition).chunk(9,-1)
            q=modulate(block.norms[0](h),s1,c1)
            h=h+g1[:,None]*block.self_attention(q,q,q,key_padding_mask=joint_padding,need_weights=False)[0]
            q=modulate(block.norms[1](h),s2,c2)
            key_mask=memory_padding if self.error_attention_strength==0 else bias
            h=h+g2[:,None]*block.cross_attention(q,memory,memory,key_padding_mask=key_mask,need_weights=False)[0]
            q=modulate(block.norms[2](h),s3,c3)
            h=h+g3[:,None]*block.mlp(q)
        return self.head(self.norm(h)).reshape(len(x),17,21,3)*self.token_mask

"""Add observation-only temporal evidence without replacing any base pose."""
import torch
from torch import nn
from recovery_model_v15 import RecoveryHand3D
from anchor_reference_v14 import trajectory_reference
from hand3d_temporal_v7 import pack

class ReferenceConditionHand3D(RecoveryHand3D):
    def __init__(self,policy,enabled=True):
        super().__init__(policy,True);self.reference_enabled=enabled
        self.reference_project=nn.Sequential(nn.Linear(8,self.width),nn.GELU(),nn.Linear(self.width,self.width))
        nn.init.zeros_(self.reference_project[-1].weight);nn.init.zeros_(self.reference_project[-1].bias)

    def encode(self,b):
        joint,memory,c,heat=super().encode(b)
        if not self.reference_enabled:return joint,memory,c,heat
        ref,usable=trajectory_reference(b,b['dt'],horizon=.5)
        delta=b['dt'][:,None,:]-b['dt'][:,:,None]
        valid=b['available'].all(-1)&b['rgb_valid'];near=valid[:,None]&(delta.abs()<=.5);near[:,:,8]=False
        support=near.sum(-1).float()/8
        left=(near&(delta<0)).any(-1);right=(near&(delta>0)).any(-1)
        left_gap=delta.abs().masked_fill(~(near&(delta<0)),1.).amin(-1)
        right_gap=delta.abs().masked_fill(~(near&(delta>0)),1.).amin(-1)
        flags=torch.stack([usable.float(),support,(left&right).float(),left_gap,right_gap],-1)[:,:,None].expand(-1,-1,21,-1)
        features=torch.cat([(pack(ref)-pack(b['xyz'])).clamp(-10,10),flags],-1)
        evidence=self.reference_project(features)*(usable&b['rgb_valid'])[:,:,None,None]
        flat=evidence.flatten(1,2);joint=joint+flat
        # Keep the original memory length/masks and every RGB cell intact.
        memory=torch.cat([memory[:,:17*21]+flat,memory[:,17*21:]],1)
        c=c+evidence[:,8].mean(1)
        return joint,memory,c,heat

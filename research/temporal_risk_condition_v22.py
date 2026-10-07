"""Condition historical tokens on their own error risks; current gates unchanged."""
import torch
from torch import nn
from recovery_model_v15 import RecoveryHand3D

class TemporalRiskHand3D(RecoveryHand3D):
    def __init__(self,policy):
        super().__init__(policy,True)
        self.temporal_risk_project = nn.Sequential(nn.Linear(4,self.width),nn.GELU(),nn.Linear(self.width,self.width))
        nn.init.zeros_(self.temporal_risk_project[-1].weight)
        nn.init.zeros_(self.temporal_risk_project[-1].bias)

    def encode(self,b):
        joint,memory,condition,heat = super().encode(b)
        probability = b['temporal_error_risk']
        assert probability.shape==(*b['xyz'].shape[:3],2)
        root = probability[:,:,5:6].clone()
        root[...,1] = root[...,0]
        temporal = torch.cat([root,probability],2)
        original = torch.stack([b['risk_camera'],b['risk_relative']],-1)
        original_root = original[:,5:6].clone()
        original_root[...,1] = original_root[...,0]
        original = torch.cat([original_root,original],1)[:,None]
        feature = torch.cat([temporal-original,temporal],-1)
        exists = torch.cat([b['available'][:,:,5:6],b['available']],2)
        mask = exists & b['rgb_valid'][:,:,None]
        mask[:,8] = False
        evidence = self.temporal_risk_project(feature)*mask[...,None]
        flat = evidence.flatten(1,2)
        joint = joint+flat
        memory = torch.cat([memory[:,:17*21]+flat,memory[:,17*21:]],1)
        return joint,memory,condition,heat

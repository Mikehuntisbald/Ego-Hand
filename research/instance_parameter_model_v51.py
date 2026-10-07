"""Full 3D parameter DiT with predicted instance ownership conditions.

RGB is unchanged. Neither masks annotated by humans nor keypoint labels are
inference inputs. Unknown mask quality disables only the new ownership path.
"""
import torch
from torch import nn
from torch.nn import functional as F
from protected_parameter_model_v48 import ProtectedParameterHand

class InstanceParameterHand(ProtectedParameterHand):
    def __init__(self,kind='dit',device='cpu',teacher_weight=4.):
        super().__init__(kind,device,teacher_weight)
        self.instance_cell=nn.Sequential(nn.Linear(3,self.width),nn.SiLU(),nn.Linear(self.width,self.width))
        self.instance_joint=nn.Sequential(nn.Linear(2*self.width+3,self.width),nn.SiLU(),nn.Linear(self.width,self.width))
        self.ownership_head=nn.Sequential(nn.LayerNorm(1283),nn.Linear(1283,128),nn.GELU(),nn.Linear(128,1))
        # An actual bounded parameter model remains the output; the auxiliary
        # 2D instance branch cannot independently move decoded XYZ landmarks.
        for branch in [self.instance_cell,self.instance_joint,self.ownership_head]:
            nn.init.zeros_(branch[-1].weight);nn.init.zeros_(branch[-1].bias)

    def ownership(self,b):
        features=b['rgb_native'].float();B,T,S,_=features.shape
        own=b.get('instance_own',torch.zeros(B,T,S,device=features.device)).float().clamp(0,1)
        other=b.get('instance_other',torch.zeros_like(own)).float().clamp(0,1)
        q=b.get('instance_quality',torch.zeros(B,T,device=features.device)).float().clamp(0,1)
        x=torch.cat([features,own[:,:,:,None],other[:,:,:,None],q[:,:,None,None].expand(-1,-1,S,1)],-1)
        logits=torch.logit(own.clamp(.1,.9))+self.ownership_head(x).squeeze(-1)
        return dict(logits=logits,probability=logits.sigmoid(),input_own=own,other=other,quality=q)

    def ownership_loss(self,b,target,valid):
        out=self.ownership(b);logits=out['logits'];p=out['probability'];target=target.float();valid=valid.float()
        ce=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*valid).sum()/valid.sum().clamp_min(1)
        dice=1-(2*(p*target*valid).sum()+1)/((p*valid).sum()+(target*valid).sum()+1)
        return ce+.5*dice

    def encode(self,b):
        joint,memory,c,heat=super().encode(b)
        B,T,S,_=b['rgb_native'].shape
        ownership=self.ownership(b)
        own=ownership['probability'];other=ownership['other'];quality=ownership['quality']
        quality=quality*b['rgb_valid'].float()
        # A segmentation failure is never interpreted as physical invisibility.
        quality=quality*(ownership['input_own'].sum(-1)>0).float()
        vision=memory[:,T*21:].reshape(B,T,S,self.width)
        probability=heat['probability'].float()
        weighted=probability*own[:,:,None]
        local=(weighted[:,:,:,None]*vision[:,:,None]).sum(3)/weighted.sum(-1,keepdim=True).clamp_min(1e-6)
        competing=probability*other[:,:,None]
        foreign=(competing[:,:,:,None]*vision[:,:,None]).sum(3)/competing.sum(-1,keepdim=True).clamp_min(1e-6)
        support=weighted.sum(-1);interference=competing.sum(-1)
        cues=torch.cat([local,foreign,support[:,:,:,None],interference[:,:,:,None],quality[:,:,None,None].expand(-1,-1,20,1)],-1)
        delta=self.instance_joint(cues)*quality[:,:,None,None]
        delta=torch.cat([delta.mean(2,keepdim=True),delta],2)
        cell=self.instance_cell(torch.stack([own,other,quality[:,:,None].expand(-1,-1,S)],-1))*quality[:,:,None,None]
        updated=joint.reshape(B,T,21,self.width)+delta
        memory=torch.cat([updated.flatten(1,2)*b['rgb_valid'][:,:,None,None].expand(-1,-1,21,1).flatten(1,2),(vision+cell).flatten(1,2)],1)
        heat.update(instance_support=support,instance_interference=interference,instance_quality=quality)
        return updated.flatten(1,2),memory,c+delta[:,8].mean(1),heat

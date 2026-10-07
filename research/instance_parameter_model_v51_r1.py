"""Preserve predecessor BF16 arithmetic at zero ownership residuals."""
import torch
from torch.nn import functional as F
from instance_parameter_model_v51 import InstanceParameterHand as InitialInstanceHand
from protected_parameter_model_v48 import ProtectedParameterHand

class InstanceParameterHand(InitialInstanceHand):
    def encode(self,b):
        joint,memory,c,heat=ProtectedParameterHand.encode(self,b)
        B,T,S,_=b['rgb_native'].shape;ownership=self.ownership(b)
        own=ownership['probability'];other=ownership['other'];quality=ownership['quality']
        quality=quality*b['rgb_valid'].float()*(ownership['input_own'].sum(-1)>0).float()
        vision=memory[:,T*21:].reshape(B,T,S,self.width);probability=heat['probability'].float()
        weighted=probability*own[:,:,None];competing=probability*other[:,:,None]
        local=(weighted[:,:,:,None]*vision[:,:,None]).sum(3)/weighted.sum(-1,keepdim=True).clamp_min(1e-6)
        foreign=(competing[:,:,:,None]*vision[:,:,None]).sum(3)/competing.sum(-1,keepdim=True).clamp_min(1e-6)
        support=weighted.sum(-1);interference=competing.sum(-1)
        cues=torch.cat([local,foreign,support[:,:,:,None],interference[:,:,:,None],quality[:,:,None,None].expand(-1,-1,20,1)],-1)
        delta=self.instance_joint(cues)*quality[:,:,None,None];delta=torch.cat([delta.mean(2,keepdim=True),delta],2)
        cell=self.instance_cell(torch.stack([own,other,quality[:,:,None].expand(-1,-1,S)],-1))*quality[:,:,None,None]
        # Preserve the BF16 c + BF16 diffusion-time addition and BF16
        # observation queries. Upcasting a zero residual changes that rounding.
        updated=joint.reshape(B,T,21,self.width)+delta.to(joint.dtype)
        condition=c+delta[:,8].mean(1).to(c.dtype)
        memory=torch.cat([updated.flatten(1,2)*b['rgb_valid'][:,:,None,None].expand(-1,-1,21,1).flatten(1,2),(vision+cell).flatten(1,2)],1)
        valid=b['rgb_valid'].float();h=updated.mean(2)
        context=condition+(h*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        heat['side_logits']=self.side_head(context)+F.one_hot(b['predicted_right'].long(),2).float()*4
        heat.update(instance_support=support,instance_interference=interference,instance_quality=quality)
        return updated.flatten(1,2),memory,condition,heat

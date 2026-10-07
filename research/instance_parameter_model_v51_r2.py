"""Instance-conditioned full 3D model with supervised landmark visibility."""
import torch
from torch import nn
from instance_parameter_model_v51_r1 import InstanceParameterHand as Parent

class InstanceParameterHand(Parent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.instance_visibility=nn.Sequential(nn.LayerNorm(self.width),nn.Linear(self.width,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.instance_visibility[-1].weight);nn.init.constant_(self.instance_visibility[-1].bias,2.)
    def encode(self,b):
        joint,memory,c,heat=super().encode(b)
        B,T=b['rgb_native'].shape[:2]
        logits=self.instance_visibility(joint.reshape(B,T,21,self.width)[:,:,1:]).squeeze(-1).float()
        heat.update(visibility_logits=logits,visibility_probability=logits.sigmoid())
        return joint,memory,c,heat

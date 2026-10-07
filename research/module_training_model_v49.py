"""Matched warmstart ablations; GT/teacher are loss targets only."""
import torch
from torch.nn import functional as F
from protected_parameter_model_v48 import ProtectedParameterHand

ARMS=['full','frozen_same_start','no_teacher','no_original_protection','no_protection','no_motion_supervision','no_rgb','center_only','pooled_rgb']

def transform_input(b,arm):
    b=dict(b)
    if arm=='no_rgb':b['rgb_native']=torch.zeros_like(b['rgb_native'])
    elif arm=='pooled_rgb':b['rgb_native']=b['rgb_native'].mean(2,keepdim=True).expand_as(b['rgb_native'])
    elif arm=='center_only':
        valid=b['rgb_valid'].clone();valid[:,:8]=False;valid[:,9:]=False;b['rgb_valid']=valid
        available=b['available'].clone();available[:,:8]=False;available[:,9:]=False;b['available']=available
    return b

class ModuleAblationHand(ProtectedParameterHand):
    def __init__(self,kind='dit',device='cpu',arm='full'):
        assert arm in ARMS;super().__init__(kind,device,teacher_weight=0. if arm in ['no_teacher','no_protection'] else 4.)
        self.arm=arm;self.last_generated=None
    def generate(self,*args,**kwargs):
        result=super().generate(*args,**kwargs);self.last_generated=result;return result
    def objective(self,b,target,seed):
        loss,parts=super().objective(b,target,seed);generated=self.last_generated;xyz=generated['xyz'];gt=target['gt'];valid=target['valid']
        if self.arm in ['no_original_protection','no_protection']:
            ce=(xyz-gt).norm(dim=-1);rel=lambda x:x-x[...,5:6,:]
            re=(rel(xyz)-rel(gt)).norm(dim=-1);base=b['base'];g=gt[:,8];m=valid[:,8].clone();m[:,5]=False
            be=(base-g).norm(dim=-1);br=(rel(base)-rel(g)).norm(dim=-1);gc=m&(be<=.01);gr=m&(br<=.01)
            guard=((ce[:,8]-be-.001).clamp_min(0)/.005*gc).sum()/gc.sum().clamp_min(1)
            guard+=((re[:,8]-br-.001).clamp_min(0)/.005*gr).sum()/gr.sum().clamp_min(1)
            loss=loss-2*guard;parts['removed_original_guard']=float(guard.detach())
        if self.arm=='no_motion_supervision':
            dt=b['dt'][:,1:]-b['dt'][:,:-1];mask=(valid[:,1:]&valid[:,:-1]&(dt>0)[:,:,None]).float()
            delta=(xyz[:,1:]-xyz[:,:-1])-(gt[:,1:]-gt[:,:-1])
            velocity=(F.smooth_l1_loss(delta/(dt.clamp_min(.001)[:,:,None,None]*.1),torch.zeros_like(delta),reduction='none',beta=.5).sum(-1)*mask).sum()/mask.sum().clamp_min(1)
            loss=loss-.05*velocity;parts['removed_velocity']=float(velocity.detach())
        self.last_generated=None
        return loss,parts

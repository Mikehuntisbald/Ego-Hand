"""Guard teacher-correct points through FK; inference architecture unchanged."""
import torch
from semantic_parameter_model_v36 import SemanticParameterHand

class ProtectedParameterHand(SemanticParameterHand):
    def __init__(self,kind='dit',device='cpu',teacher_weight=4.):
        super().__init__(kind,device);self.teacher_weight=teacher_weight;self._capture=False;self._generated=None
    def generate(self,*args,**kwargs):
        result=super().generate(*args,**kwargs)
        if self._capture:self._generated=result
        return result
    def objective(self,b,target,seed):
        self._capture=True
        try:
            base,parts=super().objective(b,target,seed);generated=self._generated
        finally:self._capture=False;self._generated=None
        guard,extra=self.teacher_penalty(generated['xyz'][:,8],target)
        return base+self.teacher_weight*guard,dict(parts,**extra)
    @staticmethod
    def teacher_penalty(xyz,target):
        gt=target['gt'][:,8];teacher=target['teacher_xyz'].detach()
        mask=target['valid'][:,8].clone();mask[:,5]=False
        teacher_c=(teacher-gt).norm(dim=-1);student_c=(xyz-gt).norm(dim=-1)
        relative=lambda x:x-x[:,5:6]
        teacher_r=(relative(teacher)-relative(gt)).norm(dim=-1);student_r=(relative(xyz)-relative(gt)).norm(dim=-1)
        mc=mask&(teacher_c<=.01);mr=mask&(teacher_r<=.01)
        guard=((student_c-teacher_c-.001).clamp_min(0)/.005*mc).sum()/mc.sum().clamp_min(1)
        guard+=((student_r-teacher_r-.001).clamp_min(0)/.005*mr).sum()/mr.sum().clamp_min(1)
        parts=dict(teacher_guard=float(guard.detach()),teacher_camera_good=int(mc.sum()),teacher_relative_good=int(mr.sum()))
        return guard,parts

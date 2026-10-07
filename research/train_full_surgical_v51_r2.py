"""Full-model continuation with genuine glove keypoint supervision.

External 2D labels supervise projected FK outputs and the spatial localizer.
There are no fabricated 3D or diffusion x0 targets for the surgical domain.
The original timed HOT3D 3D/diffusion/protection objectives remain active.
"""
import json,hashlib,os
from pathlib import Path
import torch,numpy as np
from torch.nn import functional as F
import train_full_instance_v51_r1 as trainer
from instance_parameter_model_v51_r2 import InstanceParameterHand
from online_parameter_model_v47 import OnlineVisual
from complete_instance_v51_r1 import pinhole_project
from cache_instance_conditions_v51 import RUN

from prepare_paired_domain_v51 import PAIRED
PHASE=PAIRED
LATEST=None

class SurgicalFullHand(InstanceParameterHand):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        global LATEST;LATEST=self
        object.__setattr__(self,'aux_visual',None)
        saved=torch.load(RUN/'surgical_pose/inputs.pt',weights_only=False,mmap=True)
        object.__setattr__(self,'aux_inputs',saved['inputs']);object.__setattr__(self,'aux_metadata',saved['metadata'])
        object.__setattr__(self,'aux_targets',torch.load(RUN/'surgical_pose/targets.pt',weights_only=False,mmap=True))
        object.__setattr__(self,'aux_pixels',np.load(RUN/'surgical_pose/pixels.npy',mmap_mode='r'))
        object.__setattr__(self,'aux_train',[i for i,r in enumerate(saved['metadata']) if r['split']=='train'])
        object.__setattr__(self,'aux_dev',[i for i,r in enumerate(saved['metadata']) if r['split']=='dev'])

    def auxiliary_batch(self,ids,device):
        b={k:v[ids].to(device) for k,v in self.aux_inputs.items()};live=self.aux_visual.encode_pixels([self.aux_pixels[i] for i in ids],device)
        native=torch.zeros(len(ids),17,192,1280,device=device);native[:,8]=live;b['rgb_native']=native
        return b

    def objective(self,b,target,seed):
        native,parts=super().objective(b,target,seed)
        ids=[self.aux_train[(seed*37+j*101)%len(self.aux_train)] for j in range(2)]
        external=self.auxiliary_batch(ids,b['base'].device);generated=self.generate(external,seed+51000)
        uv=pinhole_project(generated['xyz'][:,8],external['camera_params'])/1408
        # Annotated pixels/visibility/handedness are targets only, never inputs.
        gt=self.aux_targets['xy'][ids].to(uv.device);valid=self.aux_targets['valid'][ids].to(uv.device)
        visible=self.aux_targets['visible'][ids].to(uv.device);right=self.aux_targets['right'][ids].to(uv.device)
        scale=(external['roi'][:,8,2:]-external['roi'][:,8,:2]).mean(-1).clamp_min(.01)
        weight=valid.float()*torch.where(visible,1.,.5)
        error=F.smooth_l1_loss((uv-gt)/scale[:,None,None]*10,torch.zeros_like(uv),reduction='none',beta=.5).sum(-1)
        pose=(error*weight).sum()/weight.sum().clamp_min(1)
        heat={k:v[:,8] for k,v in generated['encoded'][3].items() if k in ['xy','logits','probability','features']}
        localization=self.visual_head.loss(heat,gt,valid,external['positions'][:,8],external['roi'][:,8])
        side=F.cross_entropy(generated['side_logits'],right)
        visibility=(F.binary_cross_entropy_with_logits(generated['encoded'][3]['visibility_logits'][:,8],visible.float(),reduction='none')*valid).sum()/valid.sum().clamp_min(1)
        return native+.3*pose+.025*localization+.1*side+.05*visibility,dict(parts,surgical_FK_projection=float(pose.detach()),surgical_localization=float(localization.detach()),surgical_side=float(side.detach()),surgical_visibility=float(visibility.detach()),surgical_annotated_points=int(valid.sum()),surgical_occluded_annotated_points=int((valid&~visible).sum()))

class SurgicalVisual(OnlineVisual):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);object.__setattr__(LATEST,'aux_visual',self)

def main():
    PHASE.mkdir(exist_ok=True)
    for name in ['mask_conditions.pt']:
        dest=PHASE/name
        if not dest.exists():os.symlink(RUN/name,dest)
    meta=json.loads((RUN/'surgical_pose/cache_done.json').read_text());assert meta['complete'] and not meta['GT_3D']
    (PHASE/'supervision_protocol.json').write_text(json.dumps(dict(full_model=True,natural_glove_2D_annotation=meta,GT_3D_fabricated=False,diffusion_parameter_supervision='HOT3D real GT parameters only',external_diffusion_supervision='Differentiable DDIM sample -> FK -> pinhole projection -> human 2D coordinates',GT_or_teacher_inference_inputs=False,default_changed=False),indent=2))
    trainer.RUN=PHASE;trainer.INITIAL=Path('/mnt/why/HOT3D/experiments/online_rgb_iterative_v48/protected/best.pt');trainer.InstanceParameterHand=SurgicalFullHand;trainer.OnlineVisual=SurgicalVisual
    trainer.main()

if __name__=='__main__':main()

"""Inference from RGB hand crops and camera geometry only; no GT required."""
import torch
from coarse_pose3d import CoarsePose3D
from residual_models import ResidualModel
from oof_calibration import calibrated_confidence,frozen_gates
from oof_common import RUN,OLD

class SubjectOOFPoseRefiner:
    def __init__(self,method='oof_dit',device='cuda:0'):
        assert method in ['oof_dit','oof_regression','in_subject_dit']
        self.device=device;self.checkpoint=torch.load(RUN/method/'best.pt',map_location='cpu',weights_only=False)
        self.coarse=CoarsePose3D().to(device).eval()
        self.coarse.load_state_dict(torch.load(OLD/'coarse_fine/best.pt',map_location='cpu',weights_only=False)['model'])
        self.rgb_encoder=CoarsePose3D().to(device).eval()
        self.rgb_encoder.load_state_dict(torch.load(RUN/'rgb_encoder.pt',map_location='cpu',weights_only=False)['model'])
        self.refiner=ResidualModel(kind='regression' if method=='oof_regression' else 'dit').to(device).eval()
        self.refiner.load_state_dict(self.checkpoint['model'])
    @torch.inference_mode()
    def __call__(self,images,geometry,seed=901003):
        images=images.to(self.device);geometry=geometry.to(self.device)
        generator=torch.Generator(device=self.device).manual_seed(seed)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            # The study caches observations in batches of 256. BF16 coarse
            # convolutions can shift by millimeters with a different batch
            # shape, so pad observation batches to the same shape at deployment.
            coarse=[];confidences=[];tokens=[]
            for start in range(0,len(images),256):
                im=images[start:start+256];geo=geometry[start:start+256];n=len(im)
                if n<256:
                    im=torch.cat([im,torch.zeros(256-n,*im.shape[1:],device=im.device,dtype=im.dtype)])
                    geo=torch.cat([geo,torch.zeros(256-n,geo.shape[1],device=geo.device,dtype=geo.dtype)])
                base=self.coarse(im,geo);features=self.rgb_encoder(im,geo)['rgb_tokens']
                coarse.append(base['xyz'][:n].float())
                confidences.append(torch.cat([base['root_confidence'][:n,None],base['confidence'][:n]],1).float())
                tokens.append(features[:n].half().float()) # match the cache's FP16 storage
            c=torch.cat(coarse);conf=torch.cat(confidences);features=torch.cat(tokens)
            calibrated=calibrated_confidence(conf,self.checkpoint['uncertainty_calibration'])
            proposal=self.refiner.propose(c,calibrated,features.float(),steps=10,samples=4 if self.refiner.kind=='dit' else 1,generator=generator)
            logits,_=self.refiner.gates(proposal,c,calibrated,features.float())
            gates=frozen_gates(self.refiner,logits,self.checkpoint['calibration'])
            xyz=self.refiner.apply_gates(proposal,c,gates)
        return dict(coarse_xyz_m=c.float(),refined_xyz_m=xyz.float(),confidence=calibrated.float(),gates=gates.float())

"""Prediction-only observation encoding and gated 3D inference for v3."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import cv2
import wilor_eval_common as common
from cache_dit_v3 import observed_crop
from coarse_pose3d import CoarsePose3D
from prepare_pose_training import crop_and_geometry
from dit_v5_model import VisualResidual
from dit_v3_sampling import propose
from compare_detectors import iou

ROOT=common.ROOT
RUN=ROOT/'experiments/dit_wilor_v5'


class ObservationEncoder:
    def __init__(self,device='cuda:0'):
        self.device=device
        self.wilor,_=common.load_model(device)
        self.capture={}
        self.handle=self.wilor.backbone.register_forward_hook(lambda module,inputs,output:self.capture.update(features=output[-1]))
        self.coarse=CoarsePose3D(str(ROOT/'weights/yolo26s.pt')).to(device).eval()
        self.coarse.load_state_dict(torch.load(ROOT/'experiments/dit_lowconfidence_v1/coarse_fine/best.pt',map_location='cpu',weights_only=False)['model'])

    @torch.inference_mode()
    def features(self,crops,rotations,focals,rights,side_conf,side_iou,reproj,fallback):
        inputs=common.input_tensor(crops,rights,1).to(self.device)
        out=self.wilor({'img':inputs});features=self.capture['features']
        global_features=F.adaptive_avg_pool2d(features,(8,6)).flatten(2).transpose(1,2)
        xy=out['pred_keypoints_2d'][:,common.MAPPING].float()
        grid=torch.stack([xy[...,0]*256/192*2,xy[...,1]*2],-1)
        local=F.grid_sample(features,grid[:,:,None],mode='bilinear',padding_mode='zeros',align_corners=False)[:,:,:,0].transpose(1,2)
        joints=out['pred_keypoints_3d'][:,common.MAPPING].float()
        cam=out['pred_cam'].float();f=torch.tensor(focals,device=self.device)
        translation=torch.stack([cam[:,1],cam[:,2],2*f/(256*cam[:,0].clamp_min(1e-6))],-1)
        Q=np.array([[0,1,0],[-1,0,0],[0,0,1]],np.float32)
        transform=np.asarray(rotations)@Q.T;transform[:,:,0]*=(2*np.asarray(rights)-1)[:,None]
        transform=torch.tensor(transform,device=self.device)
        result=dict(global_features=global_features.half(),local=local.half(),
                    wilor=torch.einsum('bij,bkj->bki',transform,joints+translation[:,None]),transform=transform,
                    geometry=torch.tensor(np.column_stack([focals,rights,side_conf,side_iou,reproj]),device=self.device).float(),
                    wilor_2d=xy,shape=out['pred_mano_params']['betas'].float(),crop_fallback=torch.tensor(fallback,device=self.device))
        result['global']=result.pop('global_features')
        return result

    @torch.inference_mode()
    def coarse_predictions(self,images,geometries):
        n=len(images)
        x=torch.from_numpy(np.stack(images)[...,::-1].transpose(0,3,1,2).copy()).to(self.device).float()/255
        g=torch.from_numpy(np.stack(geometries)).to(self.device).float()
        assert n<=256
        if n<256:
            x=torch.cat([x,torch.zeros(256-n,3,256,256,device=self.device)])
            g=torch.cat([g,torch.zeros(256-n,21,device=self.device)])
        with torch.autocast('cuda',dtype=torch.bfloat16):out=self.coarse(x,g)
        return dict(coarse=out['xyz'][:n].float(),confidence=torch.cat([out['root_confidence'][:n,None],out['confidence'][:n]],1).float())


class Refiner:
    def __init__(self,checkpoint,device='cuda:0'):
        self.device=device
        self.checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model=VisualResidual(self.checkpoint['kind']).to(device).eval()
        self.model.coherent_gate=self.checkpoint.get('coherent_gate',False)
        self.model.load_state_dict(self.checkpoint['model'])

    @torch.inference_mode()
    def __call__(self,observation,seed=916003,generator=None):
        allowed=['coarse','confidence','wilor','transform','geometry','wilor_2d','shape','global','local']
        b={k:observation[k].to(self.device) for k in allowed}
        sampling=self.checkpoint.get('sampling',dict(policy='independent',samples=2,steps=10))
        if generator is None:generator=torch.Generator(device=self.device).manual_seed(seed)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            proposal=propose(self.model,b,generator=generator,**sampling)
            raw=self.model.gate_values(proposal,b).float()
        op=self.checkpoint['operating'];gates=raw*(raw>=op['threshold'])*op['strength']
        pose=self.model.apply(proposal,b,gates)
        return dict(xyz=pose,coarse=b['coarse'],gates=gates,proposal=proposal.float())


def prepare_observation(image,box,camera,side_predictions):
    coarse_crop,roi,geometry=crop_and_geometry(image,box,camera)
    coarse_crop=cv2.imdecode(cv2.imencode('.jpg',coarse_crop,[cv2.IMWRITE_JPEG_QUALITY,95])[1],cv2.IMREAD_COLOR)
    side=side_predictions;overlap=iou([roi],side['boxes'])[0]
    if len(overlap) and overlap.max()>=.05:
        j=int(overlap.argmax());right=int(side['classes'][j]);side_conf=float(side['scores'][j]);side_iou=float(overlap[j])
    else:right=1;side_conf=0.;side_iou=0.
    warped,R,f,error,fallback=observed_crop(image,roi,camera)
    return dict(coarse_crop=coarse_crop,coarse_geometry=geometry,roi=roi,
                wilor_crop=warped,rotation=R,focal=f,right=right,side_conf=side_conf,side_iou=side_iou,reprojection_error=error,crop_fallback=fallback)


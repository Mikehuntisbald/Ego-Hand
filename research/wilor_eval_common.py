"""WiLoR inference adapter with calibrated, prediction-only perspective crops."""
import inspect
import json
import os
import sys
from pathlib import Path
import numpy as np

RUN=Path('/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003')
ROOT=Path('/mnt/why/HOT3D')
SOURCE=Path((RUN/'source_path.txt').read_text())
sys.path.insert(0,str(SOURCE))
sys.path.append('/mnt/why/HOT3D/.venv/lib/python3.12/site-packages')
sys.path.append('/mnt/why/HOT3D-hand-tracking-toolkit')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
# Legacy chumpy assets, not a change to network computation.
if not hasattr(inspect,'getargspec'):
    inspect.getargspec=inspect.getfullargspec
for k,v in {'bool':np.bool_,'int':int,'float':float,'complex':complex,'object':object,'str':str,'unicode':str}.items():
    if k not in np.__dict__:setattr(np,k,v)
import cv2
import torch
from hand_tracking_toolkit.camera import from_json
from scipy.optimize import root

MAPPING=[4,8,12,16,20,0,2,3,5,6,7,9,10,11,13,14,15,17,18,19]
MEAN=np.array([.485,.456,.406],np.float32)[None,None]
STD=np.array([.229,.224,.225],np.float32)[None,None]


def unproject(cam,uv):
    target=(np.asarray(uv,float)-cam.c)/cam.f
    initial=target*min(1.,.65/max(np.linalg.norm(target),1e-9))
    solution=root(lambda p:cam.distort.evaluate(p)-target,initial,tol=1e-10)
    ray=cam.unproject(solution.x)
    error=np.linalg.norm(cam.eye_to_window(ray)-uv)
    assert np.isfinite(ray).all() and error<.01,(uv,error,solution.message)
    return ray


def load_model(device='cuda:2'):
    from wilor.configs import get_config
    from wilor.models import WiLoR
    cfg=get_config(str(RUN/'assets/model_config.yaml'))
    cfg.defrost()
    cfg.MODEL.BBOX_SHAPE=[192,256]
    cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS',None)
    cfg.MANO.MODEL_PATH=str(RUN/'assets/MANO_RIGHT.pkl')
    cfg.MANO.DATA_DIR=str(RUN/'assets')
    cfg.MANO.MEAN_PARAMS=str(RUN/'assets/mano_mean_params.npz')
    cfg.LOSS_WEIGHTS.ADVERSARIAL=0
    cfg.freeze()
    os.chdir(SOURCE)
    model=WiLoR(cfg,init_renderer=False)
    checkpoint=RUN/'assets/wilor_final.mirror.ckpt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=ck.get('state_dict',ck)
    mismatch=model.load_state_dict(state,strict=False)
    missing=list(mismatch.missing_keys)
    unexpected=[k for k in mismatch.unexpected_keys if not k.startswith('discriminator.')]
    assert not missing and not unexpected,(missing,unexpected)
    model.to(device).eval()
    return model,cfg


def crop(image,box,camera_json,padding=1.3):
    cam=from_json(camera_json)
    box=np.asarray(box,float);center=(box[:2]+box[2:])/2
    ray=unproject(cam,center);z=ray/np.linalg.norm(ray)
    x=np.array([1.,0.,0.]);x-=z*np.dot(x,z);x/=np.linalg.norm(x)
    y=np.cross(z,x);R=np.stack([x,y,z],axis=1)
    corners=np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]],
                      [center[0],box[1]],[center[0],box[3]],[box[0],center[1]],[box[2],center[1]]])
    rays=np.stack([unproject(cam,p) for p in corners])@R
    xy=rays[:,:2]/rays[:,2:3].clip(.05)
    halfspan=max(float(np.abs(xy).max()),.015)*padding
    focal=95.5/halfspan # hand fits inside WiLoR's 192-pixel central image band
    yy,xx=np.mgrid[:256,:256]
    virtual=np.stack([(xx-127.5)/focal,(yy-127.5)/focal,np.ones_like(xx)],-1).reshape(-1,3)
    native=virtual@R.T
    uv=cam.eye_to_window(native).astype(np.float32).reshape(256,256,2)
    warped=cv2.remap(image,uv[:,:,0],uv[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    reprojection_error=float(np.linalg.norm(cam.eye_to_window(z)-center))
    return warped,R.astype(np.float32),float(focal),reprojection_error


def input_tensor(crops,rights,rotation):
    images=[]
    for image,right in zip(crops,rights):
        image=np.rot90(image,rotation)
        if not right:image=image[:,::-1]
        rgb=image[:,:,::-1].astype(np.float32)/255
        images.append(((rgb-MEAN)/STD).transpose(2,0,1).copy())
    return torch.from_numpy(np.stack(images))


@torch.inference_mode()
def predict(model,crops,rights,rotations,focals,rotation=0,device='cuda:2'):
    inp=input_tensor(crops,rights,rotation).to(device)
    out=model({'img':inp})
    joints=out['pred_keypoints_3d'].float().cpu().numpy()[:,MAPPING]
    cam=out['pred_cam'].float().cpu().numpy()
    sign=2*np.asarray(rights)-1
    joints[:,:,0]*=sign[:,None]
    cam[:,1]*=sign
    translation=np.stack([cam[:,1],cam[:,2],2*np.asarray(focals)/(256*np.maximum(cam[:,0],1e-6))],axis=-1)
    Q=np.linalg.matrix_power(np.array([[0,1,0],[-1,0,0],[0,0,1]],np.float32),rotation)
    R=np.asarray(rotations)@Q.T
    xyz=np.einsum('bij,bkj->bki',R,joints+translation[:,None])
    relative=np.einsum('bij,bkj->bki',R,joints-joints[:,5:6])
    return dict(xyz=xyz.astype(np.float32),relative=relative.astype(np.float32),cam=cam,
                side=np.asarray(rights),focals=np.asarray(focals))


if __name__=='__main__':
    model,cfg=load_model()
    print(json.dumps(dict(loaded=True,parameters=sum(p.numel() for p in model.parameters()))),flush=True)

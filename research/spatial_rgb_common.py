"""Dense hand-pretrained visual conditions for offline 2D annotation.

All geometry below uses the supplied detector ROI and camera only. No current
unmasked handedness prediction, GT pose, or visibility enters the encoder.
"""
import json
from pathlib import Path
import wilor_eval_common as common
import cv2
import numpy as np
import torch
from cache_dit_v3 import observed_crop

OLD=common.ROOT/'experiments/offline_keypoint_rgb_v2'
RUN=common.ROOT/'experiments/offline_keypoint_spatial_v3'
Q=np.array([[0,1,0],[-1,0,0],[0,0,1]],np.float32)

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def prepare(record,roi,rects,color=None):
    image=cv2.imread(record['image']);assert image is not None
    h,w=image.shape[:2];assert (h,w)==(1408,1408)
    cam=common.from_json(record['camera'])
    _,R,f,error,fallback=observed_crop(image,roi,record['camera'])
    yy,xx=np.mgrid[:256,:256]
    rays=np.stack([(xx-127.5)/f,(yy-127.5)/f,np.ones_like(xx)],-1)@R.T
    uv=cam.eye_to_window(rays.reshape(-1,3)).reshape(256,256,2).astype(np.float32)
    crops=[];color=32+64*((record['clip']+1)%4) if color is None else color
    for rect in rects:
        covered=image.copy()
        if rect[2]>rect[0] and rect[3]>rect[1]:
            native=roi[:2]+np.asarray(rect).reshape(2,2)*(roi[2:]-roi[:2])/256
            # Cover the bilinear remapper's one-pixel support too. Otherwise a
            # fully hidden ROI could retain source pixels along fractional edges.
            lo=np.maximum(np.floor(native[0]).astype(int)-1,0)
            hi=np.minimum(np.ceil(native[1]).astype(int)+1,[w,h])
            covered[lo[1]:hi[1],lo[0]:hi[0]]=color
        warped=cv2.remap(covered,uv[...,0],uv[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        # Perspective correction can otherwise expose pixels beyond the supplied
        # square ROI. Match the old experiment's available image region exactly.
        inside=(uv[...,0]>=roi[0])&(uv[...,0]<roi[2])&(uv[...,1]>=roi[1])&(uv[...,1]<roi[3])
        warped[~inside]=0
        crops.append(warped)
    # Actual patch embedding: kernel16, stride16, padding2, central 192 band.
    gy,gx=np.mgrid[:16,:12];canonical=np.stack([gx*16+37.5,gy*16+5.5],-1).reshape(-1,2)
    transform=R@Q.T
    rays=np.c_[(canonical-127.5)/f,np.ones(192)]@transform.T
    positions=cam.eye_to_window(rays).astype(np.float32)/1408
    assert np.isfinite(positions).all() and error<.01
    return crops,positions,transform,f,error,fallback

def records_and_index():
    records=json.loads((OLD/'rgb_records.json').read_text())
    index=torch.load(OLD/'rgb_index.pt',weights_only=False)
    return records,index

def targets(records,index,geometry):
    # Called by training/evaluation only; never by image preparation/inference.
    n=len(records)+1;gt=np.zeros((n,20,2),np.float32);valid=np.zeros((n,20),bool)
    for i,r in enumerate(records,1):
        if not r['matched']:continue
        uv=common.from_json(r['camera']).eye_to_window(np.asarray(r['gt']))/1408
        gt[i]=np.nan_to_num(uv);valid[i]=np.asarray(r['projection_valid'])&np.isfinite(uv).all(-1)
    return torch.from_numpy(gt),torch.from_numpy(valid)

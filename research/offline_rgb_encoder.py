"""Shared frozen YOLO RGB encoder; every occluder is applied before encoding."""
import hashlib
from pathlib import Path
import wilor_eval_common as common
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO

WEIGHT=common.ROOT/'experiments/dit_lowconfidence_v1/detector/weights/best.pt'

def crop_roi(box):
    b=np.asarray(box,float);center=(b[:2]+b[2:])/2;side=max(b[2:]-b[:2])*1.4
    side=max(side,16.)
    return np.r_[center-side/2,center+side/2].astype(np.float32)

def crop_image(image,roi):
    x1,y1,x2,y2=roi;s=256/(x2-x1)
    return cv2.warpAffine(image,np.array([[s,0,-s*x1],[0,s,-s*y1]],np.float32),(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

def rectangles(sequence,clip):
    # Determined ONLY by sequence/clip metadata, never keypoints or GT.
    seed=int(hashlib.sha256(f'{sequence}/{clip}'.encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed);f=rng.uniform(.40,.70,4);n=(f*256).astype(int)
    return np.array([[0,0,0,0],[0,0,n[0],256],[256-n[1],0,256,256],[0,0,256,n[2]],[0,256-n[3],256,256],[0,0,256,256]],np.int64)

def cover(crop,rect,color=128):
    result=crop.copy();x1,y1,x2,y2=map(int,rect);result[y1:y2,x1:x2]=color;return result

class RGBEncoder(nn.Module):
    def __init__(self):
        super().__init__();full=YOLO(str(WEIGHT)).model
        self.layers=nn.ModuleList(list(full.model[:7]));self.eval()
        for p in self.parameters():p.requires_grad_(False)
    def forward(self,image):
        low=None;x=image
        for i,layer in enumerate(self.layers):
            assert layer.f==-1;x=layer(x)
            if i==4:low=x
        return torch.cat([F.adaptive_avg_pool2d(low,4),F.adaptive_avg_pool2d(x,4)],1).flatten(2).transpose(1,2)

def input_tensor(images,device):
    return torch.from_numpy(np.stack(images)[...,::-1].transpose(0,3,1,2).copy()).to(device).float()/255

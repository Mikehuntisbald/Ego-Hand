"""Resumable frozen WiLoR conditions on the existing OOF data lineage."""
import argparse
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import defaultdict
import wilor_eval_common as common
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from ultralytics import YOLO
from compare_detectors import iou

ROOT=common.ROOT
RUN=ROOT/'experiments/dit_wilor_v3'
SOURCE_CACHE=ROOT/'experiments/dit_subject_oof_v2/oof_cache.pt'


def observed_crop(image,roi,camera_json):
    # An expanded observed ROI can extend beyond the sensor. Never invert an
    # unobserved far-off-sensor corner as though it were a valid calibrated ray.
    h,w=image.shape[:2]
    box=np.clip(roi,[0,0,0,0],[w-1,h-1,w-1,h-1])
    try:
        return (*common.crop(image,box,camera_json,padding=1.),False)
    except AssertionError:
        cam=common.from_json(camera_json);center=(box[:2]+box[2:])/2
        z=common.unproject(cam,center);z/=np.linalg.norm(z)
        x=np.array([1.,0.,0.]);x-=z*np.dot(x,z);x/=np.linalg.norm(x)
        y=np.cross(z,x);R=np.stack([x,y,z],1);eps=1e-4
        jac=np.stack([(cam.eye_to_window(z+eps*a)-cam.eye_to_window(z-eps*a))/(2*eps) for a in [x,y]],1)
        corners=np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]])
        xy=np.linalg.solve(jac,(corners-center).T).T
        focal=95.5/max(float(np.abs(xy).max()),.015)
        yy,xx=np.mgrid[:256,:256]
        rays=np.stack([(xx-127.5)/focal,(yy-127.5)/focal,np.ones_like(xx)],-1).reshape(-1,3)@R.T
        uv=cam.eye_to_window(rays).astype(np.float32).reshape(256,256,2)
        warped=cv2.remap(image,uv[:,:,0],uv[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        return warped,R.astype(np.float32),focal,float(np.linalg.norm(cam.eye_to_window(z)-center)),True


def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def side_predictions(rows,device):
    destination=RUN/'side_predictions.json'
    images=sorted({r['image'] for r in rows})
    if destination.exists():
        result=json.loads(destination.read_text());assert set(result)==set(images);return result
    model=YOLO(str(ROOT/'experiments/detector_compare_wilor_20261003/wilor_detector.pt'))
    results={}
    partial=RUN/'side_predictions.partial.json'
    if partial.exists():results=json.loads(partial.read_text())
    pending=[im for im in images if im not in results]
    for start in range(0,len(pending),32):
        paths=pending[start:start+32]
        output=model.predict(paths,imgsz=960,device=device,batch=32,conf=.001,max_det=100,verbose=False)
        for path,p in zip(paths,output):
            results[path]=dict(boxes=p.boxes.xyxy.cpu().tolist(),classes=p.boxes.cls.cpu().tolist(),scores=p.boxes.conf.cpu().tolist())
        if start%320==0:
            save(partial,results);print(json.dumps(dict(stage='side',done=len(results),total=len(images))),flush=True)
    save(destination,results)
    del model;torch.cuda.empty_cache()
    return results


def annotations(rows):
    files={}
    for r in rows:
        image=Path(r['image']);split=image.parent.parent.parent.name
        path=ROOT/'export/annotations'/split/r['sequence']/f'clip-{r["clip"]:06d}.jsonl'
        if path not in files:files[path]=None
    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded=list(pool.map(lambda p:[json.loads(l) for l in p.read_text().splitlines()],files))
    return {(r['clip'],r['frame']):r['camera'] for clip in loaded for r in clip}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='3');parser.add_argument('--batch',type=int,default=16);args=parser.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(0)
    device='cuda:'+args.device
    raw=torch.load(SOURCE_CACHE,map_location='cpu',weights_only=False,mmap=True)
    rows=raw['rows']
    meta=[]
    for i,r in enumerate(rows):
        role=r['role'] if r['role'] in ['denoise','gate_fit'] else 'development'
        meta.append(dict(index=i,role=role,subject=r['subject'],sequence=r['sequence'],clip=r['clip'],frame=r['frame'],
                         image=r['image'],roi=r['roi'],side_label=r['side'],visibility_label=r['visible_fraction'],projection_valid=r['projection_valid']))
    save(RUN/'rows.json',meta)
    observation=dict(gt=raw['gt'].float(),coarse=raw['coarse'].float(),confidence=raw['confidence'].float(),
                     final_coarse=raw['in_subject_coarse'].float(),final_confidence=raw['in_subject_confidence'].float())
    torch.save(observation,RUN/'observations.pt')
    del raw
    sides=side_predictions(rows,args.device)
    cameras=annotations(rows)
    def prepare(i):
        row=rows[i];roi=np.array(row['roi']);side=sides[row['image']]
        overlap=iou([roi],side['boxes'])[0]
        if len(overlap) and overlap.max()>=.05:
            j=int(overlap.argmax());right=int(side['classes'][j]);confidence=float(side['scores'][j]);matched=float(overlap[j])
        else:right=1;confidence=0.;matched=0.
        image=cv2.imread(row['image'])
        # roi already includes the original 1.3 expansion.
        warped,R,f,error,fallback=observed_crop(image,roi,cameras[(row['clip'],row['frame'])])
        return warped,R,f,right,confidence,matched,error,fallback
    model,cfg=common.load_model(device)
    captured={}
    def hook(module,inputs,output):captured['features']=output[-1]
    handle=model.backbone.register_forward_hook(hook)
    folder=RUN/'feature_chunks';folder.mkdir(exist_ok=True)
    Q=np.array([[0,1,0],[-1,0,0],[0,0,1]],np.float32)
    chunk_size=512;started=time.time()
    with ThreadPoolExecutor(max_workers=8) as pool,torch.inference_mode():
        for chunk_start in range(0,len(rows),chunk_size):
            dest=folder/f'{chunk_start:06d}.pt'
            if dest.exists():continue
            values=defaultdict(list)
            end=min(len(rows),chunk_start+chunk_size)
            for start in range(chunk_start,end,args.batch):
                ix=list(range(start,min(end,start+args.batch)))
                prepared=list(pool.map(prepare,ix))
                crops,R,focal,rights,side_conf,side_iou,reproj,fallback=map(list,zip(*prepared))
                x=common.input_tensor(crops,rights,rotation=1).to(device)
                out=model({'img':x})
                features=captured['features']
                global_features=F.adaptive_avg_pool2d(features,(8,6)).flatten(2).transpose(1,2)
                xy=out['pred_keypoints_2d'][:,common.MAPPING].float()
                grid=torch.stack([xy[...,0]*256/192*2,xy[...,1]*2],dim=-1)
                local_features=F.grid_sample(features,grid[:,:,None],mode='bilinear',padding_mode='zeros',align_corners=False)[:,:,:,0].transpose(1,2)
                joints=out['pred_keypoints_3d'][:,common.MAPPING].float().cpu()
                cam=out['pred_cam'].float().cpu();f=torch.tensor(focal)
                translation=torch.stack([cam[:,1],cam[:,2],2*f/(256*cam[:,0].clamp_min(1e-6))],-1)
                sign=2*np.asarray(rights)-1
                transform=np.asarray(R)@Q.T
                transform[:,:,0]*=sign[:,None]
                canonical_pose=joints+translation[:,None]
                xyz=torch.einsum('bij,bkj->bki',torch.tensor(transform),canonical_pose)
                values['global'].append(global_features.half().cpu())
                values['local'].append(local_features.half().cpu())
                values['wilor'].append(xyz.float())
                values['transform'].append(torch.tensor(transform).float())
                values['geometry'].append(torch.tensor(np.column_stack([focal,rights,side_conf,side_iou,reproj])).float())
                values['wilor_2d'].append(xy.cpu())
                values['shape'].append(out['pred_mano_params']['betas'].float().cpu())
                values['crop_fallback'].append(torch.tensor(fallback))
            chunk={k:torch.cat(v) for k,v in values.items()}
            assert all(torch.isfinite(v).all() for v in chunk.values())
            chunk['start']=chunk_start;chunk['end']=end
            tmp=dest.with_suffix('.partial');torch.save(chunk,tmp);tmp.replace(dest)
            status=dict(stage='condition_cache',done=end,total=len(rows),seconds=time.time()-started,complete=False,acceptance_passed=False)
            save(RUN/'status.json',status);print(json.dumps(status),flush=True)
    handle.remove()
    expected=list(range(0,len(rows),chunk_size))
    assert all((folder/f'{i:06d}.pt').exists() for i in expected)
    save(RUN/'condition_cache_done.json',dict(complete=True,samples=len(rows),chunks=len(expected),global_shape=[48,1280],local_shape=[20,1280],
        source_cache=str(SOURCE_CACHE),feature_basis='Frozen WiLoR pretrained backbone; no random frozen adapter',rotation=1,padding_on_observed_roi=1.,
        gt_fields='Only observations.gt and rows.*_label/projection_valid; none passed to prediction',
        coarse_training='Existing six-fold subject OOF; final estimator retained as alternate train observation and deployment baseline'))
    save(RUN/'status.json',dict(stage='condition_cache_complete',complete=False,acceptance_passed=False))
    print('CONDITION_CACHE_COMPLETE',flush=True)


if __name__=='__main__':main()

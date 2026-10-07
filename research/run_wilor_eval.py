"""Observe -> development rotation selection -> frozen 3D evaluation; no DiT."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from wilor_eval_common import ROOT,RUN,crop,load_model,predict
import cv2
import numpy as np
import torch
from compare_detectors import save
from coarse_pose3d import CoarsePose3D
from prepare_pose_training import crop_and_geometry
from metrics_3d import EVAL_INDICES,pose_metrics


def prepare(samples):
    receipt=RUN/'crop_cache_done.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('geometry_version')==2:
        assert json.loads(receipt.read_text())['manifest_sha256']==hashlib.sha256((RUN/'samples.json').read_bytes()).hexdigest()
        return
    def worker(s):
        im=cv2.imread(s['image'])
        warped,R,f,e=crop(im,s['box'],s['camera'])
        old,roi,geo=crop_and_geometry(im,s['box'],s['camera'])
        # Match old coarse estimator's JPEG95 training observation format.
        old=cv2.imdecode(cv2.imencode('.jpg',old,[cv2.IMWRITE_JPEG_QUALITY,95])[1],cv2.IMREAD_COLOR)
        return warped,R,f,e,old,geo
    n=len(samples)
    wc=np.lib.format.open_memmap(RUN/'wilor_crops.npy',mode='w+',dtype=np.uint8,shape=(n,256,256,3))
    cc=np.lib.format.open_memmap(RUN/'coarse_crops.npy',mode='w+',dtype=np.uint8,shape=(n,256,256,3))
    rotations=[];focals=[];errors=[];geometry=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,(w,R,f,e,c,g) in enumerate(pool.map(worker,samples)):
            wc[i]=w;cc[i]=c;rotations.append(R);focals.append(f);errors.append(e);geometry.append(g)
            if i%200==0:print(json.dumps(dict(stage='crops',done=i,total=n)),flush=True)
    wc.flush();cc.flush()
    np.savez(RUN/'crop_geometry.npz',rotations=np.stack(rotations),focals=focals,ray_reprojection_error=errors,geometry=np.stack(geometry))
    save(receipt,dict(geometry_version=2,samples=n,manifest_sha256=hashlib.sha256((RUN/'samples.json').read_bytes()).hexdigest(),
                      maximum_ray_reprojection_error_px=float(max(errors)),rays_over_1px=int(sum(e>1 for e in errors))))


def infer_wilor(model,samples,ids,rotation,batch=16):
    cache=np.load(RUN/'wilor_crops.npy',mmap_mode='r')
    geometry=np.load(RUN/'crop_geometry.npz')
    parts=[]
    for start in range(0,len(ids),batch):
        ix=ids[start:start+batch]
        p=predict(model,cache[ix],[samples[i]['right'] for i in ix],geometry['rotations'][ix],geometry['focals'][ix],rotation=rotation)
        parts.append(p['xyz'])
        if start%256==0:print(json.dumps(dict(stage='wilor',rotation=rotation,done=start,total=len(ids))),flush=True)
    return np.concatenate(parts)


def metrics(pred,gt):
    r=pose_metrics(pred,gt);r.pop('sample_mpjpe19_mm')
    return r


def coarse(samples):
    path=RUN/'coarse_predictions.npy'
    if path.exists():return np.load(path)
    model=CoarsePose3D(weight=str(ROOT/'weights/yolo26s.pt')).to('cuda:2').eval()
    model.load_state_dict(torch.load(ROOT/'experiments/dit_lowconfidence_v1/coarse_fine/best.pt',map_location='cpu',weights_only=False)['model'])
    images=np.load(RUN/'coarse_crops.npy',mmap_mode='r');g=np.load(RUN/'crop_geometry.npz')['geometry'];parts=[]
    with torch.inference_mode():
        for start in range(0,len(samples),256):
            x=torch.from_numpy(images[start:start+256,...,::-1].transpose(0,3,1,2).copy()).to('cuda:2').float()/255
            geo=torch.from_numpy(g[start:start+256]).to('cuda:2');n=len(x)
            if n<256:
                x=torch.cat([x,torch.zeros(256-n,3,256,256,device='cuda:2')]);geo=torch.cat([geo,torch.zeros(256-n,21,device='cuda:2')])
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x,geo)
            parts.append(p['xyz'][:n].float().cpu().numpy())
    pred=np.concatenate(parts);np.save(path,pred)
    del model;torch.cuda.empty_cache()
    return pred


def evaluate(samples,base,wilor):
    report={}
    for split in ['tune','test']:
        ids=np.array([i for i,s in enumerate(samples) if s['split']==split and s['matched'] and s['right']>=0])
        gt=np.array([samples[i]['gt'] for i in ids],np.float32)
        vis=np.array([samples[i]['visibility'] for i in ids]);valid=np.array([samples[i]['projection_valid'] for i in ids],bool)
        groups=dict(all=np.ones(len(ids),bool),occlusion_lt_0_5=vis<.5,severe_lt_0_25=vis<.25,
                    occlusion_in_view=(vis<.5)&(valid.sum(1)>=18))
        group_results={}
        for name,mask in groups.items():
            group_results[name]=dict(samples=int(mask.sum()),coarse=metrics(base[ids][mask],gt[mask]) if mask.any() else None,
                                      wilor=metrics(wilor[ids][mask],gt[mask]) if mask.any() else None)
        correct_side=np.array([samples[i]['right']==int(samples[i]['gt_side']=='right') for i in ids])
        row=dict(samples=len(ids),side_accuracy=float(correct_side.mean()),groups=group_results,
                 correct_side_subset=dict(samples=int(correct_side.sum()),wilor=metrics(wilor[ids][correct_side],gt[correct_side])),
                 note='Correct-side subset is diagnostic only; primary metrics include incorrectly predicted side.')
        before=np.linalg.norm((base[ids]-base[ids,5:6])-(gt-gt[:,5:6]),axis=-1)[:,EVAL_INDICES]*1000
        after=np.linalg.norm((wilor[ids]-wilor[ids,5:6])-(gt-gt[:,5:6]),axis=-1)[:,EVAL_INDICES]*1000
        precise=before<=10
        row['previous_correct_relative_joints']=int(precise.sum())
        row['previous_correct_relative_joints_harmed_gt1mm']=float(((after>before+1)&precise).sum()/max(1,precise.sum()))
        row['paired_relative_improvement_mm']=float((before-after).mean())
        sequences=np.array([samples[i]['sequence'] for i in ids]);counts=[]
        for seq in np.unique(sequences):
            m=sequences==seq;counts.append([(before[m]-after[m]).sum(),int(m.sum())*19])
        counts=np.array(counts);rng=np.random.default_rng(20261003)
        boot=counts[rng.integers(len(counts),size=(2000,len(counts)))].sum(1)
        row['paired_relative_improvement_ci95_mm']=np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist()
        report[split]=row
    save(RUN/'results.json',report)
    print(json.dumps(report,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','evaluate'],default='evaluate');a=ap.parse_args()
    cv2.setNumThreads(0);torch.set_num_threads(4)
    samples=json.loads((RUN/'samples.json').read_text())
    prepare(samples)
    if a.stage=='prepare':return
    base=coarse(samples)
    model,cfg=load_model()
    selection_path=RUN/'selection.json'
    if selection_path.exists():
        rotation=json.loads(selection_path.read_text())['rotation']
    else:
        candidates=np.array([i for i,s in enumerate(samples) if s['split']=='tune' and s['matched'] and s['right']>=0])
        ids=np.sort(np.random.default_rng(20261003).choice(candidates,min(384,len(candidates)),replace=False))
        gt=np.array([samples[i]['gt'] for i in ids],np.float32);scores=[]
        for rotation in range(4):
            p=infer_wilor(model,samples,ids,rotation)
            score=metrics(p,gt);scores.append(dict(rotation=rotation,metrics=score))
            print(json.dumps(scores[-1]),flush=True)
        rotation=min(scores,key=lambda s:s['metrics']['wrist_relative_mpjpe19_mm'])['rotation']
        save(selection_path,dict(rotation=rotation,development_indices=ids.tolist(),scores=scores,
                                  criterion='Lowest wrist-relative MPJPE19 on fixed P0003 subset; no test scoring before freezing'))
    path=RUN/'wilor_predictions.npy'
    if path.exists():pred=np.load(path)
    else:
        ids=np.array([i for i,s in enumerate(samples) if s['right']>=0])
        pred=np.full((len(samples),20,3),np.nan,np.float32)
        pred[ids]=infer_wilor(model,samples,ids,rotation)
        np.save(path,pred)
    evaluate(samples,base,pred)


if __name__=='__main__':main()

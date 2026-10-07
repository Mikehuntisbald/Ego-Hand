"""Build prediction-only instance conditions for a bounded full-model pilot."""
import argparse,collections,hashlib,json,time
from pathlib import Path
import numpy as np,cv2,torch
from instance_masks_v51 import HandInstanceSegmenter,cell_conditions
from hand3d_v8_common import V7

RUN=V7.parent/'full_model_gloves_multihand_v51_20261007'
SOURCE=V7.parent/'online_rgb_3d_v47'

def overlap_score(box,other):
    box=np.asarray(box);other=np.asarray(other)
    lo=np.maximum(box[:2],other[:2]);hi=np.minimum(box[2:],other[2:]);area=np.maximum(hi-lo,0).prod()
    return float(area/max(min(np.maximum(box[2:]-box[:2],0).prod(),np.maximum(other[2:]-other[:2],0).prod()),1))

def choose_indices(data,records,train_count,dev_count):
    byimage=collections.defaultdict(list)
    for i,r in enumerate(records,1):byimage[r['image']].append((i,r['box']))
    centers=data['feature_ids'][:,8].tolist();selected={}
    for role,count in [('train',train_count),('dev_select',dev_count)]:
        groups=collections.defaultdict(list)
        for wi,(fid,r) in enumerate(zip(centers,data['roles'])):
            if r!=role:continue
            box=records[fid-1]['box'];score=max([overlap_score(box,z) for j,z in byimage[records[fid-1]['image']] if j!=fid and not np.allclose(box,z,atol=.1)]+[0])
            group=data['rows'][wi].get('sequence',data['subjects'][wi]);groups[group].append((score,wi))
        picked=[]
        for g in groups:groups[g].sort(reverse=True)
        keys=sorted(groups)
        while len(picked)<count and any(groups.values()):
            for key in keys:
                if groups[key] and len(picked)<count:picked.append(groups[key].pop(0)[1])
        selected[role]=picked
    return selected,byimage

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');ap.add_argument('--train',type=int,default=64);ap.add_argument('--dev',type=int,default=32);a=ap.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(0);RUN.mkdir(exist_ok=True)
    if (RUN/'mask_conditions_done.json').exists():return
    data=torch.load(SOURCE/'inputs.pt',weights_only=False,mmap=True);records=json.loads((SOURCE/'records.json').read_text())
    assert len(records)+1==len(data['world'])
    indices,byimage=choose_indices(data,records,a.train,a.dev)
    needed=torch.unique(data['feature_ids'][indices['train']+indices['dev_select']]);needed=needed[needed>0].tolist()
    bank=np.zeros((len(records)+1,192),np.float16);other=np.zeros_like(bank);quality=np.zeros(len(bank),np.float32)
    grouped=collections.defaultdict(list)
    for fid in needed:grouped[records[fid-1]['image']].append(fid)
    segmenter=HandInstanceSegmenter(a.device);start=time.time();folder=RUN/'predicted_masks_hot3d';folder.mkdir(exist_ok=True)
    for n,(image,fids) in enumerate(grouped.items()):
        instances=byimage[image];distinct={tuple(np.round(box,3)):box for _,box in instances};boxes=list(distinct.values());keys=list(distinct)
        out=segmenter(cv2.imread(image),boxes)
        for fid in fids:
            record=records[fid-1];k=keys.index(tuple(np.round(record['box'],3)));pred=out[k]
            x,y=cell_conditions(pred['mask'],pred['other'],data['positions_bank'][fid].numpy(),[1408,1408]);bank[fid]=x;other[fid]=y;quality[fid]=pred['quality']
        if n%20==0:
            status=dict(stage='hot3d_predicted_masks',done=n+1,total=len(grouped),seconds=time.time()-start)
            (RUN/'mask_progress.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
    torch.save(dict(own=torch.from_numpy(bank),other=torch.from_numpy(other),quality=torch.from_numpy(quality),indices=indices),RUN/'mask_conditions.pt')
    (RUN/'mask_conditions_done.json').write_text(json.dumps(dict(complete=True,windows={k:len(v) for k,v in indices.items()},observations=len(needed),images=len(grouped),selection='Predicted-box overlap, stratified by sequence; train/dev only',GT_mask_or_keypoints_used=False,seconds=time.time()-start),indent=2))

if __name__=='__main__':main()

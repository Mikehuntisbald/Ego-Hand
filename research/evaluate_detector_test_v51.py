"""Freeze GT-free full-pipeline front-end outputs before held-out scoring."""
import sys,json,hashlib,collections
sys.path.append('/mnt/why/HOT3D/.venv/lib/python3.12/site-packages')
from pathlib import Path
import torch,numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from train_domain_detector_v51 import INITIAL
from prepare_paired_domain_v51 import PAIRED

def main():
    torch.set_num_threads(4)
    folder=PAIRED/'heldout_detector';folder.mkdir(exist_ok=True)
    if (folder/'evaluation.json').exists():return
    rows=[json.loads(x) for x in (PAIRED/'domain_records.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['split']=='test']
    selected=json.loads((PAIRED/'detector/selection.json').read_text())['selected']
    frozen={}
    for name,path in [('reference',INITIAL),('candidate',selected)]:
        model=YOLO(str(path));pred=[]
        for start in range(0,len(rows),16):
            batch=rows[start:start+16]
            out=model.predict([r['image'] for r in batch],imgsz=960,conf=.05,iou=.7,max_det=20,batch=16,device='0',verbose=False)
            pred.extend([dict(boxes=p.boxes.xyxy.cpu().tolist(),scores=p.boxes.conf.cpu().tolist()) for p in out])
        frozen[name]=pred;del model;torch.cuda.empty_cache()
    target=folder/'sealed_predictions.json';target.write_text(json.dumps(dict(ids=[r['id'] for r in rows],predictions=frozen)))
    (folder/'freeze.json').write_text(json.dumps(dict(sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),candidate=selected,predictions_sealed_before_scoring=True,scope='Detector stage of complete pipeline; not 3D accuracy'),indent=2))
    stats={};group=collections.defaultdict(lambda:collections.defaultdict(lambda:[0,0,0]))
    for name,preds in frozen.items():
        stats[name]={}
        for row,p in zip(rows,preds):
            gt=np.asarray([h['box_xyxy'] for h in row['hands']],float).reshape(-1,4);boxes=np.asarray(p['boxes'],float).reshape(-1,4)
            hits=0
            if len(gt) and len(boxes):
                lo=np.maximum(gt[:,None,:2],boxes[None,:,:2]);hi=np.minimum(gt[:,None,2:],boxes[None,:,2:]);inter=np.maximum(hi-lo,0).prod(-1)
                ov=inter/np.maximum((gt[:,2:]-gt[:,:2]).prod(-1)[:,None]+(boxes[:,2:]-boxes[:,:2]).prod(-1)[None]-inter,1e-9)
                a,b=linear_sum_assignment(-((ov>=.5).astype(float)+ov*1e-3));hits=int((ov[a,b]>=.5).sum())
            s=stats[name].setdefault(row['dataset'],dict(images=0,hands=0,matched=0,predicted=0))
            s['images']+=1;s['hands']+=len(gt);s['matched']+=hits;s['predicted']+=len(boxes)
            g=group[row['dataset']][(name,row['group'])];g[0]+=hits;g[1]+=len(gt);g[2]+=len(boxes)
        for dataset,s in stats[name].items():
            s['box_coverage_IoU50']=s['matched']/max(s['hands'],1)
            if dataset=='egohands':s['precision_IoU50']=s['matched']/max(s['predicted'],1)
            s['precision_scope_complete']=dataset=='egohands'
    cis={};rng=np.random.default_rng(2026100755)
    for dataset,g in group.items():
        keys=sorted({key[1] for key in g});deltas=[]
        for key in keys:
            old=g['reference',key];new=g['candidate',key]
            deltas.append(new[0]/max(new[1],1)-old[0]/max(old[1],1))
        samples=np.mean(rng.choice(deltas,(2000,len(deltas)),replace=True),axis=1)
        cis[dataset]=dict(groups=len(keys),paired_coverage_CI95=np.quantile(samples,[.025,.975]).tolist(),mean_group_delta=float(np.mean(deltas)))
    result=dict(statistics=stats,paired_intervals=cis,test_used_for_training=False,default_changed=False,GT_3D=False)
    (folder/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()

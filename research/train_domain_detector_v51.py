"""Adapt the complete pipeline's detector using real glove/multiple-hand labels."""
import sys
sys.path.append('/mnt/why/HOT3D/.venv/lib/python3.12/site-packages')
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from ultralytics import YOLO
from cache_instance_conditions_v51 import RUN

INITIAL=Path('/mnt/why/HOT3D/experiments/dit_lowconfidence_v1/detector/weights/best.pt')

def coverage(model,records,device):
    statistics={};sealed=[]
    for start in range(0,len(records),16):
        batch=records[start:start+16];out=model.predict([r['image'] for r in batch],imgsz=960,conf=.05,iou=.7,max_det=20,batch=16,device=device,verbose=False)
        for r,p in zip(batch,out):sealed.append(dict(id=r['id'],dataset=r['dataset'],group=r['group'],boxes=p.boxes.xyxy.cpu().tolist(),scores=p.boxes.conf.cpu().tolist()))
    # All predictions are materialized before labels are used for the metrics.
    for r,p in zip(records,sealed):
        gt=np.asarray([h['box_xyxy'] for h in r['hands']]);boxes=np.asarray(p['boxes']).reshape(-1,4);used=set();match=0
        for box in boxes:
            if not len(gt):continue
            lo=np.maximum(gt[:,:2],box[:2]);hi=np.minimum(gt[:,2:],box[2:]);intersection=np.maximum(hi-lo,0).prod(-1)
            union=np.maximum(gt[:,2:]-gt[:,:2],0).prod(-1)+np.maximum(box[2:]-box[:2],0).prod()-intersection
            ov=intersection/np.maximum(union,1e-9)
            for k in used:ov[k]=0
            k=int(ov.argmax())
            if ov[k]>=.5:used.add(k);match+=1
        s=statistics.setdefault(r['dataset'],dict(images=0,hands=0,matched=0,predicted=0));s['images']+=1;s['hands']+=len(gt);s['matched']+=match;s['predicted']+=len(boxes)
    for dataset,s in statistics.items():
        s['box_coverage_IoU50']=s['matched']/max(s['hands'],1)
        if dataset=='egohands':s['precision_IoU50']=s['matched']/max(s['predicted'],1)
        s['label_scope']='All labelled hands' if dataset=='egohands' else 'Annotated gloves only; other bare hands are not complete GT'
    return statistics,sealed

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=3);p.add_argument('--device',default='0');a=p.parse_args();torch.set_num_threads(4)
    folder=RUN/'detector';folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():return
    rows=[json.loads(x) for x in (RUN/'domain_records.jsonl').read_text().splitlines()]
    development=[r for r in rows if r['split']=='dev'];test=[r for r in rows if r['split']=='test']
    # Hold test labels closed until the development-selected checkpoint is fixed.
    old=YOLO(str(INITIAL));baseline,preds=coverage(old,development,a.device)
    (folder/'baseline_dev.json').write_text(json.dumps(baseline,indent=2));(folder/'baseline_dev_predictions.json').write_text(json.dumps(preds));del old;torch.cuda.empty_cache()
    model=YOLO(str(INITIAL));model.train(data=str(RUN/'detector_data/data.yaml'),epochs=a.epochs,imgsz=960,batch=16,workers=4,device=a.device,optimizer='AdamW',lr0=.0003,lrf=.2,weight_decay=.01,warmup_epochs=.2,
        mosaic=0,mixup=0,copy_paste=0,cutmix=0,erasing=0,hsv_h=0,hsv_s=0,hsv_v=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,flipud=0,fliplr=0,project=str(folder),name='pilot',exist_ok=False,plots=False,seed=2026100751,deterministic=True,save=True)
    best=folder/'pilot/weights/best.pt';model=YOLO(str(best));new,preds=coverage(model,development,a.device)
    (folder/'trained_dev.json').write_text(json.dumps(new,indent=2));(folder/'trained_dev_predictions.json').write_text(json.dumps(preds))
    admitted=all(new[d]['box_coverage_IoU50']>=baseline[d]['box_coverage_IoU50'] for d in baseline)
    chosen=best if admitted else INITIAL
    (folder/'selection.json').write_text(json.dumps(dict(selected=str(chosen),candidate=str(best),development_passed=admitted,full_3D_validation_pending=True,default_changed=False),indent=2))
    # A detector-only result cannot claim complete-model 3D success.
    (folder/'done.json').write_text(json.dumps(dict(complete=True,epochs=a.epochs,baseline=baseline,trained=new,development_admitted=admitted,external_test_not_read=True,default_changed=False),indent=2))

if __name__=='__main__':main()

"""Whole pipeline detector adaptation; inspect every saved epoch for protection."""
import sys,json,hashlib
sys.path.append('/mnt/why/HOT3D/.venv/lib/python3.12/site-packages')
from pathlib import Path
import torch
from ultralytics import YOLO
from train_domain_detector_v51 import coverage,INITIAL
from prepare_paired_domain_v51 import PAIRED

def main():
    torch.set_num_threads(4);folder=PAIRED/'detector';folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():return
    rows=[json.loads(x) for x in (PAIRED/'domain_records.jsonl').read_text().splitlines()];dev=[r for r in rows if r['split']=='dev']
    old=YOLO(str(INITIAL));baseline,preds=coverage(old,dev,'0');del old;torch.cuda.empty_cache()
    (folder/'baseline_dev.json').write_text(json.dumps(baseline,indent=2));(folder/'baseline_dev_predictions.json').write_text(json.dumps(preds))
    model=YOLO(str(INITIAL));model.train(data=str(PAIRED/'detector_data/data.yaml'),epochs=4,imgsz=960,batch=16,workers=4,device='0',optimizer='AdamW',lr0=.0001,lrf=.2,weight_decay=.01,warmup_epochs=.2,
        mosaic=0,mixup=0,copy_paste=0,cutmix=0,erasing=0,hsv_h=0,hsv_s=0,hsv_v=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,flipud=0,fliplr=0,project=str(folder),name='balanced',exist_ok=False,plots=False,seed=2026100752,deterministic=True,save=True,save_period=1)
    candidates=list((folder/'balanced/weights').glob('epoch*.pt'))+list((folder/'balanced/weights').glob('best.pt'));history=[];admitted=[]
    for path in candidates:
        model=YOLO(str(path));metrics,preds=coverage(model,dev,'0')
        feasible=all(metrics[d]['box_coverage_IoU50']>=baseline[d]['box_coverage_IoU50'] for d in baseline)
        feasible=feasible and metrics['egohands']['precision_IoU50']>=baseline['egohands']['precision_IoU50']
        entry=dict(checkpoint=str(path),metrics=metrics,admitted=feasible);history.append(entry)
        if feasible:admitted.append((sum(metrics[d]['box_coverage_IoU50'] for d in metrics),str(path),preds))
    (folder/'history.json').write_text(json.dumps(history,indent=2))
    selected=max(admitted,key=lambda x:x[0]) if admitted else (0.,str(INITIAL),None)
    chosen=Path(selected[1]);digest=hashlib.file_digest(chosen.open('rb'),'sha256').hexdigest()
    (folder/'selection.json').write_text(json.dumps(dict(selected=str(chosen),sha256=digest,development_passed=bool(admitted),native_coverage_must_not_drop=True,paired_interactions=True,default_changed=False),indent=2))
    if selected[2]:(folder/'selected_dev_predictions.json').write_text(json.dumps(selected[2]))
    (folder/'done.json').write_text(json.dumps(dict(complete=True,epochs=4,admitted=bool(admitted),baseline=baseline,selected=str(chosen),external_test_closed=True,default_changed=False),indent=2))

if __name__=='__main__':main()

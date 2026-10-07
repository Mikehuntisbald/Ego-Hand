"""Freeze the existing YOLO predictions and exact per-hand development successes."""
import json,sys,hashlib
from pathlib import Path
import torch
ROOT=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')

def main():
    sys.path.append('/mnt/why/HOT3D/.venv/lib/python3.12/site-packages');from ultralytics import YOLO
    from rfdetr_dev_selection_v53 import score
    rows=[r for r in map(json.loads,(ROOT/'domain_records.jsonl').read_text().splitlines()) if r['split']=='dev'];ck=json.loads(Path('/mnt/why/HOT3D/experiments/full_model_gloves_multihand_v51_20261007/paired_protocol/detector/selection.json').read_text())['selected'];model=YOLO(ck);out=model.predict([r['image'] for r in rows],imgsz=960,conf=.05,iou=.7,max_det=100,batch=16,device='0',verbose=False);pred=[dict(id=r['id'],boxes=d.boxes.xyxy.cpu().tolist(),scores=d.boxes.conf.cpu().tolist()) for r,d in zip(rows,out)];(ROOT/'yolo_development_sealed.json').write_text(json.dumps(pred));stats,ok=score(rows,pred,.05);(ROOT/'yolo_development_baseline.json').write_text(json.dumps(dict(statistics=stats,success=ok,checkpoint=ck,checkpoint_sha256=hashlib.file_digest(open(ck,'rb'),'sha256').hexdigest(),GT_free_inference=True,threshold=.05),indent=2));print(json.dumps(stats),flush=True)

if __name__=='__main__':main()

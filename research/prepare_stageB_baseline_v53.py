"""Same live YOLO baseline for new development groups and reviewed backgrounds."""
import json,sys,hashlib
from pathlib import Path
import torch
ROOT=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53B_20261007')
def main():
    sys.path.append('/mnt/why/HOT3D/.venv/lib/python3.12/site-packages');from ultralytics import YOLO
    from rfdetr_dev_selection_v53 import score
    torch.set_num_threads(8)
    rows=[r for r in map(json.loads,(ROOT/'domain_records.jsonl').read_text().splitlines()) if r['split']=='dev']
    ck='/mnt/why/HOT3D/experiments/full_model_gloves_multihand_v51_20261007/paired_protocol/detector/balanced/weights/epoch2.pt';model=YOLO(ck)
    predictions=[]
    for start in range(0,len(rows),16):
        batch=rows[start:start+16];out=model.predict([r['image'] for r in batch],imgsz=960,conf=.05,iou=.7,max_det=100,batch=16,device='0',verbose=False)
        predictions.extend(dict(id=r['id'],boxes=d.boxes.xyxy.cpu().tolist(),scores=d.boxes.conf.cpu().tolist()) for r,d in zip(batch,out))
    (ROOT/'yolo_development_sealed.json').write_text(json.dumps(predictions));stats,ok=score(rows,predictions,.05)
    (ROOT/'yolo_development_baseline.json').write_text(json.dumps(dict(statistics=stats,success=ok,checkpoint=ck,checkpoint_sha256=hashlib.file_digest(open(ck,'rb'),'sha256').hexdigest(),GT_free_inference=True,threshold=.05),indent=2));print(json.dumps(stats),flush=True)
if __name__=='__main__':main()

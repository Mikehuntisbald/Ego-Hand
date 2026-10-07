"""RGB-only direct frontend prediction, sealed before scoring test targets."""
import os,json,argparse,hashlib,time
from pathlib import Path
import torch,numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--kind',choices=['rfdetr','yolo'],required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--threshold',type=float,required=True);p.add_argument('--policy',default='raw');a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    frames=json.loads(Path(a.input).read_text())['frames'];assert all(set(f)<={'id','image','dataset','group','split'} for f in frames)
    out=Path(a.output);out.mkdir(exist_ok=True);assert not (out/'freeze.json').exists(),'Never overwrite sealed inference'
    torch.set_num_threads(8);started=time.time();pred=[]
    if a.kind=='rfdetr':
        from rfdetr import RFDETR
        from rfdetr_dev_selection_v53 import BoxesOnly
        from rfdetr_dev_selection_stageB_v53 import nms_predictions
        model=RFDETR.from_checkpoint(a.checkpoint,device='cuda:0',trust_checkpoint=True);model.model.postprocess=BoxesOnly(model.model.postprocess)
        for i in range(0,len(frames),8):
            batch=frames[i:i+8]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):ds=model.predict([v['image'] for v in batch],threshold=.001,include_source_image=False)
            pp=[dict(id=v['id'],boxes=d.xyxy.tolist(),scores=d.confidence.tolist()) for v,d in zip(batch,ds)]
            if a.policy=='box_nms_0.7':pp=nms_predictions(pp)
            for v in pp:
                keep=np.asarray(v['scores'])>=a.threshold;v['boxes']=np.asarray(v['boxes']).reshape(-1,4)[keep].tolist();v['scores']=np.asarray(v['scores'])[keep].tolist()
            pred+=pp
            (out/'progress.json').write_text(json.dumps(dict(images=len(pred),total=len(frames),seconds=time.time()-started)))
    else:
        import sys
        sys.path.append('/mnt/why/HOT3D/.venv/lib/python3.12/site-packages');from ultralytics import YOLO
        model=YOLO(a.checkpoint)
        for i in range(0,len(frames),16):
            batch=frames[i:i+16];ds=model.predict([v['image'] for v in batch],imgsz=960,conf=a.threshold,iou=.7,max_det=100,rect=False,batch=16,device='0',verbose=False)
            pred.extend(dict(id=v['id'],boxes=d.boxes.xyxy.cpu().tolist(),scores=d.boxes.conf.cpu().tolist()) for v,d in zip(batch,ds))
            (out/'progress.json').write_text(json.dumps(dict(images=len(pred),total=len(frames),seconds=time.time()-started)))
    raw=json.dumps(pred);(out/'predictions.json').write_text(raw)
    (out/'freeze.json').write_text(json.dumps(dict(complete=True,kind=a.kind,checkpoint=a.checkpoint,checkpoint_sha256=hashlib.file_digest(open(a.checkpoint,'rb'),'sha256').hexdigest(),input_sha256=hashlib.file_digest(open(a.input,'rb'),'sha256').hexdigest(),prediction_sha256=hashlib.sha256(raw.encode()).hexdigest(),images=len(frames),seconds=time.time()-started,threshold=a.threshold,policy=a.policy,GT_free=True,test_XML_read=False,precision='BF16' if a.kind=='rfdetr' else 'YOLO native'),indent=2))
    print('SEALED',a.kind,len(pred),flush=True)

if __name__=='__main__':main()

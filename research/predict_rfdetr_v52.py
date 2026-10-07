"""RF-DETR isolated-runtime inference; input contract is RGB metadata only."""
import json, argparse, hashlib, time
from pathlib import Path
import cv2, numpy as np, torch

ROOT=Path('/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007')

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint',default=str(ROOT/'full/checkpoint_best_total.pth'));p.add_argument('--threshold',type=float,default=.05);a=p.parse_args()
    torch.set_num_threads(8);cv2.setNumThreads(0)
    from rfdetr import RFDETR
    model=RFDETR.from_checkpoint(a.checkpoint,device='cuda:0',trust_checkpoint=True)
    request=json.loads(Path(a.input).read_text());frames=request['frames']
    assert all(set(f)<= {'id','image','group','dataset','split','timestamp_s','camera','camera_calibrated','source_image','image_transform'} for f in frames), 'Inference must not receive target annotations'
    folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True);records=[];started=time.time()
    for begin in range(0,len(frames),4):
        batch=frames[begin:begin+4]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictions=model.predict([f['image'] for f in batch],threshold=a.threshold,include_source_image=False)
        for offset,(f,d) in enumerate(zip(batch,predictions)):
            assert d.mask is not None
            path=folder/f'{begin+offset:06d}.npz'
            np.savez_compressed(path,masks=d.mask.astype(bool),boxes=d.xyxy.astype(np.float32),scores=d.confidence.astype(np.float32))
            records.append(dict(f,output=str(path),detections=len(d)))
        if begin%100==0:
            (folder/'progress.json').write_text(json.dumps(dict(images=min(begin+4,len(frames)),total=len(frames),elapsed_s=time.time()-started)))
    (folder/'predictions.json').write_text(json.dumps(dict(frames=records,threshold=a.threshold,checkpoint=a.checkpoint,checkpoint_sha256=hashlib.file_digest(open(a.checkpoint,'rb'),'sha256').hexdigest(),GT_free=True,real_RGB=True)))
    (folder/'freeze.json').write_text(json.dumps(dict(complete=True,metadata_sha256=hashlib.file_digest(open(folder/'predictions.json','rb'),'sha256').hexdigest(),array_sha256={Path(r['output']).name:hashlib.file_digest(open(r['output'],'rb'),'sha256').hexdigest() for r in records},sealed_before_labels=True),indent=2))
    print(json.dumps(dict(complete=True,images=len(frames),elapsed_s=time.time()-started)),flush=True)

if __name__=='__main__':main()

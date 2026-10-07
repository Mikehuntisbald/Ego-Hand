"""Selected RF frontend supplies both hand boxes and instance masks to 3D."""
import json,argparse,hashlib,time,os
from pathlib import Path
import numpy as np,torch,cv2
from rfdetr import RFDETR
from rfdetr_dev_selection_stageB_v53 import nms_predictions

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--threshold',type=float,required=True);p.add_argument('--policy',default='raw');a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3';torch.set_num_threads(8);cv2.setNumThreads(0)
    folder=Path(a.output);folder.mkdir(exist_ok=True);assert not (folder/'freeze.json').exists()
    frames=json.loads(Path(a.input).read_text())['frames'];assert all(set(f)<={'id','image','group','dataset','split','timestamp_s','camera','camera_calibrated','source_image','image_transform'} for f in frames)
    model=RFDETR.from_checkpoint(a.checkpoint,device='cuda:0',trust_checkpoint=True);records=[];started=time.time()
    for begin in range(0,len(frames),2):
        batch=frames[begin:begin+2]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):ds=model.predict([v['image'] for v in batch],threshold=.001,include_source_image=False)
        for offset,(f,d) in enumerate(zip(batch,ds)):
            indices=np.arange(len(d))
            if a.policy=='box_nms_0.7':indices=np.array(nms_predictions([dict(boxes=d.xyxy.tolist(),scores=d.confidence.tolist())])[0]['keep_indices'],dtype=int)
            indices=indices[d.confidence[indices]>=a.threshold];path=folder/f'{begin+offset:06d}.npz'
            np.savez_compressed(path,masks=d.mask[indices].astype(bool),boxes=d.xyxy[indices].astype(np.float32),scores=d.confidence[indices].astype(np.float32));records.append(dict(f,output=str(path),detections=len(indices)))
        (folder/'progress.json').write_text(json.dumps(dict(images=len(records),total=len(frames),seconds=time.time()-started)))
    (folder/'predictions.json').write_text(json.dumps(dict(frames=records,threshold=a.threshold,policy=a.policy,checkpoint=a.checkpoint,checkpoint_sha256=hashlib.file_digest(open(a.checkpoint,'rb'),'sha256').hexdigest(),GT_free=True,real_RGB=True)))
    (folder/'freeze.json').write_text(json.dumps(dict(complete=True,metadata_sha256=hashlib.file_digest(open(folder/'predictions.json','rb'),'sha256').hexdigest(),array_sha256={Path(v['output']).name:hashlib.file_digest(open(v['output'],'rb'),'sha256').hexdigest() for v in records},sealed_before_labels=True),indent=2))
    print('INSTANCES SEALED',len(records),flush=True)
if __name__=='__main__':main()

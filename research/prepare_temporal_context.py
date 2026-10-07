"""Track all YOLO proposals before GT assignment; cache strictly past contexts."""
import json,time
from pathlib import Path
import cv2,numpy as np,torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from coarse_pose3d import CoarsePose3D
from prepare_pose_training import ROOT,RUN,crop_and_geometry
from predict_hand_crops import ious

def main():
    torch.set_num_threads(4);cv2.setNumThreads(0);device='cuda:1'
    data=torch.load(RUN/'coarse_cache.pt',map_location='cpu',weights_only=False);rows=data['rows']
    current={}
    for i,r in enumerate(rows):current.setdefault((r['clip'],r['frame']),[]).append(i)
    groups={}
    for r in rows:groups[r['clip']]=r
    detector=YOLO(str(RUN/'detector/weights/best.pt'))
    coarse=CoarsePose3D().to(device);coarse.load_state_dict(torch.load(RUN/'coarse_fine/best.pt',map_location='cpu',weights_only=False)['model']);coarse.eval()
    n=len(rows);S=4
    history_rgb=torch.zeros(n,S,64,128,dtype=torch.float16);history_pose=torch.zeros(n,S,20,3)
    history_conf=torch.zeros(n,S,21);history_dt=torch.zeros(n,S);history_valid=torch.zeros(n,S,dtype=torch.bool)
    history_rgb[:,0]=data['rgb'];history_pose[:,0]=data['coarse'];history_conf[:,0]=data['confidence'];history_valid[:,0]=True
    started=time.time();assigned=0
    for count,(clip,example) in enumerate(sorted(groups.items())):
        split='train' if example['role'] in ['coarse','residual'] else 'val'
        path=ROOT/'export/annotations'/split/example['sequence']/f'clip-{clip:06d}.jsonl'
        stride=3 if example['role']=='residual' else 5
        frames=[json.loads(line) for frame,line in enumerate(path.read_text().splitlines()) if frame%stride==0]
        tracks={};next_id=0
        for start in range(0,len(frames),32):
            chunk=frames[start:start+32]
            predictions=detector.predict([str(ROOT/'export'/r['image']) for r in chunk],device='1',imgsz=960,batch=32,
                conf=.01,max_det=10,verbose=False,save=False)
            for frame,pred in zip(chunk,predictions):
                boxes=pred.boxes.xyxy.cpu().numpy();now=frame['timestamp_ns']*1e-9
                crops=[];rois=[];geometry=[]
                for box in boxes:
                    crop,roi,g=crop_and_geometry(pred.orig_img,box,frame['camera'])
                    # Match the JPEG input distribution of the frozen pose model.
                    encoded=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95])[1]
                    crop=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
                    crops.append(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));rois.append(roi);geometry.append(g)
                if not crops:continue
                active=[k for k,v in tracks.items() if now-v[-1]['time']<=.5]
                assigned_tracks={}
                if active:
                    previous=[]
                    for k in active:
                        tr=tracks[k];box=tr[-1]['roi'].copy()
                        if len(tr)>=2:
                            dt=tr[-1]['time']-tr[-2]['time'];velocity=(tr[-1]['roi']-tr[-2]['roi'])/max(.01,dt)
                            box+=np.clip(velocity*(now-tr[-1]['time']),-100,100)
                        previous.append(box)
                    overlap=ious(rois,previous);ii,jj=linear_sum_assignment(-overlap)
                    assigned_tracks={i:active[j] for i,j in zip(ii,jj) if overlap[i,j]>.1}
                images=torch.from_numpy(np.stack(crops).transpose(0,3,1,2).copy()).to(device).float()/255
                geo=torch.tensor(np.stack(geometry),device=device)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):output=coarse(images,geo)
                poses=output['xyz'].float().cpu();rgbs=output['rgb_tokens'].half().cpu()
                confs=torch.cat([output['root_confidence'][:,None],output['confidence']],1).float().cpu()
                for i,roi in enumerate(rois):
                    if i not in assigned_tracks:assigned_tracks[i]=next_id;tracks[next_id]=[];next_id+=1
                    tid=assigned_tracks[i]
                    tracks[tid].append(dict(time=now,roi=roi,pose=poses[i],rgb=rgbs[i],confidence=confs[i]))
                # Current labeled sample is linked by its observed predicted ROI,
                # not GT side, GT pose, GT box, or visibility.
                sample_ids=current.get((clip,frame['frame']),[])
                for sample in sample_ids:
                    observed_roi=np.asarray(rows[sample]['roi']);match=ious([observed_roi],rois)[0]
                    i=int(match.argmax())
                    if match[i]<.9:continue
                    track=tracks[assigned_tracks[i]]
                    for slot,lag in enumerate([1/6,1/3,2/3],1):
                        candidates=[o for o in track if 0<now-o['time']<=1. and o['time']<=now-lag+.025]
                        if not candidates:continue
                        obs=max(candidates,key=lambda o:o['time'])
                        history_rgb[sample,slot]=obs['rgb'];history_pose[sample,slot]=obs['pose'];history_conf[sample,slot]=obs['confidence']
                        history_dt[sample,slot]=obs['time']-now;history_valid[sample,slot]=True;assigned+=1
        if count%30==0:print(json.dumps(dict(clips=count+1,total=len(groups),past_contexts=assigned,seconds=time.time()-started)),flush=True)
    torch.save(dict(rgb=history_rgb,pose=history_pose,confidence=history_conf,dt=history_dt,valid=history_valid),RUN/'temporal_context.pt')
    (RUN/'temporal_context_done.json').write_text(json.dumps(dict(completed=True,samples=n,past_contexts=assigned,
        policy='All detector proposals tracked with ROI IoU and velocity; reset each clip; strictly causal; no GT association used in tracking')))

if __name__=='__main__':main()

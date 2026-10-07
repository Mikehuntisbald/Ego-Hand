"""Global and interior-only ROI proposals with a conservative background veto."""
import json
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torchvision.ops import nms
from online_parameter_model_v47 import OnlineVisual
from cache_domain_masks_v51 import crop_array
from cache_instance_conditions_v51 import RUN
from ultralytics import YOLO

def detect(frames,path,device,folder):
    detector=YOLO(str(path));global_boxes=[]
    for start in range(0,len(frames),16):
        out=detector.predict([f['image'] for f in frames[start:start+16]],imgsz=960,conf=.05,iou=.7,max_det=20,batch=16,device=device.split(':')[-1],verbose=False)
        global_boxes.extend([dict(boxes=x.boxes.xyxy.cpu().tolist(),scores=x.boxes.conf.cpu().tolist()) for x in out])
    visual=OnlineVisual(device,False);ck=torch.load(RUN/'paired_protocol/core_r1/dit_joint/best.pt',weights_only=False,map_location='cpu');visual.load_tail(ck['visual_tail'])
    head=nn.Sequential(nn.LayerNorm(1280),nn.Linear(1280,64),nn.GELU(),nn.Linear(64,1)).to(device).eval()
    ck=torch.load(RUN/'paired_protocol/handness_r1/model.pt',weights_only=False,map_location='cpu');head.load_state_dict(ck['model'])
    @torch.no_grad()
    def handness(images,boxes):
        if not len(boxes):return []
        pixels=[crop_array(image,box) for image,box in zip(images,boxes)];prob=[]
        for start in range(0,len(pixels),48):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                feature=visual.encode_pixels(pixels[start:start+48],device).float().reshape(-1,16,12,1280)[:,4:12,3:9].mean((1,2));prob.extend(head(feature).squeeze(-1).sigmoid().float().cpu().tolist())
        return prob
    answer=[];unfiltered=[];review=[]
    for index,(frame,global_) in enumerate(zip(frames,global_boxes)):
        image=cv2.imread(frame['image']);boxes=global_['boxes'];scores=global_['scores'];prob=handness([image]*len(boxes),boxes)
        seeds=sorted([i for i in range(len(boxes)) if prob[i]>=.2 and scores[i]>=.08],key=lambda i:scores[i],reverse=True)[:3]
        proposals=[]
        for i in seeds:
            box=np.asarray(boxes[i]);centre=(box[:2]+box[2:])/2;edge=max(box[2:]-box[:2])*1.6
            roi=np.r_[centre-edge/2,centre+edge/2];roi[:2]=np.maximum(roi[:2],0);roi[2:]=np.minimum(roi[2:],[image.shape[1],image.shape[0]]);x,y,x2,y2=roi.astype(int)
            if min(x2-x,y2-y)<40:continue
            out=detector.predict(image[y:y2,x:x2],imgsz=1280,conf=.05,iou=.7,max_det=12,device=device.split(':')[-1],verbose=False)[0]
            for local,score in zip(out.boxes.xyxy.cpu().numpy(),out.boxes.conf.cpu().numpy()):
                if min(local[:2])<.03*min(x2-x,y2-y) or min([x2-x-local[2],y2-y-local[3]])<.03*min(x2-x,y2-y):continue
                proposals.append((local+[x,y,x,y],float(score)))
        if proposals:
            extra=[p[0].tolist() for p in proposals];prob+=handness([image]*len(extra),extra);boxes=boxes+extra;scores=scores+[p[1] for p in proposals]
        unfiltered.append(dict(image=frame['image'],timestamp_s=frame['timestamp_s'],boxes=boxes,scores=scores,handness=prob))
        # Semantic probabilities are not physical instance certainty. Keep all
        # vetoed candidates in the output for review, never call this recall.
        keep=[i for i,p in enumerate(prob) if p>=.1]
        review.extend(dict(frame=frame,image=frame['image'],box=boxes[i],score=scores[i],handness=prob[i],reason='Predicted background; candidate retained for review') for i in range(len(boxes)) if i not in keep)
        if keep:
            select=nms(torch.tensor([boxes[i] for i in keep]),torch.tensor([scores[i] for i in keep]),.65).tolist();keep=[keep[i] for i in select]
        answer.append(dict(boxes=[boxes[i] for i in keep],scores=[scores[i] for i in keep],handness=[prob[i] for i in keep]))
        if index%20==0:print(json.dumps(dict(stage='crowded_proposals',done=index+1,total=len(frames),remaining=len(keep))),flush=True)
    (Path(folder)/'unfiltered_detection_candidates.json').write_text(json.dumps(unfiltered));(Path(folder)/'background_review_candidates.json').write_text(json.dumps(review))
    del detector,visual,head;torch.cuda.empty_cache();return answer

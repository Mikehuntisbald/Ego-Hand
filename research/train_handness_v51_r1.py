"""Natural hard-negative hand verification; preserve every rejected candidate."""
import json,collections,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from online_parameter_model_v47 import OnlineVisual
from cache_domain_masks_v51 import crop_array
from cache_instance_conditions_v51 import RUN
from prepare_paired_domain_v51 import PAIRED
import wilor_eval_common
from ultralytics import YOLO

def iou(box,gt):
    lo=np.maximum(gt[:,:2],box[:2]);hi=np.minimum(gt[:,2:],box[2:]);inter=np.maximum(hi-lo,0).prod(-1)
    return inter/np.maximum((gt[:,2:]-gt[:,:2]).prod(-1)+(box[2:]-box[:2]).prod()-inter,1e-9)

def main():
    torch.set_num_threads(4);cv2.setNumThreads(0);torch.manual_seed(2026100758);rng=np.random.default_rng(2026100758);device='cuda:0'
    folder=PAIRED/'handness_r1';folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():return
    rows=[json.loads(x) for x in (PAIRED/'domain_records.jsonl').read_text().splitlines()];groups=collections.defaultdict(list)
    for r in rows:
        if r['dataset']=='egohands' and r['split']!='test':groups[r['group']].append(r)
    chosen=[]
    for records in groups.values():chosen.extend(records[i] for i in np.linspace(0,len(records)-1,min(8,len(records))).round().astype(int))
    detector=YOLO(json.loads((PAIRED/'detector/selection.json').read_text())['selected']);preds=[]
    for i in range(0,len(chosen),16):preds.extend(detector.predict([r['image'] for r in chosen[i:i+16]],imgsz=960,conf=.05,iou=.7,max_det=20,device='0',verbose=False))
    del detector;torch.cuda.empty_cache();pixels=[];labels=[];roles=[];metadata=[]
    for row,pred in zip(chosen,preds):
        image=cv2.imread(row['image']);h,w=image.shape[:2];gt=np.asarray([x['box_xyxy'] for x in row['hands']],float).reshape(-1,4)
        if not len(gt):continue
        candidates=[(box,1,'annotated_hand') for box in gt]
        candidates.extend((box,0,'detector_hard_negative') for box in pred.boxes.xyxy.cpu().numpy() if iou(box,gt).max()<.05)
        # Cropping true background regions does not synthesize an occlusion.
        for _ in range(30):
            bw,bh=rng.uniform(.12,.45,2)*[w,h];x,y=rng.uniform([0,0],[w-bw,h-bh]);box=np.array([x,y,x+bw,y+bh])
            if iou(box,gt).max()<.01 and sum(np.maximum(np.minimum(gt[:,2:],box[2:])-np.maximum(gt[:,:2],box[:2]),0).prod(-1))<.005*bw*bh:
                candidates.append((box,0,'natural_background_crop'))
                if sum(y==0 for _,y,_ in candidates)>=6:break
        for box,label,kind in candidates:
            pixels.append(crop_array(image,box));labels.append(label);roles.append(row['split']);metadata.append(dict(image=row['image'],group=row['group'],label=label,kind=kind))
    curated=[]
    for frame in [24,48,72,96,119]:
        path=RUN/'natural_nail_care_final_r1/normalized'/f'{frame:06d}.jpg';image=cv2.imread(str(path))
        for box,label,kind in [([0,690,430,1100],0,'visually_reviewed_tray_weak_background'),([15,775,400,1090],0,'visually_reviewed_tray_interior'),([460,600,730,760],1,'visually_reviewed_patient_hand_presence'),([700,440,890,630],1,'visually_reviewed_upper_glove_presence'),([710,580,990,760],1,'visually_reviewed_lower_glove_presence')]:
            pixels.append(crop_array(image,box));labels.append(label);roles.append('train');curated.append(dict(image=str(path),box=box,label=label,kind=kind,source='Assistant visual audit; weak presence labels, no keypoint or 3D targets'))
    (folder/'curated_training_labels.json').write_text(json.dumps(curated,indent=2))
    visual=OnlineVisual(device,False);ck=torch.load(PAIRED/'core_r1/dit_joint/best.pt',weights_only=False,map_location='cpu');visual.load_tail(ck['visual_tail']);features=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for begin in range(0,len(pixels),48):features.append(visual.encode_pixels(pixels[begin:begin+48],device).float().reshape(-1,16,12,1280)[:,4:12,3:9].mean((1,2)).cpu())
    del visual;torch.cuda.empty_cache();features=torch.cat(features)
    surgery=torch.load(RUN/'surgical_pose/inputs.pt',weights_only=False,mmap=True);sfeat=torch.load(PAIRED/'localizer_isolated_r2/features.pt',weights_only=False,mmap=True)['surgical'].float().reshape(-1,16,12,1280)[:,4:12,3:9].mean((1,2))
    features=torch.cat([features,sfeat]);labels.extend([1]*len(sfeat));roles.extend(r['split'] for r in surgery['metadata'])
    target=torch.tensor(labels,device=device,dtype=torch.float32);x=features.to(device);train=torch.tensor([i for i,r in enumerate(roles) if r=='train'],device=device);dev=torch.tensor([i for i,r in enumerate(roles) if r=='dev'],device=device)
    head=nn.Sequential(nn.LayerNorm(1280),nn.Linear(1280,64),nn.GELU(),nn.Linear(64,1)).to(device);optimizer=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.01)
    positives=train[target[train]==1];negatives=train[target[train]==0]
    curated_ids=torch.arange(len(pixels)-25,len(pixels),device=device)
    for step in range(500):
        ids=torch.cat([positives[torch.randint(len(positives),(24,),device=device)],negatives[torch.randint(len(negatives),(24,),device=device)],curated_ids[torch.randint(25,(16,),device=device)]])
        loss=nn.functional.binary_cross_entropy_with_logits(head(x[ids]).squeeze(-1),target[ids]);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    head.eval()
    with torch.no_grad():prob=head(x[dev]).squeeze(-1).sigmoid().cpu();y=target[dev].cpu()
    torch.save(dict(probability=prob,labels=y,indices=dev.cpu(),predictions_sealed=True),folder/'development.pt')
    thresholds=np.linspace(.05,.95,19);history=[]
    for threshold in thresholds:
        accepted=prob>=threshold;recall=float(accepted[y==1].float().mean());rejection=float((~accepted[y==0]).float().mean())
        history.append(dict(threshold=float(threshold),hand_recall=recall,background_rejection=rejection,admitted=recall>=.98 and rejection>=.5))
    choices=[r for r in history if r['admitted']];selection=max(choices,key=lambda r:r['background_rejection']) if choices else None
    torch.save(dict(model=head.state_dict(),selection=selection,base_checkpoint=str(PAIRED/'core_r1/dit_joint/best.pt')),folder/'model.pt')
    (folder/'history.json').write_text(json.dumps(history,indent=2));(folder/'metadata.json').write_text(json.dumps(metadata))
    (folder/'done.json').write_text(json.dumps(dict(complete=True,admitted=bool(selection),selection=selection,dev_positive=int((y==1).sum()),dev_negative=int((y==0).sum()),all_rejected_detections_review_only=True,synthetic_occlusion=False,default_changed=False,scope='Centre-spatial verifier; target video is now scene-adaptation training material, not held-out evidence. Other dataset groups remain held out; not a guaranteed physical hand probability'),indent=2))
    print((folder/'done.json').read_text(),flush=True)

if __name__=='__main__':main()

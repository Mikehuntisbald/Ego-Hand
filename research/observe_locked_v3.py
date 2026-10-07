"""Encode every detector proposal on sealed, previously unused source sequences."""
import hashlib
import json
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
ROOT=Path('/mnt/why/HOT3D');RUN=Path(os.environ.get('HOT3D_DIT_RUN',str(ROOT/'experiments/dit_wilor_v3')))
sys.path.insert(0,str(RUN/'sealed'))
import wilor_eval_common
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from dit_v3_inference import ObservationEncoder,prepare_observation
from compare_detectors import iou


def save(path,obj):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)


def main():
    torch.set_num_threads(4);cv2.setNumThreads(0)
    assert (RUN/'sealed/selection.json').exists() and (RUN/'locked_data_ready.json').exists()
    if (RUN/'locked_observations_done.json').exists():return
    seal=json.loads((RUN/'sealed/selection.json').read_text())
    assert hashlib.sha256((RUN/'locked_manifest.json').read_bytes()).hexdigest()==seal['locked_manifest_sha256']
    for name,digest in seal['code'].items():assert hashlib.sha256((RUN/'sealed'/name).read_bytes()).hexdigest()==digest
    manifest=json.loads((RUN/'locked_manifest.json').read_text());frames=[]
    for s in manifest['sequences']:
        for clip in s['clips']:
            path=ROOT/'export/annotations'/s['split']/s['sequence']/f'clip-{clip:06d}.jsonl'
            frames.extend(json.loads(line) for i,line in enumerate(path.read_text().splitlines()) if i%5==0)
    save(RUN/'locked_frames.json',frames)
    detections_path=RUN/'locked_detections.json'
    if detections_path.exists():detections=json.loads(detections_path.read_text())
    else:
        detector=YOLO(str(ROOT/'experiments/dit_lowconfidence_v1/detector/weights/best.pt'))
        side_detector=YOLO(str(ROOT/'experiments/detector_compare_wilor_20261003/wilor_detector.pt'))
        detections=[]
        for start in range(0,len(frames),32):
            chunk=frames[start:start+32];images=[str(ROOT/'export'/f['image']) for f in chunk]
            boxes=detector.predict(images,imgsz=960,device='0',batch=32,conf=.01,max_det=10,verbose=False)
            sides=side_detector.predict(images,imgsz=960,device='0',batch=32,conf=.001,max_det=100,verbose=False)
            for p,s in zip(boxes,sides):
                detections.append(dict(boxes=p.boxes.xyxy.cpu().tolist(),scores=p.boxes.conf.cpu().tolist(),
                    side=dict(boxes=s.boxes.xyxy.cpu().tolist(),classes=s.boxes.cls.cpu().tolist(),scores=s.boxes.conf.cpu().tolist())))
            if start%320==0:print(json.dumps(dict(stage='locked_detection',done=len(detections),total=len(frames))),flush=True)
        save(detections_path,detections)
        del detector,side_detector;torch.cuda.empty_cache()
    assert len(detections)==len(frames)
    rows=[];total_gt=0;recalled50=0;matched30=0
    for fi,(frame,pred) in enumerate(zip(frames,detections)):
        eligible=[]
        for h in frame['hands']:
            if h['box_amodal_xyxy'] is None or (h['modeled_hand_visible_fraction'] or 0)<=0 or h['xyz_camera_m'][5][2]<=.05:continue
            b=np.clip(h['box_amodal_xyxy'],0,1408)
            if min(b[2:]-b[:2])<2:continue
            eligible.append((h,b))
        total_gt+=len(eligible)
        overlap=iou(pred['boxes'],[b for h,b in eligible]);matched={}
        if len(pred['boxes']) and eligible:
            ii,jj=linear_sum_assignment(-overlap)
            matched={int(i):int(j) for i,j in zip(ii,jj) if overlap[i,j]>=.3}
            matched30+=len(matched);recalled50+=sum(overlap[i,j]>=.5 for i,j in zip(ii,jj))
        for pi,box in enumerate(pred['boxes']):
            target=eligible[matched[pi]][0] if pi in matched else None
            rows.append(dict(index=len(rows),frame_index=fi,sequence=frame['sequence'],subject=frame['subject'],clip=frame['clip'],frame=frame['frame'],
                image=str(ROOT/'export'/frame['image']),box=box,score=pred['scores'][pi],camera=frame['camera'],side_predictions=pred['side'],
                matched=target is not None,gt=target['xyz_camera_m'] if target else None,visibility_label=target['modeled_hand_visible_fraction'] if target else None,
                projection_valid=target['keypoint_projection_valid'] if target else None,side_label=target['side'] if target else None))
    save(RUN/'locked_rows.json',rows)
    save(RUN/'locked_detection_counts.json',dict(frames=len(frames),eligible_hands=total_gt,proposals=len(rows),matched_iou30=matched30,recalled_iou50=int(recalled50),
        note='Every detector proposal is encoded; GT is used only after detection for evaluation association.'))
    encoder=ObservationEncoder('cuda:0');folder=RUN/'locked_observations';folder.mkdir(exist_ok=True)
    def worker(row):
        return prepare_observation(cv2.imread(row['image']),row['box'],row['camera'],row['side_predictions'])
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0,len(rows),512):
            dest=folder/f'{start:06d}.pt'
            if dest.exists():continue
            end=min(start+512,len(rows));prepared=list(pool.map(worker,rows[start:end]));fields=defaultdict(list)
            for offset in range(0,len(prepared),16):
                part=prepared[offset:offset+16]
                result=encoder.features(*[[p[k] for p in part] for k in ['wilor_crop','rotation','focal','right','side_conf','side_iou','reprojection_error','crop_fallback']])
                for k,v in result.items():fields[k].append(v.cpu())
            for offset in range(0,len(prepared),256):
                part=prepared[offset:offset+256]
                result=encoder.coarse_predictions([p['coarse_crop'] for p in part],[p['coarse_geometry'] for p in part])
                for k,v in result.items():fields[k].append(v.cpu())
            chunk={k:torch.cat(v) for k,v in fields.items()}
            assert all(torch.isfinite(v).all() for v in chunk.values())
            chunk.update(start=start,end=end)
            tmp=dest.with_suffix('.partial');torch.save(chunk,tmp);tmp.replace(dest)
            status=dict(stage='locked_observation_encoding',done=end,total=len(rows),complete=False,acceptance_passed=False)
            save(RUN/'status.json',status);print(json.dumps(status),flush=True)
    save(RUN/'locked_observations_done.json',dict(complete=True,proposals=len(rows),frames=len(frames),rows_sha256=hashlib.sha256((RUN/'locked_rows.json').read_bytes()).hexdigest(),
        frozen_selection_sha256=hashlib.sha256((RUN/'sealed/selection.json').read_bytes()).hexdigest()))


if __name__=='__main__':main()

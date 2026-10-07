"""Predict boxes first; GT is used only for assignment, labels, and missed-GT counts."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from prepare_pose_training import ROOT,RUN,crop_and_geometry

def ious(a,b):
    a=np.asarray(a).reshape(-1,4);b=np.asarray(b).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(-1)
    return inter/(np.maximum(a[:,2:]-a[:,:2],0).prod(-1)[:,None]+np.maximum(b[:,2:]-b[:,:2],0).prod(-1)[None]-inter+1e-9)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='2');a=ap.parse_args()
    while not (RUN/'detector_done.json').exists():time.sleep(20)
    part=json.loads((RUN/'partition.json').read_text());manifest=json.loads((ROOT/'subset_manifest.json').read_text())
    sets={role:set(part[key]) for role,key in [('coarse','coarse_clips'),('residual','residual_clips'),('tune','tune_clips'),('test','locked_test_clips')]}
    frames=[]
    for seq in manifest['sequences']:
        for clip in seq['clips']:
            role=next(r for r,ids in sets.items() if clip in ids)
            stride=3 if role=='residual' else 5
            path=ROOT/'export/annotations'/seq['split']/seq['sequence']/f'clip-{clip:06d}.jsonl'
            frames.extend((json.loads(line),role) for i,line in enumerate(path.read_text().splitlines()) if i%stride==0)
    model=YOLO(str(RUN/'detector/weights/best.pt'));rows=[];detection=[];started=time.time()
    for start in range(0,len(frames),32):
        chunk=frames[start:start+32]
        predictions=model.predict([str(ROOT/'export'/r['image']) for r,_ in chunk],device=a.device,imgsz=960,batch=32,
                                  conf=.01,max_det=10,verbose=False,save=False)
        for (frame,role),result in zip(chunk,predictions):
            boxes=result.boxes.xyxy.cpu().numpy();scores=result.boxes.conf.cpu().numpy()
            # GT assignments happen after every prediction is fixed.
            hands=[h for h in frame['hands'] if h['box_amodal_xyxy'] is not None and (h['modeled_hand_visible_fraction'] or 0)>0 and h['xyz_camera_m'][5][2]>.05]
            gt_boxes=[np.clip(h['box_amodal_xyxy'],[0,0,0,0],[1408,1408,1408,1408]) for h in hands]
            overlap=ious(boxes,gt_boxes)
            assignments=[]
            if len(boxes) and len(hands):
                ii,jj=linear_sum_assignment(-overlap)
                assignments=[(i,j) for i,j in zip(ii,jj) if overlap[i,j]>=.3]
            matched={j:i for i,j in assignments}
            for j,hand in enumerate(hands):
                detection.append(dict(role=role,subject=frame['subject'],sequence=frame['sequence'],clip=frame['clip'],frame=frame['frame'],
                    side=hand['side'],visible_fraction=hand['modeled_hand_visible_fraction'],
                    box_matched=j in matched,box_iou=float(overlap[matched[j],j]) if j in matched else 0.,
                    box_score=float(scores[matched[j]]) if j in matched else 0.))
            if not assignments:continue
            image=cv2.imread(str(ROOT/'export'/frame['image']))
            for i,j in assignments:
                hand=hands[j];crop,roi,geo=crop_and_geometry(image,boxes[i],frame['camera'])
                folder=RUN/'predicted_crops'/role/frame['sequence']/f"clip-{frame['clip']:06d}";folder.mkdir(parents=True,exist_ok=True)
                dest=folder/f"{frame['frame']:06d}_{i}.jpg";assert cv2.imwrite(str(dest),crop,[cv2.IMWRITE_JPEG_QUALITY,95])
                rows.append(dict(role=role,subject=frame['subject'],sequence=frame['sequence'],clip=frame['clip'],frame=frame['frame'],side=hand['side'],
                    crop=str(dest),image=str(ROOT/'export'/frame['image']),roi=roi.tolist(),geometry=geo.tolist(),
                    xyz_camera_m=hand['xyz_camera_m'],uv_pixels=hand['uv_pixels'],projection_valid=hand['keypoint_projection_valid'],
                    visible_fraction=hand['modeled_hand_visible_fraction'],box_score=float(scores[i]),box_iou=float(overlap[i,j]),
                    box_source='predicted',timestamp_ns=frame['timestamp_ns']))
        if start%320==0:print(json.dumps(dict(frames=min(start+32,len(frames)),total=len(frames),matched_pose_samples=len(rows),seconds=time.time()-started)),flush=True)
    dest=RUN/'predicted_manifest.jsonl';dest.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n')
    (RUN/'detector_associations.jsonl').write_text('\n'.join(json.dumps(r) for r in detection)+'\n')
    (RUN/'predicted_crops_done.json').write_text(json.dumps(dict(completed=True,frames=len(frames),samples=len(rows),by_role={r:sum(x['role']==r for x in rows) for r in sets})))

if __name__=='__main__':main()

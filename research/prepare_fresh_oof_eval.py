"""Download/export only a preregistered RGB/calibration/hand-GT extension."""
import json,time
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from download_rgb_subset import download
from export_hand_labels import export_clip
from prepare_pose_training import crop_and_geometry
from predict_hand_crops import ious
from oof_common import ROOT,OLD,RUN,save,sha

def main():
    if (RUN/'fresh_done.json').exists():return
    m=json.loads((RUN/'fresh_manifest.json').read_text())
    tree={x['path']:x for x in json.loads((ROOT/'provenance/train_aria_tree.json').read_text())}
    jobs=[(s['split'],s['subject'],s['sequence'],c) for s in m['sequences'] for c in s['clips']]
    def task(job):
        return download(job,m,tree)
    receipts=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        fs={pool.submit(task,j):j for j in jobs}
        for f in as_completed(fs):
            receipts.append(f.result());save(RUN/'fresh_download_status.json',dict(done=len(receipts),total=len(jobs)))
            print(json.dumps(dict(downloaded_exported=len(receipts),total=len(jobs))),flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:exports=list(pool.map(export_clip,jobs))
    save(RUN/'fresh_receipts.json',dict(download=receipts,export=exports))
    model=YOLO(str(OLD/'detector/weights/best.pt'));frames=[]
    for split,subject,seq,clip in jobs:
        path=ROOT/'export/annotations'/split/seq/f'clip-{clip:06d}.jsonl'
        frames.extend(json.loads(line) for i,line in enumerate(path.read_text().splitlines()) if i%5==0)
    rows=[];associations=[];cv2.setNumThreads(0)
    for start in range(0,len(frames),32):
        chunk=frames[start:start+32]
        preds=model.predict([str(ROOT/'export'/f['image']) for f in chunk],device='0',imgsz=960,batch=32,conf=.01,max_det=10,verbose=False,save=False)
        for frame,p in zip(chunk,preds):
            boxes=p.boxes.xyxy.cpu().numpy();scores=p.boxes.conf.cpu().numpy()
            hands=[h for h in frame['hands'] if h['box_amodal_xyxy'] is not None and (h['modeled_hand_visible_fraction'] or 0)>0 and h['xyz_camera_m'][5][2]>.05]
            overlap=ious(boxes,[np.clip(h['box_amodal_xyxy'],[0,0,0,0],[1408,1408,1408,1408]) for h in hands])
            pairs=[]
            if len(boxes) and hands:
                ii,jj=linear_sum_assignment(-overlap);pairs=[(i,j) for i,j in zip(ii,jj) if overlap[i,j]>=.3]
            matched={j:i for i,j in pairs}
            for j,h in enumerate(hands):
                associations.append(dict(subject=frame['subject'],sequence=frame['sequence'],clip=frame['clip'],frame=frame['frame'],
                    matched=j in matched,iou=float(overlap[matched[j],j]) if j in matched else 0.))
            image=cv2.imread(str(ROOT/'export'/frame['image']))
            for i,j in pairs:
                h=hands[j];crop,roi,geo=crop_and_geometry(image,boxes[i],frame['camera'])
                folder=RUN/'fresh_crops'/frame['sequence']/f"clip-{frame['clip']:06d}";folder.mkdir(parents=True,exist_ok=True)
                dest=folder/f"{frame['frame']:06d}_{i}.jpg";assert cv2.imwrite(str(dest),crop,[cv2.IMWRITE_JPEG_QUALITY,95])
                rows.append(dict(role='fresh',subject=frame['subject'],sequence=frame['sequence'],clip=frame['clip'],frame=frame['frame'],side=h['side'],
                    crop=str(dest),image=str(ROOT/'export'/frame['image']),roi=roi.tolist(),geometry=geo.tolist(),
                    xyz_camera_m=h['xyz_camera_m'],uv_pixels=h['uv_pixels'],projection_valid=h['keypoint_projection_valid'],
                    visible_fraction=h['modeled_hand_visible_fraction'],box_score=float(scores[i]),box_iou=float(overlap[i,j]),box_source='predicted',timestamp_ns=frame['timestamp_ns']))
    dest=RUN/'fresh_predicted_manifest.jsonl';dest.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    save(RUN/'fresh_associations.json',associations)
    save(RUN/'fresh_done.json',dict(complete=True,clips=len(jobs),frames=len(frames),matched_hands=len(rows),manifest_sha256=sha(dest)))
    print((RUN/'fresh_done.json').read_text(),flush=True)
if __name__=='__main__':main()

"""Actual predicted instances -> full 3D -> sealed outputs -> held-out 2D GT."""
import json,hashlib,collections
from pathlib import Path
import numpy as np,torch,cv2
from scipy.optimize import linear_sum_assignment
from annotate_instances_v51_r6 import normalize_frame
from crowded_detector_v51 import detect
from complete_instance_v51_r3 import complete as reference_complete
from complete_instance_v51_r5 import complete as candidate_complete
from cache_instance_conditions_v51 import RUN
from prepare_paired_domain_v51 import PAIRED
from build_surgical_pose_cache_v51 import MAPPING

def main():
    torch.set_num_threads(4);cv2.setNumThreads(0);device='cuda:0';folder=RUN/'complete_glove_test';folder.mkdir(exist_ok=True)
    if (folder/'evaluation.json').exists():return
    records=[json.loads(x) for x in (PAIRED/'domain_records.jsonl').read_text().splitlines()];groups=collections.defaultdict(list)
    for row in records:
        if row['dataset']=='surgical_hands' and row['split']=='test':groups[row['group']].append(row)
    chosen=[]
    for g,rows in sorted(groups.items()):
        rows.sort(key=lambda r:r['id']);chosen.extend(rows[i] for i in np.linspace(0,len(rows)-1,min(2,len(rows))).round().astype(int))
    # Only image/camera metadata is sent to inference; no annotated boxes,
    # keypoints, visibility, tracks or sides enter either complete pipeline.
    normal=folder/'normalized';normal.mkdir(exist_ok=True);frames=[normalize_frame(dict(image=r['image'],timestamp_s=0),normal,i) for i,r in enumerate(chosen)]
    selection=json.loads((PAIRED/'detector/selection.json').read_text())['selected']
    from ultralytics import YOLO
    model=YOLO(selection);out=model.predict([f['image'] for f in frames],imgsz=960,conf=.05,iou=.7,max_det=20,batch=12,device='0',verbose=False)
    original=[dict(boxes=p.boxes.xyxy.cpu().tolist(),scores=p.boxes.conf.cpu().tolist()) for p in out];del model;torch.cuda.empty_cache()
    improved=detect(frames,selection,device,folder);frozen={};observations={}
    for name,detector,complete in [('reference',original,reference_complete),('candidate',improved,candidate_complete)]:
        tracks=[]
        for i,(frame,pred) in enumerate(zip(frames,detector)):
            for j,(box,score) in enumerate(zip(pred['boxes'],pred['scores'])):tracks.append(dict(id=f'{i}_{j}',frames=[dict(frame,box_xyxy=box,box_confidence=score)]))
        source=dict(image_size=[1408,1408],tracks=tracks);observations[name]=source
        result=complete(source,device,mask_folder=folder/name/'predicted_masks');assert result['constraint_checks_passed']
        path=folder/(name+'_sealed.json');path.write_text(json.dumps(result));frozen[name]=result
    (folder/'freeze.json').write_text(json.dumps(dict(complete=True,images=[r['id'] for r in chosen],GT_free=True,actual_detected_ROIs=True,sha256={n:hashlib.file_digest((folder/(n+'_sealed.json')).open('rb'),'sha256').hexdigest() for n in frozen},scope='12 independent still images; timestamp 0 is each isolated segment origin, no video timing claim'),indent=2))
    # Keypoint labels read only after both complete output arrays are sealed.
    raw=Path('/mnt/why/HOT3D/domain_data_v51/surgical_hands/selected_records.jsonl');target={r['id']:r for r in map(json.loads,raw.read_text().splitlines())}
    per_image=[];total={n:dict(labelled_points=0,correct_PCK10=0,hands=0,matched_hands=0,matched_labelled_points=0,matched_correct_PCK10=0,visible_points=0,visible_correct=0,occluded_points=0,occluded_correct=0) for n in frozen}
    for i,row in enumerate(chosen):
        label=target[row['id']];offset=np.asarray(frames[i]['image_transform']['offset']);scale=frames[i]['image_transform']['scale'];hands=label['hands'];gtboxes=np.array([h['bbox'] for h in hands])*scale+np.tile(offset,2)
        xy=[];valid=[];visible=[]
        for hand in hands:
            kp=np.asarray(hand['keypoints']).reshape(21,3)[MAPPING];xy.append(kp[:,:2]*scale+offset);valid.append(kp[:,2]>0);visible.append(kp[:,2]==2)
        xy=np.asarray(xy);valid=np.asarray(valid);visible=np.asarray(visible);denom=np.mean(gtboxes[:,2:]-gtboxes[:,:2],axis=-1)
        report=dict(id=row['id'],group=row['group'],labelled_points=int(valid.sum()))
        for name,result in frozen.items():
            indices=[k for k,tr in enumerate(observations[name]['tracks']) if tr['id'].startswith(str(i)+'_')];boxes=np.array([observations[name]['tracks'][k]['frames'][0]['box_xyxy'] for k in indices]).reshape(-1,4)
            success=np.zeros_like(valid);matched_points=matched_correct=0;matched=0
            if len(boxes) and len(gtboxes):
                lo=np.maximum(gtboxes[:,None,:2],boxes[None,:,:2]);hi=np.minimum(gtboxes[:,None,2:],boxes[None,:,2:]);inter=np.maximum(hi-lo,0).prod(-1);ov=inter/np.maximum(np.prod(gtboxes[:,2:]-gtboxes[:,:2],-1)[:,None]+np.prod(boxes[:,2:]-boxes[:,:2],-1)[None]-inter,1e-9)
                a,b=linear_sum_assignment(-((ov>=.5).astype(float)+ov*1e-3))
                for gi,pj in zip(a,b):
                    if ov[gi,pj]<.5:continue
                    matched+=1;f=result['tracks'][indices[pj]]['frames'][0];xyz=np.asarray(f['candidate_xyz_camera_m']);fx,fy,cx,cy=frames[i]['camera']['calibration']['projection_params'];uv=xyz[:,:2]/xyz[:,2:]*[fx,fy]+[cx,cy]
                    success[gi]=(np.linalg.norm(uv-xy[gi],axis=-1)/denom[gi]<=.1)&valid[gi];matched_points+=int(valid[gi].sum());matched_correct+=int(success[gi].sum())
            s=total[name];s['hands']+=len(hands);s['matched_hands']+=matched;s['labelled_points']+=int(valid.sum());s['correct_PCK10']+=int(success.sum());s['matched_labelled_points']+=matched_points;s['matched_correct_PCK10']+=matched_correct
            s['visible_points']+=int((valid&visible).sum());s['visible_correct']+=int((success&visible).sum());s['occluded_points']+=int((valid&~visible).sum());s['occluded_correct']+=int((success&~visible).sum())
            report[name]=dict(correct=int(success.sum()),success=success.tolist())
        per_image.append(report)
    for s in total.values():
        s['complete_pipeline_PCK10']=s['correct_PCK10']/max(s['labelled_points'],1);s['conditional_matched_PCK10']=s['matched_correct_PCK10']/max(s['matched_labelled_points'],1)
    grouped=collections.defaultdict(list)
    for r in per_image:grouped[r['group']].append(r)
    delta=[sum(x['candidate']['correct']-x['reference']['correct'] for x in rows)/max(sum(x['labelled_points'] for x in rows),1) for rows in grouped.values()];rng=np.random.default_rng(2026100759);ci=np.quantile(np.mean(rng.choice(delta,(2000,len(delta)),replace=True),axis=1),[.025,.975]).tolist()
    result=dict(statistics=total,images=len(chosen),source_groups=len(groups),group_paired_PCK_delta_CI95=ci,GT_3D=False,GT_used_for_inference=False,missed_hands_count_as_incorrect_points=True,temporal_accuracy_test=False,reference='Previous admitted full v51 YOLO/mask/parameter-DiT/FK; same detector weights; no spatial-guide/ROI/semantic enhancements',default_changed=False)
    (folder/'evaluation.json').write_text(json.dumps(result,indent=2));(folder/'per_image.json').write_text(json.dumps(per_image));print(json.dumps(result),flush=True)

if __name__=='__main__':main()

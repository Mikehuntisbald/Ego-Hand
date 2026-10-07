"""Audit already-opened data for the newly requested front-end recovery task.

This script uses GT ONLY to count targets and score detector recall. Its history
statistics are oracle diagnostic ceilings, never model inputs or tracking IDs.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path('/mnt/why/HOT3D')
SOURCE=ROOT/'experiments/dit_wilor_v5'
OUT=ROOT/'experiments/frontend_diffusion_recovery_v1'
OUT.mkdir(exist_ok=True)

def iou(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    overlap=np.maximum(hi-lo,0).prod(-1)
    return overlap/(np.maximum(a[:,2:]-a[:,:2],0).prod(-1)[:,None]+np.maximum(b[:,2:]-b[:,:2],0).prod(-1)[None]-overlap+1e-9)

def group(h):
    vis=h.get('modeled_hand_visible_fraction')
    if vis is None:return 'visibility_unknown'
    if vis<=0:return 'modeled_visibility_zero'
    if vis<.25:return 'severe_partial_0_to_25pct'
    if vis<.5:return 'partial_25_to_50pct'
    return 'at_least_50pct'

frames=json.loads((SOURCE/'locked_frames.json').read_text())
predictions=json.loads((SOURCE/'locked_detections.json').read_text())
assert len(frames)==len(predictions)
results={};examples=[];targets=[]
for threshold in [.01,.25]:
    stats={};history={};absent_targets=0;hand_records=0;total_predictions=0
    for fi,(frame,det) in enumerate(zip(frames,predictions)):
        now=frame['timestamp_ns']*1e-9;w,h=frame['image_size']
        boxes=[b for b,s in zip(det['boxes'],det['scores']) if s>=threshold];total_predictions+=len(boxes)
        eligible=[]
        for hand in frame['hands']:
            hand_records+=1
            valid=np.asarray(hand['keypoint_projection_valid'],bool)
            # Keep fully hidden in-view hands; distinguish them from offscreen hands.
            inview=valid.sum()>=18 and hand['xyz_camera_m'][5][2]>.05
            if not inview:absent_targets+=1;continue
            g=group(hand)
            st=stats.setdefault(g,dict(in_view_hands=0,with_amodal_box=0,recalled_iou50=0,
                missed_with_prior_recalled_same_gt_hand_within_1s=0,missed_with_no_such_prior=0))
            st['in_view_hands']+=1
            if hand['box_amodal_xyxy'] is None:continue
            b=np.clip(hand['box_amodal_xyxy'],[0,0,0,0],[w,h,w,h])
            if min(b[2:]-b[:2])<2:continue
            st['with_amodal_box']+=1;eligible.append((hand,b,g))
        matched=set()
        if boxes and eligible:
            overlap=iou(boxes,[b for hand,b,g in eligible]);ii,jj=linear_sum_assignment(-overlap)
            matched={int(j) for i,j in zip(ii,jj) if overlap[i,j]>=.5}
        updates=[]
        for j,(hand,b,g) in enumerate(eligible):
            st=stats[g];key=(frame['sequence'],frame['clip'],hand['side'])
            if j in matched:
                st['recalled_iou50']+=1;updates.append(key)
            else:
                previous=history.get(key)
                has_prior=previous is not None and 0<now-previous<=1.
                st['missed_with_prior_recalled_same_gt_hand_within_1s' if has_prior else 'missed_with_no_such_prior']+=1
                if threshold==.25 and g=='modeled_visibility_zero' and len(examples)<12:
                    examples.append(dict(frame_index=fi,image=frame['image'],sequence=frame['sequence'],clip=frame['clip'],frame=frame['frame'],
                        gt_box=b.tolist(),past_same_gt_hand_recalled_within_1s=has_prior,used_for='diagnosis only, not inference'))
            if threshold==.01:
                targets.append(dict(frame_index=fi,sequence=frame['sequence'],clip=frame['clip'],frame=frame['frame'],group=g,
                    gt_box=b.tolist(),detected=j in matched,labels_for_supervision_and_evaluation_only=True))
        for key in updates:history[key]=now
    for st in stats.values():
        st['recall_iou50']=st['recalled_iou50']/max(1,st['with_amodal_box'])
    results[str(threshold)]=dict(groups=stats,total_proposals=total_predictions,hand_records=hand_records,
        excluded_offscreen_or_fewer_than_18_valid_joints=absent_targets)
report=dict(task='YOLO front-end missing-target recovery, distinct from downstream pose refinement',
    source=str(SOURCE),already_opened_development_data=True,frames=len(frames),frame_stride=5,
    source_sequences=len({f['sequence'] for f in frames}),thresholds_are_diagnostics_not_selection=True,
    diagnostics=results,examples=examples,
    important=['modeled visibility zero is a dataset proxy, not proof that every contextual cue is absent',
        'no per-keypoint occlusion labels; joint-level zero-information performance cannot be established from this label',
        'same-GT-hand history counts are oracle diagnostic ceilings; real tracking must use predictions only',
        'no-history or out-of-view targets must not be silently counted as successfully tracked detections',
        'existing YOLO training labels exclude visibility<=0; old matched-only 3D evaluation excludes detector misses'])
(OUT/'missing_target_audit.json').write_text(json.dumps(report,indent=2))
(OUT/'development_target_labels.json').write_text(json.dumps(targets))
print(json.dumps({k:v for k,v in report.items() if k!='examples'},indent=2))

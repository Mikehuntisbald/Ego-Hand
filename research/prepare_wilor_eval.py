"""Fix YOLO observations and inferred side before reconstruction scoring."""
import json
from pathlib import Path
import numpy as np
from compare_detectors import iou,matches,save

ROOT=Path('/mnt/why/HOT3D')
RUN=ROOT/'experiments/yolo26_wilor_3d_20261003'
DET=ROOT/'experiments/detector_compare_wilor_20261003'
frames=json.loads((DET/'frames.json').read_text())
yolo=json.loads((DET/'yolo26_hot3d_predictions.json').read_text())['predictions']
wilor=json.loads((DET/'wilor_released_predictions.json').read_text())['predictions']
det_results=json.loads((DET/'comparison.json').read_text())
threshold=det_results['results']['yolo26_hot3d']['metrics']['operating_points']['tune_best_f1']['test']['threshold']
source_cache={};samples=[];counts={}
for i,(f,yp,wp) in enumerate(zip(frames,yolo,wilor)):
    image=Path(f['image']);split=image.parent.parent.parent.name
    ann=ROOT/'export/annotations'/split/f['sequence']/f'clip-{f["clip"]:06d}.jsonl'
    if str(ann) not in source_cache:source_cache[str(ann)]=[json.loads(l) for l in ann.read_text().splitlines()]
    raw=source_cache[str(ann)][f['frame']]
    raw_by_side={h['side']:h for h in raw['hands']}
    order,assigned,used=matches(f,yp,threshold)
    # Inference side association uses only predicted boxes/classes, never GT.
    overlaps=iou([yp['boxes'][j] for j in order],wp['boxes'])
    for pos,yp_index in enumerate(order):
        if len(wp['boxes']) and overlaps[pos].max()>=.1:
            wi=int(overlaps[pos].argmax());right=int(wp['classes'][wi]);side_iou=float(overlaps[pos,wi])
        else:right=-1;side_iou=0.
        target=assigned[pos]
        hand=raw_by_side[f['hands'][target]['side']] if target>=0 else None
        samples.append(dict(index=len(samples),frame_index=i,split=f['split'],image=f['image'],sequence=f['sequence'],clip=f['clip'],frame=f['frame'],
                            box=yp['boxes'][yp_index],score=yp['scores'][yp_index],right=right,side_iou=side_iou,camera=raw['camera'],
                            matched=hand is not None,gt=hand['xyz_camera_m'] if hand else None,gt_side=hand['side'] if hand else None,
                            visibility=hand['modeled_hand_visible_fraction'] if hand else None,
                            projection_valid=hand['keypoint_projection_valid'] if hand else None))
    counts.setdefault(f['split'],dict(frames=0,gt_hands=0,matched=0,missed=0,false_positives=0))
    c=counts[f['split']];c['frames']+=1;c['gt_hands']+=len(f['hands']);c['matched']+=int(used.sum());c['missed']+=int((~used).sum());c['false_positives']+=len(order)-int(used.sum())
for split,c in counts.items():
    part=[s for s in samples if s['split']==split and s['matched']]
    c['side_unavailable']=sum(s['right']<0 for s in part)
    # Test side accuracy is intentionally deferred until the reconstruction protocol is frozen.
    if split=='tune':
        valid=[s for s in part if s['right']>=0]
        c['side_accuracy']=sum(s['right']==int(s['gt_side']=='right') for s in valid)/max(1,len(valid))
save(RUN/'samples.json',samples)
save(RUN/'observation_audit.json',counts)
save(RUN/'protocol.json',dict(detector='Frozen YOLO26 HOT3D at P0003 best-F1 threshold',threshold=threshold,
     handedness='WiLoR detector class at maximum box IoU >=0.1; no GT side as inference input; report missing-side failures',
     crop='Prediction-only FISHEYE624 -> virtual pinhole crop, padding1.3; central192-pixel margin; native camera extrinsics not used',
     selection='Four fixed 90-degree crop rotations compared only on deterministic development subset; freeze winner before test',
     joint_mapping='Official MANO->canonical20 mapping via WiLoR openpose21 order; same UmeTrack GT as original coarse',
     metrics='Camera MPJPE19, wrist-relative MPJPE19, wrist translation, PCK and hard groups; no GT alignment for primary metrics',
     limitations=['WiLoR released training config lists HOT3D-TRAIN: upstream overlap with these subjects is unknown.',
                  'Official MANO and UmeTrack joint definitions/models have residual anatomical differences.',
                  'This reuses existing test sequences and is a deployment comparison, not new-subject generalization evidence.']))
print(json.dumps(counts,indent=2),flush=True)

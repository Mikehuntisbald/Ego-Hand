"""Offline confirmation/association; keep tentative boxes as review candidates."""
import numpy as np

def box_iou(a,b):
    a=np.asarray(a);b=np.asarray(b);inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();return float(inter/max((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter,1))

def confirm_and_merge(tracks,descriptors):
    accepted={};tentative=[]
    for key,tr in tracks.items():
        confidence=np.asarray([f['box_confidence'] for f in tr['frames']]);strong=int((confidence>=.3).sum())
        # Persistence alone cannot turn a static false alarm into a real hand.
        if strong>=2 or (len(confidence)>=12 and confidence.mean()>=.2):accepted[key]=tr
        else:tentative.append(dict(id=tr['id'],frames=tr['frames'],status='unconfirmed_detection_review_only',strong_observations=strong))
    links=[]
    keys=list(accepted)
    for a in keys:
        end=accepted[a]['frames'][-1];candidates=[]
        for b in keys:
            if a==b:continue
            start=accepted[b]['frames'][0];gap=start['timestamp_s']-end['timestamp_s']
            if not 0<gap<=.55:continue
            xa=descriptors[(a,len(accepted[a]['frames'])-1)];xb=descriptors[(b,0)]
            cosine=float(xa@xb/max(np.linalg.norm(xa)*np.linalg.norm(xb),1e-9));iou=box_iou(end['box_xyxy'],start['box_xyxy'])
            if cosine>=.92 and iou>=.65:candidates.append((.6*(1-iou)+.4*(1-cosine),b,gap,cosine,iou))
        candidates.sort()
        if candidates and (len(candidates)==1 or candidates[1][0]-candidates[0][0]>=.1):links.append((a,*candidates[0]))
    targets={};used=set();root={k:k for k in accepted};merged=[]
    for a,cost,b,gap,cosine,iou in sorted(links,key=lambda x:x[1]):
        if b in used:continue
        used.add(b);targets[a]=b
    for a in sorted(accepted,key=lambda k:accepted[k]['frames'][0]['timestamp_s']):
        if a in used:continue
        chain=[a];cursor=a
        while cursor in targets and targets[cursor] not in chain:
            cursor=targets[cursor];chain.append(cursor)
        frames=[f for k in chain for f in accepted[k]['frames']];frames.sort(key=lambda f:f['timestamp_s'])
        assert all(frames[i]['timestamp_s']>frames[i-1]['timestamp_s'] for i in range(1,len(frames)))
        merged.append(dict(id=accepted[a]['id'],frames=frames,merged_from=[accepted[k]['id'] for k in chain]))
    return merged,tentative,dict(input_tracklets=len(tracks),confirmed_tracklets=len(accepted),confirmed_trajectories=len(merged),unconfirmed_tracklets=len(tentative),offline_links=links,thresholds=dict(anchor_confidence=.3,min_anchors=2,min_persistence_frames=12,min_mean_confidence=.2,merge_cosine=.92,merge_IoU=.65,max_gap_s=.55),association_GT_used=False,unconfirmed_detections_preserved=True)

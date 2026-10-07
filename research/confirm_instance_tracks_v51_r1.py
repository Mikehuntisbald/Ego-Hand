"""Use future anchors before confirming tracks; preserve alternate detections."""
import numpy as np
from confirm_instance_tracks_v51 import box_iou

def confirm_and_merge(tracks,descriptors):
    keys=sorted(tracks,key=lambda k:tracks[k]['frames'][0]['timestamp_s']);choices=[]
    for a in keys:
        ta=tracks[a]['frames'];end=ta[-1];candidates=[]
        for b in keys:
            if a==b or tracks[b]['frames'][0]['timestamp_s']<=ta[0]['timestamp_s']:continue
            tb=tracks[b]['frames'];start=tb[0];gap=start['timestamp_s']-end['timestamp_s']
            if gap>.55 or gap<-.15:continue
            if gap>0:
                xa=descriptors[(a,len(ta)-1)];xb=descriptors[(b,0)];iou=box_iou(end['box_xyxy'],start['box_xyxy'])
            else:
                lookup={round(f['timestamp_s'],6):(i,f) for i,f in enumerate(ta)};common=[(lookup[round(f['timestamp_s'],6)],j,f) for j,f in enumerate(tb) if round(f['timestamp_s'],6) in lookup]
                if not common:continue
                (i,fa),j,fb=common[-1];xa=descriptors[(a,i)];xb=descriptors[(b,j)];iou=float(np.median([box_iou(x[0][1]['box_xyxy'],x[2]['box_xyxy']) for x in common]))
            cosine=float(xa@xb/max(np.linalg.norm(xa)*np.linalg.norm(xb),1e-9))
            if cosine>=(.97 if gap<=0 else .92) and iou>=(.75 if gap<=0 else .65):candidates.append((.6*(1-iou)+.4*(1-cosine),b,gap,cosine,iou))
        candidates.sort()
        if candidates and (len(candidates)==1 or candidates[1][0]-candidates[0][0]>=.1):choices.append((a,*candidates[0]))
    targets={};incoming=set()
    for a,cost,b,gap,cosine,iou in sorted(choices,key=lambda x:x[1]):
        if b not in incoming:targets[a]=b;incoming.add(b)
    accepted=[];tentative=[];duplicate_observations=[];chains=[]
    for a in keys:
        if a in incoming:continue
        chain=[a];cursor=a
        while cursor in targets and targets[cursor] not in chain:cursor=targets[cursor];chain.append(cursor)
        bytime={}
        for key in chain:
            for frame in tracks[key]['frames']:
                t=round(frame['timestamp_s'],6)
                if t in bytime:
                    previous=bytime[t]
                    if frame['box_confidence']>previous['box_confidence']:bytime[t]=frame;duplicate_observations.append(previous)
                    else:duplicate_observations.append(frame)
                else:bytime[t]=frame
        frames=[bytime[t] for t in sorted(bytime)];confidence=np.asarray([f['box_confidence'] for f in frames]);strong=int((confidence>=.3).sum());record=dict(id=tracks[a]['id'],frames=frames,merged_from=[tracks[k]['id'] for k in chain])
        if strong>=2 or (len(frames)>=12 and confidence.mean()>=.2):accepted.append(record)
        else:tentative.append(dict(record,status='unconfirmed_detection_review_only',strong_observations=strong))
        chains.append(dict(original=record['merged_from'],frames=len(frames),strong_observations=strong,confirmed=strong>=2 or (len(frames)>=12 and confidence.mean()>=.2)))
    if duplicate_observations:tentative.append(dict(id='alternate_overlapping_hypotheses',frames=duplicate_observations,status='alternate_detection_review_only'))
    return accepted,tentative,dict(input_tracklets=len(tracks),confirmed_trajectories=len(accepted),unconfirmed_groups=len(tentative),offline_links=choices,chains=chains,association_GT_used=False,all_detection_hypotheses_preserved=True,future_anchors_used_for_confirmation=True,temporal_overlap_merge_uncalibrated=True)

"""Prediction-only, bounded 17-slot context resampling for v43 experiments."""
import collections
import numpy as np
import torch
from compare_detectors import iou

LONG_OFFSETS = [-120,-80,-50,-30,-10,-3,-2,-1,0,1,2,3,10,30,50,80,120]
SHORT_OFFSETS = [-48,-30,-18,-10,-6,-3,-2,-1,0,1,2,3,6,10,18,30,48]
MAX_CONTEXT_S = 1.6


class ObservationSampler:
    """All grouping and aliases use detector tracks, timestamps and box IoU."""
    def __init__(self, primary, full):
        self.full=full
        self.groups=collections.defaultdict(list)
        self.by_image=collections.defaultdict(list)
        for fid,r in enumerate(primary,1):
            self.groups[(r['sequence'],int(r['clip']),int(r['track_id']))].append(fid)
            self.by_image[r['image']].append(fid)
        self.lookup={}
        for key,ids in self.groups.items():
            ids.sort(key=lambda f:full[f-1]['timestamp_ns'])
            self.lookup[key]=(np.asarray(ids),np.asarray([full[f-1]['timestamp_ns'] for f in ids],np.int64))

    def primary_center(self,fid):
        r=self.full[fid-1]
        candidates=self.by_image[r['image']]
        assert candidates, 'Center has no prediction-only dense observation'
        overlap=iou([r['box']],[self.full[x-1]['box'] for x in candidates])[0]
        best=int(overlap.argmax())
        assert overlap[best]>.99, 'Do not infer a track from a weak box match'
        return candidates[best]

    def sample(self,centers,offsets=SHORT_OFFSETS):
        fids=np.zeros((len(centers),17),np.int64)
        dts=np.tile(np.asarray(offsets,np.float64)/30.,(len(centers),1))
        for index,center in enumerate(centers):
            center=int(center);anchor=self.primary_center(center)
            r=self.full[anchor-1];key=(r['sequence'],int(r['clip']),int(r['track_id']))
            # Legacy center aliases carry the same image/box but no timestamp.
            # Use the exactly matched dense observation's timestamp.
            ids,times=self.lookup[key];now=int(r['timestamp_ns'])
            used={anchor};fids[index,8]=center;dts[index,8]=0.
            for slot in sorted((x for x in range(17) if x!=8),key=lambda x:abs(offsets[x])):
                target=now+int(round(offsets[slot]*1e9/30.))
                nearest=int(np.abs(times-target).argmin());candidate=int(ids[nearest])
                delta=(int(times[nearest])-now)/1e9
                if abs(int(times[nearest])-target)>16_668_667 or candidate in used:continue
                # The observed timestamp, not just the requested slot, is bounded.
                if abs(delta)>max(abs(x) for x in offsets)/30.+2e-6:continue
                fids[index,slot]=candidate;dts[index,slot]=delta;used.add(candidate)
        return torch.from_numpy(fids),torch.tensor(dts,dtype=torch.float32)


def replace_windows(data,feature_ids,dt,xy_bank,observed_bank):
    return dict(data,feature_ids=feature_ids,dt=dt,
                xy=xy_bank[feature_ids],observed_2d=observed_bank[feature_ids])


def window_statistics(feature_ids,dt):
    valid=feature_ids>0
    count=valid.sum(1).float()
    near=dt[:,[7,9]].abs()[valid[:,[7,9]]]
    return dict(windows=len(feature_ids),valid_slots_mean=float(count.mean()),
                valid_slots_median=float(count.median()),valid_slots_min=int(count.min()),
                valid_slots_max=int(count.max()),max_observed_context_s=float(dt.abs()[valid].max()),
                near_interval_median_s=float(near.median()) if len(near) else None,
                source_frame_ids_unique=all(len(set(row[row>0].tolist()))==int((row>0).sum()) for row in feature_ids))

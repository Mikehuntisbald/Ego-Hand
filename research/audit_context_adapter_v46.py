import json,collections,numpy as np
from hand3d_v8_common import V7,save
import encode_hand3d_dense_v14 as sampling
from temporal_window_v44 import SHORT_OFFSETS
from complete_hand_tracks_acceleration_v46 import bounded_sample
run=V7.parent/'acceleration_validation_v46';rows=json.loads((run/'fresh_rows.json').read_text());groups=collections.defaultdict(list)
for i,r in enumerate(rows):groups[(r['sequence'],r['clip'],r['track_id'])].append(i)
sampling.OFFSETS=SHORT_OFFSETS;affected=[];maxdt=0;slots=0
for key,ids in groups.items():
    times=np.array([rows[i]['timestamp_ns']*1e-9 for i in ids]);n=len(ids)
    z=dict(times=times,xy=np.zeros((n,20,2),np.float32),available=np.ones((n,20),bool),positions=np.zeros((n,192,2),np.float32),roi=np.zeros((n,4),np.float32),scores=np.ones(n,np.float32))
    original,_=sampling.sample(z,'cpu');dt=original['dt'].abs();valid=original['rgb_valid'];bad=valid&(dt>1.600002)
    pass
    maxdt=max(maxdt,float(dt[valid].max()))
    sampled,chosen=bounded_sample(z,'cpu',sampling.sample)
    bad=valid&~sampled['rgb_valid']
    if bad.any():affected.append('_'.join(map(str,key)));slots+=int(bad.sum())
    assert (sampled['dt'].abs()[sampled['rgb_valid']]<=1.600002).all()
save(run/'context_adapter_audit.json',dict(affected_tracks=affected,removed_context_slots=slots,original_max_context_s=maxdt,corrected_max_context_s=1.600002,gt_used=False))
print(json.dumps(dict(affected_tracks=len(affected),slots=slots,original_max_context_s=maxdt)))

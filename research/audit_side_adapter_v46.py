"""Check side metadata against the exact WiLoR frontend ROI, before evaluation."""
import json
import numpy as np,cv2
from dit_v3_inference import prepare_observation
from hand3d_v8_common import V7,save
from compare_detectors import iou
from offline_rgb_encoder import crop_roi
run=V7.parent/'acceleration_validation_v46';rows=json.loads((run/'fresh_rows.json').read_text());changed=[];right=[]
for i,r in enumerate(rows):
    box=np.asarray(r['box'],float);center=(box[:2]+box[2:])/2;size=max(float((box[2:]-box[:2]).max())*1.3,24.)
    roi=np.r_[center-size/2,center+size/2].astype(np.float32);s=r['side_predictions']
    def choose(roi):
        overlap=iou([roi],s['boxes'])[0]
        return int(s['classes'][int(overlap.argmax())]) if len(overlap) and overlap.max()>=.05 else 1
    exact=choose(roi);right.append(exact)
    if choose(crop_roi(box))!=exact:changed.append(dict(index=i,sequence=r['sequence'],clip=r['clip'],track_id=r['track_id']))
checked=sorted(set([r['index'] for r in changed]+list(range(min(16,len(rows))))))
for i in checked:
    r=rows[i];p=prepare_observation(cv2.imread(r['image']),r['box'],r['camera'],r['side_predictions'])
    assert p['right']==right[i],i
save(run/'side_adapter_audit.json',dict(exact_frontend_roi='1.3*maxboxsize,min24; as prepare_pose_training.crop_and_geometry',changed=changed,total=len(rows),predicted_right=right,native_function_checked=len(checked),native_function_parity=True))
print(json.dumps(dict(changed=len(changed),affected_tracks=sorted({str((r['sequence'],r['clip'],r['track_id'])) for r in changed}))))

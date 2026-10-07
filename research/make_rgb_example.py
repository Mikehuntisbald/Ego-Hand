import json
import wilor_eval_common
import torch
from offline_rgb_data import RUN,save

rows=json.loads((RUN/'rows.json').read_text());data=torch.load(RUN/'windows.pt',weights_only=False);index=torch.load(RUN/'rgb_index.pt',weights_only=False)
records=json.loads((RUN/'rgb_records.json').read_text())
i=next(i for i,r in enumerate(rows) if r['role']=='test' and 40<=r['frame']<=105)
variant=1+i%4;frames=[]
for t in range(17):
    fid=int(index['feature_ids'][i,t])
    if not fid:continue
    record=records[fid-1];seen=data['observed'][i,t].clone();masked=6<=t<12
    if masked:seen[:]=False
    xy=data['xy'][i,t]*1408
    frame=dict(timestamp_s=float(data['dt'][i,t])+10,image=record['image'],box_xyxy=record['box'],
        observed=seen.tolist(),xy_px=[p.tolist() if s else None for p,s in zip(xy,seen)])
    if masked:frame.update(occluder_crop_xyxy=index['rectangles'][fid,variant].tolist(),occluder_color=32+64*((record['clip']+1)%4))
    frames.append(frame)
out=RUN/'delivery';out.mkdir(exist_ok=True)
save(out/'example_input.json',dict(image_size=[1408,1408],tracks=[dict(id='example',frames=frames)]))
save(out/'example_provenance.json',dict(row_index=i,variant=variant,selection='First task-test window with central frame in [40,105], not selected by prediction error',
    gt_inputs=False,pixel_mask_before_encoder=True,all_old_keypoints_erased_in_masked_frames=True))
print(str(out))

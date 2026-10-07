"""Inspect all outputs, including frames excluded from GT motion metrics."""
import json
import torch
from hand3d_v8_common import V7,save

root=V7.parent;run=root/'motion_threshold_v45'
strict=torch.load(run/'strict.pt',weights_only=False);hard=torch.load(run/'hard_only_x2.pt',weights_only=False)
allrows=json.loads((root/'joint_mano_v28/rows.json').read_text());ids=torch.tensor([i for i,r in enumerate(allrows) if r['role']=='dev_select']);rows=[allrows[i] for i in ids]
labels=torch.load(root/'joint_mano_v28/evaluation_labels.pt',weights_only=False,mmap=True);valid=labels['valid'][ids];gt=labels['gt'][ids]
delta=(hard['prediction']-strict['prediction']).norm(dim=-1)*1000
frame=delta.max(-1).values;changed=frame>.01
events=[]
for i in torch.nonzero(changed).flatten().tolist():
    r=rows[i];event=dict(index=i,sequence=r['sequence'],clip=r['clip'],track_id=r['track_id'],frame=r['frame'],
        image=r['image'],maximum_point_change_mm=float(frame[i]),gt_valid_points=int(valid[i].sum()),
        is_selected_center=r['window_index'] is not None)
    if valid[i].any():
        for name,result in [('strict',strict),('hard_x2',hard)]:
            camera=(result['prediction'][i]-gt[i]).norm(dim=-1)*1000
            rel=((result['prediction'][i]-result['prediction'][i,5])-(gt[i]-gt[i,5])).norm(dim=-1)*1000
            mask=valid[i].clone();mask[5]=False
            event[name+'_camera_error_mm']=float(camera[mask].mean())
            event[name+'_relative_error_mm']=float(rel[mask].mean())
    events.append(event)
events.sort(key=lambda r:r['maximum_point_change_mm'],reverse=True)
summary=dict(all_observations=len(rows),changed_frames_over_001mm=int(changed.sum()),
    changed_frames_with_any_gt=int((changed&valid.any(-1)).sum()),changed_frames_without_any_gt=int((changed&~valid.any(-1)).sum()),
    max_all_points_change_mm=float(delta.max()),max_gt_valid_points_change_mm=float(delta[valid].max()),
    strict_pose_contracted_segments=sum(x['pose_scale']<1 for x in strict['restoration']),
    strict_root_contracted_segments=sum(x['root_scale']<1 for x in strict['restoration']),
    hard_x2_pose_contracted_segments=sum(x['pose_scale']<1 for x in hard['restoration']),
    hard_x2_root_contracted_segments=sum(x['root_scale']<1 for x in hard['restoration']),
    gt_motion_metric_unchanged=True,events=events,changed_gt_frame_details=[x for x in events if x['gt_valid_points']>0],
    interpretation='Hard restoration changes a few mostly unmatched/no-valid-GT outputs substantially. Unchanged GT-based retention does not imply unchanged all-frame output.')
save(run/'hard_restore_effect.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='events'}));print(json.dumps(events[:3]))

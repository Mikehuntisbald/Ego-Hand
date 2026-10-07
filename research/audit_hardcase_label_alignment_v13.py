"""Verify hard-case label/image lineage before fitting more models.

All GT checks below are offline diagnostics; no label or oracle enters inference.
"""
import json
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,load,save
from hand3d_trajectory_data_v9 import RUN as V9,targets
import spatial_rgb_common as s
RUN=V7.parent/'hardcase_alignment_v13'

def main():
    data=load('cpu');extra=torch.load(V9/'trajectory_targets.pt',weights_only=False);records,_=s.records_and_index();queue=json.loads((V7.parent/'natural_reliability_v4/delivery/hardcase_review_queue.json').read_text());classification=json.loads((V7.parent/'natural_reliability_v4/temporal_clue_audit/classification.json').read_text());classes={r['id']:r for r in classification['cases']};cases=[];annotations={}
    for q in queue:
        fid=int(data['feature_ids'][q['window_index'],8]);r=records[fid-1];image=Path(r['image']);parts=image.relative_to(s.common.ROOT/'export/images').parts;path=s.common.ROOT/'export/annotations'/parts[0]/parts[1]/(parts[2]+'.jsonl')
        if path not in annotations:annotations[path]=[json.loads(line) for line in path.read_text().splitlines()]
        frame=annotations[path][int(r['frame'])];assert int(image.stem)==int(frame['frame'])==int(r['frame']);assert frame['sequence']==r['sequence'] and frame['clip']==r['clip']
        camera_equal=frame['camera']==r['camera'];distances=[float(np.max(np.abs(np.asarray(h['xyz_camera_m'])-data['gt'][q['window_index']].numpy()))) for h in frame['hands']];closest=int(np.argmin(distances));matched=frame['hands'][closest]
        ids=torch.tensor([q['window_index']]);gt,valid,_,_=targets(data,extra,ids);f=data['feature_ids'][ids];center=f[:,8];aligned=torch.einsum('btjc,bck->btjk',data['world'][f]-data['translation'][center,None,None],data['rotation'][center]);mask=valid[0].clone();mask[:,5]=False
        error=(((aligned[0]-aligned[0,:,5:6])-(gt[0]-gt[0,:,5:6])).norm(dim=-1)*1000*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        gt_displacement=(gt[0,:,5]-gt[0,8,5]).norm(dim=-1);dt=data['dt'][ids][0];usable=valid[0,:,5]&(dt.abs()>1e-6);speeds=gt_displacement[usable]/dt[usable].abs()
        clues=[dict(slot=j,dt_s=float(dt[j]),relative_coarse_error_mm=float(error[j]),gt_wrist_displacement_mm=float(gt_displacement[j]*1000)) for j in range(17) if j!=8 and bool(valid[0,j,5])]
        cases.append(dict(id=q['id'],window_index=q['window_index'],category=classes.get(q['id'],{}).get('category'),image=str(image),official_annotation=str(path),frame=int(r['frame']),camera_metadata_exact=camera_equal,official_label_max_difference_m=distances[closest],official_side=matched['side'],center_wilor_relative_mm=float(error[8]),context=clues,min_context_relative_mm=min((c['relative_coarse_error_mm'] for c in clues),default=None),median_gt_wrist_speed_m_s=float(speeds.median()) if len(speeds) else None))
    maximum=max(c['official_label_max_difference_m'] for c in cases);same_camera=all(c['camera_metadata_exact'] for c in cases)
    save(RUN/'results.json',dict(complete=True,cases=cases,official_labels_agree=maximum<2e-6,official_label_max_difference_m=maximum,camera_metadata_exact=same_camera,scope='Check against exported official frame labels only, not independent ground-truth quality validation; history minima use GT diagnostically and never select output'))
    print(json.dumps(dict(cases=len(cases),official_labels_agree=maximum<2e-6,max_difference_m=maximum,camera_metadata_exact=same_camera,severe=[c for c in cases if c['id'] in classes]),indent=2),flush=True)

if __name__=='__main__':main()

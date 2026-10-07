"""GT side is diagnostic only; prediction-side lineage remains separate."""
import json
from collections import defaultdict
import torch
from hand3d_trajectory_data_v9 import RUN
from hand3d_v8_common import V7,save
import spatial_rgb_common as s

def main():
    data=torch.load(V7/'data.pt',weights_only=False,mmap=True);records,_=s.records_and_index();queue=json.loads((V7.parent/'natural_reliability_v4/delivery/hardcase_review_queue.json').read_text());sources={}
    for source in sorted({r['source'] for r in records}):
        rows=json.loads((V7.parent/source/'locked_rows.json').read_text());parts=[]
        for p in sorted((V7.parent/source/'locked_observations').glob('*.pt')):parts.append(torch.load(p,weights_only=False,mmap=True)['geometry'][:,1:4].float())
        sources[source]=(rows,torch.cat(parts))
    cases=[]
    for q in queue:
        fids=data['feature_ids'][q['window_index']];center=records[int(fids[8])-1];sr,geo=sources[center['source']];r=sr[center['index']];side=r.get('side_label');observations=[]
        for f,dt in zip(fids.tolist(),data['dt'][q['window_index']].tolist()):
            if not f:continue
            rec=records[f-1];rows,ge=sources[rec['source']];original=rows[rec['index']];g=ge[rec['index']];observations.append(dict(frame=rec['frame'],dt=dt,gt_side_diagnostic=original.get('side_label'),matched=original.get('matched'),predicted_right=float(g[0]),side_confidence=float(g[1]),side_iou=float(g[2])))
        matched=[x for x in observations if x['matched'] and x['gt_side_diagnostic'] is not None];wrong=[x for x in matched if side is not None and x['gt_side_diagnostic']!=side];predwrong=[x for x in matched if ('right' if x['predicted_right']>=.5 else 'left')!=x['gt_side_diagnostic']]
        cases.append(dict(id=q['id'],center_gt_side_diagnostic=side,history_wrong_gt_side=len(wrong),matched_context=len(matched),predicted_side_wrong=len(predwrong),severe=q['after_px']>40,observations=observations))
    summary=dict(hardcases=len(cases),contexts_with_other_gt_hand=sum(x['history_wrong_gt_side']>0 for x in cases),severe_contexts_with_other_gt_hand=sum(x['history_wrong_gt_side']>0 and x['severe'] for x in cases),center_side_prediction_wrong=sum(('right' if next(r for r in x['observations'] if r['dt']==0)['predicted_right']>=.5 else 'left')!=x['center_gt_side_diagnostic'] for x in cases),cases=cases,scope='GT side used only to audit identity; never used for tracking or inference conditioning')
    save(RUN/'track_identity_audit.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='cases'}),flush=True)

if __name__=='__main__':main()

"""Prediction-only handedness consensus audit before changing upstream XYZ.

GT side and pose are attached to the audit after choices; neither selects votes.
All retained-failure results remain diagnostics. No new model is adopted here.
"""
import collections,json
import numpy as np,torch
from hand3d_v8_common import V7,load,save
from compare_detectors import iou
from offline_rgb_encoder import crop_roi
import spatial_rgb_common as s
RUN=V7.parent/'side_consensus_v16'
POLICY=dict(min_side_iou=.05,min_side_confidence=.1,min_support=3,min_dominance=.8,max_center_confidence_to_override=.7)

def votes(records):
    groups=collections.defaultdict(list)
    for fid,r in enumerate(records,1):
        side=r['side_predictions'];overlap=iou([crop_roi(r['box'])],side['boxes'])[0]
        if len(overlap) and overlap.max()>=POLICY['min_side_iou']:
            j=int(overlap.argmax());right=int(side['classes'][j]);confidence=float(side['scores'][j]);weight=float(overlap[j])*confidence*float(r['score'])
            if confidence>=POLICY['min_side_confidence']:groups[(r['sequence'],r['clip'],r['track_id'])].append((fid,right,weight))
    out={}
    for key,rows in groups.items():
        weights=[sum(w for fid,c,w in rows if c==side) for side in [0,1]];side=int(weights[1]>weights[0]);dominance=weights[side]/max(sum(weights),1e-9);support=sum(c==side for fid,c,w in rows)
        out[key]=dict(right=side,dominance=dominance,support=support,eligible=support>=POLICY['min_support'] and dominance>=POLICY['min_dominance'])
    return out

def main():
    torch.set_num_threads(4);RUN.mkdir(exist_ok=True);save(RUN/'policy_before_audit.json',POLICY)
    old=load('cpu');records,_=s.records_and_index();sources={}
    for src in {r['source'] for r in records}:
        rows=json.loads((V7.parent/src/'locked_rows.json').read_text());chunks=[]
        for path in sorted((V7.parent/src/'locked_observations').glob('*.pt')):chunks.append(torch.load(path,weights_only=False,mmap=True)['geometry'][:,1:4].float())
        sources[src]=(rows,torch.cat(chunks))
    summary={}
    for label,folder,aligned in [('train_development','dense_sampling_v13','aligned_density_v13'),('retained_failures','hard_dense_v14','hard_aligned_v14')]:
        dense=json.loads((V7.parent/folder/'fresh_rows.json').read_text());consensus=votes(dense);association=json.loads((V7.parent/aligned/'center_association.json').read_text());results=[]
        for a in association:
            ix=a['source_window_index'];r=records[int(old['feature_ids'][ix,8])-1];sr,geo=sources[r['source']];g=geo[r['index']];original=sr[r['index']];candidate=int(g[0]);proposal=None
            if a['matched']:
                rec=dense[a['new_center_detection']-1];proposal=consensus.get((rec['sequence'],rec['clip'],rec['track_id']))
                if proposal and proposal['eligible'] and float(g[1])<POLICY['max_center_confidence_to_override']:candidate=proposal['right']
            gt_side=original.get('side_label');base=old['xyz_camera_bank'][old['feature_ids'][ix,8]];gt=old['gt'][ix];mask=old['valid'][ix].clone();mask[5]=False
            error=float(((((base-base[5])-(gt-gt[5])).norm(dim=-1)*1000)*mask).sum()/mask.sum().clamp_min(1))
            expected=1 if gt_side=='right' else 0 if gt_side=='left' else None
            results.append(dict(window_index=ix,role=old['roles'][ix],sequence=r['sequence'],clip=r['clip'],frame=r['frame'],original_right=int(g[0]),original_confidence=float(g[1]),candidate_right=candidate,changed=candidate!=int(g[0]),consensus=proposal,GT_side_diagnostic=gt_side,original_wrong=expected is not None and int(g[0])!=expected,candidate_wrong=expected is not None and candidate!=expected,baseline_relative_mm=error))
        roles={}
        for role in sorted({r['role'] for r in results}):
            group=[r for r in results if r['role']==role];changed=[r for r in group if r['changed']]
            roles[role]=dict(windows=len(group),changed=len(changed),original_wrong=sum(r['original_wrong'] for r in group),candidate_wrong=sum(r['candidate_wrong'] for r in group),fixed=sum(r['original_wrong'] and not r['candidate_wrong'] for r in changed),introduced=sum(not r['original_wrong'] and r['candidate_wrong'] for r in changed),hard_changed=sum(r['baseline_relative_mm']>40 for r in changed))
        summary[label]=dict(roles=roles,rows=results)
    save(RUN/'audit_results.json',dict(complete=True,policy=POLICY,sets=summary,scope='Prediction-only votes; GT used after choices only for diagnosis. Candidate handedness has not yet been used to recompute XYZ or train/validate a replacement. Retained47already inspected; no fresh accuracy claim.'))
    print(json.dumps({label:value['roles'] for label,value in summary.items()},indent=2),flush=True)

if __name__=='__main__':main()

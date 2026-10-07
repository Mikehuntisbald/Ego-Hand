"""Offline gap-filling candidates from other prediction-only tracklets.

No GT participates in selection. This only audits possible real RGB context;
it does not change the accepted v16 model or report fresh accuracy evidence.
"""
import collections,json
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,save
from audit_side_consensus_v16 import votes,POLICY as SIDE_POLICY
from compare_detectors import iou
import spatial_rgb_common as s
RUN=V7.parent/'context_merge_v17'
POLICY=dict(min_box_score=.1,side_support=3,side_dominance=.8,root_margin_m=.3,root_speed_allowance_m_s=.25,fill_existing_missing_slots_only=True,center_identity='Same predicted side as the actualv16centerXYZ; preservehighconfidencecenterclass')

def native_side(r):
    b=np.asarray(r['box'],float);c=(b[:2]+b[2:])/2;size=max(float((b[2:]-b[:2]).max())*1.3,24.);roi=np.r_[c-size/2,c+size/2].astype(np.float32)
    p=r['side_predictions'];overlap=iou([roi],p['boxes'])[0]
    if len(overlap) and overlap.max()>=SIDE_POLICY['min_side_iou']:
        j=int(overlap.argmax());return int(p['classes'][j]),float(p['scores'][j])
    return 1,0.

def main():
    torch.set_num_threads(4);RUN.mkdir(exist_ok=True);save(RUN/'policy_before_audit.json',POLICY);summary={}
    # Only the already-frozen predicted class is read from this audit payload.
    # GT diagnostics in that file never participate in frame selection.
    side_audit=json.loads((V7.parent/'side_consensus_v16/audit_results.json').read_text())
    for label,root,observations,aligned in [('train_development','side_data_v16/consensus','dense_sampling_v13','aligned_density_v13'),('retained_failures','hard_side_v16/consensus','hard_dense_v14','hard_aligned_v14')]:
        data=torch.load(V7.parent/root/'dense_data.pt',weights_only=False,mmap=True);records=json.loads((V7.parent/observations/'fresh_rows.json').read_text());association=json.loads((V7.parent/aligned/'center_association.json').read_text());consensus=votes(records);perframe=collections.defaultdict(list);qualified={}
        for fid,r in enumerate(records,1):
            key=(r['sequence'],r['clip'],r['track_id']);vote=consensus.get(key)
            if vote and vote['eligible'] and r['score']>=POLICY['min_box_score']:
                qualified[fid]=vote;perframe[(r['sequence'],r['clip'],r['frame'])].append(fid)
        predicted_identity={r['window_index']:r['candidate_right'] for r in side_audit['sets'][label]['rows']}
        fids=data['feature_ids'].clone();dt=data['dt'].clone();rows=[]
        for k,a in enumerate(association):
            if not a['matched']:continue
            centre=records[a['new_center_detection']-1];vote=consensus.get((centre['sequence'],centre['clip'],centre['track_id']))
            if not vote or not vote['eligible']:continue
            identity=predicted_identity[int(data['source_window_indices'][k])];existing=data['feature_ids'][k];existing_times=data['dt'][k];added=[]
            for slot in range(17):
                if existing[slot]>0:continue
                target_frame=centre['frame']+int(round(float(existing_times[slot])*30));candidates=perframe.get((centre['sequence'],centre['clip'],target_frame),[])
                candidates=[f for f in candidates if qualified[f]['right']==identity]
                if not candidates:continue
                valid_slots=torch.where(existing>0)[0];near=valid_slots[(existing_times[valid_slots]-existing_times[slot]).abs().argmin()];anchor=data['world'][existing[near],5];gap=abs(float(existing_times[slot]-existing_times[near]));maximum=POLICY['root_margin_m']+POLICY['root_speed_allowance_m_s']*gap
                options=[]
                for fid in candidates:
                    distance=float((data['world'][fid,5]-anchor).norm())
                    if distance<=maximum:options.append((records[fid-1]['score']*qualified[fid]['dominance']*np.exp(-distance/.3),fid,distance))
                if not options:continue
                quality,fid,distance=max(options);fids[k,slot]=fid;dt[k,slot]=(records[fid-1]['timestamp_ns']-centre['timestamp_ns'])*1e-9
                added.append(dict(slot=slot,frame=target_frame,fid=fid,predicted_side=identity,score=records[fid-1]['score'],root_distance_m=distance,source_track=records[fid-1]['track_id']))
            if added:rows.append(dict(window_index=int(data['source_window_indices'][k]),role=data['roles'][k],sequence=centre['sequence'],clip=centre['clip'],frame=centre['frame'],original_context=int((existing>0).sum()),new_context=int((fids[k]>0).sum()),added=added))
        assert torch.equal(fids[:,8],data['feature_ids'][:,8])
        torch.save(dict(feature_ids=fids,dt=dt,source_window_indices=data['source_window_indices']),RUN/f'{label}_proposed_slots.pt')
        roles={}
        for role in sorted(set(data['roles'])):
            rr=[r for r in rows if r['role']==role];roles[role]=dict(windows_with_added_frames=len(rr),added_real_contexts=sum(len(r['added']) for r in rr))
        summary[label]=dict(roles=roles,changed_windows=rows,original_observed_fraction=float((data['feature_ids']>0).float().mean()),new_observed_fraction=float((fids>0).float().mean()))
    save(RUN/'audit_results.json',dict(complete=True,policy=POLICY,sets=summary,scope='Prediction-only missing-context proposals; no GT/side/visibility/oracle in choices; no model trained or recovery metric computed; independently check identity/labels before retraining. Retained failures already inspected.'))
    print(json.dumps({k:{'roles':v['roles'],'before':v['original_observed_fraction'],'after':v['new_observed_fraction']} for k,v in summary.items()},indent=2),flush=True)

if __name__=='__main__':main()

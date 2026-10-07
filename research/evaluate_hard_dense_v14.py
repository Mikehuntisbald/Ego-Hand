"""Frozen v14 on retained old failures; diagnostic evaluation only."""
import json
import torch
from hand3d_v8_common import V7,save,metrics
from hand3d_data_v7 import batch,risk_features
from hand3d_risk_v7 import Risk3D
from density_model_v13 import DensityTrajectoryHand3D,POLICY
from train_aligned_density_v13 import make
from native_projection_policy_v11 import apply as conservative
from adaptive_projection_v14 import apply

RUN=V7.parent/'adaptive_projection_v14/dit_dense';DATA=V7.parent/'hard_aligned_v14';RISK=V7.parent/'aligned_density_v13/risk_dense'

@torch.inference_mode()
def main():
    torch.set_num_threads(4);device='cuda:3'
    ready=json.loads((DATA/'ready.json').read_text());assert ready['complete'] and ready['center_observations_and_targets_exact'] and ready['center_native_rgb_exact']
    frozen=json.loads((RUN/'fourth_seal.json').read_text());assert json.loads((RUN/'fourth_results.json').read_text())['passed']
    raw_data=torch.load(DATA/'dense_data.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw_data.items()}
    bank=torch.load(DATA/'native_bank.pt',weights_only=False,mmap=True).to(device);ids=torch.arange(len(data['roles']),device=device)
    ck=torch.load(RISK/'risk_all.pt',weights_only=False,map_location=device);risk=Risk3D(ck['dim']).to(device).eval();risk.load_state_dict(ck['model']);temp=torch.tensor(json.loads((RISK/'calibration.json').read_text())['temperature'],device=device)
    prob=(risk(risk_features(batch(data,ids)))/temp).sigmoid()
    ck=torch.load(V7.parent/'native_density_v13/dit_dense/best.pt',weights_only=False,map_location=device);model=DensityTrajectoryHand3D('dit',True).to(device).eval();model.load_state_dict(ck['model'])
    outputs={}
    for start in range(0,len(ids),8):
        b=make(data,bank,ids[start:start+8],prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(b,seed=202610114+start)
        raw=p['xyz_camera_m'].float()
        for key,v in [('prediction',apply(raw,b,frozen['policy'])),('fallback',conservative(raw,b['base'],POLICY)),('proposal',raw),('std',p['std_m'].float())]:outputs.setdefault(key,[]).append(v)
    out={k:torch.cat(v) for k,v in outputs.items()};base=data['xyz_camera_bank'][data['feature_ids'][:,8]];gt=data['gt'];valid=data['valid']
    queue=json.loads((DATA/'case_queue.json').read_text());classification=json.loads((V7.parent/'natural_reliability_v4/temporal_clue_audit/classification.json').read_text());classes={q['id']:q for q in classification['cases']}
    v11=torch.load(V7.parent/'native_projection_v11/old_predictions.pt',weights_only=False);oldlookup={int(x):i for i,x in enumerate(v11['indices'])};prior=torch.stack([v11['prediction'][oldlookup[q['window_index']]] for q in queue]).to(device)
    assert torch.equal(base.cpu(),torch.stack([v11['base'][oldlookup[q['window_index']]] for q in queue]))
    cases=[]
    for i,q in enumerate(queue):
        rec=data['rows'][i];fids=data['feature_ids'][i];dt=data['dt'][i]
        c=dict(id=q['id'],window_index=q['window_index'],category=classes.get(q['id'],{}).get('category','unclassified'),reason=classes.get(q['id'],{}).get('reason',''),row=rec,valid_context_count=int((fids>0).sum()),context_times_s=[float(dt[k]) if fids[k]>0 else None for k in range(17)])
        for name,p in [('baseline',base),('final',out['prediction']),('fallback',out['fallback']),('v11_same_center',prior),('raw',out['proposal'])]:c[name]=metrics(p[i:i+1],base[i:i+1],gt[i:i+1],valid[i:i+1])
        cases.append(c)
    groups={}
    for label,keep in [('all47',torch.ones(len(queue),dtype=torch.bool,device=device)),('severe8',torch.tensor([q['id'] in classes for q in queue],device=device)),('A_nearby_clues',torch.tensor([q['id'] in classes and classes[q['id']]['category']=='A' for q in queue],device=device)),('B_clip_missing_clues',torch.tensor([q['id'] in classes and classes[q['id']]['category']=='B' for q in queue],device=device))]:
        groups[label]=dict(windows=int(keep.sum()),**{name:metrics(p[keep],base[keep],gt[keep],valid[keep]) for name,p in [('baseline',base),('final',out['prediction']),('fallback',out['fallback']),('v11_same_center',prior),('raw',out['proposal'])]})
    save(RUN/'hard_results.json',dict(complete=True,policy=frozen['policy'],groups=groups,cases=cases,original_centers_exact=True,association=ready['association'],scope='All47 retained failures, already inspected. Diagnostic only; no post-hoc model/gate changes; A/B labels are manual visual audit, not per-finger visibility truth.'))
    torch.save(dict(**{k:v.cpu() for k,v in out.items()},base=base.cpu(),gt=gt.cpu(),valid=valid.cpu(),probability=prob.cpu(),indices=data['source_window_indices'].cpu(),rows=data['rows']),RUN/'hard_predictions.pt')
    print(json.dumps(dict(complete=True,groups=groups)),flush=True)

if __name__=='__main__':main()

import json,torch
from hand3d_v8_common import V7,save,metrics
from hand3d_data_v7 import batch,risk_features
from hand3d_risk_v7 import Risk3D
from density_model_v13 import DensityTrajectoryHand3D
from adaptive_projection_v14 import apply
RUN=V7.parent/'side_native_v16';DATA=V7.parent/'hard_side_v16/consensus';RISK=V7.parent/'side_data_v16/consensus/risk_dense'

@torch.inference_mode()
def main():
    torch.set_num_threads(4);device='cuda:3';assert json.loads((DATA/'ready.json').read_text())['complete']
    frozen=json.loads((RUN/'fifth_seal.json').read_text());raw=torch.load(DATA/'dense_data.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw.items()};bank=torch.load(DATA/'native_bank.pt',weights_only=False,mmap=True).to(device);ids=torch.arange(len(data['roles']),device=device)
    ck=torch.load(RISK/'risk_all.pt',weights_only=False,map_location=device);risk=Risk3D(ck['dim']).to(device).eval();risk.load_state_dict(ck['model']);temp=torch.tensor(json.loads((RISK/'calibration.json').read_text())['temperature'],device=device);prob=(risk(risk_features(batch(data,ids)))/temp).sigmoid()
    ck=torch.load(RUN/'consensus/uniform_adaptive/best.pt',weights_only=False,map_location=device);model=DensityTrajectoryHand3D('dit',True).to(device).eval();model.load_state_dict(ck['model']);outputs={}
    for start in range(0,len(ids),8):
        b=batch(data,ids[start:start+8],prob);b['rgb_native']=bank[data['feature_ids'][start:start+8]]
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(b,seed=202610114+start)
        proposal=p['xyz_camera_m'].float()
        for key,x in [('prediction',apply(proposal,b,frozen['policy'])),('proposal',proposal),('std',p['std_m'].float())]:outputs.setdefault(key,[]).append(x)
    out={k:torch.cat(v) for k,v in outputs.items()};base=data['original_base_for_evaluation'];gt=data['gt'];valid=data['valid'];coarse=data['xyz_camera_bank'][data['feature_ids'][:,8]]
    q=json.loads((V7.parent/'hard_aligned_v14/case_queue.json').read_text());classes={r['id']:r for r in json.loads((V7.parent/'natural_reliability_v4/temporal_clue_audit/classification.json').read_text())['cases']};old=torch.load(V7.parent/'adaptive_projection_v14/dit_dense/hard_predictions.pt',weights_only=False);assert torch.equal(old['base'],base.cpu());v14=old['prediction'].to(device)
    groups={};cases=[]
    for label,keep in [('all47',torch.ones(len(q),dtype=torch.bool,device=device)),('severe8',torch.tensor([r['id'] in classes for r in q],device=device)),('A_nearby_clues',torch.tensor([classes.get(r['id'],{}).get('category')=='A' for r in q],device=device)),('B_clip_missing_clues',torch.tensor([classes.get(r['id'],{}).get('category')=='B' for r in q],device=device))]:
        groups[label]=dict(windows=int(keep.sum()),**{k:metrics(p[keep],base[keep],gt[keep],valid[keep]) for k,p in [('baseline',base),('v14',v14),('corrected_wilor',coarse),('final',out['prediction']),('raw',out['proposal'])]})
    for i,r in enumerate(q):
        c=dict(id=r['id'],window_index=r['window_index'],category=classes.get(r['id'],{}).get('category','unclassified'),reason=classes.get(r['id'],{}).get('reason',''))
        for k,p in [('baseline',base),('v14',v14),('corrected_wilor',coarse),('final',out['prediction']),('raw',out['proposal'])]:c[k]=metrics(p[i:i+1],base[i:i+1],gt[i:i+1],valid[i:i+1])
        cases.append(c)
    save(RUN/'hard_results.json',dict(complete=True,groups=groups,cases=cases,scope='Retained47failures; alreadyinspected; fixedv16notretuned; no new independent claim'))
    torch.save(dict(**{k:v.cpu() for k,v in out.items()},base=base.cpu(),coarse=coarse.cpu(),gt=gt.cpu(),valid=valid.cpu(),indices=data['source_window_indices'].cpu(),rows=data['rows']),RUN/'hard_predictions.pt');print(json.dumps(groups,indent=2),flush=True)

if __name__=='__main__':main()

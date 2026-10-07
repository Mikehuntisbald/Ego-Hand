"""Check proposal reliability on calibration clips; test only after locking refinement."""
import json,shutil,hashlib,itertools
import numpy as np,torch
import spatial_rgb_common as s
from natural_reliability import RUN,RiskHead,risk_features
from natural_policy import apply_policy
from train_spatial_temporal import load
from evaluate_natural_reliability import proposals,measure
from evaluate_offline_kp import bootstrap

@torch.inference_mode()
def candidate_scores(data,raw,model,temp,ids,candidates):
    output={}
    for arm in ['dit','regression','rgb_probe']:
        values=[]
        for start in range(0,len(ids),96):
            ix=ids[start:start+96];f=data['feature_ids'][ix];xy=data['xy'][ix].clone();available=data['observed'][ix].clone()
            xy[:,8]=candidates[arm][start:start+len(ix)];available[:,8]=True
            x=risk_features(xy,available,data['dt'][ix],data['roi'][f],raw['scores'][f],raw['frame_bank'][f],data['positions_bank'][f])
            values.append((model(x)/temp).sigmoid())
        output[arm]=torch.cat(values)
    return output

def main():
    torch.set_num_threads(4);device='cuda:0'
    if (RUN/'policy_refinement.json').exists():return
    rows,data=load(device);raw=torch.load(RUN/'features.pt',weights_only=False,mmap=True)
    roles=np.array(raw['roles']);raw={k:(v.to(device) if k in ['scores','frame_bank'] else v) for k,v in raw.items()}
    prob=torch.load(RUN/'probabilities.pt',weights_only=False)['joint'].to(device)
    ck=torch.load(RUN/'risk_joint.pt',weights_only=False,map_location=device);risk=RiskHead(ck['dim']).to(device).eval();risk.load_state_dict(ck['model'])
    temp=json.loads((RUN/'risk_calibration.json').read_text())['joint']['temperature']
    ids=torch.tensor(np.where(roles=='dev_calibrate')[0],device=device);base,cand=proposals(data,prob,ids,device);scores=candidate_scores(data,raw,risk,temp,ids,cand)
    gt=data['gt'][ids];available=data['observed'][ids,8];confirmed=torch.zeros_like(available);roi=data['roi'][data['feature_ids'][ids,8]]
    m=data['valid'][ids].clone()&available;m[:,5]=False
    previous=json.loads((RUN/'policy.json').read_text());best=previous;trials=[]
    for arm,tau,blend,limit,margin,maximum in itertools.product(['dit','regression','rgb_probe'],[.2,.35,.5,.65],[.5,1.],[.1,.25,.5],[.05,.15,.3],[.2,.35,.5]):
        policy=dict(arm=arm,threshold=tau,blend=blend,max_delta_roi=limit,agreement_roi=None,risk_margin=margin,proposal_risk_max=maximum)
        p,selected=apply_policy(base,cand,prob[ids],available,confirmed,roi,policy,scores);met=measure(p,base,gt,m,selected)
        passed=met['harm_good_to_over20_rate']<=.01 and met['selection_rate']<=.3 and met['bad_mean_px']<=met['base_bad_mean_px']*.95
        trials.append(dict(policy=policy,metrics=met,passed=passed))
        if passed and met['mean_px']<best['metrics']['mean_px']:best=trials[-1]
    improved=best['metrics']['mean_px']<=previous['metrics']['mean_px']-.1
    s.save(RUN/'policy_refinement.json',dict(accepted=improved,previous=previous,selected=best if improved else previous,
        rationale='Assess candidate as well as input; calibration clips only; >=0.1px incremental improvement required',trials=trials))
    if not improved:print(json.dumps(dict(accepted=False,best=best)));return
    # Preserve the first sealed experiment and its already opened audit.
    shutil.copytree(RUN/'sealed',RUN/'sealed_v1',dirs_exist_ok=True)
    s.save(RUN/'policy.json',best);shutil.copy2(RUN/'policy.json',RUN/'sealed/policy.json')
    for name in ['natural_policy.py','refine_natural_policy.py']:shutil.copy2(s.Path(__file__).parent/name,RUN/'sealed'/name)
    s.save(RUN/'sealed/manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (RUN/'sealed').iterdir() if p.is_file() and p.name!='manifest.json'})
    ids=torch.tensor(np.where(roles=='test')[0],device=device);old=np.load(RUN/'predictions.npz')
    assert np.array_equal(ids.cpu().numpy(),old['window_indices'])
    base=torch.tensor(old['base'],device=device);cand={a:torch.tensor(old[a],device=device) for a in ['dit','regression','rgb_probe']}
    scores=candidate_scores(data,raw,risk,temp,ids,cand);gt=data['gt'][ids];available=data['observed'][ids,8];confirmed=torch.zeros_like(available);roi=data['roi'][data['feature_ids'][ids,8]]
    valid=data['valid'][ids].clone();valid[:,5]=False;m=valid&available
    p,selected=apply_policy(base,cand,prob[ids],available,confirmed,roi,best['policy'],scores)
    report=json.loads((RUN/'test_results.json').read_text());shutil.copy2(RUN/'test_results.json',RUN/'test_results_v1.json')
    report['policy']=best;report['all_test']=measure(p,base,gt,m,selected)
    clusters=np.array([rows[i]['sequence'] for i in ids.tolist()]);delta=((p-gt).norm(dim=-1)-(base-gt).norm(dim=-1))*1408
    report['mean_change_ci95_px']=bootstrap(delta.cpu().numpy(),m.cpu().numpy(),clusters)
    lookup={wi:j for j,wi in enumerate(ids.tolist())};subset=json.loads((s.RUN/'natural_finger_audit/candidates.json').read_text());ix=torch.tensor([lookup[q['window_index']] for q in subset],device=device)
    report['hardcases']=measure(p[ix],base[ix],gt[ix],m[ix],selected[ix]);report['examples']=[]
    for q in json.loads((s.RUN/'natural_finger_audit/reviewed_examples.json').read_text()):
        j=lookup[q['window_index']];one=torch.tensor([j],device=device)
        report['examples'].append(dict(id=q['id'],category=q['visual_review_category'],**measure(p[one],base[one],gt[one],m[one],selected[one]),mean_p_bad=float(prob[ids[j]][m[j]].mean())))
    report['refinement_scope']='Candidate-risk refinement motivated by prior audit, selected on dev_calibrate only. Test set already inspected; not a fresh confirmatory evaluation.'
    s.save(RUN/'test_results.json',report)
    payload={k:old[k] for k in old.files};payload.update(prediction=p.cpu().numpy(),selected=selected.cpu().numpy(),**{a+'_risk':v.cpu().numpy() for a,v in scores.items()})
    np.savez_compressed(RUN/'predictions_final.npz',**payload);print(json.dumps(dict(accepted=True,policy=best,all_test=report['all_test'],hardcases=report['hardcases'],examples=report['examples']),indent=2))

if __name__=='__main__':main()

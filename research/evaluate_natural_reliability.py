import json,hashlib,shutil,itertools
from pathlib import Path
import numpy as np,torch
import spatial_rgb_common as s
from natural_reliability import RUN,metrics
from natural_corrector import NaturalCorrector,natural_batch
from natural_policy import apply_policy
from train_spatial_temporal import load
from spatial_rgb_model import SpatialHead
from evaluate_offline_kp import bootstrap

@torch.inference_mode()
def proposals(data,prob,ids,device):
    result={};base=[];confirmed=[]
    for arm in ['dit','regression']:
        ck=torch.load(RUN/arm/'best.pt',weights_only=False,map_location=device)
        model=NaturalCorrector(arm).to(device).eval();model.load_state_dict(ck['model']);parts=[];std=[]
        for start in range(0,len(ids),48):
            ix=ids[start:start+48];b=natural_batch(data,ix,prob)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model.predict(b)
            parts.append(out['xy'].float());std.append(out['std'].norm(dim=-1))
            if arm=='dit':base.append(b['linear'])
        result[arm]=torch.cat(parts);result[arm+'_std']=torch.cat(std)
    model=SpatialHead().to(device).eval();model.load_state_dict(torch.load(s.RUN/'sealed/rgb_probe.pt',weights_only=False,map_location=device)['model'])
    parts=[]
    for start in range(0,len(ids),96):
        ix=ids[start:start+96];f=data['feature_ids'][ix,8];x=data['bank'][f,0].float().transpose(1,2).reshape(-1,128,16,12)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model.decode(x,data['positions_bank'][f],data['roi'][f])
        parts.append(out['xy'].float())
    result['rgb_probe']=torch.cat(parts)
    return torch.cat(base),result

def measure(p,base,gt,m,selected=None):
    err=(p-gt).norm(dim=-1)*1408;be=(base-gt).norm(dim=-1)*1408;good=m&(be<=10);bad=m&(be>20)
    out=dict(points=int(m.sum()),mean_px=float(err[m].mean()),base_mean_px=float(be[m].mean()),
        originally_bad_points=int(bad.sum()),bad_mean_px=float(err[bad].mean()),base_bad_mean_px=float(be[bad].mean()),
        bad_recovered_to20=int((bad&(err<=20)).sum()),bad_improved_over5=int((bad&(be-err>5)).sum()),
        good_points=int(good.sum()),good_harmed_to_over20=int((good&(err>20)).sum()),
        harm_good_to_over20_rate=float((err[good]>20).float().mean()),pck20=float((err[m]<=20).float().mean()),p95_px=float(torch.quantile(err[m],.95)))
    if selected is not None:
        out.update(selected_points=int((selected&m).sum()),selection_rate=float(selected[m].float().mean()),
            selected_bad_precision=float((be[selected&m]>20).float().mean()) if (selected&m).any() else None)
    return out

def main():
    torch.set_num_threads(4);device='cuda:0'
    assert all((RUN/a/'done.json').exists() for a in ['dit','regression'])
    if (RUN/'test_results.json').exists():print('Already evaluated; no repeat selection');return
    rows,data=load(device);r=torch.load(RUN/'features.pt',weights_only=False,mmap=True);roles=np.array(r['roles'])
    probs=torch.load(RUN/'probabilities.pt',weights_only=False);prob=probs['joint'].to(device)
    ids=torch.tensor(np.where(roles=='dev_calibrate')[0],device=device);base,candidates=proposals(data,prob,ids,device)
    gt=data['gt'][ids];available=data['observed'][ids,8];confirmed=torch.zeros_like(available);roi=data['roi'][data['feature_ids'][ids,8]]
    valid=data['valid'][ids].clone();valid[:,5]=False;m=valid&available
    identity=measure(base,base,gt,m);configs=[]
    for arm,tau,blend,limit,agree in itertools.product(['dit','regression','rgb_probe'],[.2,.35,.5,.65,.8,.9],[.25,.5,1.],[.1,.25,.5],[None,.05]):
        policy=dict(arm=arm,threshold=tau,blend=blend,max_delta_roi=limit,agreement_roi=agree)
        p,selected=apply_policy(base,candidates,prob[ids],available,confirmed,roi,policy);score=measure(p,base,gt,m,selected)
        passed=(score['harm_good_to_over20_rate']<=.01 and score['mean_px']<=identity['mean_px']-.2
            and score['bad_mean_px']<=identity['bad_mean_px']*.95 and score['selection_rate']<=.30)
        configs.append(dict(policy=policy,metrics=score,passed=passed))
    approved=[q for q in configs if q['passed']]
    chosen=min(approved,key=lambda q:q['metrics']['mean_px']) if approved else dict(policy=dict(arm='regression',review_only=True),metrics=identity,passed=False)
    s.save(RUN/'policy.json',chosen)
    s.save(RUN/'calibration_search.json',dict(identity=identity,criteria='<=1% originally <=10px points become >20px; >=0.2px all-point improvement; >=5% reduction on >20px points; <=30% edited',configs=configs))
    # Freeze every fitted component and policy before any test scores are read.
    seal=RUN/'sealed';seal.mkdir(exist_ok=True)
    for arm in ['dit','regression']:shutil.copy2(RUN/arm/'best.pt',seal/f'{arm}.pt')
    for name in ['risk_joint.pt','risk_box.pt','risk_calibration.json','policy.json','protocol.json']:shutil.copy2(RUN/name,seal/name)
    torch.save(r['projection'],seal/'risk_projection.pt');shutil.copy2(s.RUN/'sealed/rgb_probe.pt',seal/'rgb_probe.pt')
    code=['natural_reliability.py','natural_corrector.py','natural_policy.py','train_natural_corrector.py','evaluate_natural_reliability.py',
        'spatial_rgb_common.py','spatial_rgb_model.py','spatial_temporal_model.py','offline_kp_model.py','pose_residual_dit.py']
    for name in code:shutil.copy2(Path(__file__).parent/name,seal/name)
    s.save(seal/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in seal.iterdir() if p.is_file() and p.name!='manifest.json'})
    ids=torch.tensor(np.where(roles=='test')[0],device=device);base,candidates=proposals(data,prob,ids,device)
    available=data['observed'][ids,8];confirmed=torch.zeros_like(available);roi=data['roi'][data['feature_ids'][ids,8]]
    gt=data['gt'][ids];valid=data['valid'][ids].clone();valid[:,5]=False;m=valid&available
    p,selected=apply_policy(base,candidates,prob[ids],available,confirmed,roi,chosen['policy'])
    be=(base-gt).norm(dim=-1)*1408;ee=(p-gt).norm(dim=-1)*1408
    clusters=np.array([rows[i]['sequence'] for i in ids.tolist()])
    report=dict(policy=chosen,all_test=measure(p,base,gt,m,selected),
        reliability={arm:metrics(probs[arm][ids.cpu()],be.cpu(),m.cpu()) for arm in ['box','joint']},
        raw_proposals={arm:measure(candidates[arm],base,gt,m) for arm in ['dit','regression','rgb_probe']},
        mean_change_ci95_px=bootstrap((ee-be).cpu().numpy(),m.cpu().numpy(),clusters),
        scope='Previously inspected test subjects; no artificial masking. Point reliability is error risk, not visibility. 47 hard cases excluded from all fitting/calibration.')
    subset=json.loads((s.RUN/'natural_finger_audit/candidates.json').read_text());examples=json.loads((s.RUN/'natural_finger_audit/reviewed_examples.json').read_text())
    lookup={wi:j for j,wi in enumerate(ids.tolist())};ix=torch.tensor([lookup[q['window_index']] for q in subset],device=device)
    report['hardcases']=measure(p[ix],base[ix],gt[ix],m[ix],selected[ix]);report['examples']=[]
    for q in examples:
        j=lookup[q['window_index']];one=torch.tensor([j],device=device)
        report['examples'].append(dict(id=q['id'],category=q['visual_review_category'],**measure(p[one],base[one],gt[one],m[one],selected[one]),
            mean_p_bad=float(prob[ids[j]][m[j]].mean())))
    np.savez_compressed(RUN/'predictions.npz',window_indices=ids.cpu().numpy(),base=base.cpu().numpy(),prediction=p.cpu().numpy(),
        gt=gt.cpu().numpy(),valid=valid.cpu().numpy(),available=available.cpu().numpy(),selected=selected.cpu().numpy(),p_bad=prob[ids].cpu().numpy(),
        **{k:v.cpu().numpy() for k,v in candidates.items()})
    s.save(RUN/'test_results.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()

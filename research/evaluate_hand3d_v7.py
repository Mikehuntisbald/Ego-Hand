import json,itertools,shutil,hashlib,time
from pathlib import Path
import numpy as np,torch
from hand3d_data_v7 import RUN,V4,load,batch,save
from hand3d_temporal_v7 import TemporalHand3D,WRIST
from train_hand3d_v7 import metrics,proposals
from evaluate_offline_kp import bootstrap

def apply_gate(candidate,b,policy):
    base=b['base'];root_delta=candidate[:,WRIST]-base[:,WRIST]
    pose_delta=(candidate-candidate[:,WRIST:WRIST+1])-(base-base[:,WRIST:WRIST+1])
    root=(b['risk_camera'][:,WRIST]>=policy['threshold'])&(root_delta.norm(dim=-1)<=policy['root_limit_m'])&~b['confirmed'][:,WRIST]
    root=root&(b['risk_camera'].amin(1)>=.2)&(candidate[:,:,2]>0).all(1)
    pose=(b['risk_relative']>=policy['threshold'])&(b['risk_camera']>=.2)&(pose_delta.norm(dim=-1)<=policy['pose_limit_m'])&~b['confirmed']
    pose[:,WRIST]=False;pose=pose&(candidate[:,:,2]>0)
    if policy.get('review_only'):root.zero_();pose.zero_()
    out=base+policy['blend']*root_delta[:,None]*root[:,None,None]+policy['blend']*pose_delta*pose[...,None]
    out=torch.where(b['confirmed'][...,None],base,out)
    return out,dict(root=root,pose=pose)

def main():
    torch.set_num_threads(4);device='cuda:0';data=load(device);roles=np.array(data['roles']);prob=torch.load(RUN/'risk_probabilities.pt',weights_only=False)['joint'].to(device)
    cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=device);cb=batch(data,cal,prob);base=cb['base'];gt=data['gt'][cal];valid=data['valid'][cal];identity=metrics(base,base,gt,valid)
    policies={};cal_reports={};arms=['rgb_regression','rgb_dit','tracks_regression','tracks_dit']
    for arm in arms:
        assert (RUN/arm/'training_done.json').exists()
        cp=np.load(RUN/arm/'calibration.npz');assert np.array_equal(cp['indices'],cal.cpu().numpy());candidate=torch.from_numpy(cp['prediction']).to(device)
        configs=[]
        for tau,blend,root_limit,pose_limit in itertools.product([.35,.5,.65,.8,.9,.95],[.25,.5,1.],[.005,.01,.02,.04],[.01,.02,.04,.08]):
            policy=dict(threshold=tau,blend=blend,root_limit_m=root_limit,pose_limit_m=pose_limit)
            corrected,_=apply_gate(candidate,cb,policy);r=metrics(corrected,base,gt,valid)
            passed=(r['camera_good_harm_rate']<=.01 and r['relative_good_harm_rate']<=.01 and r['camera_mm']<=identity['camera_mm']-.2
                and r['relative_mm']<=identity['relative_mm']+.1 and r['camera_bad_mean_mm']<=identity['camera_bad_mean_mm']*.95)
            configs.append(dict(policy=policy,metrics=r,passed=passed))
        approved=[r for r in configs if r['passed']]
        chosen=min(approved,key=lambda r:r['metrics']['camera_mm']+.5*r['metrics']['relative_mm']) if approved else dict(policy=dict(threshold=1.1,blend=0.,root_limit_m=0.,pose_limit_m=0.,review_only=True),metrics=identity,passed=False)
        policies[arm]=chosen;cal_reports[arm]=dict(raw=metrics(candidate,base,gt,valid),chosen=chosen)
        save(RUN/arm/'gate_search.json',dict(identity=identity,configs=configs,chosen=chosen))
        print(json.dumps(dict(stage='calibrate',arm=arm,approved=bool(approved),metrics=chosen['metrics'])),flush=True)
    save(RUN/'policies.json',policies)
    approved_rgb=[(k,v) for k,v in policies.items() if k.startswith('rgb_') and v['passed']]
    chosen=min(approved_rgb,key=lambda q:q[1]['metrics']['camera_mm']+.5*q[1]['metrics']['relative_mm'])[0] if approved_rgb else 'rgb_regression'
    save(RUN/'selection.json',dict(arm=chosen,approved=policies[chosen]['passed'],selected_on='dev_calibrate only; selection locked before reading test metrics',calibration=cal_reports))
    seal=RUN/'sealed';seal.mkdir(exist_ok=True)
    for arm in arms:shutil.copy2(RUN/arm/'best.pt',seal/f'{arm}.pt')
    for name in ['risk_all.pt','risk_calibration.json','protocol.json','policies.json','selection.json']:shutil.copy2(RUN/name,seal/name)
    for name in ['hand3d_temporal_v7.py','hand3d_data_v7.py','hand3d_risk_v7.py','train_hand3d_v7.py','evaluate_hand3d_v7.py']:shutil.copy2(Path(__file__).parent/name,seal/name)
    save(seal/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in seal.iterdir() if p.is_file() and p.name!='manifest.json'})
    # The held-out labels below never affect checkpoints, thresholds or arm selection.
    test=torch.tensor(np.where(roles=='test')[0],device=device);b=batch(data,test,prob);gt=data['gt'][test];valid=data['valid'][test];base=b['base']
    rows=data['rows'];clusters=np.array([rows[i]['sequence'] for i in test.tolist()]);subjects=np.array([rows[i]['subject'] for i in test.tolist()]);canonical=valid.clone();canonical[:,WRIST]=False
    hard=json.loads((V4/'delivery/hardcase_review_queue.json').read_text());lookup={v:i for i,v in enumerate(test.tolist())};hi=torch.tensor([lookup[q['window_index']] for q in hard],device=device)
    records,_=s_records();center=data['feature_ids'][test,8].tolist();visibility=torch.tensor([records[i-1]['visibility_label'] if records[i-1]['visibility_label'] is not None else 1. for i in center],device=device)
    results={};preds={}
    for arm in arms:
        ck=torch.load(seal/f'{arm}.pt',weights_only=False,map_location=device);model=TemporalHand3D(ck['kind'],ck['use_rgb'],ck['width'],ck['depth']).to(device);model.load_state_dict(ck['model'])
        pred,std=proposals(model,data,prob,test);final,gates=apply_gate(pred,b,policies[arm]['policy']);preds[arm]=final
        camera_change=((final-gt).norm(dim=-1)-(base-gt).norm(dim=-1))*1000
        relative_change=(((final-final[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])).norm(dim=-1)-((base-base[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])).norm(dim=-1))*1000
        result=dict(selected_step=ck['step'],policy=policies[arm],raw=metrics(pred,base,gt,valid),gated=metrics(final,base,gt,valid),hardcases=metrics(final[hi],base[hi],gt[hi],valid[hi]),
            camera_change_ci95_mm=bootstrap(camera_change.cpu().numpy(),canonical.cpu().numpy(),clusters),relative_change_ci95_mm=bootstrap(relative_change.cpu().numpy(),canonical.cpu().numpy(),clusters),per_subject={},groups={},cases=[],
            corrected_roots=int(gates['root'].sum()),corrected_relative_joints=int((gates['pose']&canonical).sum()))
        for subject in sorted(set(subjects)):
            idx=torch.tensor(np.where(subjects==subject)[0],device=device);result['per_subject'][subject]=metrics(final[idx],base[idx],gt[idx],valid[idx])
        for name,selector in [('whole_hand_visibility_below_0.5',visibility<.5),('whole_hand_visibility_above_0.8',visibility>.8)]:
            idx=torch.where(selector)[0]
            if len(idx):result['groups'][name]=metrics(final[idx],base[idx],gt[idx],valid[idx])
        for q in hard:
            j=lookup[q['window_index']];ix=torch.tensor([j],device=device);result['cases'].append(dict(id=q['id'],**metrics(final[ix],base[ix],gt[ix],valid[ix])))
        save(RUN/arm/'test_results.json',result);results[arm]=result
        np.savez_compressed(RUN/arm/'test_predictions.npz',indices=test.cpu().numpy(),base=base.cpu().numpy(),gt=gt.cpu().numpy(),raw=pred.cpu().numpy(),prediction=final.cpu().numpy(),std=std.cpu().numpy(),valid=valid.cpu().numpy())
        del model;torch.cuda.empty_cache();print(json.dumps(dict(stage='test',arm=arm,gated=result['gated'])),flush=True)
    comparisons={}
    for kind in ['regression','dit']:
        rgb=preds['rgb_'+kind];tracks=preds['tracks_'+kind];delta=((rgb-gt).norm(dim=-1)-(tracks-gt).norm(dim=-1))*1000
        comparisons['rgb_vs_tracks_'+kind]=dict(camera_delta_mm=float(delta[canonical].mean()),camera_ci95=bootstrap(delta.cpu().numpy(),canonical.cpu().numpy(),clusters))
    delta=((preds['rgb_dit']-gt).norm(dim=-1)-(preds['rgb_regression']-gt).norm(dim=-1))*1000
    comparisons['dit_vs_regression']=dict(camera_delta_mm=float(delta[canonical].mean()),camera_ci95=bootstrap(delta.cpu().numpy(),canonical.cpu().numpy(),clusters))
    report=dict(selection=json.loads((RUN/'selection.json').read_text()),baseline=metrics(base,base,gt,valid),methods=results,comparisons=comparisons,
        scope='Correct 3D-output task, existing previously inspected P0010/P0015 test sequences; not fresh new-subject proof. Hardcase selection was based on earlier 2D failures, not per-finger visibility truth.')
    save(RUN/'test_results.json',report);save(RUN/'evaluation_done.json',dict(complete=True,time=time.time()));print(json.dumps(comparisons,indent=2))

def s_records():
    import spatial_rgb_common as s
    return s.records_and_index()

if __name__=='__main__':main()

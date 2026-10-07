"""Final 3D policy with a geometric correct-point safety bound.
At most 5mm root + 5mm relative update implies <=10mm camera displacement.
Thus an originally <=10mm point cannot cross 20mm under the declared metrics.
"""
import json,itertools,shutil,hashlib
from pathlib import Path
import numpy as np,torch
from hand3d_data_v7 import RUN as BASE,load,batch,save
from hand3d_temporal_v7 import WRIST
from hand3d_protected_v7 import ProtectedHand3D
from train_hand3d_v7 import metrics,proposals
from evaluate_offline_kp import bootstrap
PARENT=BASE.parent/'offline_hand3d_v7_protected';RUN=BASE.parent/'offline_hand3d_v7_feasible';RUN.mkdir(exist_ok=True)

from hand3d_feasible_gate_v7 import apply_feasible as apply_bounded

def main():
    torch.set_num_threads(4);device='cuda:0';data=load(device);roles=np.array(data['roles']);prob=torch.load(BASE/'risk_probabilities.pt',weights_only=False)['joint'].to(device)
    cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=device);cb=batch(data,cal,prob);base=cb['base'];gt=data['gt'][cal];valid=data['valid'][cal];identity=metrics(base,base,gt,valid)
    arms=['rgb_regression','rgb_dit','tracks_regression','tracks_dit'];policies={};calibration={}
    protocol=json.loads((PARENT/'protocol.json').read_text());protocol.update(parent_stage=str(PARENT),final_policy='Project full XYZ update onto intersecting camera and wrist-relative displacement balls, both <=9.9mm; original protection/recovery gates unchanged',scope='This bound permits modest correction, not full recovery of arbitrarily large errors')
    save(RUN/'protocol.json',protocol)
    for arm in arms:
        cp=np.load(PARENT/arm/'calibration.npz');assert np.array_equal(cp['indices'],cal.cpu().numpy());candidate=torch.from_numpy(cp['prediction']).to(device);configs=[]
        for tau,blend,root_limit,pose_limit in itertools.product([.35,.5,.65,.8,.9,.95],[.25,.5,1.],[.005,.0075,.0099],[.005,.0075,.0099]):
            policy=dict(threshold=tau,blend=blend,root_limit_m=root_limit,pose_limit_m=pose_limit,bounded=True)
            corrected,_=apply_bounded(candidate,cb,policy);r=metrics(corrected,base,gt,valid)
            passed=(r['camera_good_harm_rate']<=.01 and r['relative_good_harm_rate']<=.01 and r['camera_mm']<=identity['camera_mm']-.2
                and r['relative_mm']<=identity['relative_mm']+.1 and r['camera_bad_mean_mm']<=identity['camera_bad_mean_mm']*.95)
            configs.append(dict(policy=policy,metrics=r,passed=passed))
        approved=[q for q in configs if q['passed']]
        chosen=min(approved,key=lambda q:q['metrics']['camera_mm']+.5*q['metrics']['relative_mm']) if approved else dict(policy=dict(threshold=1.1,blend=0.,root_limit_m=.0099,pose_limit_m=.0099,bounded=True,review_only=True),metrics=identity,passed=False)
        policies[arm]=chosen;calibration[arm]=dict(chosen=chosen,raw=metrics(candidate,base,gt,valid));save(RUN/f'{arm}_calibration.json',dict(configs=configs,chosen=chosen))
        print(json.dumps(dict(stage='calibrate',arm=arm,passed=chosen['passed'],metrics=chosen['metrics'])),flush=True)
    approved=[(k,v) for k,v in policies.items() if k.startswith('rgb_') and v['passed']]
    selected=min(approved,key=lambda q:q[1]['metrics']['camera_mm']+.5*q[1]['metrics']['relative_mm'])[0] if approved else 'rgb_regression'
    selection=dict(arm=selected,approved=bool(approved),selected_on='dev_calibrate only before test evaluation',calibration=calibration)
    save(RUN/'selection.json',selection);save(RUN/'policies.json',policies)
    seal=RUN/'sealed';seal.mkdir(exist_ok=True)
    for arm in arms:shutil.copy2(PARENT/arm/'best.pt',seal/f'{arm}.pt')
    for name in ['risk_all.pt','risk_calibration.json']:shutil.copy2(PARENT/name,seal/name)
    for name in ['protocol.json','policies.json','selection.json']:shutil.copy2(RUN/name,seal/name)
    for name in ['hand3d_temporal_v7.py','hand3d_protected_v7.py','hand3d_data_v7.py','hand3d_risk_v7.py','train_hand3d_v7.py','train_hand3d_protected_v7.py','evaluate_hand3d_feasible_v7.py','hand3d_feasible_gate_v7.py']:shutil.copy2(Path(__file__).parent/name,seal/name)
    save(seal/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in seal.iterdir() if p.is_file() and p.name!='manifest.json'})
    test=torch.tensor(np.where(roles=='test')[0],device=device);b=batch(data,test,prob);base=b['base'];gt=data['gt'][test];valid=data['valid'][test];canonical=valid.clone();canonical[:,WRIST]=False
    clusters=np.array([data['rows'][i]['sequence'] for i in test.tolist()]);subjects=np.array([data['rows'][i]['subject'] for i in test.tolist()]);results={};predictions={}
    hard=json.loads((BASE.parent/'natural_reliability_v4/delivery/hardcase_review_queue.json').read_text());lookup={v:i for i,v in enumerate(test.tolist())};hi=torch.tensor([lookup[q['window_index']] for q in hard],device=device)
    for arm in arms:
        ck=torch.load(seal/f'{arm}.pt',weights_only=False,map_location=device);model=ProtectedHand3D(ck['kind'],ck['use_rgb'],ck['width'],ck['depth']).to(device);model.load_state_dict(ck['model'])
        candidate,std=proposals(model,data,prob,test);out,gates=apply_bounded(candidate,b,policies[arm]['policy']);predictions[arm]=out
        assert float((out-base).norm(dim=-1).max())<=.010001
        cam_change=((out-gt).norm(dim=-1)-(base-gt).norm(dim=-1))*1000
        rel_change=(((out-out[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])).norm(dim=-1)-((base-base[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])).norm(dim=-1))*1000
        result=dict(selected_step=ck['step'],raw=metrics(candidate,base,gt,valid),gated=metrics(out,base,gt,valid),policy=policies[arm],hardcases=metrics(out[hi],base[hi],gt[hi],valid[hi]),
            camera_change_ci95_mm=bootstrap(cam_change.cpu().numpy(),canonical.cpu().numpy(),clusters),relative_change_ci95_mm=bootstrap(rel_change.cpu().numpy(),canonical.cpu().numpy(),clusters),per_subject={},cases=[],
            corrected_roots=int(gates['root'].sum()),corrected_pose_joints=int((gates['pose']&canonical).sum()),max_camera_displacement_mm=float((out-base).norm(dim=-1).max()*1000))
        for subject in sorted(set(subjects)):
            ix=torch.tensor(np.where(subjects==subject)[0],device=device);result['per_subject'][subject]=metrics(out[ix],base[ix],gt[ix],valid[ix])
        for q in hard:
            j=lookup[q['window_index']];ix=torch.tensor([j],device=device);result['cases'].append(dict(id=q['id'],**metrics(out[ix],base[ix],gt[ix],valid[ix])))
        save(RUN/f'{arm}_test_results.json',result);results[arm]=result
        np.savez_compressed(RUN/f'{arm}_predictions.npz',indices=test.cpu().numpy(),base=base.cpu().numpy(),gt=gt.cpu().numpy(),raw=candidate.cpu().numpy(),prediction=out.cpu().numpy(),std=std.cpu().numpy(),valid=valid.cpu().numpy())
        print(json.dumps(dict(stage='test',arm=arm,gated=result['gated'])),flush=True);del model
    comparison={}
    for kind in ['regression','dit']:
        x=predictions['rgb_'+kind];y=predictions['tracks_'+kind];delta=((x-gt).norm(dim=-1)-(y-gt).norm(dim=-1))*1000;comparison['rgb_vs_tracks_'+kind]=dict(mean_delta_mm=float(delta[canonical].mean()),ci95=bootstrap(delta.cpu().numpy(),canonical.cpu().numpy(),clusters))
    delta=((predictions['rgb_dit']-gt).norm(dim=-1)-(predictions['rgb_regression']-gt).norm(dim=-1))*1000;comparison['dit_vs_regression']=dict(mean_delta_mm=float(delta[canonical].mean()),ci95=bootstrap(delta.cpu().numpy(),canonical.cpu().numpy(),clusters))
    save(RUN/'test_results.json',dict(selection=selection,baseline=metrics(base,base,gt,valid),methods=results,comparisons=comparison,scope='Existing inspected sequences and subjects; no independent new-subject claim; 3D metrics in mm. Bound guarantees only the declared <=10mm-to->20mm protection, not that every correction is beneficial.'))
    save(RUN/'evaluation_done.json',dict(complete=True,approved=selection['approved']))

if __name__=='__main__':main()


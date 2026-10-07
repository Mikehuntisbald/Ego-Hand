"""Separate fixed proposal quality from operating-point limitations.

Choose exactly one policy using dev_select. Dev_calibrate may reject it but
cannot choose a different one. No fresh/retained failure metrics are read.
"""
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,save,metrics,score
from hand3d_data_v7 import batch
from density_model_v13 import DensityTrajectoryHand3D
from adaptive_projection_v14 import apply
from calibrate_hand3d_v8 import paired_ci

def hard_mask(base,gt,valid):
    mask=valid.clone();mask[:,5]=False
    error=((base-base[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000
    return (error*mask).sum(-1)/mask.sum(-1).clamp_min(1)>40

def summarize(pred,base,gt,valid,hard):
    m=metrics(pred,base,gt,valid);_,ok=score(m)
    return dict(final=m,hard=metrics(pred[hard],base[hard],gt[hard],valid[hard]),qualified=ok)

def improves(candidate,prior):
    return (candidate['qualified'] and
        candidate['final']['camera_mm']<=prior['final']['camera_mm']+.1 and
        candidate['final']['relative_mm']<=prior['final']['relative_mm']+.1 and
        candidate['hard']['relative_mm']<prior['hard']['relative_mm'] and
        candidate['hard']['relative_bad_recovered20']>prior['hard']['relative_bad_recovered20'])

@torch.inference_mode()
def main():
    torch.set_num_threads(4);device='cuda:0';root=V7.parent/'projection_operating_v24';root.mkdir(exist_ok=True)
    assert not (root/'development_results.json').exists()
    source=V7.parent/'side_data_v16/consensus'
    raw_data=torch.load(source/'dense_data.pt',weights_only=False,mmap=True)
    data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw_data.items()}
    risk=torch.load(source/'risk_dense/risk_probabilities.pt',weights_only=False)['joint'].to(device)
    ids=torch.tensor(np.where(np.asarray(data['roles'])=='dev_select')[0],device=device)
    checkpoint_path=V7.parent/'side_native_v16/consensus/uniform_adaptive/best.pt'
    checkpoint=torch.load(checkpoint_path,weights_only=False,map_location=device)
    model=DensityTrajectoryHand3D('dit',True).to(device).eval();model.load_state_dict(checkpoint['model'])
    bank=torch.load(source/'native_bank.pt',weights_only=False,mmap=True).to(device)
    proposals=[]
    for start in range(0,len(ids),8):
        ix=ids[start:start+8];b=batch(data,ix,risk);b['rgb_native']=bank[data['feature_ids'][ix]]
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model.predict_rollout(b,seed=202610114+start)
        proposals.append(out['xyz_camera_m'].float())
    proposal=torch.cat(proposals);b=batch(data,ids,risk)
    base,gt,valid=data['original_base_for_evaluation'][ids],data['gt'][ids],data['valid'][ids]
    hard=hard_mask(base,gt,valid)
    original=json.loads((V7.parent/'side_native_v16/fifth_seal.json').read_text())['policy']
    prior=summarize(apply(proposal,b,original),base,gt,valid,hard)
    rows=[];winner=None
    for strength,cap,camera,relative in itertools.product([.5,.75,1.],[.05,.075],[.9,.95,.975],[.65,.75,.85]):
        policy=dict(original,strength=strength,large_cap_m=cap,camera_risk_threshold=camera,relative_risk_threshold=relative)
        result=summarize(apply(proposal,b,policy),base,gt,valid,hard)
        accepted=improves(result,prior)
        row=dict(policy=policy,**result,eligible=accepted)
        rows.append(row)
        if accepted:
            key=(-result['hard']['relative_bad_recovered20'],result['hard']['relative_mm'],result['final']['relative_mm'],result['final']['camera_mm'])
            if winner is None or key<winner[0]:winner=(key,row)
    selection=dict(complete=True,original=prior,grid=rows,selected_policy=winner[1]['policy'] if winner else None,
        selected_result=winner[1] if winner else None,criteria='Bothharmrates<=1%, originalgain>=5%/camera preservation; <=0.1mm overall regression vs v16; more hard recovery and lower hardmean',
        scope='Dev_select only, fixedv16model. SameRGB/XYZ/risks/proposal/seed. Trusted9.5mm cap unchanged. This tests operatingpoint, not learnednewinformation. No unseenclips or oldfailures read.')
    save(root/'selection.json',selection)
    torch.save(dict(indices=ids.cpu(),proposal=proposal.cpu()),root/'selection_proposal.pt')
    if winner is None:
        save(root/'development_results.json',dict(complete=True,approved=False,reason='No feasible dev_select policy with extra difficult recovery',selection=selection))
        print('No eligible policy; calibration not opened',flush=True);return
    policy=winner[1]['policy']
    # Freeze selection, model/risk/input/code lineage before calibration.
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    save(root/'selection_seal.json',dict(policy=policy,model_sha256=sha(checkpoint_path),
        risk_sha256=sha(source/'risk_dense/risk_all.pt'),temperature_sha256=sha(source/'risk_dense/calibration.json'),
        data_sha256=sha(source/'dense_data.pt'),code_sha256=sha(__file__),
        current_projection_sha256=sha(Path(__file__).resolve().parent/'adaptive_projection_v14.py'),selection_sha256=sha(root/'selection.json'),
        created_unix=time.time(),scope='One dev_select winner; calibration may reject only, no alternatives selected'))
    cal=torch.load(V7.parent/'side_native_v16/consensus/uniform_adaptive/calibration.pt',weights_only=False)
    ci=cal['indices'].to(device)
    assert all(data['roles'][int(i)]=='dev_calibrate' for i in ci)
    raw=cal['proposal'].to(device);cb=batch(data,ci,risk);base=data['original_base_for_evaluation'][ci];gt=data['gt'][ci];valid=data['valid'][ci]
    hard=hard_mask(base,gt,valid);old=apply(raw,cb,original);pred=apply(raw,cb,policy)
    reference=summarize(old,base,gt,valid,hard);candidate=summarize(pred,base,gt,valid,hard)
    records=[data['rows'][int(i)] for i in ci]
    paired=paired_ci(pred,base,gt,valid,records)
    approved=improves(candidate,reference) and paired['relative']['ci95_delta_mm'][1]<0
    report=dict(complete=True,approved=approved,policy=policy,candidate=candidate,baseline_v16=reference,
        paired_vs_original=paired,paired_vs_v16=paired_ci(pred,old,gt,valid,records),
        scope='Dev_calibrate operating check only. Policy frozen from dev_select; no reselection after results. Fixed model, protection retained. Sixth unusedclips and engineering checks required before deployment. No broad newsubject/fullOOFclaim.')
    save(root/'development_results.json',report)
    torch.save(dict(indices=ci.cpu(),candidate=pred.cpu(),baseline=old.cpu()),root/'development_predictions.pt')
    print(json.dumps(dict(approved=approved,policy=policy,candidate=candidate,prior=reference),indent=2),flush=True)

if __name__=='__main__':main()

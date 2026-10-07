import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,save,metrics,score
from hand3d_data_v7 import batch,risk_features
from hand3d_risk_v7 import Risk3D
from density_model_v13 import DensityTrajectoryHand3D,POLICY
from train_aligned_density_v13 import make
from native_projection_policy_v11 import apply as conservative
from adaptive_projection_v14 import apply
from calibrate_hand3d_v8 import paired_ci
from hand3d_temporal_v7 import EDGES
DATA=V7.parent/'aligned_density_v13';FRESH=V7.parent/'fourth_dense_v14';RUN=V7.parent/'adaptive_projection_v14/dit_dense';MODEL=V7.parent/'native_density_v13/dit_dense/best.pt';CODE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def seal():
    assert not (RUN/'fourth_results.json').exists();assert json.loads((FRESH/'prepared.json').read_text())['complete'];assert json.loads((FRESH/'unseen_check.json').read_text())['passed'];selection=json.loads((RUN/'selection.json').read_text());assert selection['approved']
    names=['density_model_v13.py','density_prior_loss_v13.py','hand3d_native_v10.py','hand3d_trajectory_v9.py','hand3d_temporal_v7.py','pose_residual_dit.py','spatial_rgb_model.py','hand3d_data_v7.py','hand3d_rollout_v8.py','hand3d_risk_v7.py','adaptive_projection_v14.py','evaluate_fourth_adaptive_v14.py']
    obj=dict(created_unix=time.time(),selection_closed=True,model_sha256=sha(MODEL),risk_sha256=sha(DATA/'risk_dense/risk_all.pt'),temperature_sha256=sha(DATA/'risk_dense/calibration.json'),manifest_sha256=sha(FRESH/'fresh_manifest.json'),policy=selection['selected']['policy'],code_sha256={n:sha(CODE/n) for n in names},scope='Fourth12unused clips; source sequences/subjects previously encountered; no new-subject or full-pipelineOOF claim')
    save(RUN/'fourth_seal.json',obj);print(json.dumps(obj),flush=True)

@torch.no_grad()
def evaluate():
    torch.set_num_threads(4);device='cuda:3';frozen=json.loads((RUN/'fourth_seal.json').read_text());assert sha(MODEL)==frozen['model_sha256'] and sha(DATA/'risk_dense/risk_all.pt')==frozen['risk_sha256'];assert sha(FRESH/'fresh_manifest.json')==frozen['manifest_sha256']
    for n,h in frozen['code_sha256'].items():assert sha(CODE/n)==h
    raw_data=torch.load(FRESH/'dense_data.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw_data.items()};bank=torch.load(FRESH/'fresh_dense.pt',weights_only=False,mmap=True).to(device);ids=torch.arange(len(data['roles']),device=device)
    rck=torch.load(DATA/'risk_dense/risk_all.pt',weights_only=False,map_location=device);risk=Risk3D(rck['dim']).to(device).eval();risk.load_state_dict(rck['model']);temperature=torch.tensor(json.loads((DATA/'risk_dense/calibration.json').read_text())['temperature'],device=device)
    prob=torch.cat([(risk(risk_features(batch(data,ids[start:start+64])))/temperature).sigmoid() for start in range(0,len(ids),64)])
    ck=torch.load(MODEL,weights_only=False,map_location=device);model=DensityTrajectoryHand3D('dit',True).to(device).eval();model.load_state_dict(ck['model']);outputs={}
    for start in range(0,len(ids),8):
        ix=ids[start:start+8];b=make(data,bank,ix,prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(b,seed=202610114+start)
        raw=p['xyz_camera_m'].float();pred=apply(raw,b,frozen['policy']);fallback=conservative(raw,b['base'],POLICY)
        for key,value in [('prediction',pred),('proposal',raw),('fallback',fallback),('std',p['std_m'].float())]:outputs.setdefault(key,[]).append(value)
    out={k:torch.cat(v) for k,v in outputs.items()};base=data['xyz_camera_bank'][data['feature_ids'][:,8]];gt=data['gt'];valid=data['valid'];rows=data['rows'];final=metrics(out['prediction'],base,gt,valid);fallback=metrics(out['fallback'],base,gt,valid);ci=paired_ci(out['prediction'],base,gt,valid,rows);_,ok=score(final);ok=ok and ci['relative']['ci95_delta_mm'][1]<0 and final['relative_bad_recovered20']>0;mask=valid.clone();mask[:,5]=False
    re=((base-base[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000;hard=(re*mask).sum(-1)/mask.sum(-1).clamp_min(1)>40;groups={}
    if hard.any():groups['baseline_relative_over40mm']=dict(windows=int(hard.sum()),baseline=metrics(base[hard],base[hard],gt[hard],valid[hard]),final=metrics(out['prediction'][hard],base[hard],gt[hard],valid[hard]),fallback=metrics(out['fallback'][hard],base[hard],gt[hard],valid[hard]),raw=metrics(out['proposal'][hard],base[hard],gt[hard],valid[hard]))
    bone={}
    for label,pred in [('baseline',base),('final',out['prediction'])]:
        values=[]
        for u,v in EDGES:
            keep=valid[:,u]&valid[:,v];values.append((((pred[:,u]-pred[:,v]).norm(dim=-1)-(gt[:,u]-gt[:,v]).norm(dim=-1)).abs()*1000)[keep])
        z=torch.cat(values);bone[label]=dict(mean_length_error_mm=float(z.mean()),p90_mm=float(z.quantile(.9)))
    report=dict(complete=True,windows=len(ids),points=int(mask.sum()),model_step=ck['step'],policy=frozen['policy'],baseline=metrics(base,base,gt,valid),final=final,fallback=fallback,raw=metrics(out['proposal'],base,gt,valid),paired_ci=ci,adaptive_minus_fallback_ci=paired_ci(out['prediction'],out['fallback'],gt,valid,rows),groups=groups,bones=bone,passed=ok,scope=frozen['scope'],protection='Large radii empirical; fully conservative radii give geometry guarantee. Point-error risks are not visibility truth or calibrated post-correction probability.')
    save(RUN/'fourth_results.json',report);torch.save(dict(**{k:v.cpu() for k,v in out.items()},base=base.cpu(),gt=gt.cpu(),valid=valid.cpu(),probability=prob.cpu(),rows=rows),RUN/'fourth_predictions.pt');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['seal','evaluate']);a=p.parse_args();seal() if a.stage=='seal' else evaluate()

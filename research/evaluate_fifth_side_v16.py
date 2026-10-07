"""Seal every model/policy/input before reading fifth independent results."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from hand3d_v8_common import V7,save,metrics,score
from hand3d_data_v7 import batch,risk_features
from hand3d_risk_v7 import Risk3D
from density_model_v13 import DensityTrajectoryHand3D
from adaptive_projection_v14 import apply
from calibrate_hand3d_v8 import paired_ci
from audit_side_consensus_v16 import POLICY as SIDE_POLICY
RUN=V7.parent/'side_native_v16';FRESH=V7.parent/'fifth_dense_v16';DATA=V7.parent/'side_data_v16';CODE=Path(__file__).resolve().parent
MODELS={v:RUN/v/'uniform_adaptive/best.pt' for v in ['control','consensus']};MODELS['v14']=V7.parent/'native_density_v13/dit_dense/best.pt'
RISKS={v:DATA/v/'risk_dense' for v in ['control','consensus']};RISKS['v14']=V7.parent/'aligned_density_v13/risk_dense'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def seal():
    assert not (RUN/'fifth_results.json').exists();assert json.loads((RUN/'development_results.json').read_text())['approved'];assert json.loads((FRESH/'side_ready.json').read_text())['complete'];assert json.loads((FRESH/'unseen_check.json').read_text())['passed']
    policy=json.loads((RUN/'development_results.json').read_text())['policy'];names=['density_model_v13.py','hand3d_native_v10.py','hand3d_trajectory_v9.py','hand3d_temporal_v7.py','pose_residual_dit.py','spatial_rgb_model.py','hand3d_data_v7.py','hand3d_risk_v7.py','adaptive_projection_v14.py','audit_side_consensus_v16.py','prepare_fifth_side_v16.py','evaluate_fifth_side_v16.py']
    obj=dict(created_unix=time.time(),models={v:sha(p) for v,p in MODELS.items()},risk_sha256={v:sha(p/'risk_all.pt') for v,p in RISKS.items()},temperature_sha256={v:sha(p/'calibration.json') for v,p in RISKS.items()},data_sha256={v:sha(FRESH/('consensus_data.pt' if v=='consensus' else 'dense_data.pt')) for v in ['control','consensus']},native_bank_sha256=sha(FRESH/'fresh_dense.pt'),manifest_sha256=sha(FRESH/'fresh_manifest.json'),code_sha256={n:sha(CODE/n) for n in names},policy=policy,side_policy=SIDE_POLICY,scope='Fifth12unused clips; previously encountered subjects/source sequences. Frozen full-pipeline results compared against identical originalWiLoR20XYZ; no fullOOF/newsubjectclaim.')
    save(RUN/'fifth_seal.json',obj);print(json.dumps(dict(sealed=True,models=len(MODELS))),flush=True)

@torch.inference_mode()
def evaluate():
    torch.set_num_threads(4);device='cuda:3';frozen=json.loads((RUN/'fifth_seal.json').read_text())
    for name,h in frozen['code_sha256'].items():assert sha(CODE/name)==h,name
    assert sha(FRESH/'fresh_manifest.json')==frozen['manifest_sha256'] and sha(FRESH/'fresh_dense.pt')==frozen['native_bank_sha256']
    bank=torch.load(FRESH/'fresh_dense.pt',weights_only=False,mmap=True).to(device);results={};predictions={};base=gt=valid=rows=None
    for variant in ['v14','control','consensus']:
        assert sha(MODELS[variant])==frozen['models'][variant] and sha(RISKS[variant]/'risk_all.pt')==frozen['risk_sha256'][variant] and sha(RISKS[variant]/'calibration.json')==frozen['temperature_sha256'][variant]
        path=FRESH/('consensus_data.pt' if variant=='consensus' else 'dense_data.pt');assert sha(path)==frozen['data_sha256']['consensus' if variant=='consensus' else 'control']
        raw_data=torch.load(path,weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw_data.items()};ids=torch.arange(len(data['roles']),device=device)
        if base is None:
            base=data['xyz_camera_bank'][data['feature_ids'][:,8]];gt=data['gt'];valid=data['valid'];rows=data['rows']
        else:assert torch.equal(gt,data['gt']) and torch.equal(valid,data['valid']) and rows==data['rows']
        ck=torch.load(RISKS[variant]/'risk_all.pt',weights_only=False,map_location=device);risk=Risk3D(ck['dim']).to(device).eval();risk.load_state_dict(ck['model']);temp=torch.tensor(json.loads((RISKS[variant]/'calibration.json').read_text())['temperature'],device=device)
        prob=torch.cat([(risk(risk_features(batch(data,ids[start:start+64])))/temp).sigmoid() for start in range(0,len(ids),64)])
        ck=torch.load(MODELS[variant],weights_only=False,map_location=device);model=DensityTrajectoryHand3D('dit',True).to(device).eval();model.load_state_dict(ck['model']);outs={}
        for start in range(0,len(ids),8):
            b=batch(data,ids[start:start+8],prob);b['rgb_native']=bank[data['feature_ids'][start:start+8]]
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(b,seed=202610114+start)
            raw=p['xyz_camera_m'].float()
            for key,x in [('prediction',apply(raw,b,frozen['policy'])),('proposal',raw),('std',p['std_m'].float())]:outs.setdefault(key,[]).append(x)
        out={k:torch.cat(v) for k,v in outs.items()};predictions[variant]=out;mask=valid.clone();mask[:,5]=False
        re=(((base-base[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000*mask).sum(-1)/mask.sum(-1).clamp_min(1);hard=re>40
        m=metrics(out['prediction'],base,gt,valid);_,ok=score(m);ci=paired_ci(out['prediction'],base,gt,valid,rows)
        results[variant]=dict(step=ck['step'],final=m,raw=metrics(out['proposal'],base,gt,valid),coarse_input=metrics(data['xyz_camera_bank'][data['feature_ids'][:,8]],base,gt,valid),paired_vs_original=ci,hard_windows=int(hard.sum()),hard=metrics(out['prediction'][hard],base[hard],gt[hard],valid[hard]),passed=ok and ci['relative']['ci95_delta_mm'][1]<0 and metrics(out['prediction'][hard],base[hard],gt[hard],valid[hard])['relative_bad_recovered20']>0)
        del model,risk,data;torch.cuda.empty_cache()
    comparisons={v:paired_ci(predictions['consensus']['prediction'],predictions[v]['prediction'],gt,valid,rows) for v in ['v14','control']}
    report=dict(complete=True,windows=len(rows),baseline=metrics(base,base,gt,valid),variants=results,consensus_comparisons=comparisons,passed=results['consensus']['passed'],scope=frozen['scope'])
    save(RUN/'fifth_results.json',report);torch.save(dict(variants={v:{k:x.cpu() for k,x in p.items()} for v,p in predictions.items()},base=base.cpu(),gt=gt.cpu(),valid=valid.cpu(),rows=rows),RUN/'fifth_predictions.pt')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['seal','evaluate']);a=p.parse_args();seal() if a.stage=='seal' else evaluate()

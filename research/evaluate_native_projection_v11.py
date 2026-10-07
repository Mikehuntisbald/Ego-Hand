"""Sealed unused-clip validation of native3D DiT plus coupled safe projection."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,load,save,metrics,score,camera_bank
from hand3d_data_v7 import batch,risk_features
from hand3d_native_v10 import NativeTrajectoryHand3D
from hand3d_visual_v8 import dense_bank
from train_hand3d_native_v10 import RUN as DATA,make_batch
from train_native_rollout_v10 import RUN as PROPOSAL
from native_projection_policy_v11 import apply
from calibrate_hand3d_v8 import paired_ci
from hand3d_risk_v7 import Risk3D
from hand3d_temporal_v7 import EDGES
RUN=V7.parent/'native_projection_v11';CODE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def seal():
    assert not (RUN/'fresh_results.json').exists()
    selection=json.loads((RUN/'development_selection.json').read_text());assert selection['approved']
    assert json.loads((DATA/'fresh_ready.json').read_text())['complete']
    code=['hand3d_native_v10.py','hand3d_trajectory_v9.py','native_projection_policy_v11.py','evaluate_native_projection_v11.py','hand3d_data_v7.py','train_hand3d_native_v10.py']
    obj=dict(created_unix=time.time(),selection_closed=True,policy=selection['selected']['policy'],
             model_sha256={arm:sha(PROPOSAL/f'rgb_{arm}/best.pt') for arm in ['dit','regression']},
             manifest_sha256=sha(DATA/'fresh_manifest.json'),code_sha256={n:sha(CODE/n) for n in code},
             scope='Third12clips unused by development. Reused subjects/source sequences. Head OOF does not establish full pipeline OOF.',
             control='Native direct3D temporal regression with exactly the same projection; no fresh-dependent selection')
    save(RUN/'fresh_evaluation_seal.json',obj);print(json.dumps(obj),flush=True)

def bone_metrics(pred,base,gt,valid):
    def values(x):
        out=[]
        for u,v in EDGES:
            mask=valid[:,u]&valid[:,v]
            out.append((((x[:,u]-x[:,v]).norm(dim=-1)-(gt[:,u]-gt[:,v]).norm(dim=-1)).abs()*1000)[mask])
        return torch.cat(out)
    a=values(pred);b=values(base)
    return dict(baseline_mean_length_error_mm=float(b.mean()),final_mean_length_error_mm=float(a.mean()),baseline_p90_mm=float(b.quantile(.9)),final_p90_mm=float(a.quantile(.9)),scope='Bone-length accuracy only; not full anatomical validity')

@torch.no_grad()
def evaluate(fresh,arm,device):
    frozen=json.loads((RUN/'fresh_evaluation_seal.json').read_text());assert sha(PROPOSAL/f'rgb_{arm}/best.pt')==frozen['model_sha256'][arm]
    for name,digest in frozen['code_sha256'].items():assert sha(CODE/name)==digest
    if fresh:
        assert sha(DATA/'fresh_manifest.json')==frozen['manifest_sha256']
        raw=torch.load(DATA/'fresh_data.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw.items()}
        bank=torch.load(DATA/'fresh_dense.pt',weights_only=False,mmap=True).to(device);ids=torch.arange(len(data['roles']),device=device);records=json.loads((DATA/'fresh_rows.json').read_text());params=torch.zeros(len(records)+1,16,device=device)
        import spatial_rgb_common as s
        for i,r in enumerate(records,1):
            cam=s.common.from_json(r['camera']);params[i]=torch.tensor(list(cam.f)+list(cam.c)+list(cam.distort),device=device)
        ck=torch.load(V7/'risk_all.pt',weights_only=False,map_location=device);risk=Risk3D(ck['dim']).to(device).eval();risk.load_state_dict(ck['model']);temperature=torch.tensor(json.loads((V7/'risk_calibration.json').read_text())['temperature'],device=device)
        probability=torch.cat([(risk(risk_features(batch(data,ids[start:start+64])))/temperature).sigmoid() for start in range(0,len(ids),64)])
    else:
        data=load(device);bank=dense_bank(device,len(data['world']));ids=torch.tensor(np.where(np.asarray(data['roles'])=='test')[0],device=device);params=camera_bank().to(device);probability=torch.load(V7/'risk_probabilities.pt',weights_only=False)['joint'].to(device)
    ck=torch.load(PROPOSAL/f'rgb_{arm}/best.pt',weights_only=False,map_location=device);model=NativeTrajectoryHand3D(arm,True).to(device).eval();model.load_state_dict(ck['model']);out={}
    for start in range(0,len(ids),8):
        ix=ids[start:start+8];b=make_batch(data,bank,ix,probability)
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(b,seed=202610103+start)
        raw=p['xyz_camera_m'].float();pred=apply(raw,b['base'],frozen['policy'])
        for key,value in [('prediction',pred),('proposal',raw),('std',p['std_m'].float())]:out.setdefault(key,[]).append(value)
    out={k:torch.cat(v) for k,v in out.items()};pred=out['prediction'];raw=out['proposal'];base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids]
    final=metrics(pred,base,gt,valid);baseline=metrics(base,base,gt,valid);ci=paired_ci(pred,base,gt,valid,rows);_,ok=score(final);ok=ok and ci['relative']['ci95_delta_mm'][1]<0 and final['relative_bad_recovered20']>0 and final['relative_bad_mean_mm']<baseline['relative_bad_mean_mm']
    groups={};cases=[]
    if not fresh:
        queue=json.loads((V7.parent/'natural_reliability_v4/delivery/hardcase_review_queue.json').read_text());lookup={int(i):j for j,i in enumerate(ids)};classification=json.loads((V7.parent/'natural_reliability_v4/temporal_clue_audit/classification.json').read_text());a_ids={r['id'] for r in classification['cases'] if r['category']=='A'};b_ids={r['id'] for r in classification['cases'] if r['category']=='B'}
        for label,q in [('hard47',queue),('severe8',[r for r in queue if r['after_px']>40]),('nearby_clues',[r for r in queue if r['id'] in a_ids]),('clip_lacks_clues',[r for r in queue if r['id'] in b_ids])]:
            ix=[lookup[r['window_index']] for r in q]
            if ix:groups[label]=dict(baseline=metrics(base[ix],base[ix],gt[ix],valid[ix]),final=metrics(pred[ix],base[ix],gt[ix],valid[ix]),raw=metrics(raw[ix],base[ix],gt[ix],valid[ix]))
        for q in queue:
            j=lookup[q['window_index']];cases.append(dict(id=q['id'],window_index=q['window_index'],category='A' if q['id'] in a_ids else 'B' if q['id'] in b_ids else 'unclassified',baseline=metrics(base[j:j+1],base[j:j+1],gt[j:j+1],valid[j:j+1]),final=metrics(pred[j:j+1],base[j:j+1],gt[j:j+1],valid[j:j+1]),raw=metrics(raw[j:j+1],base[j:j+1],gt[j:j+1],valid[j:j+1])))
    else:
        mask=valid.clone();mask[:,5]=False;r=((base-base[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000;hard=(r*mask).sum(-1)/mask.sum(-1).clamp_min(1)>40
        if hard.any():groups['baseline_relative_over40mm']=dict(windows=int(hard.sum()),baseline=metrics(base[hard],base[hard],gt[hard],valid[hard]),final=metrics(pred[hard],base[hard],gt[hard],valid[hard]),raw=metrics(raw[hard],base[hard],gt[hard],valid[hard]))
    dc=(pred-base).norm(dim=-1).max();dr=((pred-pred[:,5:6])-(base-base[:,5:6])).norm(dim=-1).max();assert dc<=.009501 and dr<=.009501
    report=dict(arm=arm,split='fresh' if fresh else 'old_diagnostic',windows=len(ids),model_step=ck['step'],policy=frozen['policy'],baseline=baseline,final=final,raw=metrics(raw,base,gt,valid),paired_ci=ci,groups=groups,cases=cases,bones=bone_metrics(pred,base,gt,valid),passed=ok,max_camera_displacement_mm=float(dc*1000),max_relative_displacement_mm=float(dr*1000),scope=frozen['scope'] if fresh else 'Previously inspected diagnostic set; no independent evidence')
    name=('fresh' if fresh else 'old')+('' if arm=='dit' else '_regression');save(RUN/f'{name}_results.json',report);torch.save(dict(indices=ids.cpu(),**{k:v.cpu() for k,v in out.items()},base=base.cpu(),gt=gt.cpu(),valid=valid.cpu(),rows=rows),RUN/f'{name}_predictions.pt')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('stage',choices=['seal','fresh','old']);p.add_argument('--arm',choices=['dit','regression'],default='dit');p.add_argument('--device',default='cuda:0');a=p.parse_args();seal() if a.stage=='seal' else evaluate(a.stage=='fresh',a.arm,a.device)

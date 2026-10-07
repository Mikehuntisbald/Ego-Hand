import json,time,hashlib,argparse
import numpy as np,torch
from hand3d_v8_common import V7,load,batch,save,metrics,score,camera_bank
from hand3d_native_v10 import NativeTrajectoryHand3D
from hand3d_visual_v8 import dense_bank
from train_native_rollout_v10 import RUN as PROPOSAL
from train_hand3d_native_v10 import RUN as DATA,make_batch
from proposal_critic_v9 import ProposalCritic,features
from calibrate_proposal_critic_v9 import choose
from calibrate_hand3d_v8 import paired_ci
from hand3d_risk_v7 import Risk3D
from hand3d_data_v7 import risk_features
from bounded_policy_v8 import apply
RUN=V7.parent/'native_critic_v10'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def seal():
    selection=json.loads((RUN/'native_selection.json').read_text());assert selection['approved'];assert not (RUN/'fresh_results.json').exists();assert json.loads((DATA/'fresh_ready.json').read_text())['complete'];root=PROPOSAL/'rgb_dit';critic=RUN/'critic.pt';code=__import__('pathlib').Path(__file__).resolve().parent
    obj=dict(proposal_sha256=sha(root/'best.pt'),critic_sha256=sha(critic),manifest_sha256=sha(DATA/'fresh_manifest.json'),policy=selection['selected']['policy'],created_unix=time.time(),selection_closed=True,code_sha256={name:sha(code/name) for name in ['hand3d_native_v10.py','hand3d_trajectory_v9.py','proposal_critic_v9.py','evaluate_native_critic_v10.py']},scope='Third12unused clips; reused subjects/sequences, unknown upstream overlap; no full-pipeline OOF claim')
    save(RUN/'fresh_evaluation_seal.json',obj);print(json.dumps(obj),flush=True)

@torch.no_grad()
def evaluate(fresh,device):
    seal=json.loads((RUN/'fresh_evaluation_seal.json').read_text());assert sha(PROPOSAL/'rgb_dit/best.pt')==seal['proposal_sha256'] and sha(RUN/'critic.pt')==seal['critic_sha256'];rck=torch.load(V7/'risk_all.pt',weights_only=False,map_location=device);risk=Risk3D(rck['dim']).to(device).eval();risk.load_state_dict(rck['model']);temperature=torch.tensor(json.loads((V7/'risk_calibration.json').read_text())['temperature'],device=device)
    if fresh:
        assert sha(DATA/'fresh_manifest.json')==seal['manifest_sha256'];raw=torch.load(DATA/'fresh_data.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw.items()};bank=torch.load(DATA/'fresh_dense.pt',weights_only=False,mmap=True).to(device);ids=torch.arange(len(data['roles']),device=device);rows=json.loads((DATA/'fresh_rows.json').read_text());params=torch.zeros(len(rows)+1,16,device=device)
        import spatial_rgb_common as s
        for i,r in enumerate(rows,1):cam=s.common.from_json(r['camera']);params[i]=torch.tensor(list(cam.f)+list(cam.c)+list(cam.distort),device=device)
        prob=torch.cat([(risk(risk_features(batch(data,ids[start:start+64])))/temperature).sigmoid() for start in range(0,len(ids),64)])
    else:
        data=load(device);bank=dense_bank(device,len(data['world']));ids=torch.tensor(np.where(np.asarray(data['roles'])=='test')[0],device=device);params=camera_bank().to(device);prob=torch.load(V7/'risk_probabilities.pt',weights_only=False)['joint'].to(device)
    ck=torch.load(PROPOSAL/'rgb_dit/best.pt',weights_only=False,map_location=device);model=NativeTrajectoryHand3D('dit',True).to(device).eval();model.load_state_dict(ck['model']);cck=torch.load(RUN/'critic.pt',weights_only=False,map_location=device);critic=ProposalCritic(cck['dim']).to(device).eval();critic.load_state_dict(cck['model']);out={}
    for start in range(0,len(ids),8):
        ix=ids[start:start+8];b=make_batch(data,bank,ix,prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(b,seed=202610103+start)
        raw=p['xyz_camera_m'].float();std=p['std_m'].float();scores=torch.stack([critic(features(b,raw,std,params[data['feature_ids'][ix,8]],a)) for a in [.25,.5,.75,1.]])
        pred,accepted,alpha=choose(raw,b['base'],scores,seal['policy'])
        for key,value in [('prediction',pred),('proposal',raw),('std',std),('accepted',accepted),('alpha',alpha)]:out.setdefault(key,[]).append(value)
    out={k:torch.cat(v) for k,v in out.items()};pred=out['prediction'];raw=out['proposal'];accepted=out['accepted'];base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];fallback=apply(raw,base,dict(cap_m=.0095,strength=.5));m=metrics(pred,base,gt,valid);baseline=metrics(base,base,gt,valid);ci=paired_ci(pred,base,gt,valid,rows);_,ok=score(m);ok=ok and ci['relative']['ci95_delta_mm'][1]<0 and m['relative_bad_recovered20']>0 and m['relative_bad_mean_mm']<baseline['relative_bad_mean_mm'];groups={}
    if not fresh:
        queue=json.loads((V7.parent/'natural_reliability_v4/delivery/hardcase_review_queue.json').read_text());lookup={int(i):j for j,i in enumerate(ids)};classification=json.loads((V7.parent/'natural_reliability_v4/temporal_clue_audit/classification.json').read_text());a_ids={r['id'] for r in classification['cases'] if r['category']=='A'};b_ids={r['id'] for r in classification['cases'] if r['category']=='B'}
        for label,q in [('hard47',queue),('severe8',[r for r in queue if r['after_px']>40]),('nearby_clues',[r for r in queue if r['id'] in a_ids]),('clip_lacks_clues',[r for r in queue if r['id'] in b_ids])]:
            ix=[lookup[r['window_index']] for r in q]
            if ix:groups[label]=dict(final=metrics(pred[ix],base[ix],gt[ix],valid[ix]),raw=metrics(raw[ix],base[ix],gt[ix],valid[ix]),accepted_hands=int(accepted[ix].sum()))
    else:
        from hand3d_temporal_v7 import WRIST
        mask=valid.clone();mask[:,WRIST]=False;relative=((base-base[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])).norm(dim=-1)*1000;hard=(relative*mask).sum(-1)/mask.sum(-1).clamp_min(1)>40
        if hard.any():groups['baseline_relative_over40mm']=dict(final=metrics(pred[hard],base[hard],gt[hard],valid[hard]),raw=metrics(raw[hard],base[hard],gt[hard],valid[hard]),accepted_hands=int(accepted[hard].sum()))
    report=dict(split='fresh' if fresh else 'old_diagnostic',windows=len(ids),accepted_hands=int(accepted.sum()),proposal_step=ck['step'],critic_step=cck['step'],policy=seal['policy'],baseline=baseline,final=m,fallback=metrics(fallback,base,gt,valid),raw=metrics(raw,base,gt,valid),paired_ci=ci,groups=groups,passed=ok,scope='Third12new clips; reused subjects and source sequences' if fresh else 'Previously inspected diagnostic set')
    name='fresh' if fresh else 'old';save(RUN/f'{name}_results.json',report);torch.save(dict(indices=ids.cpu(),**{k:v.cpu() for k,v in out.items()},base=base.cpu(),gt=gt.cpu(),valid=valid.cpu(),rows=rows),RUN/f'{name}_predictions.pt');print(json.dumps(report),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('stage',choices=['seal','fresh','old']);p.add_argument('--device',default='cuda:3');a=p.parse_args();seal() if a.stage=='seal' else evaluate(a.stage=='fresh',a.device)

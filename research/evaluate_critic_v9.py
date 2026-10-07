import json,hashlib,time,argparse
import numpy as np,torch
from hand3d_trajectory_data_v9 import RUN as V9
from train_trajectory_rollout_v9 import RUN as ROLLOUT
from hand3d_trajectory_v9 import TrajectoryHand3D
from hand3d_v8_common import V7,load,batch,save,metrics,score,camera_bank
from hand3d_data_v7 import risk_features
from hand3d_risk_v7 import Risk3D
from proposal_critic_v9 import ProposalCritic,features
from calibrate_proposal_critic_v9 import choose,RUN
from calibrate_hand3d_v8 import paired_ci
from bounded_policy_v8 import apply

def seal():
    selected=json.loads((RUN/'critic_selection.json').read_text());assert selected['large_correction_approved_on_development'];assert not (RUN/'fresh_results.json').exists();assert json.loads((V9/'fresh_ready.json').read_text())['complete']
    obj=dict(proposal_sha256=hashlib.sha256((ROLLOUT/'rgb_dit/best.pt').read_bytes()).hexdigest(),critic_sha256=hashlib.sha256((RUN/'critic.pt').read_bytes()).hexdigest(),policy=selected['selected']['policy'],manifest_sha256=hashlib.sha256((V9/'fresh_manifest.json').read_bytes()).hexdigest(),created_unix=time.time(),selection_closed=True,scope='Another12unused clips; reused subjects/sequences; encoder not full-pipeline OOF')
    save(RUN/'fresh_evaluation_seal.json',obj);print(json.dumps(obj),flush=True)

@torch.no_grad()
def evaluate(fresh):
    device='cuda:2';seal=json.loads((RUN/'fresh_evaluation_seal.json').read_text());assert hashlib.sha256((ROLLOUT/'rgb_dit/best.pt').read_bytes()).hexdigest()==seal['proposal_sha256'];assert hashlib.sha256((RUN/'critic.pt').read_bytes()).hexdigest()==seal['critic_sha256']
    risk_ck=torch.load(V7/'risk_all.pt',weights_only=False,map_location=device);risk=Risk3D(risk_ck['dim']).to(device).eval();risk.load_state_dict(risk_ck['model']);temperature=torch.tensor(json.loads((V7/'risk_calibration.json').read_text())['temperature'],device=device)
    if fresh:
        raw=torch.load(V9/'fresh_data.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw.items()};ids=torch.arange(len(data['roles']),device=device)
        rows=json.loads((V9/'fresh_rows.json').read_text());params=torch.zeros(len(rows)+1,16,device=device)
        import spatial_rgb_common as s
        for j,r in enumerate(rows,1):cam=s.common.from_json(r['camera']);params[j]=torch.tensor(list(cam.f)+list(cam.c)+list(cam.distort),device=device)
        prob=torch.cat([(risk(risk_features(batch(data,ids[start:start+64])))/temperature).sigmoid() for start in range(0,len(ids),64)])
    else:
        data=load(device);ids=torch.tensor(np.where(np.asarray(data['roles'])=='test')[0],device=device);params=camera_bank().to(device);prob=torch.load(V7/'risk_probabilities.pt',weights_only=False)['joint'].to(device)
    ck=torch.load(ROLLOUT/'rgb_dit/best.pt',weights_only=False,map_location=device);model=TrajectoryHand3D('dit').to(device).eval();model.load_state_dict(ck['model']);cck=torch.load(RUN/'critic.pt',weights_only=False,map_location=device);critic=ProposalCritic(cck['dim']).to(device).eval();critic.load_state_dict(cck['model']);predictions=[];raws=[];accepts=[];stds=[]
    for start in range(0,len(ids),8):
        ix=ids[start:start+8];b=batch(data,ix,prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):output=model.predict_rollout(b,seed=202610093+start)
        proposal=output['xyz_camera_m'].float();std=output['std_m'].float();scores=torch.stack([critic(features(b,proposal,std,params[data['feature_ids'][ix,8]],a)) for a in [.25,.5,.75,1.]])
        pred,accepted,alpha=choose(proposal,b['base'],scores,seal['policy']);predictions.append(pred);raws.append(proposal);accepts.append(accepted);stds.append(std)
    pred=torch.cat(predictions);proposal=torch.cat(raws);accepted=torch.cat(accepts);base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];fallback=apply(proposal,base,dict(cap_m=.0095,strength=.5));m=metrics(pred,base,gt,valid);baseline=metrics(base,base,gt,valid);ci=paired_ci(pred,base,gt,valid,rows);_,ok=score(m);ok=ok and ci['relative']['ci95_delta_mm'][1]<0 and m['relative_bad_recovered20']>0 and m['relative_bad_mean_mm']<baseline['relative_bad_mean_mm'];groups={}
    if not fresh:
        queue=json.loads((V7.parent/'natural_reliability_v4/delivery/hardcase_review_queue.json').read_text());lookup={int(i):j for j,i in enumerate(ids)}
        for label,qs in [('hard47',queue),('severe8',[q for q in queue if q['after_px']>40])]:
            ix=[lookup[q['window_index']] for q in qs];groups[label]=dict(final=metrics(pred[ix],base[ix],gt[ix],valid[ix]),raw=metrics(proposal[ix],base[ix],gt[ix],valid[ix]),accepted_hands=int(accepted[ix].sum()))
    report=dict(split='fresh' if fresh else 'old_diagnostic',windows=len(ids),accepted_hands=int(accepted.sum()),policy=seal['policy'],proposal_step=ck['step'],critic_step=cck['step'],baseline=baseline,final=m,fallback=metrics(fallback,base,gt,valid),raw=metrics(proposal,base,gt,valid),paired_ci=ci,groups=groups,passed=ok,scope='Fresh12source clips of reused subjects/sequences' if fresh else 'Previously inspected diagnostic set')
    name='fresh' if fresh else 'old';save(RUN/f'{name}_results.json',report);torch.save(dict(indices=ids.cpu(),prediction=pred.cpu(),proposal=proposal.cpu(),std=torch.cat(stds).cpu(),base=base.cpu(),gt=gt.cpu(),valid=valid.cpu(),accepted=accepted.cpu(),rows=rows),RUN/f'{name}_predictions.pt');print(json.dumps(report),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('stage',choices=['seal','fresh','old']);a=p.parse_args();seal() if a.stage=='seal' else evaluate(a.stage=='fresh')

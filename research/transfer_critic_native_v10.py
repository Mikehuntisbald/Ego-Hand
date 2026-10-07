"""Frozen OOF critic transferred to the new native1280 proposal distribution.
All model/policy choices use development; the third12clip benchmark stays closed.
"""
import argparse,json,hashlib
import numpy as np,torch
from hand3d_native_v10 import NativeTrajectoryHand3D
from train_native_rollout_v10 import RUN as PROPOSAL
from train_hand3d_native_v10 import RUN as DATA,make_batch
from hand3d_v8_common import V7,load,save,metrics,score,camera_bank
from hand3d_visual_v8 import dense_bank
from proposal_critic_v9 import ProposalCritic,features
from calibrate_proposal_critic_v9 import choose
from calibrate_hand3d_v8 import paired_ci
from bounded_policy_v8 import apply
RUN=V7.parent/'native_critic_v10';RUN.mkdir(exist_ok=True)

@torch.no_grad()
def collect(kind,device):
    folder=PROPOSAL/('rgb_'+kind);assert json.loads((folder/'training_done.json').read_text())['complete']
    data=load(device);bank=dense_bank(device,len(data['world']));prob=torch.load(V7/'risk_probabilities.pt',weights_only=False)['joint'].to(device);params=camera_bank().to(device);ids=torch.tensor(np.where(np.asarray(data['roles'])=='dev_calibrate')[0],device=device)
    ck=torch.load(folder/'best.pt',weights_only=False,map_location=device);model=NativeTrajectoryHand3D(kind,True).to(device).eval();model.load_state_dict(ck['model']);cck=torch.load(V7.parent/'proposal_critic_v9/critic.pt',weights_only=False,map_location=device);critic=ProposalCritic(cck['dim']).to(device).eval();critic.load_state_dict(cck['model']);out={}
    for start in range(0,len(ids),8):
        ix=ids[start:start+8];b=make_batch(data,bank,ix,prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(b,seed=202610103+start)
        raw=p['xyz_camera_m'].float();std=p['std_m'].float();scores=torch.stack([critic(features(b,raw,std,params[data['feature_ids'][ix,8]],a)) for a in [.25,.5,.75,1.]])
        for key,value in [('proposal',raw),('std',std),('scores',scores.transpose(0,1))]:out.setdefault(key,[]).append(value.cpu())
    out={k:torch.cat(v) for k,v in out.items()};out.update(indices=ids.cpu(),step=ck['step'],kind=kind,proposal_sha256=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest())
    torch.save(out,RUN/f'{kind}_calibration.pt');print(json.dumps(dict(kind=kind,selected_step=ck['step'],windows=len(ids))),flush=True)

def select():
    data=load('cpu');trials={};best=None
    for kind in ['dit','regression']:
        c=torch.load(RUN/f'{kind}_calibration.pt',weights_only=False);ids=c['indices'];raw=c['proposal'];scores=c['scores'].transpose(0,1);base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];fallback=apply(raw,base,dict(cap_m=.0095,strength=.5));fm=metrics(fallback,base,gt,valid);values=[]
        for hazard in [.0001,.001,.005,.01,.025,.05,.1,.2]:
            for useful in [.5,.7,.85,.95]:
                for gain in [.5,2.,5.]:
                    policy=dict(hazard_max=hazard,useful_min=useful,gain_min_mm=gain);pred,accepted,alpha=choose(raw,base,scores,policy);m=metrics(pred,base,gt,valid);_,feasible=score(m);rank=(m['relative_mm'],m['camera_mm']);record=dict(policy=policy,metrics=m,accepted_hands=int(accepted.sum()),feasible=feasible);values.append(record)
                    if feasible and accepted.any() and m['relative_mm']<fm['relative_mm']-.05 and (best is None or rank<tuple(best['rank'])):
                        ci=paired_ci(pred,base,gt,valid,rows)
                        if ci['relative']['ci95_delta_mm'][1]<0:best=dict(kind=kind,policy=policy,metrics=m,accepted_hands=int(accepted.sum()),rank=rank,paired_ci=ci,proposal_sha256=c['proposal_sha256'],step=c['step'])
        trials[kind]=dict(fallback=fm,trials=values)
    save(RUN/'calibration_trials.json',trials);save(RUN/'development_selection.json',dict(approved=best is not None,selected=best,selection='Main finger-relative error first; camera within0.1mm baseline; both harm<=1%; require relative benefit beyond bounded fallback',critic='Frozen v9 critic from actual OOF proposal heads; new generator distribution, so independent validation required',scope='Development only; third12clips metrics unopened'))
    print((RUN/'development_selection.json').read_text(),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','select']);p.add_argument('--kind',choices=['dit','regression'],default='dit');p.add_argument('--device',default='cuda:2');a=p.parse_args();collect(a.kind,a.device) if a.stage=='collect' else select()

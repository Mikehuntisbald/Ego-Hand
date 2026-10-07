import json
import torch
from proposal_critic_v9 import ProposalCritic
from hand3d_v8_common import V7,load,save,metrics,score
from calibrate_proposal_critic_v9 import choose
from calibrate_hand3d_v8 import paired_ci
from bounded_policy_v8 import apply
RUN=V7.parent/'native_critic_v10'

@torch.no_grad()
def main():
    torch.set_num_threads(4);device='cuda:3';data=load(device);c=torch.load(RUN/'critic_calibration.pt',weights_only=False);ids=c['indices'].to(device);raw=c['proposal'].to(device);base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];ck=torch.load(RUN/'critic.pt',weights_only=False,map_location=device);critic=ProposalCritic(ck['dim']).to(device).eval();critic.load_state_dict(ck['model']);scores=critic(c['x'].to(device).float()).reshape(4,len(ids),5);fallback=apply(raw,base,dict(cap_m=.0095,strength=.5));fm=metrics(fallback,base,gt,valid);best=None;trials=[]
    for hazard in [.0001,.001,.005,.01,.025,.05,.1,.2]:
        for useful in [.5,.7,.85,.95]:
            for gain in [.5,2.,5.]:
                policy=dict(hazard_max=hazard,useful_min=useful,gain_min_mm=gain);pred,accepted,_=choose(raw,base,scores,policy);m=metrics(pred,base,gt,valid);_,ok=score(m);rank=(m['relative_mm'],m['camera_mm']);trials.append(dict(policy=policy,metrics=m,accepted_hands=int(accepted.sum()),feasible=ok))
                if ok and accepted.any() and m['relative_mm']<fm['relative_mm']-.05 and (best is None or rank<tuple(best['rank'])):
                    ci=paired_ci(pred,base,gt,valid,rows)
                    if ci['relative']['ci95_delta_mm'][1]<0:best=dict(policy=policy,metrics=m,accepted_hands=int(accepted.sum()),rank=rank,paired_ci=ci)
    save(RUN/'native_calibration_trials.json',trials);save(RUN/'native_selection.json',dict(approved=best is not None,selected=best,fallback_metrics=fm,selection='Main finger-relative error first; camera no regression beyond0.1mm; both harm<=1%; require recovery beyond bounded fallback',scope='Native proposal-head OOF training; dev-only selection; third12clips untouched'))
    print((RUN/'native_selection.json').read_text(),flush=True)

if __name__=='__main__':main()

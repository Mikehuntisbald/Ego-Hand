import json
import numpy as np,torch
from proposal_critic_v9 import ProposalCritic
from hand3d_v8_common import V7,load,save,metrics,score
from calibrate_hand3d_v8 import paired_ci
from bounded_policy_v8 import apply
RUN=V7.parent/'proposal_critic_v9'

def choose(predictions,base,critic_scores,policy):
    # Every option moves the complete hand with one scalar, including its root.
    strengths=torch.tensor([.25,.5,.75,1.],device=base.device);gain=(critic_scores[:,:,0]+.75*critic_scores[:,:,1])*30
    hazard=critic_scores[:,:,2:4].sigmoid().amax(-1);useful=critic_scores[:,:,4].sigmoid()
    eligible=(hazard<=policy['hazard_max'])&(useful>=policy['useful_min'])&(gain>=policy['gain_min_mm'])&(critic_scores[:,:,1]>0)
    rank=gain.masked_fill(~eligible,float('-inf'));best=rank.argmax(0);accepted=eligible.any(0);alpha=strengths[best]
    full=base+alpha[:,None,None]*(predictions-base);fallback=apply(predictions,base,dict(cap_m=.0095,strength=.5));return torch.where(accepted[:,None,None],full,fallback),accepted,alpha

@torch.no_grad()
def main():
    torch.set_num_threads(4);device='cuda:2';data=load(device);c=torch.load(RUN/'critic_calibration.pt',weights_only=False);ids=c['indices'].to(device);raw=c['proposal'].to(device);base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids]
    ck=torch.load(RUN/'critic.pt',weights_only=False,map_location=device);model=ProposalCritic(ck['dim']).to(device).eval();model.load_state_dict(ck['model']);x=c['x'].to(device).float();scores=model(x).reshape(4,len(ids),5)
    fallback=apply(raw,base,dict(cap_m=.0095,strength=.5));fm=metrics(fallback,base,gt,valid);fallback_score=fm['camera_mm']+.75*fm['relative_mm'];best=None;trials=[]
    for hazard in [.0001,.001,.005,.01,.025,.05,.1,.2]:
        for useful in [.5,.7,.85,.95]:
            for gain in [.5,2.,5.]:
                policy=dict(hazard_max=hazard,useful_min=useful,gain_min_mm=gain);pred,accepted,alpha=choose(raw,base,scores,policy);m=metrics(pred,base,gt,valid);_,ok=score(m);rank=m['camera_mm']+.75*m['relative_mm'];trial=dict(policy=policy,accepted_hands=int(accepted.sum()),metrics=m,feasible=ok);trials.append(trial)
                if ok and accepted.any() and rank<fallback_score-.05 and (best is None or rank<best['score']):
                    ci=paired_ci(pred,base,gt,valid,rows)
                    if ci['relative']['ci95_delta_mm'][1]<0:best=dict(policy=policy,metrics=m,score=rank,accepted_hands=int(accepted.sum()),paired_ci=ci)
    save(RUN/'critic_calibration_trials.json',trials);save(RUN/'critic_selection.json',dict(large_correction_approved_on_development=best is not None,selected=best,fallback_metrics=fm,fallback_score=fallback_score,criterion='Maintain<=1% harm in camera/relative, camera no regression, relative>=5%, improvement over bounded fallback; fresh remains unopened'))
    print((RUN/'critic_selection.json').read_text(),flush=True)

if __name__=='__main__':main()

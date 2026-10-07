"""Development-only proposal ceiling. Oracle choices are diagnostic, never inference."""
import json
import torch
from hand3d_v8_common import V7,load,metrics,save
from bounded_policy_v8 import apply
from proposal_critic_v9 import ProposalCritic,labels
RUN=V7.parent/'native_critic_v10'

@torch.no_grad()
def main():
    torch.set_num_threads(4);device='cuda:3';data=load(device)
    c=torch.load(RUN/'critic_calibration.pt',weights_only=False);ids=c['indices'].to(device);raw=c['proposal'].to(device);base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids]
    fallback=apply(raw,base,dict(cap_m=.0095,strength=.5));mask=valid.clone();mask[:,5]=False
    def errors(p):
        return (p-gt).norm(dim=-1),((p-p[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)
    be,br=errors(base);fe,fr=errors(fallback)
    candidates=torch.stack([base+a*(raw-base) for a in [.25,.5,.75,1.]])
    pe=[];pr=[];safe=[];details=[]
    for pred in candidates:
        e,r=errors(pred);pe.append(e);pr.append(r);safe.append(~((mask&(be<=.01)&(e>.02))|(mask&(br<=.01)&(r>.02))).any(-1))
        details.append(metrics(pred,base,gt,valid))
    pe=torch.stack(pe);pr=torch.stack(pr);safe=torch.stack(safe)
    rg=((fr[None]-pr)*mask[None]).sum(-1)/mask.sum(-1).clamp_min(1)
    cg=((fe[None]-pe)*mask[None]).sum(-1)/mask.sum(-1).clamp_min(1)
    eligible=safe&(rg>.0002)&(cg>-.0001)
    rank=rg.masked_fill(~eligible,float('-inf'));best=rank.argmax(0);accepted=eligible.any(0);oracle=torch.where(accepted[:,None,None],candidates[best,torch.arange(len(ids),device=device)],fallback)
    ck=torch.load(RUN/'critic.pt',weights_only=False,map_location=device);model=ProposalCritic(ck['dim']).to(device).eval();model.load_state_dict(ck['model']);scores=model(c['x'].to(device).float()).reshape(4,len(ids),5)
    hazard=scores[:,:,2:4].sigmoid().amax(-1);useful=scores[:,:,4].sigmoid()
    q=lambda x: [float(t) for t in torch.quantile(x.float(),torch.tensor([0,.1,.5,.9,1.],device=device))] if x.numel() else []
    trials=json.loads((RUN/'native_calibration_trials.json').read_text());selected=[t for t in trials if t['accepted_hands']]
    selected.sort(key=lambda t:(not t['feasible'],t['metrics']['relative_mm']))
    out=dict(scope='Development-only GT oracle; not a deployable selector or independent evidence',windows=len(ids),fallback=metrics(fallback,base,gt,valid),raw_alpha_metrics=details,
             safely_better_hands=int(accepted.sum()),oracle=metrics(oracle,base,gt,valid),
             safe_better_hazard_quantiles=q(hazard[eligible]),safe_better_useful_quantiles=q(useful[eligible]),
             unsafe_hazard_quantiles=q(hazard[~safe]),all_hazard_quantiles=q(hazard),all_useful_quantiles=q(useful),
             nonzero_trial_count=len(selected),top_nonzero_trials=selected[:8])
    save(RUN/'ceiling_diagnosis.json',out);print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()

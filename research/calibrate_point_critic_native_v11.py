import argparse,json
import torch
from hand3d_v8_common import V7,load,save,metrics,score,camera_bank
from hand3d_data_v7 import batch
from hand3d_visual_v8 import dense_bank
from point_critic_native_v11 import JointProposalCritic,visual_queries
from calibrate_proposal_critic_v9 import choose
from calibrate_hand3d_v8 import paired_ci
from bounded_policy_v8 import apply
ROOT=V7.parent/'point_critic_native_v11';SOURCE=V7.parent/'native_critic_v10'

@torch.no_grad()
def get_scores(model,data,bank,prob,ids,raw,x,params):
    n=len(ids);scores=[]
    for j,alpha in enumerate([.25,.5,.75,1.]):
        parts=[]
        for start in range(0,n,32):
            ix=ids[start:start+32];visual=None
            if model.native:
                b=batch(data,ix,prob);b['rgb_native']=bank[data['feature_ids'][ix]]
                visual=visual_queries(b,raw[start:start+32],params[data['feature_ids'][ix,8]],alpha)
            parts.append(model(x[j*n+start:j*n+start+len(ix)].float(),visual))
        scores.append(torch.cat(parts))
    return torch.stack(scores)

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['native','geometry'],required=True);ap.add_argument('--device',default='cuda:3');args=ap.parse_args()
    torch.set_num_threads(4);device=args.device;run=ROOT/args.arm
    assert json.loads((run/'done.json').read_text())['complete']
    data=load(device);params=camera_bank().to(device);c=torch.load(SOURCE/'critic_calibration.pt',weights_only=False);ids=c['indices'].to(device);raw=c['proposal'].to(device)
    base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids]
    ck=torch.load(run/'critic.pt',weights_only=False,map_location=device);model=JointProposalCritic(ck['dim'],ck['native']).to(device).eval();model.load_state_dict(ck['model'])
    bank=dense_bank(device,len(data['world'])) if ck['native'] else None;prob=torch.load(V7/'risk_probabilities.pt',weights_only=False)['joint'].to(device)
    scores=get_scores(model,data,bank,prob,ids,raw,c['x'].to(device),params);torch.save(dict(scores=scores.cpu(),indices=ids.cpu()),run/'calibration_scores.pt')
    fallback=apply(raw,base,dict(cap_m=.0095,strength=.5));fm=metrics(fallback,base,gt,valid);best=None;trials=[]
    for hazard in [.0001,.001,.005,.01,.025,.05,.1,.2,.3]:
        for useful in [.3,.5,.7,.85,.95]:
            for gain in [.5,2.,5.]:
                policy=dict(hazard_max=hazard,useful_min=useful,gain_min_mm=gain);pred,accepted,_=choose(raw,base,scores,policy);m=metrics(pred,base,gt,valid);_,ok=score(m);trials.append(dict(policy=policy,metrics=m,accepted_hands=int(accepted.sum()),feasible=ok))
                if ok and accepted.any() and m['relative_mm']<fm['relative_mm']-.05 and (best is None or m['relative_mm']<best['metrics']['relative_mm']):
                    ci=paired_ci(pred,base,gt,valid,rows)
                    if ci['relative']['ci95_delta_mm'][1]<0:best=dict(policy=policy,metrics=m,accepted_hands=int(accepted.sum()),paired_ci=ci)
    save(run/'calibration_trials.json',trials);save(run/'development_selection.json',dict(approved=best is not None,selected=best,fallback_metrics=fm,scope='Dev-only. Native1280 local/global visual queries; joint supervision; third12clips untouched'))
    print(json.dumps(dict(arm=args.arm,critic_step=ck['step'],approved=best is not None,selected=best)),flush=True)

if __name__=='__main__':main()

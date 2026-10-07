"""Development-only whole-hand strength/bound control, preserving root/pose coupling."""
import torch
from hand3d_v8_common import V7,load,metrics,score,save
from bounded_policy_v8 import apply
from calibrate_hand3d_v8 import paired_ci
RUN=V7.parent/'native_scalar_v10'

@torch.no_grad()
def main():
    torch.set_num_threads(4);device='cuda:0';data=load(device);RUN.mkdir(exist_ok=True)
    c=torch.load(V7.parent/'native_critic_v10/critic_calibration.pt',weights_only=False);ids=c['indices'].to(device);raw=c['proposal'].to(device);base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids]
    trials=[];best=None
    for strength in [.125,.2,.25,.3,.35,.4,.5,.75,1.]:
        for cap in [.0095,.0125,.015,.02,.03,.05,None]:
            p=dict(strength=strength,cap_m=cap);pred=apply(raw,base,p);m=metrics(pred,base,gt,valid);_,ok=score(m);t=dict(policy=p,metrics=m,feasible=ok);trials.append(t)
            if ok and (best is None or m['relative_mm']<best['metrics']['relative_mm']):
                ci=paired_ci(pred,base,gt,valid,rows)
                if ci['relative']['ci95_delta_mm'][1]<0:best={**t,'paired_ci':ci}
    save(RUN/'calibration_trials.json',trials);save(RUN/'development_selection.json',dict(approved=best is not None,selected=best,scope='Dev-calibration only. For caps>9.5mm protection is empirical, not a guaranteed triangle-inequality bound; third12clips untouched'))
    print(best,flush=True)

if __name__=='__main__':main()

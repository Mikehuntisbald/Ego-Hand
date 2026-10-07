import json,itertools
import torch
from hand3d_v8_common import V7,load,save,metrics,score
from bounded_policy_v8 import apply as scalar
from native_projection_policy_v11 import apply
from calibrate_hand3d_v8 import paired_ci
RUN=V7.parent/'native_projection_v11';SOURCE=V7.parent/'native_critic_v10'

@torch.no_grad()
def main():
    torch.set_num_threads(4);device='cuda:0';data=load(device);RUN.mkdir(exist_ok=True)
    c=torch.load(SOURCE/'critic_calibration.pt',weights_only=False);ids=c['indices'].to(device);raw=c['proposal'].to(device);base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids]
    trials=[];best=None;fm=metrics(scalar(raw,base,dict(cap_m=.0095,strength=.5)),base,gt,valid)
    for strength,cap,root_cap,weight in itertools.product([.25,.5,.75,1.],[.0049,.0075,.0095],[.0049,.0075,.0095],[.25,1.,4.]):
        if root_cap>cap:continue
        policy=dict(strength=strength,cap_m=cap,root_cap_m=root_cap,relative_weight=weight);pred=apply(raw,base,policy);m=metrics(pred,base,gt,valid);_,ok=score(m)
        dc=(pred-base).norm(dim=-1).max();dr=((pred-pred[:,5:6])-(base-base[:,5:6])).norm(dim=-1).max();assert dc<.009501 and dr<.009501
        trial=dict(policy=policy,metrics=m,feasible=ok,max_camera_displacement_m=float(dc),max_relative_displacement_m=float(dr));trials.append(trial)
        if ok and m['relative_mm']<fm['relative_mm']-.05 and (best is None or m['relative_mm']<best['metrics']['relative_mm']):
            ci=paired_ci(pred,base,gt,valid,rows)
            if ci['relative']['ci95_delta_mm'][1]<0:best={**trial,'paired_ci':ci}
    save(RUN/'calibration_trials.json',trials);save(RUN/'development_selection.json',dict(approved=best is not None,selected=best,fallback=fm,scope='Dev only; camera/relative displacement both<=9.5mm; third12clips untouched'))
    print(json.dumps(dict(approved=best is not None,selected=best)),flush=True)

if __name__=='__main__':main()

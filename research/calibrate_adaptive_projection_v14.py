import json,itertools,argparse
import torch
from hand3d_v8_common import V7,save,metrics,score
from hand3d_data_v7 import batch
from native_projection_policy_v11 import apply as conservative
from density_model_v13 import POLICY
from adaptive_projection_v14 import apply
from calibrate_hand3d_v8 import paired_ci
DATA=V7.parent/'aligned_density_v13';RUN=V7.parent/'adaptive_projection_v14'

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',default='dit_dense');ap.add_argument('--generator',choices=['density','anchor'],default='density');ap.add_argument('--device',default='cuda:3');args=ap.parse_args();torch.set_num_threads(4);device=args.device;source=V7.parent/('native_density_v13' if args.generator=='density' else 'anchor_native_v14_closed')/args.arm;density=args.arm.split('_')[-1] if args.generator=='density' else 'dense';raw_data=torch.load(DATA/f'{density}_data.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw_data.items()};c=torch.load(source/'calibration.pt',weights_only=False);ids=c['indices'].to(device);raw=c['proposal'].to(device);prob=torch.load(DATA/f'risk_{density}/risk_probabilities.pt',weights_only=False)['joint'].to(device);b=batch(data,ids,prob);base=b['base'];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];fallback=conservative(raw,base,POLICY);fm=metrics(fallback,base,gt,valid);best=None;trials=[];out=RUN/(args.arm if args.generator=='density' else 'anchor_'+args.arm);out.mkdir(parents=True,exist_ok=True)
    if (out/'selection.json').exists() and not (out/'initial_weight4_selection.json').exists():
        __import__('shutil').copy2(out/'selection.json',out/'initial_weight4_selection.json');__import__('shutil').copy2(out/'trials.json',out/'initial_weight4_trials.json')
    for strength,large,pc,pr,weight in itertools.product([.25,.5,1.],[.03,.05,.1,.2],[.5,.65,.8,.9,.95,.99],[.5,.65,.8,.9,.95,.99],[.25,1.,4.]):
        policy=dict(strength=strength,large_cap_m=large,camera_risk_threshold=pc,relative_risk_threshold=pr,relative_weight=weight);pred=apply(raw,b,policy);m=metrics(pred,base,gt,valid);_,ok=score(m);preserves_fallback_camera=m['camera_mm']<=fm['camera_mm']+.1;trials.append(dict(policy=policy,metrics=m,feasible=ok,preserves_fallback_camera=preserves_fallback_camera))
        if ok and preserves_fallback_camera and m['relative_mm']<fm['relative_mm']-.1 and m['relative_bad_recovered20']>fm['relative_bad_recovered20'] and (best is None or m['relative_mm']<best['metrics']['relative_mm']):
            ci=paired_ci(pred,base,gt,valid,rows)
            if ci['relative']['ci95_delta_mm'][1]<0:best=dict(policy=policy,metrics=m,paired_ci=ci)
    save(out/'trials.json',trials);save(out/'selection.json',dict(approved=best is not None,selected=best,fallback=fm,criterion='Original recovery/protection criteria plus camera error no more than0.1mm worse than current conservative fallback',scope='Dev-only. Coupled constraints, per-point error-risk radii; large movement has empirical protection only. Independent evaluation not opened.'))
    print(json.dumps(dict(arm=args.arm,approved=best is not None,selected=best),indent=2),flush=True)

if __name__=='__main__':main()

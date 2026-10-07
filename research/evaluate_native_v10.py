import json,argparse
import numpy as np,torch
from hand3d_native_v10 import NativeTrajectoryHand3D
from train_hand3d_native_v10 import RUN,predict,make_batch
from hand3d_v8_common import V7,load,save,metrics,score
from hand3d_visual_v8 import dense_bank
from bounded_policy_v8 import apply
from calibrate_hand3d_v8 import paired_ci

def select():
    data=load('cpu');trials={};best=None
    for arm in ['dit_c1280','dit_c128','regression_c1280','regression_c128']:
        c=torch.load(RUN/arm/'calibration.pt',weights_only=False,map_location='cpu');ids=c['indices'];base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];values=[]
        for cap in [.0049,.0075,.0095]:
            for strength in [.25,.5,1.]:
                policy=dict(cap_m=cap,strength=strength);pred=apply(c['raw'],base,policy);m=metrics(pred,base,gt,valid);ci=paired_ci(pred,base,gt,valid,rows);_,ok=score(m);ok=ok and ci['relative']['ci95_delta_mm'][1]<0;rank=m['camera_mm']+.75*m['relative_mm'];values.append(dict(policy=policy,metrics=m,paired_ci=ci,feasible=ok))
                if ok and (best is None or rank<best['score']):best=dict(arm=arm,policy=policy,metrics=m,score=rank,paired_ci=ci)
        trials[arm]=values
    save(RUN/'calibration_trials.json',trials);save(RUN/'development_selection.json',dict(approved_on_development=best is not None,selected=best,scope='Development only; all v8/v9 data already inspected; no new independent claim'))
    print(json.dumps(best),flush=True)

@torch.no_grad()
def diagnose():
    device='cuda:0';data=load(device);bank=dense_bank(device,len(data['world']));ep=torch.load(V7/'risk_probabilities.pt',weights_only=False)['joint'].to(device);selection=json.loads((RUN/'development_selection.json').read_text())['selected'];assert selection
    ck=torch.load(RUN/selection['arm']/'best.pt',weights_only=False,map_location=device);model=NativeTrajectoryHand3D(ck['kind'],ck['native']).to(device).eval();model.load_state_dict(ck['model']);test=torch.tensor(np.where(np.asarray(data['roles'])=='test')[0],device=device);raw=predict(model,data,bank,ep,test);base=data['xyz_camera_bank'][data['feature_ids'][test,8]];pred=apply(raw,base,selection['policy']);gt=data['gt'][test];valid=data['valid'][test]
    queue=json.loads((V7.parent/'natural_reliability_v4/delivery/hardcase_review_queue.json').read_text());lookup={int(i):j for j,i in enumerate(test)};groups={}
    for name,q in [('hard47',queue),('severe8',[x for x in queue if x['after_px']>40])]:
        ix=[lookup[x['window_index']] for x in q];groups[name]=dict(bounded=metrics(pred[ix],base[ix],gt[ix],valid[ix]),raw=metrics(raw[ix],base[ix],gt[ix],valid[ix]))
    save(RUN/'old_diagnostic.json',dict(arm=selection['arm'],selected_step=ck['step'],policy=selection['policy'],overall=metrics(pred,base,gt,valid),raw=metrics(raw,base,gt,valid),groups=groups,scope='Previously inspected diagnostic set; not fresh'))
    print(json.dumps(dict(arm=selection['arm'],groups=groups)),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','diagnose']);a=p.parse_args();select() if a.stage=='select' else diagnose()

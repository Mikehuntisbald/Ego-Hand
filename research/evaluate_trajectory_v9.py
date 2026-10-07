import argparse,json,hashlib
import numpy as np,torch
from hand3d_trajectory_data_v9 import RUN,center_only
from hand3d_trajectory_v9 import TrajectoryHand3D
from train_hand3d_trajectory_v9 import predict
from hand3d_v8_common import V7,load,batch,save,metrics,score
from hand3d_data_v7 import risk_features
from hand3d_risk_v7 import Risk3D
from bounded_policy_v8 import apply
from calibrate_hand3d_v8 import paired_ci

def select():
    data=load('cpu');trials={};selected=None
    for arm in ['rgb_dit','rgb_regression']:
        c=torch.load(RUN/arm/'calibration.pt',weights_only=False,map_location='cpu');ids=c['indices'];raw=c['raw'];base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];results=[]
        for cap in [.0049,.0075,.0095]:
            for strength in [.25,.5,1.]:
                policy=dict(cap_m=cap,strength=strength);pred=apply(raw,base,policy);m=metrics(pred,base,gt,valid);ci=paired_ci(pred,base,gt,valid,rows);_,ok=score(m);ok=ok and ci['relative']['ci95_delta_mm'][1]<0
                results.append(dict(policy=policy,metrics=m,paired_ci=ci,feasible=ok));rank=m['camera_mm']+.75*m['relative_mm']
                if ok and (selected is None or rank<selected['score']):selected=dict(arm=arm,policy=policy,metrics=m,score=rank,paired_ci=ci)
        trials[arm]=results
    save(RUN/'calibration_trials.json',trials);save(RUN/'development_selection.json',dict(approved_on_development=selected is not None,selected=selected,scope='Development only; v9 fresh metrics unopened'));print(json.dumps(selected),flush=True)

@torch.no_grad()
def diagnose():
    device='cuda:0';data=load(device);ids=torch.tensor(np.where(np.asarray(data['roles'])=='dev_calibrate')[0],device=device);selection=json.loads((RUN/'development_selection.json').read_text())['selected'];assert selection is not None
    ck=torch.load(RUN/selection['arm']/'best.pt',weights_only=False,map_location=device);model=TrajectoryHand3D(ck['kind']).to(device).eval();model.load_state_dict(ck['model']);rck=torch.load(V7/'risk_all.pt',weights_only=False,map_location=device);risk=Risk3D(rck['dim']).to(device).eval();risk.load_state_dict(rck['model']);temp=torch.tensor(json.loads((V7/'risk_calibration.json').read_text())['temperature'],device=device)
    base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];reports={};predictions={}
    for mode in ['full','no_history','zero_rgb']:
        parts=[]
        for start in range(0,len(ids),8):
            b=batch(data,ids[start:start+8])
            if mode=='no_history':b=center_only(b)
            pr=(risk(risk_features(b))/temp).sigmoid();b['risk_camera']=pr[:,:,0];b['risk_relative']=pr[:,:,1]
            if mode=='zero_rgb':b['rgb']=torch.zeros_like(b['rgb'])
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model.predict(b,seed=202610091+start)
            parts.append(output['xyz_camera_m'].float())
        raw=torch.cat(parts);pred=apply(raw,base,selection['policy']);predictions[mode]=(pred,raw);reports[mode]=dict(bounded=metrics(pred,base,gt,valid),raw=metrics(raw,base,gt,valid))
    ci=paired_ci(predictions['no_history'][0],predictions['full'][0],gt,valid,rows);rawci=paired_ci(predictions['no_history'][1],predictions['full'][1],gt,valid,rows);save(RUN/'conditioning_diagnosis.json',dict(results=reports,no_history_minus_full_ci=ci,raw_no_history_minus_full_ci=rawci));print(json.dumps(dict(results=reports,no_history_minus_full_ci=ci)),flush=True)
    test=torch.tensor(np.where(np.asarray(data['roles'])=='test')[0],device=device);prob=torch.load(V7/'risk_probabilities.pt',weights_only=False)['joint'].to(device);raw=predict(model,data,prob,test);base=data['xyz_camera_bank'][data['feature_ids'][test,8]];pred=apply(raw,base,selection['policy']);gt=data['gt'][test];valid=data['valid'][test]
    queue=json.loads((V7.parent/'natural_reliability_v4/delivery/hardcase_review_queue.json').read_text());lookup={int(i):j for j,i in enumerate(test)};groups={}
    for label,qset in [('natural_hardcases_47',queue),('severe_8',[q for q in queue if q['after_px']>40])]:
        ix=[lookup[q['window_index']] for q in qset];groups[label]=dict(bounded=metrics(pred[ix],base[ix],gt[ix],valid[ix]),raw=metrics(raw[ix],base[ix],gt[ix],valid[ix]))
    save(RUN/'old_diagnostic.json',dict(arm=selection['arm'],selected_step=ck['step'],overall=metrics(pred,base,gt,valid),raw=metrics(raw,base,gt,valid),groups=groups,scope='Previously inspected data; diagnostic only'))

if __name__=='__main__':
    torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','diagnose']);a=p.parse_args();select() if a.stage=='select' else diagnose()

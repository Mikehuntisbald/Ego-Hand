import json,argparse
import numpy as np,torch
from hand3d_v8_common import RUN,V7,load,metrics,score,save
from bounded_policy_v8 import apply
from hand3d_temporal_v7 import WRIST

def paired_ci(pred,base,gt,valid,rows,iterations=5000):
    mask=valid.clone();mask[:,WRIST]=False
    rel=lambda x:(x-x[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])
    deltas={'camera':((pred-gt).norm(dim=-1)-(base-gt).norm(dim=-1))*1000,'relative':(rel(pred).norm(dim=-1)-rel(base).norm(dim=-1))*1000}
    seqs=sorted({r['sequence'] for r in rows});rng=np.random.default_rng(202610081);draw=rng.integers(len(seqs),size=(iterations,len(seqs)));result={}
    for name,values in deltas.items():
        sums=[];counts=[];per={}
        for seq in seqs:
            select=torch.tensor([r['sequence']==seq for r in rows],device=mask.device);m=mask[select];v=values[select][m]
            sums.append(float(v.sum()));counts.append(len(v));per[seq]=float(v.mean())
        total=np.asarray(sums)[draw].sum(1)/np.maximum(np.asarray(counts)[draw].sum(1),1);result[name]=dict(ci95_delta_mm=np.quantile(total,[.025,.975]).tolist(),per_sequence_delta_mm=per)
    return result

def calibrate(folders):
    data=load('cpu');basebank=data['xyz_camera_bank'];results={};selected=None
    for folder in folders:
        cache=torch.load(RUN/folder/'calibration.pt',weights_only=False,map_location='cpu');ids=cache['indices'].cpu();preds=cache['predictions'];base=basebank[data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];trials=[]
        for source in ['xyz_camera_m','raw_xyz_camera_m']:
            for cap in [None,.0049,.0075,.0095]:
                for strength in [.25,.5,1.]:
                    policy=dict(source=source,cap_m=cap,strength=strength);pred=apply(preds[source],base,policy);m=metrics(pred,base,gt,valid);ci=paired_ci(pred,base,gt,valid,rows);key,feasible=score(m)
                    feasible=feasible and ci['relative']['ci95_delta_mm'][1]<0 and m['relative_bad_recovered20']>0 and m['relative_bad_mean_mm']<metrics(base,base,gt,valid)['relative_bad_mean_mm']
                    trial=dict(policy=policy,feasible=feasible,metrics=m,paired_ci=ci);trials.append(trial)
                    # Automatic use includes a direction-preserving bound. At 9.5mm,
                    # triangle inequality protects both <=10mm error categories
                    # from crossing20mm, independently of risk-head calibration.
                    if feasible and cap is not None and cap<=.0095:
                        rank=m['camera_mm']+.75*m['relative_mm']
                        if selected is None or rank<selected['score']:selected=dict(folder=folder,policy=policy,score=rank,metrics=m,paired_ci=ci)
        results[folder]=trials
    save(RUN/'calibration_trials.json',results);save(RUN/'development_selection.json',dict(approved_on_development=selected is not None,selected=selected,automatic_requires_bound=True,scope='Development calibration only; no fresh/test metrics used'))
    print(json.dumps(dict(approved_on_development=selected is not None,selected=selected)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folders',nargs='+',default=['dit_rollout','dit_one_step','regression_rollout']);a=p.parse_args();calibrate(a.folders)

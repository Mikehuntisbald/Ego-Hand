import json,hashlib
import numpy as np,torch
from hand3d_v8_common import RUN,V7,predict,metrics,save
from hand3d_data_v7 import batch,risk_features
from hand3d_risk_v7 import Risk3D
from evaluate_hand3d_v8 import restore
from calibrate_hand3d_v8 import paired_ci
from bounded_policy_v8 import apply

@torch.no_grad()
def main():
    torch.set_num_threads(4);device='cuda:0';seal=json.loads((RUN/'fresh_evaluation_seal.json').read_text());raw=torch.load(RUN/'fresh_data.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw.items()};ids=torch.arange(len(data['rows']),device=device)
    risk_ck=torch.load(V7/'risk_all.pt',weights_only=False,map_location=device);risk=Risk3D(risk_ck['dim']).to(device).eval();risk.load_state_dict(risk_ck['model']);temp=torch.tensor(json.loads((V7/'risk_calibration.json').read_text())['temperature'],device=device)
    prob=torch.cat([(risk(risk_features(batch(data,ids[start:start+64])))/temp).sigmoid() for start in range(0,len(ids),64)])
    selected=torch.load(RUN/'fresh_predictions.pt',weights_only=False)['prediction'].to(device);base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'];valid=data['valid'];results={}
    for folder in ['regression_rollout','dit_one_step']:
        assert hashlib.sha256((RUN/folder/'best.pt').read_bytes()).hexdigest()==seal['controls_sha256'][folder]
        model,visual,ck=restore(folder,device,data,True);output=predict(model,data,prob,ids);policy=seal['controls'][folder];pred=apply(output[policy['source']],base,policy);m=metrics(pred,base,gt,valid);ci=paired_ci(pred,selected,gt,valid,data['rows'])
        results[folder]=dict(policy=policy,metrics=m,paired_vs_selected_dit=ci);del model,visual;torch.cuda.empty_cache()
    save(RUN/'fresh_comparisons.json',dict(results=results,selection_frozen=True));print(json.dumps(results),flush=True)

if __name__=='__main__':main()

"""Frozen OOF risk models re-evaluated on observation-only +/-1.6s windows."""
import json,torch
from hand3d_v8_common import V7,save
from hand3d_data_v7 import batch,risk_features
from hand3d_risk_v7 import Risk3D
from online_parameter_model_v47 import RUN

def main():
    torch.set_num_threads(4);device='cuda:3'
    if (RUN/'risk.pt').exists():return
    data=torch.load(RUN/'inputs.pt',weights_only=False,mmap=True)
    data={k:v.to(device) if torch.is_tensor(v) else v for k,v in data.items()}
    root=V7.parent/'side_data_v16/consensus/risk_dense';cal=json.loads((root/'calibration.json').read_text());folds=cal['folds'];temp=torch.tensor(cal['temperature'],device=device)
    N=len(data['roles']);joint=torch.zeros(N,20,2,device=device);oof=torch.zeros_like(joint)
    with torch.no_grad():
        for name in ['all','fold0','fold1','fold2']:
            ck=torch.load(root/f'risk_{name}.pt',weights_only=False,map_location=device)
            model=Risk3D(ck['dim']).to(device).eval();model.load_state_dict(ck['model'])
            ids=[i for i,r in enumerate(data['roles']) if name=='all' or (r=='train' and folds[data['subjects'][i]]==int(name[-1]))]
            if name!='all':assert not ({data['subjects'][i] for i in ids}&set(ck['trained_subjects']))
            for begin in range(0,len(ids),64):
                ix=torch.tensor(ids[begin:begin+64],device=device);prob=(model(risk_features(batch(data,ix)))/temp).sigmoid()
                (joint if name=='all' else oof)[ix]=prob
    train=torch.tensor([r=='train' for r in data['roles']],device=device);oof[~train]=joint[~train]
    torch.save(dict(joint=joint.cpu(),train_oof=oof.cpu()),RUN/'risk.pt')
    save(RUN/'risk_protocol.json',dict(passed=True,original_frozen_weights=True,original_temperature=True,training_subject_OOF_preserved=True,
        risk_context_s=1.6,new_calibration=False,learned_visual_features_not_used_by_original_risk_head=True))
    print('Short-window OOF risk ready',flush=True)
if __name__=='__main__':main()

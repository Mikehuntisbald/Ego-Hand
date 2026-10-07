import json,itertools
import numpy as np,torch
from hand3d_v8_common import V7,load,save,metrics
from hand3d_data_v7 import batch
from native_projection_policy_v11 import apply
from density_model_v13 import POLICY
from temporal_reference_v14 import reference
from calibrate_hand3d_v8 import paired_ci
RUN=V7.parent/'temporal_reference_v14'

@torch.no_grad()
def main():
    torch.set_num_threads(4);data=load('cuda:3');roles=np.array(data['roles']);prob=torch.load(V7/'risk_probabilities.pt',weights_only=False)['joint'].to('cuda:3');queue=json.loads((V7.parent/'natural_reliability_v4/delivery/hardcase_review_queue.json').read_text());groups={};selected=None
    for split in ['dev_select','dev_calibrate','test']:
        ids=torch.tensor(np.where(roles==split)[0],device='cuda:3');b=batch(data,ids,prob);base=b['base'];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];reports=[]
        configs=list(itertools.product([.5,1.,2.],[False,True])) if split!='test' else [(selected['horizon'],selected['robust'])]
        for horizon,robust in configs:
            raw,usable=reference(b,horizon,robust);pred=apply(raw,base,POLICY);m=metrics(pred,base,gt,valid);raw_m=metrics(raw,base,gt,valid);report=dict(horizon=horizon,robust=robust,raw=raw_m,bounded=m,usable_hands=int(usable.sum()))
            if split=='dev_select' and (selected is None or raw_m['relative_mm']<selected['score']):selected=dict(horizon=horizon,robust=robust,score=raw_m['relative_mm'])
            if split=='test':
                lookup={int(i):j for j,i in enumerate(ids)}
                for name,q in [('hard47',queue),('severe8',[r for r in queue if r['after_px']>40])]:
                    ix=[lookup[r['window_index']] for r in q];report[name]=dict(raw=metrics(raw[ix],base[ix],gt[ix],valid[ix]),bounded=metrics(pred[ix],base[ix],gt[ix],valid[ix]))
            reports.append(report)
        groups[split]=reports
    out=dict(selected_on='Dev-select raw finger-relative error only',selected=selected,results=groups,scope='Observation-only reference, no RGB or GT input. Oldtest/hard cases previously inspected; diagnostic only. This does not replace RGB3D DiT without further training and frozen validation.')
    save(RUN/'results.json',out);print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()

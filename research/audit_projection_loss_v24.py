"""Separate useful protection from blocked correct proposals; no policy tuning."""
import json,torch
from hand3d_v8_common import V7,save,metrics
from hand3d_data_v7 import batch
from adaptive_projection_v14 import apply

def main():
    torch.set_num_threads(4)
    root=V7.parent/'projection_diagnostic_v24';root.mkdir(exist_ok=True)
    source=V7.parent/'side_data_v16/consensus'
    data=torch.load(source/'dense_data.pt',weights_only=False,mmap=True)
    cal=torch.load(V7.parent/'side_native_v16/consensus/uniform_adaptive/calibration.pt',weights_only=False)
    ids=cal['indices'];risk=torch.load(source/'risk_dense/risk_probabilities.pt',weights_only=False)['joint']
    b=batch(data,ids,risk);policy=json.loads((V7.parent/'side_native_v16/fifth_seal.json').read_text())['policy']
    raw=cal['proposal'];pred=apply(raw,b,policy);gt=data['gt'][ids];base=data['original_base_for_evaluation'][ids]
    valid=data['valid'][ids].clone();valid[:,5]=False
    error=lambda x:((x-x[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000
    be,re,pe=error(base),error(raw),error(pred)
    hard=(be*valid).sum(-1)/valid.sum(-1).clamp_min(1)>40
    reports={}
    for name,keep in [('all',torch.ones(len(ids),dtype=torch.bool)),('hard',hard)]:
        mask=valid&keep[:,None]
        reports[name]=dict(windows=int(keep.sum()),
            raw_recovers_projection_misses=int((mask&(be>20)&(re<=20)&(pe>20)).sum()),
            projection_recovers_raw_misses=int((mask&(be>20)&(re>20)&(pe<=20)).sum()),
            raw=metrics(raw[keep],base[keep],gt[keep],data['valid'][ids][keep]),
            projected=metrics(pred[keep],base[keep],gt[keep],data['valid'][ids][keep]))
    save(root/'development_diagnostic.json',dict(complete=True,results=reports,policy=policy,
        scope='Existing dev_calibrate diagnosis only, not new independent evidence. No policy/model selected or deployed. Labels identify oracle blocked/rescued points, never inference choices. Do not remove protection based on oracle counts; any operating change needs dev_select feasibility, calibration check, source-paired CI and new sealed clips.'))
    print(json.dumps({k:{n:v for n,v in r.items() if n in ['windows','raw_recovers_projection_misses','projection_recovers_raw_misses']} for k,r in reports.items()},indent=2))

if __name__=='__main__':main()

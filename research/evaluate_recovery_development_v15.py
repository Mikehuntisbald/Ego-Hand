"""Judge v15 using development only, retaining every failed control."""
import json
import torch
from hand3d_v8_common import V7,save,metrics,score
from hand3d_data_v7 import batch
from adaptive_projection_v14 import apply
from calibrate_hand3d_v8 import paired_ci
from train_recovery_v15 import RUN,DATA,difficulty

def main():
    data=torch.load(DATA/'dense_data.pt',weights_only=False,mmap=True)
    baseline=torch.load(V7.parent/'native_density_v13/dit_dense/calibration.pt',weights_only=False)
    ids=baseline['indices'];prob=torch.load(DATA/'risk_dense/risk_probabilities.pt',weights_only=False)['joint']
    b=batch(data,ids,prob);gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids]
    policy=json.loads((V7.parent/'adaptive_projection_v14/dit_dense/fourth_seal.json').read_text())['policy']
    prior=apply(baseline['proposal'],b,policy);hard=difficulty(data,ids)
    results={};winner=None
    for arm in ['uniform_adaptive','hard_adaptive','hard_conservative']:
        run=RUN/arm;done=json.loads((run/'done.json').read_text());assert done['complete']
        c=torch.load(run/'calibration.pt',weights_only=False);assert torch.equal(c['indices'],ids)
        pred=apply(c['proposal'],b,policy);m=metrics(pred,b['base'],gt,valid);hm=metrics(pred[hard],b['base'][hard],gt[hard],valid[hard]);_,ok=score(m)
        ci=paired_ci(pred,prior,gt,valid,rows);bc=paired_ci(pred,b['base'],gt,valid,rows)
        previous_hard=metrics(prior[hard],b['base'][hard],gt[hard],valid[hard])
        useful=ok and bc['relative']['ci95_delta_mm'][1]<0 and m['camera_mm']<=metrics(prior,b['base'],gt,valid)['camera_mm']+.1 and m['relative_mm']<=metrics(prior,b['base'],gt,valid)['relative_mm']+.1 and hm['relative_bad_recovered20']>previous_hard['relative_bad_recovered20'] and hm['relative_mm']<previous_hard['relative_mm']
        result=dict(selected_step=done['selected_step'],metrics=m,hard_windows=int(hard.sum()),hard=hm,prior_hard=previous_hard,paired_vs_v14=ci,paired_vs_base=bc,qualified=useful,policy=policy)
        results[arm]=result
        if useful:
            key=(-hm['relative_bad_recovered20'],hm['relative_mm'],m['relative_mm'])
            if winner is None or key<winner[0]:winner=(key,arm)
    report=dict(complete=True,approved=winner is not None,selected_arm=winner[1] if winner else None,baseline_v14=metrics(prior,b['base'],gt,valid),arms=results,criterion='Original recovery/harm gates plus no>0.1mm overall camera/relative regression againstv14 and more actual hard-point recovery with lower hard mean; policy unchanged',scope='Dev_calibrate only; no independent adoption claim; fourth batch/old failures are already inspected and cannot be reused as new evidence')
    save(RUN/'development_results.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

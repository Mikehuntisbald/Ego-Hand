"""Separate label-consistency control from extra-RGB context benefit."""
import json,torch
from hand3d_v8_common import V7,save,metrics,score
from hand3d_data_v7 import batch
from adaptive_projection_v14 import apply
from calibrate_hand3d_v8 import paired_ci
RUN=V7.parent/'context_native_v17';DATA=V7.parent/'context_data_v17'

def main():
    policy=json.loads((V7.parent/'side_native_v16/fifth_seal.json').read_text())['policy'];outputs={};reports={};base=gt=valid=rows=None
    for variant in ['control','bridge']:
        path=RUN/variant/'uniform_adaptive';done=json.loads((path/'done.json').read_text());assert done['complete']
        data=torch.load(DATA/variant/'dense_data.pt',weights_only=False,mmap=True);c=torch.load(path/'calibration.pt',weights_only=False);ids=c['indices'];prob=torch.load(DATA/variant/'risk_dense/risk_probabilities.pt',weights_only=False)['joint'];b=batch(data,ids,prob)
        original=data['original_base_for_evaluation'][ids];labels=data['gt'][ids];mask=data['valid'][ids]
        if base is not None:assert torch.equal(base,original) and torch.equal(gt,labels) and torch.equal(valid,mask)
        base=original;gt=labels;valid=mask;rows=[data['rows'][int(i)] for i in ids];pred=apply(c['proposal'],b,policy);outputs[variant]=pred
        hm=valid.clone();hm[:,5]=False;relative=(((base-base[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000*hm).sum(-1)/hm.sum(-1).clamp_min(1);hard=relative>40
        m=metrics(pred,base,gt,valid);_,ok=score(m);ci=paired_ci(pred,base,gt,valid,rows)
        reports[variant]=dict(selected_step=done['selected_step'],final=m,raw=metrics(c['proposal'],base,gt,valid),hard_windows=int(hard.sum()),hard=metrics(pred[hard],base[hard],gt[hard],valid[hard]),paired_vs_original=ci,qualified=ok and ci['relative']['ci95_delta_mm'][1]<0)
    old_data=torch.load(V7.parent/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True);prior=torch.load(V7.parent/'side_native_v16/consensus/uniform_adaptive/calibration.pt',weights_only=False);assert torch.equal(prior['indices'],ids);old_prob=torch.load(V7.parent/'side_data_v16/consensus/risk_dense/risk_probabilities.pt',weights_only=False)['joint'];v16=apply(prior['proposal'],batch(old_data,ids,old_prob),policy)
    pm=metrics(v16,base,gt,valid);ph=metrics(v16[hard],base[hard],gt[hard],valid[hard]);winner=None
    for variant,r in reports.items():
        r['paired_vs_v16']=paired_ci(outputs[variant],v16,gt,valid,rows)
        r['approved_for_independent_test']=r['qualified'] and r['final']['camera_mm']<=pm['camera_mm']+.1 and r['final']['relative_mm']<=pm['relative_mm']+.1 and r['hard']['relative_bad_recovered20']>ph['relative_bad_recovered20'] and r['hard']['relative_mm']<ph['relative_mm']
        if r['approved_for_independent_test']:
            key=(-r['hard']['relative_bad_recovered20'],r['hard']['relative_mm'],r['final']['relative_mm'])
            if winner is None or key<winner[0]:winner=(key,variant)
    comparison=paired_ci(outputs['bridge'],outputs['control'],gt,valid,rows)
    report=dict(complete=True,approved=winner is not None,selected_variant=winner[1] if winner else None,variants=reports,baseline_v16=pm,baseline_v16_hard=ph,bridge_vs_control=comparison,policy=policy,criterion='Original recovery/protection plus no>0.1mm overall regression againstv16, lower hard mean and more actual hard recovery',scope='Development only. Both arms have corrected same-hand supervision and identical centers. Bridge/control isolates addedRGBcontext; control/v16 also includes label-consistency change. New sixthunused batch required before adoption; fifth alreadyread.')
    save(RUN/'development_results.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()

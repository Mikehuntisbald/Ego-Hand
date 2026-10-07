"""Full-pipeline development comparison at identical original centers."""
import json,torch
from hand3d_v8_common import V7,save,metrics,score
from hand3d_data_v7 import batch
from adaptive_projection_v14 import apply
from calibrate_hand3d_v8 import paired_ci
RUN=V7.parent/'side_native_v16';DATA=V7.parent/'side_data_v16'

def main():
    policy=json.loads((V7.parent/'adaptive_projection_v14/dit_dense/fourth_seal.json').read_text())['policy'];outputs={};reports={};base=gt=valid=rows=None
    for variant in ['control','consensus']:
        run=RUN/variant/'uniform_adaptive';done=json.loads((run/'done.json').read_text());assert done['complete']
        data=torch.load(DATA/variant/'dense_data.pt',weights_only=False,mmap=True);c=torch.load(run/'calibration.pt',weights_only=False);ids=c['indices'];prob=torch.load(DATA/variant/'risk_dense/risk_probabilities.pt',weights_only=False)['joint'];b=batch(data,ids,prob)
        original=data['original_base_for_evaluation'][ids];labels=data['gt'][ids];mask=data['valid'][ids]
        if base is not None:assert torch.equal(original,base) and torch.equal(labels,gt) and torch.equal(mask,valid)
        base=original;gt=labels;valid=mask;rows=[data['rows'][int(i)] for i in ids];pred=apply(c['proposal'],b,policy);outputs[variant]=pred
        hmask=valid.clone();hmask[:,5]=False;err=(((base-base[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000*hmask).sum(-1)/hmask.sum(-1).clamp_min(1);hard=err>40
        m=metrics(pred,base,gt,valid);_,ok=score(m);ci=paired_ci(pred,base,gt,valid,rows)
        reports[variant]=dict(selected_step=done['selected_step'],full_pipeline=m,corrected_wilor=metrics(b['base'],base,gt,valid),dit_vs_corrected_wilor=metrics(pred,b['base'],gt,valid),raw=metrics(c['proposal'],base,gt,valid),hard_windows=int(hard.sum()),hard=metrics(pred[hard],base[hard],gt[hard],valid[hard]),paired_vs_original=ci,qualified=ok and ci['relative']['ci95_delta_mm'][1]<0)
    old_data=torch.load(V7.parent/'aligned_density_v13/dense_data.pt',weights_only=False,mmap=True);old_c=torch.load(V7.parent/'native_density_v13/dit_dense/calibration.pt',weights_only=False);assert torch.equal(old_c['indices'],ids);old_prob=torch.load(V7.parent/'aligned_density_v13/risk_dense/risk_probabilities.pt',weights_only=False)['joint'];old_b=batch(old_data,ids,old_prob);v14=apply(old_c['proposal'],old_b,policy)
    comparison=paired_ci(outputs['consensus'],outputs['control'],gt,valid,rows);vs_v14=paired_ci(outputs['consensus'],v14,gt,valid,rows);candidate=reports['consensus'];prior=metrics(v14,base,gt,valid)
    hard_prior=metrics(v14[hard],base[hard],gt[hard],valid[hard]);approved=candidate['qualified'] and candidate['full_pipeline']['camera_mm']<=prior['camera_mm']+.1 and candidate['full_pipeline']['relative_mm']<=prior['relative_mm']+.1 and candidate['hard']['relative_bad_recovered20']>hard_prior['relative_bad_recovered20'] and candidate['hard']['relative_mm']<hard_prior['relative_mm']
    save(RUN/'development_results.json',dict(complete=True,approved=approved,selected_variant='consensus' if approved else None,variants=reports,baseline_v14=prior,baseline_v14_hard=hard_prior,paired_consensus_vs_control=comparison,paired_consensus_vs_v14=vs_v14,policy=policy,criterion='Originalrecovery/protection gates plus no>0.1mm regression againstv14 and actualhardrecoverywithlowerhardmean; allharm/difficultyagainstsameoriginalWiLoR',scope='Train/development only; no independentadoptionclaim; fifthclipmetricsunopened. Currentadapter remainsv14untilindependentandengineeringvalidation.'))
    print(json.dumps(dict(approved=approved,variants=reports,baseline_v14=prior,baseline_v14_hard=hard_prior,paired_consensus_vs_control=comparison,paired_consensus_vs_v14=vs_v14),indent=2),flush=True)

if __name__=='__main__':main()

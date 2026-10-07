"""Read-only error shift and privileged oracle gate diagnosis, not deployment metrics."""
import json,numpy as np,torch
from scipy.stats import spearmanr
from train_coarse_pose import RUN
from metrics_3d import EVAL_INDICES,pose_metrics

def main():
    data=torch.load(RUN/'coarse_cache.pt',map_location='cpu',weights_only=False);report=dict(error_distribution={},oracle_diagnostics={})
    for role in ['coarse','residual','tune','test']:
        ids=[i for i,r in enumerate(data['rows']) if r['role']==role]
        c=data['coarse'][ids].numpy();gt=data['gt'][ids].numpy();confidence=data['confidence'][ids,1:].numpy()
        metric=pose_metrics(c,gt);metric.pop('sample_mpjpe19_mm')
        error=np.linalg.norm((c-c[:,5:6])-(gt-gt[:,5:6]),axis=-1)[:,EVAL_INDICES]
        sigma=-np.log(np.clip(confidence[:,EVAL_INDICES],1e-8,1))*.025
        metric.update(samples=len(ids),mean_predicted_relative_sigma_mm=float(sigma.mean()*1000),
            relative_rms_per_axis_mm=float(np.sqrt(np.mean(error**2)/3)*1000),
            uncertainty_error_rank_correlation=float(spearmanr(sigma.reshape(-1),error.reshape(-1)).statistic))
        report['error_distribution'][role]=metric
    for name in ['dit','regression','temporal_dit','temporal_regression']:
        z=np.load(RUN/f'test_predictions_selective_{name}.npz');c,gt,raw=z['coarse'],z['gt'],z['ungated']
        delta=raw-c;target=gt-c
        alpha=np.clip((target*delta).sum(-1)/(np.square(delta).sum(-1)+1e-12),0,1)
        oracle=c+alpha[...,None]*delta
        error=np.linalg.norm(c-gt,axis=-1);full=np.linalg.norm(raw-gt,axis=-1)
        metric=pose_metrics(oracle,gt);metric.pop('sample_mpjpe19_mm')
        report['oracle_diagnostics'][name]=dict(oracle_gt_dependent_not_a_model_result=metric,
            full_proposal_helpful_fraction=float((full[:,EVAL_INDICES]+.001<error[:,EVAL_INDICES]).mean()),
            proposal_direction_toward_gt_fraction=float(((target*delta).sum(-1)[:,EVAL_INDICES]>0).mean()),
            mean_ungated_delta_mm=float(np.linalg.norm(delta,axis=-1)[:,EVAL_INDICES].mean()*1000),
            note='Oracle uses GT to choose per-point strength; strictly a diagnostic upper bound and cannot be claimed as test performance')
    (RUN/'diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()

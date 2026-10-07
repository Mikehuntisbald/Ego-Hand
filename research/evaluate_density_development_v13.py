import json,itertools
import numpy as np,torch
from hand3d_v8_common import V7,save,metrics,score
from hand3d_data_v7 import batch
from density_model_v13 import POLICY
from native_projection_policy_v11 import apply
from density_candidate_policy_v13 import choose
from calibrate_hand3d_v8 import paired_ci
DATA=V7.parent/'aligned_density_v13';RUN=V7.parent/'native_density_v13'

def main():
    torch.set_num_threads(4);device='cuda:3';reports={};outputs={};sources=None
    for arm in ['dit_sparse','dit_dense','regression_dense']:
        if not (RUN/arm/'done.json').exists():continue
        density=arm.split('_')[-1];raw_data=torch.load(DATA/f'{density}_data.pt',weights_only=False,mmap=True);data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw_data.items()};cache=torch.load(RUN/arm/'calibration.pt',weights_only=False);ids=cache['indices'].to(device);raw=cache['proposal'].to(device);base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];prob=torch.load(DATA/f'risk_{density}/risk_probabilities.pt',weights_only=False)['joint'].to(device);b=batch(data,ids,prob);fallback=apply(raw,base,POLICY);fm=metrics(fallback,base,gt,valid);best=None;trials=[]
        for alpha,pc,pr,temporal in itertools.product([.25,.5,.75,1.],[.7,.8,.9,.95,.975,.99],[.5,.65,.8,.9,.95,.99],[None,0.,5.]):
            policy=dict(alpha=alpha,camera_risk_min=pc,relative_risk_min=pr,temporal_gain_min_mm=temporal);pred,accepted=choose(raw,b,policy);m=metrics(pred,base,gt,valid);_,ok=score(m);trials.append(dict(policy=policy,metrics=m,accepted_hands=int(accepted.sum()),feasible=ok))
            if ok and accepted.any() and m['relative_mm']<fm['relative_mm']-.1 and m['relative_bad_recovered20']>fm['relative_bad_recovered20'] and (best is None or m['relative_mm']<best['metrics']['relative_mm']):
                ci=paired_ci(pred,base,gt,valid,rows)
                if ci['relative']['ci95_delta_mm'][1]<0:best=dict(policy=policy,metrics=m,accepted_hands=int(accepted.sum()),paired_ci=ci)
        if best is not None:final,accepted=choose(raw,b,best['policy'])
        else:final=fallback;accepted=torch.zeros(len(ids),device=device,dtype=torch.bool)
        final_metrics=metrics(final,base,gt,valid);ci=paired_ci(final,base,gt,valid,rows);_,ok=score(final_metrics);ok=ok and ci['relative']['ci95_delta_mm'][1]<0
        reports[arm]=dict(fallback=fm,raw=metrics(raw,base,gt,valid),large_correction_selected=best is not None,selected=best,final=final_metrics,paired_ci=ci,feasible=ok,selected_step=json.loads((RUN/arm/'done.json').read_text())['selected_step'])
        sources=cache['source_window_indices'];outputs[arm]=dict(final=final.cpu(),raw=raw.cpu(),accepted=accepted.cpu(),base=base.cpu(),gt=gt.cpu(),valid=valid.cpu(),rows=rows,source_window_indices=sources)
        save(RUN/arm/'candidate_calibration_trials.json',trials);save(RUN/arm/'development_results.json',reports[arm]);torch.save(outputs[arm],RUN/arm/'development_predictions.pt')
    assert len(reports)==3,'Wait for all matched models to complete before choosing.'
    a=outputs['dit_dense'];b=outputs['dit_sparse'];assert torch.equal(a['source_window_indices'],b['source_window_indices']) and torch.equal(a['base'],b['base']) and torch.equal(a['gt'],b['gt']);density_ci=paired_ci(a['final'],b['final'],a['gt'],a['valid'],a['rows'])
    raw_density_ci=paired_ci(a['raw'],b['raw'],a['gt'],a['valid'],a['rows']);reg=outputs['regression_dense'];reg_ci=paired_ci(a['final'],reg['final'],a['gt'],a['valid'],a['rows'])
    approved=[(arm,rep) for arm,rep in reports.items() if arm.startswith('dit_') and rep['feasible']];selected=min(approved,key=lambda x:(x[1]['final']['relative_mm'],x[1]['final']['camera_mm'])) if approved else None
    out=dict(complete=True,models=reports,dense_minus_sparse_ci=density_ci,raw_dense_minus_sparse_ci=raw_density_ci,dit_minus_regression_ci=reg_ci,selected_arm=selected[0] if selected else None,selected=selected[1] if selected else None,scope='Development only on exactly the original539calibration centers. Gate choices precede any fourth-fresh evaluation. Dense/sparse share predictions/tracks; inference gate uses estimated baseline risk and3D temporal support, never GT.')
    save(RUN/'development_selection.json',out);print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()

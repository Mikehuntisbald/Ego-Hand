"""Development-only paired comparison; no new batch opened on failure."""
import json
import torch
from hand3d_v8_common import V7, save, metrics, score
from hand3d_data_v7 import batch
from adaptive_projection_v14 import apply
from calibrate_hand3d_v8 import paired_ci

RUN = V7.parent / 'canonical_native_v19'
DATA = {'control': V7.parent / 'context_data_v17/control', 'canonical': V7.parent / 'canonical_rgb_v19'}

def main():
    torch.set_num_threads(4)
    policy = json.loads((V7.parent / 'side_native_v16/fifth_seal.json').read_text())['policy']
    outputs, reports = {}, {}
    base = gt = valid = rows = None
    for variant, root in DATA.items():
        path = RUN / variant / 'uniform_adaptive'
        done = json.loads((path / 'done.json').read_text())
        assert done['complete']
        data = torch.load(root / 'dense_data.pt', weights_only=False, mmap=True)
        cal = torch.load(path / 'calibration.pt', weights_only=False)
        ids = cal['indices']
        risk = torch.load(root / 'risk_dense/risk_probabilities.pt', weights_only=False)['joint']
        b = batch(data, ids, risk)
        original, labels, mask = data['original_base_for_evaluation'][ids], data['gt'][ids], data['valid'][ids]
        if base is not None:
            assert torch.equal(base, original) and torch.equal(gt, labels) and torch.equal(valid, mask)
        base, gt, valid = original, labels, mask
        rows = [data['rows'][int(i)] for i in ids]
        pred = apply(cal['proposal'], b, policy)
        outputs[variant] = pred
        nonwrist = valid.clone()
        nonwrist[:, 5] = False
        relative = ((((base-base[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000*nonwrist).sum(-1) / nonwrist.sum(-1).clamp_min(1))
        hard = relative > 40
        m = metrics(pred, base, gt, valid)
        _, ok = score(m)
        ci = paired_ci(pred, base, gt, valid, rows)
        reports[variant] = dict(selected_step=done['selected_step'], final=m,
            raw=metrics(cal['proposal'], base, gt, valid), hard_windows=int(hard.sum()),
            hard=metrics(pred[hard], base[hard], gt[hard], valid[hard]), paired_vs_original=ci,
            qualified=ok and ci['relative']['ci95_delta_mm'][1]<0)
    old_root = V7.parent / 'side_data_v16/consensus'
    old_data = torch.load(old_root / 'dense_data.pt', weights_only=False, mmap=True)
    prior = torch.load(V7.parent / 'side_native_v16/consensus/uniform_adaptive/calibration.pt', weights_only=False)
    assert torch.equal(prior['indices'], ids)
    old_risk = torch.load(old_root / 'risk_dense/risk_probabilities.pt', weights_only=False)['joint']
    baseline = apply(prior['proposal'], batch(old_data, ids, old_risk), policy)
    pm = metrics(baseline, base, gt, valid)
    ph = metrics(baseline[hard], base[hard], gt[hard], valid[hard])
    plan = json.loads((DATA['canonical'] / 'prediction_only_encoding_plan.json').read_text())
    right = torch.tensor(plan['predicted_right'])[data['feature_ids'][ids,8]]
    for variant, r in reports.items():
        r['paired_vs_v16'] = paired_ci(outputs[variant], baseline, gt, valid, rows)
        r['predicted_side_groups'] = {name: metrics(outputs[variant][part],base[part],gt[part],valid[part])
            for name, part in [('left', ~right), ('right', right)]}
        r['approved_for_independent_test'] = (r['qualified'] and
            r['final']['camera_mm'] <= pm['camera_mm']+.1 and
            r['final']['relative_mm'] <= pm['relative_mm']+.1 and
            r['hard']['relative_bad_recovered20'] > ph['relative_bad_recovered20'] and
            r['hard']['relative_mm'] < ph['relative_mm'])
    comparison = paired_ci(outputs['canonical'], outputs['control'], gt, valid, rows)
    # Extra diagnostic only: bypass neither training conditioning nor safety.
    canonical_cal = torch.load(RUN/'canonical/uniform_adaptive/calibration.pt', weights_only=False)
    control_risk = torch.load(DATA['control']/'risk_dense/risk_probabilities.pt', weights_only=False)['joint']
    fixed_risk = apply(canonical_cal['proposal'],batch(data,ids,control_risk),policy)
    prior_control = torch.load(V7.parent/'context_native_v17/control/uniform_adaptive/calibration.pt',weights_only=False)
    replay_delta = float((prior_control['proposal']-torch.load(RUN/'control/uniform_adaptive/calibration.pt',weights_only=False)['proposal']).abs().max()*1000)
    report = dict(complete=True, approved=reports['canonical']['approved_for_independent_test'],
        selected_variant='canonical' if reports['canonical']['approved_for_independent_test'] else None,
        variants=reports, baseline_v16=pm, baseline_v16_hard=ph, canonical_vs_control=comparison,
        canonical_control_risk_projection_diagnostic=metrics(fixed_risk,base,gt,valid),
        repeated_v17_control_max_raw_delta_mm=replay_delta, policy=policy,
        criterion='Original recovery/protection, <=0.1mm overall regression versusv16, lower hard mean and more actual hard recovery',
        scope='Training/development only. Same900updates/seed/initializer/3Dlabels/XYZ/slots/policy; RGBorientation/physicalpositions/risk differ. Existing fifth/old failures excluded. No sixthbatch opened. Predicted side only.')
    save(RUN/'development_results.json',report)
    torch.save(dict(indices=ids, control=outputs['control'],canonical=outputs['canonical']),RUN/'development_predictions.pt')
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':
    main()

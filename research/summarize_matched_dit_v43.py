"""Joint matched-trial summary, with repaired and original references separate."""
import json,hashlib,torch
from hand3d_v8_common import V7,save
from parameter_candidates_v43 import load_observations
from calibrate_hand3d_v8 import paired_ci


def main():
    torch.set_num_threads(4);root=V7.parent;out=root/'matched_dit_review_v43';out.mkdir(exist_ok=True)
    base=root/'matched_stability_v43'
    paths=dict(regression=base/'regression_side_consistent',dit=base/'dit_side_consistent',denoise=base/'dit_denoise_side_consistent')
    summaries={k:json.loads((p/'summary.json').read_text()) for k,p in paths.items()}
    training={k:json.loads((root/'matched_parameter_v43'/k/'done.json').read_text()) for k in ['regression','dit','dit_denoise']}
    hashes={v['batch_plan_sha256'] for v in training.values()};assert len(hashes)==1 and all(v['steps']==4000 and v['complete'] for v in training.values())
    _,ids,allrows,cache=load_observations('cpu');rows=[allrows[i] for i in ids]
    la=torch.load(root/'joint_mano_v28/evaluation_labels.pt',weights_only=False,mmap=True);gt,valid=la['gt'][ids],la['valid'][ids]
    centers=torch.tensor([r['window_index'] is not None for r in rows]);cr=[r for r in rows if r['window_index'] is not None]
    predreg=torch.load(paths['regression']/'mean.pt',weights_only=False)['prediction']
    references=torch.load(paths['regression']/'reference_v42.pt',weights_only=False)['prediction']
    releases=torch.load(root/'stability_first_v42_sidefix/results.pt',weights_only=False)['prediction']
    predictions=dict(released_v42=releases,consistent_v42=references,matched_regression=predreg)
    methods={}
    rel=lambda p:((p-p[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000
    mask=valid.clone();mask[:,5]=False;reg_error=rel(predreg);good=mask&(reg_error<=10);bad=mask&(reg_error>20)
    for kind,run in paths.items():
        s=summaries[kind]
        names=['mean'] if kind=='regression' else ['mean','sequence']
        for variant in names:
            key='matched_regression' if kind=='regression' else ('rollout_'+variant if kind=='dit' else 'denoise_'+variant)
            report_key=('regression_' if kind=='regression' else ('dit_' if kind=='dit' else 'dit_denoise_'))+variant
            prediction=torch.load(run/(variant+'.pt'),weights_only=False)['prediction'];predictions[key]=prediction
            error=rel(prediction)
            methods[key]=dict(report=s['center_reports'][report_key],diagnostics=s['diagnostics'][report_key],coverage=s['coverage'][variant],
                paired_vs_matched_regression=paired_ci(prediction[centers],predreg[centers],gt[centers],valid[centers],cr),
                matched_reg_good_points=int(good.sum()),matched_reg_good_to_bad_points=int((good&(error>20)).sum()),
                matched_reg_bad_points=int(bad.sum()),matched_reg_bad_to_good_points=int((bad&(error<=20)).sum()),
                selected_step=s['step'],inference_seconds=s['inference_seconds'])
    modes=dict(released_v42=json.loads((root/'stability_first_v42_sidefix/summary.json').read_text())['center_reports']['v42'],
               consistent_v42=summaries['regression']['center_reports']['v42'])
    best='rollout_sequence'
    summary=dict(complete=True,date='2026-10-05',best_development_variant=best,default_changed=False,
        methods=methods,references=modes,training=training,matched_batch_plan_sha256=next(iter(hashes)),
        protocol='Same network/native192x1280RGB/GT-freeIK seeds/targets/dataorder/4000steps/optimizer/checkpointscore; kind-specific objective and denoising costs differ. Same v42 bounds and common initial-side reliability repair.',
        scope='2526train windows, 352 previously reused dev_select centerwindows; 2673 observations/2309 same-GT-hand adjacentpairs from4P0003sequences. No fresh subjects or calibration performance.',
        learned_motion_prior='No HMP or external motion prior; dataset-trained conditional parameterDiT and the same kinematic/trajectory regularization',
        candidate_selection='4DDIM draws,10steps; second-order Viterbi uses common frozen WiLoR/RGB and world motion costs; no GT at selection',
        history='Initial new regression had18.25mm generator error but31.20mm aftersolver due seed/head-side inconsistencies. All model trials were rerun with the same reliability repair; originals retained. Releasedv42=18.18mm and same-repairv42=17.98mm are separate references.',
        limitations='Single training/sampling seed, reused subject/sequences, selected/evaluated on development. Stable plausible estimates do not prove hidden-pose truth, collision freedom, cross-ID continuity, or faster-motion preservation.',
        code_sha256={n:hashlib.sha256((__import__('pathlib').Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['parameter_candidates_v43.py','stability_trajectory_v43.py','complete_hand_tracks_dit_v43.py','denoise_parameter_model_v43.py']})
    save(out/'full_summary.json',summary);torch.save(predictions,out/'predictions.pt')
    # Compatibility aliases for the existing four-panel scientific viewer;
    # captions explicitly identify matchedreg/v42/DiT rather than version names.
    selected=methods[best];viewer=dict(summary,
        center_reports=dict(v16=methods['matched_regression']['report'],v39=modes['consistent_v42'],v42=selected['report']),
        all_frame_diagnostic=dict(v42=dict(real_fast_point_pairs=selected['diagnostics']['fast_point_pairs'],
            fast_motion_amplitude_under_half=selected['diagnostics']['fast_under_half_points'],
            fast_motion_amplitude_ratio=selected['diagnostics']['fast_amplitude_ratio'])),
        coverage=dict(observations=2673,continuous_segments=96,constant_pose_segments=selected['coverage']['constant_pose_segments'],
                      pose_scale_at_most_half_segments=selected['coverage']['pose_scale_under_half_segments']),
        accuracy_tradeoff=dict(v16_relative_good_points=selected['matched_reg_good_points'],v16_good_to_v42_over20_points=selected['matched_reg_good_to_bad_points']))
    save(out/'viewer_summary.json',viewer)
    print(json.dumps(dict(best=best,table={k:dict(relative=v['report']['metrics']['relative_mm'],camera=v['report']['metrics']['camera_mm'],fast=v['diagnostics']['fast_amplitude_ratio']['median']) for k,v in methods.items()},paired=methods[best]['paired_vs_matched_regression'])),flush=True)


if __name__=='__main__':main()

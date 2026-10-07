"""Freeze context ablation evidence and verify the experimental RGB entry."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from hand3d_v8_common import V7,save
from temporal_window_v44 import SHORT_OFFSETS,LONG_OFFSETS
from encode_hand3d_dense_v14 import sample
import encode_hand3d_dense_v14 as original_sampling


def main():
    torch.set_num_threads(4);root=V7.parent;out=root/'short_context_review_v44';out.mkdir(exist_ok=True)
    trials={k:json.loads((root/'short_context_v44'/k/'summary.json').read_text()) for k in ['dit','regression']}
    methods={};predictions={}
    for kind,summary in trials.items():
        names=['mean','sequence'] if kind=='dit' else ['mean']
        for span,directory in [('long',root/'matched_stability_v43'/(kind+'_side_consistent')),('short',root/'short_context_v44'/kind)]:
            for name in names:
                key=f'{span}_{kind}_{name}';r=torch.load(directory/(name+'.pt'),weights_only=False)
                predictions[key]=r['prediction']
                methods[key]=dict(label=f'{"±4秒" if span=="long" else "±1.6秒"} {"DiT整段选择" if name=="sequence" else "DiT参数平均" if kind=="dit" else "时序回归"}',
                    report=summary['reports'][span+'_'+name],diagnostics=summary['diagnostics'][span+'_'+name])
                if span=='short':assert summary['constraint_checks'][name]['passed']
    runtime=json.loads((root/'short_context_v44/runtime_short_dit.json').read_text())
    assert runtime['context_s']==1.6 and runtime['context_retrained'] is False
    assert runtime['constraint_checks_passed'] and runtime['raw_xyz_fallback_frames']==0
    count=0
    for track in runtime['tracks']:
        assert not track['manual_anchor_blocked']
        for frame in track['frames']:
            x=torch.tensor(frame['xyz_camera_m']);assert x.shape==(20,3) and torch.isfinite(x).all();count+=1
    assert count==16
    # Runtime sampler tests cover a long real-rate timestamp span and missing
    # intervals, independent of the short RGB smoke clip.
    old=original_sampling.OFFSETS
    try:
        original_sampling.OFFSETS=SHORT_OFFSETS
        times=np.arange(241,dtype=np.float64)/30.;times=np.delete(times,np.arange(70,76))
        n=len(times);z=dict(times=times,available=np.ones((n,20),bool),xy=np.zeros((n,20,2),np.float32),
                          roi=np.ones((n,4),np.float32),positions=np.zeros((n,192,2),np.float32),scores=np.ones(n,np.float32))
        windows,chosen=sample(z,'cpu')
        maxspan=0.
        for i,ids in enumerate(chosen):
            valid=[x for x in ids if x is not None];assert len(valid)==len(set(valid))
            maxspan=max(maxspan,max(abs(times[j]-times[i]) for j in valid));assert maxspan<=1.600002
        assert chosen[len(chosen)//2][8]==len(chosen)//2
    finally:original_sampling.OFFSETS=old
    assert original_sampling.OFFSETS==LONG_OFFSETS
    default=json.loads((Path(__file__).parent/'CURRENT_PIPELINE.json').read_text());assert default['version']=='offline_stability_first_v42'
    freeze=json.loads((root/'side_native_v16/fifth_seal.json').read_text())
    for name,digest in freeze['code_sha256'].items():assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==digest
    for kind,summary in trials.items():
        c=summary['checks'];assert c['long_candidate_parameter_max_difference']==0 and c['long_risk_max_difference']==0
        assert c['generator_gt_poison_exact'] and c['center_fids_exact']
        assert c['short_context']['source_frame_ids_unique'] and c['short_context']['max_observed_context_s']<=1.600003
    assert trials['dit']['checks']['selector_gt_poison_exact']
    verification=dict(passed=True,long_candidates_exact=True,long_risk_exact=True,center_observations_unchanged=True,
        generator_and_selector_gt_poison_exact=True,all_saved_parameters_redecode=True,all_original_v42_limits_passed=True,
        raw_rgb_runtime_frames=count,runtime_context_sampling_max_s=maxspan,runtime_sampling_restored=True,
        sealed_v16_files_unchanged=len(freeze['code_sha256']),default_v42_unchanged=True,
        browser_interaction_checked=False,weights_retrained=False)
    save(out/'verification.json',verification)
    summary=dict(complete=True,experiment='short_context_v44',best_short_variant='short_dit_sequence',default_changed=False,
        context_offsets_frames=SHORT_OFFSETS,methods=methods,trials=trials,
        context_coverage=trials['dit']['checks'],comparisons=trials['dit']['comparisons'],
        protocol='Frozen v43 checkpoints, same sampling seed/draws/steps/center observations/RGB bank/shared heatmaps/hand decoder/trajectory costs and bounds. Change only temporal sampling and recompute the same risk network on its new context.',
        scope='2673 reused development observations,352 centerwindows,4P0003sequences; checkpoints trained with +/-4s and not retrained. No new subject or true invisible-finger guarantee.')
    save(out/'full_summary.json',summary);torch.save(predictions,out/'predictions.pt')
    best=methods['short_dit_sequence'];saved=torch.load(root/'short_context_v44/dit/sequence.pt',weights_only=False)
    restoration=saved['restoration'];comparison=trials['dit']['comparisons']['sequence']
    viewer=dict(summary,center_reports=dict(v16=methods['long_dit_sequence']['report'],v39=methods['short_dit_mean']['report'],v42=best['report']),
        all_frame_diagnostic=dict(v42=dict(real_fast_point_pairs=best['diagnostics']['fast_point_pairs'],
            fast_motion_amplitude_under_half=best['diagnostics']['fast_under_half_points'],
            fast_motion_amplitude_ratio=best['diagnostics']['fast_amplitude_ratio'])),
        coverage=dict(observations=2673,continuous_segments=len(restoration),constant_pose_segments=sum(x['constant_pose_fallback'] for x in restoration),
                      pose_scale_at_most_half_segments=sum(x['pose_scale']<=.5 for x in restoration)),
        accuracy_tradeoff=dict(v16_relative_good_points=comparison['long_good_points'],v16_good_to_v42_over20_points=comparison['long_good_to_short_over20']))
    save(out/'viewer_summary.json',viewer);save(out/'sequence_constraint_check.json',trials['dit']['constraint_checks']['sequence'])
    save(out/'protocol.json',dict(description=summary['protocol'],scope=summary['scope'],
        offsets_frames=SHORT_OFFSETS,weights_retrained=False,default_changed=False,
        limits=trials['dit']['constraint_checks']['sequence']['limits']))
    shutil.copyfile(root/'short_context_v44/runtime_short_dit.json',out/'runtime_short_dit.json')
    for kind in trials:shutil.copyfile(root/'short_context_v44'/kind/'summary.json',out/(kind+'_summary.json'))
    rows=json.loads((root/'short_context_v44/dit/rows.json').read_text())
    with (out/'dit_predictions.jsonl').open('w') as f:
        for r,x in zip(rows,saved['prediction']):
            f.write(json.dumps(dict(sequence=r['sequence'],clip=r['clip'],track_id=r['track_id'],frame=r['frame'],timestamp_ns=r['timestamp_ns'],
                image=r['image'],box_xyxy=r['box'],xyz_camera_m=x.tolist(),review_required=True,context_s=1.6))+'\n')
    print(json.dumps(dict(verification=verification,table={k:dict(relative=v['report']['metrics']['relative_mm'],camera=v['report']['metrics']['camera_mm'],fast=v['diagnostics']['fast_amplitude_ratio']['median']) for k,v in methods.items()},coverage=summary['context_coverage'])),flush=True)


if __name__=='__main__':main()

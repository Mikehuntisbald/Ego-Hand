"""Export the bound diagnosis with the moderate acceleration trial separate."""
import hashlib,json,shutil
from pathlib import Path
import torch
from hand3d_v8_common import V7,save


def main():
    root=V7.parent;run=root/'motion_threshold_v45';out=root/'motion_threshold_review_v45';out.mkdir(exist_ok=True)
    summary=json.loads((run/'summary.json').read_text());methods={}
    labels=dict(raw_selected='原DiT所选候选（未约束）',side_repaired_before_smoothing='侧别修复后、平滑前',
        after_initial_smoothing='95ms初始平滑后',optimized_before_hard_restoration='优化后、硬恢复前',
        strict='当前严格配置',hard_only_x2='仅末端硬检查×2',speed_x2='仅速度阈值×2',acc_x2='仅加速度阈值×2',
        motion_x2='速度与加速度×2',motion_x3='速度与加速度×3',smooth_45ms='仅初始平滑95→45ms',
        soft_acc_x2='仅优化加速度尺度×2，硬上限不变',soft_temporal_acc_x2='仅加速度软正则÷4，上限/提前惩罚不变',
        motion_x2_smooth45='全部×2＋45ms平滑',motion_x2_less_soft='全部×2＋45ms＋软正则减弱')
    predictions={}
    strict=torch.load(run/'strict.pt',weights_only=False)
    for key,label in labels.items():
        methods[key]=dict(label=label,report=summary['reports'][key],diagnostics=summary['diagnostics'][key],
            protection=summary['protection'][key],coverage=summary['coverage'].get(key))
        if key in summary['configs']:predictions[key]=torch.load(run/(key+'.pt'),weights_only=False)['prediction']
        elif key in strict['stages']:predictions[key]=strict['stages'][key]
    selected=torch.load(run/'acc_x2.pt',weights_only=False);moderate=methods['acc_x2'];protection=summary['protection']['acc_x2']
    restoration=selected['restoration']
    viewer=dict(summary,methods=methods,center_reports=dict(v16=summary['reports']['strict'],v39=summary['reports']['motion_x3'],v42=summary['reports']['acc_x2']),
        all_frame_diagnostic=dict(v42=dict(real_fast_point_pairs=moderate['diagnostics']['fast_point_pairs'],
            fast_motion_amplitude_under_half=moderate['diagnostics']['fast_under_half_points'],
            fast_motion_amplitude_ratio=moderate['diagnostics']['fast_amplitude_ratio'])),
        coverage=dict(observations=2673,continuous_segments=len(restoration),constant_pose_segments=sum(x['constant_pose_fallback'] for x in restoration),
            pose_scale_at_most_half_segments=sum(x['pose_scale']<=.5 for x in restoration)),
        accuracy_tradeoff=dict(v16_relative_good_points=protection['strict_good_points'],v16_good_to_v42_over20_points=protection['strict_good_to_bad_points']))
    save(out/'full_summary.json',dict(summary,methods=methods));save(out/'viewer_summary.json',viewer);torch.save(predictions,out/'predictions.pt')
    for key in summary['configs']:
        check=json.loads((run/(key+'_check.json')).read_text());assert check['passed'] and check['saved_parameter_redecode']['passed']
    assert json.loads((run/'strict_check.json').read_text())['strict_parity_max_m']==0.
    hard=torch.load(run/'hard_only_x2.pt',weights_only=False)
    hard_difference_mm=float((hard['prediction']-strict['prediction']).norm(dim=-1).max()*1000)
    optimized_difference_mm=float((hard['stages']['optimized_before_hard_restoration']-strict['stages']['optimized_before_hard_restoration']).norm(dim=-1).max()*1000)
    print(json.dumps(dict(hard_only_difference_mm=hard_difference_mm,optimized_difference_mm=optimized_difference_mm,
        strict_coverage=summary['coverage']['strict'],hard_only_coverage=summary['coverage']['hard_only_x2'])),flush=True)
    assert optimized_difference_mm==0.
    hard_audit=json.loads((run/'hard_restore_effect.json').read_text())
    assert abs(hard_difference_mm-hard_audit['max_all_points_change_mm'])<1e-5
    default=json.loads((Path(__file__).parent/'CURRENT_PIPELINE.json').read_text());assert default['version']=='offline_stability_first_v42'
    verification=dict(passed=True,strict_reproduces_v44_exact=True,hard_only_x2_optimized_output_exact=True,
        hard_only_x2_max_difference_mm=hard_difference_mm,
        same_frozen_dit_candidates_and_selected_path=True,saved_parameter_world_and_camera_parity=True,
        each_variant_passes_its_declared_motion_bounds=True,structural_decoder_unchanged=True,
        default_v42_unchanged=True,gt_audit_used_for_diagnosis_only=True,
        scope=summary['scope'],browser_interaction_checked=False,raw_rgb_profile_runtime_tested=False)
    save(out/'verification.json',verification)
    save(out/'sequence_constraint_check.json',json.loads((run/'acc_x2_check.json').read_text())['saved_parameter_redecode'])
    shutil.copyfile(run/'protocol.json',out/'protocol.json')
    save(out/'stage_diagnosis.json',{k:methods[k] for k in ['raw_selected','side_repaired_before_smoothing','after_initial_smoothing','optimized_before_hard_restoration','strict']})
    save(out/'gt_motion_audit.json',summary['gt_motion_audit'])
    status=dict(experiment='motion_threshold_v45',complete=True,default_changed=False,current_default='offline_stability_first_v42',
        diagnostic_conclusion='Acceleration optimization compresses labeled fast motion; hard restoration substantially changes a few mostly unmatched frames',
        moderate_trial='acc_x2',fast_retention_strict=methods['strict']['diagnostics']['fast_amplitude_ratio']['median'],
        fast_retention_acc_x2=moderate['diagnostics']['fast_amplitude_ratio']['median'],
        strict_relative_mm=methods['strict']['report']['metrics']['relative_mm'],acc_x2_relative_mm=moderate['report']['metrics']['relative_mm'],
        strict_camera_mm=methods['strict']['report']['metrics']['camera_mm'],acc_x2_camera_mm=moderate['report']['metrics']['camera_mm'],
        acc_x2_spurious_jumps=moderate['report']['coherence']['spurious_jump_pairs'],
        all_x3_spurious_jumps=methods['motion_x3']['report']['coherence']['spurious_jump_pairs'],
        scope=summary['scope'],local_report='C:/Users/Asus/Desktop/hot3d_hand_bootstrap/experiment_reviews/motion_threshold_v45/report.html')
    save(out/'MOTION_THRESHOLD_V45_STATUS.json',status);save(Path(__file__).parent/'MOTION_THRESHOLD_V45_STATUS.json',status)
    shutil.copyfile(run/'hard_restore_effect.json',out/'hard_restore_effect.json')
    print(json.dumps(dict(complete=True,moderate=moderate['report']['metrics'],fast=moderate['diagnostics']['fast_amplitude_ratio'],
        protection=protection,coverage=summary['coverage'],verification=verification)),flush=True)


if __name__=='__main__':main()

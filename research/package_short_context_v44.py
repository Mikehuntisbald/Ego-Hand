import hashlib,json,shutil,zipfile
from pathlib import Path
from hand3d_v8_common import V7,save

root=V7.parent;out=root/'short_context_review_v44';here=Path(__file__).parent
checks=json.loads((out/'offline_checks.json').read_text());assert checks['passed']
summary=json.loads((out/'full_summary.json').read_text())
files=['temporal_window_v44.py','evaluate_short_context_v44.py','complete_hand_tracks_dit_v44.py',
       'summarize_short_context_v44.py','build_short_context_report_v44.py','verify_short_context_report_v44.py',
       'short_context_compare_v44.html','SHORT_CONTEXT_V44_README.txt']
code=out/'code_snapshot';code.mkdir(exist_ok=True)
for name in files:shutil.copyfile(here/name,code/name)
shutil.copyfile(here/'SHORT_CONTEXT_V44_README.txt',out/'SHORT_CONTEXT_V44_README.txt')
best=summary['methods']['short_dit_sequence'];co=best['report']['coherence']
status=dict(experiment='short_context_v44',date='2026-10-05',complete=True,weights_retrained=False,
    checkpoint=summary['trials']['dit']['checkpoint'],checkpoint_sha256=summary['trials']['dit']['checkpoint_sha256'],
    context_s=1.6,offsets_frames=summary['context_offsets_frames'],default_changed=False,current_default='offline_stability_first_v42',
    relative_mm=best['report']['metrics']['relative_mm'],camera_mm=best['report']['metrics']['camera_mm'],
    fast_motion_median_retention=best['diagnostics']['fast_amplitude_ratio']['median'],
    spurious_jump_pairs=co['spurious_jump_pairs'],bone_extreme_frames=co['bone_extreme_frames'],bone_flicker_events=co['bone_flicker_gt_under1_pred_over10'],
    hard_recovered=summary['comparisons']['sequence']['hard_recovered_short'],hard_bad_points=summary['comparisons']['sequence']['hard_bad_points'],
    real_rgb_runtime_frames=16,observations=2673,center_windows=352,
    local_report='C:/Users/Asus/Desktop/hot3d_hand_bootstrap/experiment_reviews/short_context_v44/report.html',
    remote_entry='/mnt/why/hot3d_hand_residual/complete_hand_tracks_dit_v44.py',
    validation_scope=summary['scope'])
save(out/'SHORT_CONTEXT_V44_STATUS.json',status);save(here/'SHORT_CONTEXT_V44_STATUS.json',status)
save(out/'release.json',dict(complete=True,status=status,report_checks=checks,
    verification=json.loads((out/'verification.json').read_text()),
    code_sha256={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in files},
    predictions_sha256=hashlib.sha256((out/'predictions.pt').read_bytes()).hexdigest(),
    case_manifest_sha256=hashlib.sha256((out/'case_manifest.json').read_bytes()).hexdigest()))
with zipfile.ZipFile(root/'short_context_review_v44.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for path in out.rglob('*'):
        if path.is_file():archive.write(path,str(path.relative_to(out)))
print(json.dumps(dict(released=True,report_cases=checks['cases'],report_frames=checks['frames'],observations=2673,default_unchanged=True)))

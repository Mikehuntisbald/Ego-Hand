import hashlib,json,shutil,zipfile
from pathlib import Path
from hand3d_v8_common import V7,save

root=V7.parent;out=root/'motion_threshold_review_v45';here=Path(__file__).parent
check=json.loads((out/'offline_checks.json').read_text());assert check['passed']
verification=json.loads((out/'verification.json').read_text());assert verification['passed']
default=json.loads((here/'CURRENT_PIPELINE.json').read_text());assert default['version']=='offline_stability_first_v42'
previous=json.loads((root/'matched_dit_review_v43/release.json').read_text())['code_sha256']
preserved=['stability_trajectory_v43.py','parameter_candidates_v43.py','complete_hand_tracks_dit_v43.py','denoise_parameter_model_v43.py']
for name in preserved:assert hashlib.sha256((here/name).read_bytes()).hexdigest()==previous[name]
files=['motion_threshold_solver_v45.py','evaluate_motion_thresholds_v45.py','audit_hard_restore_effect_v45.py',
    'summarize_motion_thresholds_v45.py','build_motion_threshold_report_v45.py','verify_motion_threshold_report_v45.py',
    'motion_threshold_compare_v45.html','MOTION_THRESHOLD_V45_README.txt']
code=out/'code_snapshot';code.mkdir(exist_ok=True)
for name in files:shutil.copyfile(here/name,code/name)
shutil.copyfile(here/'MOTION_THRESHOLD_V45_README.txt',out/'MOTION_THRESHOLD_V45_README.txt')
save(out/'release.json',dict(complete=True,default_unchanged=True,original_v43_files_unchanged=preserved,
    report_checks=check,verification=verification,
    code_sha256={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in files},
    predictions_sha256=hashlib.sha256((out/'predictions.pt').read_bytes()).hexdigest(),
    status=json.loads((out/'MOTION_THRESHOLD_V45_STATUS.json').read_text())))
with zipfile.ZipFile(root/'motion_threshold_review_v45.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob('*'):
        if p.is_file():z.write(p,str(p.relative_to(out)))
print(json.dumps(dict(released=True,cases=check['cases'],frames=check['frames'],default_unchanged=True)))

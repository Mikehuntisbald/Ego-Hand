import json,subprocess,shutil
from pathlib import Path
from hand3d_v8_common import V7,save
OLD=V7.parent/'anchor_native_v14';RUN=V7.parent/'anchor_native_v14_closed';RUN.mkdir(exist_ok=True)
for blend in [.5,1.]:
    source=OLD/f'blend{blend:g}';done=json.loads((source/'done.json').read_text());assert done['selected_step']>1200
    dest=RUN/f'blend{blend:g}';dest.mkdir(exist_ok=True)
    for name in ['best.pt','calibration.pt','config.json','history.json','gradient_check.json','done.json']:shutil.copy2(source/name,dest/name)
    save(dest/'closure_lineage.json',dict(real_sampling_trained_selected=True,source=str(source),selected_step=done['selected_step'],reuse='Original global best is also minimum among post-sampling-stage checkpoints; no need to retrain these arms'))
if (RUN/'control_job.json').exists():
    previous=json.loads((RUN/'control_job.json').read_text());p=Path(f"/proc/{previous['pid']}/cmdline")
    if p.exists() and b'train_anchor_native_v14.py' in p.read_bytes():print('Closed control already running');raise SystemExit(0)
log=(RUN/'blend0.log').open('a');p=subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python','train_anchor_native_v14.py','--blend','0.0','--device','cuda:0'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);save(RUN/'control_job.json',dict(pid=p.pid,reason='Previous control best was denoising-only600; exact same seeds/updates rerun, checkpoint eligibility now requires complete sampler training'))
print(json.dumps(dict(pid=p.pid)))

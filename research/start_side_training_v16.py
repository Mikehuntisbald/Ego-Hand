import json,subprocess
from pathlib import Path
from hand3d_v8_common import V7,save
RUN=V7.parent/'side_data_v16';RUN.mkdir(exist_ok=True)
for variant in ['control','consensus']:
    for name in ['native_bank.pt','trajectory_targets.pt']:
        link=RUN/variant/name
        if not link.exists():link.symlink_to(V7.parent/'aligned_density_v13'/name)
if (RUN/'training_job.json').exists():
    old=json.loads((RUN/'training_job.json').read_text());proc=Path(f"/proc/{old['pid']}/cmdline")
    if proc.exists() and b'run_side_training_v16.py' in proc.read_bytes():print('Already training');raise SystemExit(0)
log=(RUN/'training_pipeline.log').open('a');p=subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python','run_side_training_v16.py'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
save(RUN/'training_job.json',dict(pid=p.pid,stage='First evaluate upstream ondev, then matched3D training if qualified'))
print(json.dumps(dict(pid=p.pid)))

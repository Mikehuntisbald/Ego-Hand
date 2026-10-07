import json,subprocess
from pathlib import Path
from hand3d_v8_common import V7,save
RUN=V7.parent/'natural_recovery_v15';RUN.mkdir(exist_ok=True)
PY='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python';jobs=[]
for arm,device in [('uniform_adaptive','cuda:0'),('hard_adaptive','cuda:1'),('hard_conservative','cuda:2')]:
    dest=RUN/arm;dest.mkdir(exist_ok=True)
    if (dest/'done.json').exists():continue
    if (dest/'job.json').exists():
        old=json.loads((dest/'job.json').read_text());proc=Path(f"/proc/{old['pid']}/cmdline")
        if proc.exists() and b'train_recovery_v15.py' in proc.read_bytes():jobs.append(old);continue
    log=(RUN/f'{arm}.log').open('a');p=subprocess.Popen([PY,'train_recovery_v15.py','--arm',arm,'--device',device],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    job=dict(arm=arm,device=device,pid=p.pid,steps=900);save(dest/'job.json',job);jobs.append(job)
save(RUN/'jobs.json',jobs);print(json.dumps(jobs),flush=True)

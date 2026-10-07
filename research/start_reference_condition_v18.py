import json,subprocess
from pathlib import Path
from hand3d_v8_common import V7,save
RUN=V7.parent/'reference_condition_v18';assert json.loads((RUN/'preflight.json').read_text())['passed'];jobs=[]
for variant,device in [('control','cuda:0'),('reference','cuda:1')]:
    dest=RUN/variant;dest.mkdir(exist_ok=True)
    if (dest/'uniform_adaptive/done.json').exists():continue
    if (dest/'job.json').exists():
        old=json.loads((dest/'job.json').read_text());proc=Path(f"/proc/{old['pid']}/cmdline")
        if proc.exists() and b'train_reference_condition_v18.py' in proc.read_bytes():jobs.append(old);continue
    log=(RUN/f'{variant}.log').open('a');p=subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python','train_reference_condition_v18.py','--variant',variant,'--device',device],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    job=dict(pid=p.pid,variant=variant,device=device,steps=900);save(dest/'job.json',job);jobs.append(job)
save(RUN/'jobs.json',jobs);print(json.dumps(jobs),flush=True)

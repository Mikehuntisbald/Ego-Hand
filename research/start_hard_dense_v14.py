import json,subprocess
from pathlib import Path
from hand3d_v8_common import V7,save
RUN=V7.parent/'hard_dense_v14';RUN.mkdir(exist_ok=True)
if (RUN/'job.json').exists():
    previous=json.loads((RUN/'job.json').read_text());proc=Path(f"/proc/{previous['pid']}/cmdline")
    if proc.exists() and b'prepare_hard_dense_v14.py' in proc.read_bytes():
        print('Already preparing retained failures');raise SystemExit(0)
log=(RUN/'prepare.log').open('a')
p=subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python','prepare_hard_dense_v14.py'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
save(RUN/'job.json',dict(pid=p.pid,device='cuda:3',stage='All47 previously inspected failures; real30FPS context; diagnostics only'))
print(json.dumps(dict(pid=p.pid)))

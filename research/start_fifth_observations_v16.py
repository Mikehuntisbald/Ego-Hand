import json,subprocess
from pathlib import Path
from hand3d_v8_common import V7,save
RUN=V7.parent/'fifth_dense_v16';RUN.mkdir(exist_ok=True)
if (RUN/'job.json').exists():
    old=json.loads((RUN/'job.json').read_text());proc=Path(f"/proc/{old['pid']}/cmdline")
    if proc.exists() and b'prepare_fifth_observations_v16.py' in proc.read_bytes():print('Already preparing fifth batch');raise SystemExit(0)
log=(RUN/'prepare.log').open('a');p=subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python','prepare_fifth_observations_v16.py'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
save(RUN/'job.json',dict(pid=p.pid,stage='Fifth12unused clips; metadata/observations only; model metrics unopened'))
print(json.dumps(dict(pid=p.pid)))

import json,subprocess
from pathlib import Path
from hand3d_v8_common import V7,save
RUN=V7.parent/'side_data_v16';RUN.mkdir(exist_ok=True)
if (RUN/'job.json').exists():
    old=json.loads((RUN/'job.json').read_text());p=Path(f"/proc/{old['pid']}/cmdline")
    if p.exists() and b'run_side_preparation_v16.py' in p.read_bytes():print('Already preparing');raise SystemExit(0)
log=(RUN/'pipeline.log').open('a');p=subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python','run_side_preparation_v16.py'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
save(RUN/'job.json',dict(pid=p.pid,stage='Reconstruct all changed predicted sides, then independently refit matched risk controls'))
print(json.dumps(dict(pid=p.pid)))

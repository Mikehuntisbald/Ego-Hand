import json,subprocess
from pathlib import Path
from hand3d_v8_common import V7,save
RUN=V7.parent/'context_data_v17';RUN.mkdir(exist_ok=True)
if (RUN/'job.json').exists():
    old=json.loads((RUN/'job.json').read_text());proc=Path(f"/proc/{old['pid']}/cmdline")
    if proc.exists() and b'prepare_context_data_v17.py' in proc.read_bytes():print('Already preparing');raise SystemExit(0)
log=(RUN/'prepare.log').open('a');p=subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python','prepare_context_data_v17.py'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
save(RUN/'job.json',dict(pid=p.pid,stage='Buildmatchedcontexts and same-hand supervision labels; re-fitrisks/3Dtrainingpending'))
print(json.dumps(dict(pid=p.pid)))

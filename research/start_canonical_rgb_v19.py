import json,subprocess
from pathlib import Path
from hand3d_v8_common import V7,save
RUN=V7.parent/'canonical_rgb_v19';RUN.mkdir(exist_ok=True)
if (RUN/'job.json').exists():
    old=json.loads((RUN/'job.json').read_text());proc=Path(f"/proc/{old['pid']}/cmdline")
    if proc.exists() and b'prepare_canonical_rgb_v19.py' in proc.read_bytes():print('Alreadyencoding');raise SystemExit(0)
log=(RUN/'prepare.log').open('a');p=subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python','prepare_canonical_rgb_v19.py'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
save(RUN/'job.json',dict(pid=p.pid,stage='Predicted-side canonicalRGB encoding; physicalray/position mapping; no model adoption'))
print(json.dumps(dict(pid=p.pid)))

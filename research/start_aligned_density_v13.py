import json,subprocess,time
from pathlib import Path
from hand3d_v8_common import V7,save
DATA=V7.parent/'aligned_density_v13';DATA.mkdir(exist_ok=True)
if (DATA/'pipeline_job.json').exists():
    previous=json.loads((DATA/'pipeline_job.json').read_text());path=Path(f"/proc/{previous['pid']}/cmdline")
    if path.exists() and b'run_aligned_density_v13.py' in path.read_bytes():print(json.dumps(dict(already_running=True,**previous)));raise SystemExit(0)
log=(DATA/'pipeline.log').open('a');p=subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python','run_aligned_density_v13.py'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
save(DATA/'pipeline_job.json',dict(pid=p.pid,created_unix=time.time()));print(json.dumps(dict(pid=p.pid)))

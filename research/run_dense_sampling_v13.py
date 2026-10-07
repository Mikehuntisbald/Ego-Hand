import subprocess,json,os
from hand3d_v8_common import V7
RUN=V7.parent/'dense_sampling_v13';RUN.mkdir(exist_ok=True)
if (RUN/'job.json').exists():
    previous=json.loads((RUN/'job.json').read_text());pid=previous['pid']
    path=__import__('pathlib').Path(f'/proc/{pid}/cmdline')
    if path.exists() and b'prepare_dense_sampling_v13.py' in path.read_bytes():
        print(json.dumps(dict(already_running=True,**previous)));raise SystemExit(0)
log=(RUN/'prepare.log').open('a')
p=subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python','prepare_dense_sampling_v13.py'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(RUN/'job.json').write_text(json.dumps(dict(pid=p.pid,device='cuda:3',stage='Real30FPS prediction-only observation preparation'),indent=2));print(json.dumps(dict(pid=p.pid)))

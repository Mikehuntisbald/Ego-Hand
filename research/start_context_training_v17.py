import json,subprocess
from pathlib import Path
from hand3d_v8_common import V7,save
RUN=V7.parent/'context_data_v17'
if (RUN/'training_job.json').exists():
    old=json.loads((RUN/'training_job.json').read_text());proc=Path(f"/proc/{old['pid']}/cmdline")
    if proc.exists() and b'run_context_training_v17.py' in proc.read_bytes():print('Already training');raise SystemExit(0)
log=(RUN/'training_pipeline.log').open('a');p=subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python','run_context_training_v17.py'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
save(RUN/'training_job.json',dict(pid=p.pid,stage='Re-fitOOFrisks then900stepmatched3Dcontrols;v16remainsaccepted'));print(json.dumps(dict(pid=p.pid)))

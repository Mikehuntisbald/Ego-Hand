"""Small read-only status export; no torch/CUDA initialization."""
import hashlib,json,time
from pathlib import Path
RUN=Path('/mnt/why/HOT3D/experiments/online_rgb_iterative_v48')
def read(path):return json.loads(path.read_text()) if path.exists() else None
value=dict(exported_at=time.time(),remote_run=str(RUN),controller=read(RUN/'controller_status.json'),
    teacher=read(RUN/'teacher_ready.json'),code_hashes=read(RUN/'code_hashes.json'),cpu_loss_check=read(RUN/'cpu_loss_check.json'),arms={})
code=Path('/mnt/why/hot3d_hand_residual')
value['live_source_changes']=[k for k,v in (value['code_hashes'] or {}).items() if not (code/k).exists() or hashlib.sha256((code/k).read_bytes()).hexdigest()!=v]
if value['controller']:
    pid=value['controller']['pid'];proc=Path(f'/proc/{pid}/cmdline')
    value['controller_alive']=proc.exists() and b'run_iterative_v48.py' in proc.read_bytes()
for arm in ['frozen','joint','protected']:
    folder=RUN/arm;entry={key:read(folder/f'{key}.json') for key in ['config','progress','preflight','resume_verification','done']}
    history=read(folder/'history.json');entry['history']=history or []
    entry['last_validation']=history[-1] if history else None;value['arms'][arm]=entry
value['summary_ready']=(RUN/'summary.json').exists()
if value['summary_ready']:value['summary']=read(RUN/'summary.json')
print(json.dumps(value,ensure_ascii=False))

"""Archive failed preflight evidence before fixing a not-yet-trained run."""
import datetime,json,shutil
from pathlib import Path
RUN=Path('/mnt/why/HOT3D/experiments/online_rgb_iterative_v48')
state=json.loads((RUN/'controller_status.json').read_text());assert state['stage']=='needs_attention'
pid=state['pid'];cmd=Path(f'/proc/{pid}/cmdline')
assert not cmd.exists() or b'run_iterative_v48.py' not in cmd.read_bytes(),'Controller is active'
assert not (RUN/'protected/resume.pt').exists(),'This repair is preflight-only'
archive=RUN/('preflight_failure_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
assert archive.resolve().is_relative_to(RUN.resolve());archive.mkdir()
for name in ['code_snapshot','code_hashes.json']:
    path=RUN/name
    if path.exists():shutil.move(str(path),str(archive/name))
for name in ['controller_status.json','controller.log','protected_preflight.log']:
    if (RUN/name).exists():shutil.copy2(RUN/name,archive/name)
print(json.dumps(dict(archived=str(archive),training_steps=0)))

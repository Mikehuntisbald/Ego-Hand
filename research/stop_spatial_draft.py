"""Stop only this experiment's draft jobs before correcting its input scope."""
import os,signal,json
from pathlib import Path
names={'run_spatial_stage2.py','train_spatial_temporal.py'}
found=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        args=(p/'cmdline').read_bytes().split(b'\0')
        if any(Path(x.decode(errors='replace')).name in names for x in args):
            found.append(int(p.name))
    except (OSError,ValueError):pass
for pid in found:
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
Path('/mnt/why/HOT3D/experiments/offline_keypoint_rgb_v3/superseded.json').write_text(json.dumps(dict(reason='Restrict perspective crop to original supplied ROI; prevent extra image context in comparison',draft_pids_stopped=found)))
print(json.dumps(dict(stopped=found)))

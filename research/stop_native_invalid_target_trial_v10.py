import json,os,signal,time,shutil
from pathlib import Path
root=Path('/mnt/why/HOT3D/experiments/offline_hand3d_v10_native');archive=root.with_name('offline_hand3d_v10_native_invalid_target_trial')
assert root.exists() and not archive.exists();jobs=json.loads((root/'processes.json').read_text());stopped=[]
for job in jobs:
    pid=job['pid'];proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        command=proc.read_bytes();assert b'train_hand3d_native_v10.py' in command,(pid,command[:100]);os.kill(pid,signal.SIGTERM);stopped.append(pid)
(root/'stopped_reason.json').write_text(json.dumps(dict(stopped=stopped,reason='Unlabeled trajectory targets affected diffusion training latents; retain this exploratory trial and restart after masking targets',time=time.time()),indent=2))
for name in ['hand3d_trajectory_v9.py','hand3d_trajectory_data_v9.py','hand3d_native_v10.py','train_hand3d_native_v10.py']:shutil.copy2(Path(__file__).parent/name,root/('code_'+name))
root.rename(archive);print(json.dumps(dict(archived=str(archive),stopped=stopped)),flush=True)

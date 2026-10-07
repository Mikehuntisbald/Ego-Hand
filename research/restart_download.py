"""Restart only this subset downloader; verified completed archives are reused."""
import json, os, signal, subprocess, time
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D');script='/mnt/why/HOT3D-toolkit/download_rgb_subset.py'
found=[]
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit():continue
    try:args=(directory/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if script.encode() in args:
        pid=int(directory.name);found.append(pid);os.kill(pid,signal.SIGTERM)
for pid in found:
    for _ in range(50):
        if not Path(f'/proc/{pid}').exists():break
        time.sleep(.1)
    else:raise RuntimeError(f'Downloader {pid} has not stopped')
with (ROOT/'download.log').open('a') as output:
    process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',script,'--workers','96'],
         stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(stopped_downloader_pids=found,pid=process.pid,workers=96,reused_completed_archives=True)
(ROOT/'provenance/download_restart.json').write_text(json.dumps(result,indent=2));print(result)

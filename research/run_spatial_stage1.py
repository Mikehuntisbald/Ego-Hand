"""Run the finite, staged cache/probe experiment, with checked exit codes."""
import json,subprocess,sys,time,hashlib
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D/experiments/offline_keypoint_spatial_v3')
CODE=Path(__file__).resolve().parent
ROOT.mkdir(exist_ok=True)

def run_stage(jobs):
    running=[]
    for name,args in jobs:
        log=(ROOT/f'{name}.log').open('a')
        p=subprocess.Popen([sys.executable,*args],cwd=CODE,stdout=log,stderr=subprocess.STDOUT)
        running.append((name,p,log))
    failures=[]
    for name,p,log in running:
        code=p.wait();log.close()
        if code:failures.append((name,code))
    if failures:raise RuntimeError(failures)

run_stage([(f'cache_{i}',['cache_spatial_rgb.py','--device',f'cuda:{gpu}','--shard',str(i),'--shards','3']) for i,gpu in enumerate([0,2,3])])
run_stage([('probe_rgb',['train_spatial_probe.py','--device','cuda:0']),('probe_geometry',['train_spatial_probe.py','--device','cuda:2','--geometry-only'])])
print(json.dumps(dict(stage1_complete=True)),flush=True)

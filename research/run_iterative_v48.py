"""Detached, single-instance GPU queue and resumable v48 train/evaluate controller."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

RUN=Path('/mnt/why/HOT3D/experiments/online_rgb_iterative_v48')
CODE=Path('/mnt/why/hot3d_hand_residual');DEVICE=3

def write(path,obj):
    path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));os.replace(tmp,path)
def alive(pid,needle='run_iterative_v48.py'):
    try:return needle in Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
    except OSError:return False
def occupancy():
    listing=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    gpu=[x.split(',') for x in listing.splitlines() if int(x.split(',')[0])==DEVICE][0]
    uuid=gpu[1].strip();free=int(gpu[2]);processes=[]
    listing=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    for line in listing.splitlines():
        fields=[s.strip() for s in line.split(',')]
        if len(fields)==3 and fields[1]==uuid:
            pid=int(fields[0]);cmd=Path(f'/proc/{pid}/cmdline')
            try:name=cmd.read_bytes().decode(errors='replace').replace('\x00',' ')
            except OSError:name='exited'
            processes.append(dict(pid=pid,memory_MiB=fields[2],command=name))
    return dict(device=DEVICE,free_MiB=free,compute_processes=processes)
def state(stage,**kwargs):
    value=dict(stage=stage,pid=os.getpid(),timestamp=time.time(),device=DEVICE,default_changed=False,**kwargs)
    write(RUN/'controller_status.json',value);return value
def wait_gpu(next_stage):
    while True:
        usage=occupancy()
        if not usage['compute_processes'] and usage['free_MiB']>75000:return
        state('waiting_gpu',next_stage=next_stage,occupancy=usage)
        time.sleep(30)
def invoke(name,args):
    wait_gpu(name);log=RUN/f'{name}.log'
    command=[sys.executable]+args
    with log.open('a') as output:
        child=subprocess.Popen(command,cwd=CODE,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
        state('running',task=name,child_pid=child.pid,command=command,log=str(log))
        result=child.wait()
    if result:
        tail=log.read_text(errors='replace')[-3500:]
        state('needs_attention',task=name,exit_code=result,error_tail=tail,log=str(log))
        raise RuntimeError(f'{name} failed with exit code {result}: {tail[-700:]}')
def snapshot():
    dest=RUN/'code_snapshot';dest.mkdir(exist_ok=True);hashes={}
    names=['run_iterative_v48.py','train_online_parameter_v48.py','protected_parameter_model_v48.py','prepare_teacher_protection_v48.py',
        'infer_online_diagnostic_v48.py','evaluate_online_core_v48.py','online_parameter_model_v47.py',
        'semantic_parameter_model_v36.py','parameter_temporal_model_v31.py','parameter_codec_v31.py',
        'complete_hand_tracks_acceleration_v46.py','stability_trajectory_v43.py','hand3d_v8_common.py']
    for name in names:
        src=CODE/name;out=dest/name;digest=hashlib.sha256(src.read_bytes()).hexdigest()
        if out.exists():assert hashlib.sha256(out.read_bytes()).hexdigest()==digest,f'Frozen source changed: {name}'
        else:shutil.copy2(src,out)
        hashes[name]=digest
    write(RUN/'code_hashes.json',hashes)
def worker():
    lock=(RUN/'controller.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('Existing controller owns the run');return
    try:
        snapshot()
        if not (RUN/'protected/preflight.json').exists():
            invoke('protected_preflight',['train_online_parameter_v48.py','--arm','protected','--preflight-only','--resume'])
        for arm in ['protected','frozen','joint']:
            if not (RUN/arm/'done.json').exists():invoke(f'{arm}_training',['train_online_parameter_v48.py','--arm',arm,'--resume'])
        # Freeze checkpoints before any replay-label evaluation. Keep last weights diagnostic.
        for arm in ['initial','frozen_last','joint_last','protected_last','protected_best']:
            if not (RUN/'diagnostic_v46'/arm/'sealed.json').exists():invoke(f'{arm}_inference',['infer_online_diagnostic_v48.py','--mode',arm])
        if not (RUN/'summary.json').exists():invoke('evaluation',['evaluate_online_core_v48.py'])
        state('complete',training_complete=True,evaluation_complete=True,review=str(RUN/'review/report.html'))
    except Exception:
        current=json.loads((RUN/'controller_status.json').read_text()) if (RUN/'controller_status.json').exists() else {}
        if current.get('stage')!='needs_attention':state('needs_attention',error='Controller failed; inspect controller.log')
        raise
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',action='store_true');ap.add_argument('--worker',action='store_true');ap.add_argument('--status',action='store_true');a=ap.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    if a.worker:worker();return
    if a.status:
        value=json.loads((RUN/'controller_status.json').read_text()) if (RUN/'controller_status.json').exists() else dict(stage='not_started')
        value['controller_alive']=alive(value.get('pid',-1));print(json.dumps(value,indent=2));return
    assert a.start
    lock=(RUN/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    value=json.loads((RUN/'controller_status.json').read_text()) if (RUN/'controller_status.json').exists() else {}
    if alive(value.get('pid',-1)) or value.get('stage')=='complete':print(json.dumps(value));return
    with (RUN/'controller.log').open('a') as output:
        child=subprocess.Popen([sys.executable,str(CODE/'run_iterative_v48.py'),'--worker'],cwd=CODE,
            stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        write(RUN/'controller_status.json',dict(stage='starting',pid=child.pid,timestamp=time.time(),device=DEVICE,default_changed=False))
    print(json.dumps(dict(started=True,pid=child.pid,ssh_disconnect_safe=True)))

if __name__=='__main__':main()

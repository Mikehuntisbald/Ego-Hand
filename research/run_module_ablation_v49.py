"""Single finite GPU3 queue. No recurring automation, no deployment changes."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D/experiments/module_ablation_v49_20261007')
CODE=Path('/mnt/why/hot3d_hand_residual')
ORDER=['full','wilor','initial_full','joint_full','frozen_full','raw_mean','raw_sequence','mean_solver','first_solver',
       'independent_solver','no_rgb_aux','uniform_trust','no_soft_motion','no_hard_motion','no_solver_motion','strict_acc',
       'no_rgb_input','center_only','pooled_rgb']
def write(value):
    ROOT.mkdir(parents=True,exist_ok=True);p=ROOT/'controller_status.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(pid=os.getpid(),timestamp=time.time(),default_changed=False,recurring_automation=False,**value),indent=2));os.replace(t,p)
def alive(pid):
    try:return b'run_module_ablation_v49.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:return False
def idle():
    g=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    fields=[x.split(',') for x in g.splitlines() if x.split(',')[0].strip()=='3'][0]
    p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    owners=[x for x in p.splitlines() if fields[1].strip() in x]
    return not owners and int(fields[2])>75000,owners
def invoke(task,args,gpu=True):
    if gpu:
        while True:
            ok,owners=idle()
            if ok:break
            write(dict(stage='waiting_gpu',task=task,owners=owners));time.sleep(30)
    log=ROOT/(task+'.log');cmd=[sys.executable,str(ROOT/'code_snapshot/ablate_modules_v49.py')]+args
    with log.open('a') as f:
        p=subprocess.Popen(cmd,cwd=ROOT/'code_snapshot',stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        write(dict(stage='running',task=task,child_pid=p.pid,completed=[n for n in ORDER if (ROOT/n/'sealed.json').exists()]));rc=p.wait()
    if rc:
        write(dict(stage='needs_attention',task=task,exit_code=rc,error_tail=log.read_text(errors='replace')[-3500:]));raise RuntimeError(task)
def worker():
    lock=(ROOT/'controller.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    for name in ORDER:
        if not (ROOT/name/'sealed.json').exists():invoke(name,['--variant',name],gpu=name not in ['full','wilor','initial_full','joint_full','frozen_full'])
    if not (ROOT/'summary.json').exists():invoke('evaluation',['--evaluate'],gpu=False)
    write(dict(stage='complete',completed=ORDER,review=str(ROOT/'review/report.html')))
def main():
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    if a.worker:worker();return
    assert a.start
    lock=(ROOT/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    status=json.loads((ROOT/'controller_status.json').read_text()) if (ROOT/'controller_status.json').exists() else {}
    if alive(status.get('pid',-1)) or status.get('stage')=='complete':print(json.dumps(status));return
    subprocess.run([sys.executable,str(CODE/'ablate_modules_v49.py'),'--prepare'],cwd=CODE,check=True)
    with (ROOT/'controller.log').open('a') as f:
        child=subprocess.Popen([sys.executable,str(ROOT/'code_snapshot/run_module_ablation_v49.py'),'--worker'],cwd=ROOT/'code_snapshot',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    ROOT.joinpath('controller_status.json').write_text(json.dumps(dict(stage='starting',pid=child.pid,timestamp=time.time())))
    print(json.dumps(dict(started=True,pid=child.pid,finite=True,automation_still_paused=True)))
if __name__=='__main__':main()

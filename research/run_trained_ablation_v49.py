"""Finite same-warmstart train ablations; existing exact controls are reused."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D/experiments/trained_module_ablation_v49_20261007')
FIRST=ROOT.parent/'module_ablation_v49_20261007'
V48=ROOT.parent/'online_rgb_iterative_v48'
CODE=Path('/mnt/why/hot3d_hand_residual')
TRAIN=['frozen_same_start','no_original_protection','no_protection','no_motion_supervision','no_rgb','center_only','pooled_rgb']
ARMS=['full','no_teacher']+TRAIN
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(value):
    p=ROOT/'controller_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(pid=os.getpid(),timestamp=time.time(),finite=True,recurring_automation=False,default_changed=False,**value),indent=2));os.replace(tmp,p)
def alive(pid):
    try:return b'run_trained_ablation_v49.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:return False
def idle():
    g=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    fields=[x.split(',') for x in g.splitlines() if x.split(',')[0].strip()=='3'][0]
    p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    owners=[x for x in p.splitlines() if fields[1].strip() in x];return not owners and int(fields[2])>75000,owners
def invoke(task,file,args,gpu=True):
    while gpu:
        ok,owners=idle()
        if ok:break
        write(dict(stage='waiting_gpu',task=task,owners=owners));time.sleep(30)
    with (ROOT/(task+'.log')).open('a') as f:
        cmd=[sys.executable,str(ROOT/'code_snapshot'/file)]+args
        p=subprocess.Popen(cmd,cwd=ROOT/'code_snapshot',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        write(dict(stage='running',task=task,child_pid=p.pid,command=cmd));rc=p.wait()
    if rc:
        write(dict(stage='needs_attention',task=task,exit_code=rc,error_tail=(ROOT/(task+'.log')).read_text(errors='replace')[-3500:]));raise RuntimeError(task)
def reuse():
    for arm,source in [('full','protected'),('no_teacher','joint')]:
        folder=ROOT/arm;folder.mkdir(exist_ok=True)
        for name in ['last.pt','done.json','history.json','config.json','preflight.json','resume_verification.json','batch_plan.pt']:
            src=V48/source/name;dst=folder/name
            if dst.exists():assert sha(src)==sha(dst)
            else:shutil.copy2(src,dst)
        diag=V48/'diagnostic_v46'/('protected_last' if source=='protected' else 'joint_last')
        for name in ['result.pt','sealed.json']:
            src=diag/name;dst=folder/name
            if dst.exists():assert sha(src)==sha(dst)
            else:shutil.copy2(src,dst)
        (folder/'reuse.json').write_text(json.dumps(dict(source=str(V48/source),exact_same_warmstart=True,reference='existing fixed1800 continuation endpoint, not selected last from replay')))
def pilot_check(arm):
    import torch
    folder=ROOT/arm;initial=torch.load(ROOT.parent/'online_rgb_3d_v47/dit_joint/last.pt',weights_only=False,map_location='cpu')
    ck=torch.load(folder/'resume.pt',weights_only=False,map_location='cpu')['checkpoint'];assert ck['step']>=100
    buffers={k:torch.equal(v,ck['model'][k]) for k,v in initial['model'].items() if k.startswith('codec.')}
    assert all(buffers.values());changes={}
    for i in range(28,32):
        keys=[k for k in initial['visual_tail'] if k.startswith(f'blocks.{i}.')]
        changes[str(i)]=max(float((ck['visual_tail'][k]-initial['visual_tail'][k]).abs().max()) for k in keys)
    norms=max(float((ck['visual_tail'][k]-initial['visual_tail'][k]).abs().max()) for k in initial['visual_tail'] if k.startswith('last_norm.'))
    frozen=arm in ['frozen_same_start','no_rgb']
    assert (all(v==0 for v in changes.values()) and norms==0) if frozen else (all(v>0 for v in changes.values()) and norms>0)
    head=float((ck['model']['semantic_head.weight']-initial['model']['semantic_head.weight']).abs().max());assert head>0
    checks=dict(passed=True,step=ck['step'],visual_tail_max_deltas=changes,norm_max_delta=norms,head_max_delta=head,FK_buffers_unchanged=True,mechanical_only=True)
    (folder/'pilot_verification.json').write_text(json.dumps(checks,indent=2))
def worker():
    lock=(ROOT/'controller.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    while not (FIRST/'summary.json').exists():
        write(dict(stage='waiting_module_inference',dependency=str(FIRST)));time.sleep(30)
    reuse()
    # All arms finish the same 100-step pilot before expanding. Do not prune on accuracy.
    for arm in TRAIN:
        if not (ROOT/arm/'pilot_verification.json').exists():
            args=['--arm',arm,'--steps','1800','--stop-at','100']
            if (ROOT/arm/'resume.pt').exists():args+=['--resume']
            invoke(arm+'_pilot','train_module_v49.py',args);pilot_check(arm)
    for arm in TRAIN:
        if not (ROOT/arm/'done.json').exists():invoke(arm+'_training','train_module_v49.py',['--arm',arm,'--steps','1800','--resume'])
        pilot_check(arm)
    for arm in TRAIN:
        if not (ROOT/arm/'sealed.json').exists():
            invoke(arm+'_inference','infer_module_v49.py',['--mode',arm])
            for name in ['sealed.json','result.pt']:shutil.copy2(ROOT/'diagnostic_v46'/arm/name,ROOT/arm/name)
    if not (ROOT/'summary.json').exists():invoke('evaluation','evaluate_trained_ablation_v49.py',[],gpu=False)
    write(dict(stage='complete',training_complete=True,evaluation_complete=True,review=str(ROOT/'review/report.html')))
def main():
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    if a.worker:worker();return
    assert a.start
    lock=(ROOT/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    old=json.loads((ROOT/'controller_status.json').read_text()) if (ROOT/'controller_status.json').exists() else {}
    if alive(old.get('pid',-1)) or old.get('stage')=='complete':print(json.dumps(old));return
    subprocess.run([sys.executable,str(CODE/'prepare_trained_ablation_v49.py')],cwd=CODE,check=True)
    with (ROOT/'controller.log').open('a') as f:
        p=subprocess.Popen([sys.executable,str(ROOT/'code_snapshot/run_trained_ablation_v49.py'),'--worker'],cwd=ROOT/'code_snapshot',stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (ROOT/'controller_status.json').write_text(json.dumps(dict(stage='starting',pid=p.pid,timestamp=time.time())))
    print(json.dumps(dict(started=True,pid=p.pid,finite=True,automation_still_paused=True)))
if __name__=='__main__':main()

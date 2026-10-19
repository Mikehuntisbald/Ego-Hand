"""One-shot GPU3 queue for a resumable input cache, not a scheduled task."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/why/native_mano_pool_v7_20261019');V6=Path('/mnt/why/native_mano_shape_v6_20261019')
def save(x):
    p=R/'queue_status.json';q=R/'queue_status.partial.json';q.write_text(json.dumps(x,indent=2));q.replace(p)
def current_full_process():
    target=str(V6/'eval_src/run_full_v6.py')
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if len(args)>1 and args[1].decode(errors='replace')==target:return int(p.name)
    return None
def main():
    R.mkdir(exist_ok=True);(R/'receipts').mkdir(exist_ok=True)
    lock=R/'queue.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        source=R/'src/prepare_pool.py';digest=hashlib.sha256(source.read_bytes()).hexdigest();start=time.time()
        while True:
            ready=V6/'receipts/full_pipeline_done.json'
            if ready.exists() and json.loads(ready.read_text()).get('complete'):break
            live=current_full_process();assert live is not None,'Prior full replay exited without completion; inspect its preserved logs'
            assert time.time()-start<1200,'Bounded prior-run wait expired'
            save(dict(pid=os.getpid(),stage='waiting_for_own_v6_full_replay',verified_live_prior_pid=live,started=start));time.sleep(5)
        while True:
            mem=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            apps=subprocess.check_output(['nvidia-smi','--id=3','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if mem<100 and not apps:break
            assert time.time()-start<1200,'GPU3 remained occupied; preserve queue, do not interrupt projects'
            save(dict(pid=os.getpid(),stage='waiting_for_free_GPU3',memory_MiB=mem,compute_PIDs=apps,started=start));time.sleep(5)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        command=[sys.executable,str(source)]
        if (R/'data/progress.json').exists():command+=['--resume']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH='/mnt/why/native_mano_dit_v2_20261019/src:/mnt/why/public_headhand_h20_20261019/src')
        logpath=R/'receipts'/f'cache_attempt_{int(time.time())}.log'
        with logpath.open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            save(dict(pid=os.getpid(),stage='preparing_native_inputs',child_pid=process.pid,command=command,log=str(logpath),source_sha256=digest,training_started=False))
            code=process.wait()
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,'Input-cache source changed'
        assert code==0,('Cache attempt failed; preserve log and partials',str(logpath),code)
        receipt=json.loads((R/'data/ready.json').read_text());assert receipt['complete']
        save(dict(complete=True,pid=os.getpid(),source_sha256=digest,ready_sha256=hashlib.sha256((R/'data/ready.json').read_bytes()).hexdigest(),observations=receipt['observations'],training_started=False,production_admitted=False))
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()

"""Finish this first milestone through frozen module and full-chain reviews."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/why/native_mano_pool_v7_20261019')
def main():
    start=time.time()
    while True:
        status=json.loads((R/'training_status.json').read_text())
        if status.get('complete'):break
        pid=status['pid'];args=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
        assert len(args)>1 and args[1].decode()==str(R/'src/run_training_v7.py'),'Training controller no longer owns recorded PID'
        assert time.time()-start<1200,'Bounded milestone wait exceeded; inspect actual process'
        time.sleep(3)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=f'{R}/src:{R}/eval_src:/mnt/why/native_mano_dit_v2_20261019/src:/mnt/why/native_mano_rollout_v4_20261019/src:/mnt/why/public_headhand_h20_20261019/src')
    for name in ['verify_training_v7.py','infer_v7.py','evaluate_v7.py','run_full_v7.py','audit_full_native_v7.py']:
        (R/'finish_status.json').write_text(json.dumps(dict(pid=os.getpid(),stage=name,started=time.time()),indent=2))
        with (R/'receipts'/f'{name}.log').open('w') as log:subprocess.run([sys.executable,str(R/'eval_src'/name)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(stage=name,complete=True)),flush=True)
    (R/'finish_status.json').write_text(json.dumps(dict(complete=True,first_milestone_only=True,production_admitted=False),indent=2))
if __name__=='__main__':main()

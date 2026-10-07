"""Finite stage B job, queued behind other projects; no recurring automation."""
import os,json,time,subprocess,fcntl,hashlib
from pathlib import Path
A=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')
B=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53B_20261007')
CODE=Path('/mnt/why/hot3d_hand_residual')
PY='/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007/venv/bin/python'

def free_gpu():
    uuid=subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=uuid','--format=csv,noheader']).decode().strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader']).decode()
    return not any(uuid in v for v in apps.splitlines())
def main():
    B.mkdir(exist_ok=True);lock=(B/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3');phase=None
    def status(name,**kwargs):
        nonlocal phase
        (B/'controller_status.json').write_text(json.dumps(dict(phase=name,pid=os.getpid(),time=time.time(),**kwargs),indent=2))
        if name!=phase:print(name,kwargs,flush=True);phase=name
    def run(name,args):
        status(name)
        with (B/(name+'.log')).open('a') as log:
            child=subprocess.Popen([PY,*map(str,args)],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT)
            status(name,child_pid=child.pid)
            result=child.wait()
        if result:raise RuntimeError(f'{name} exited {result}; retain frozen source/logs and repair under a new version')
    try:
        while not (A/'100doh_done.json').exists():status('waiting_verified_100doh');time.sleep(30)
        if not (B/'dataset_protocol.json').exists():run('prepare',[CODE/'prepare_stageB_v53.py'])
        source=B/'source';source.mkdir(exist_ok=True)
        for name in ['prepare_stageB_v53.py','prepare_stageB_baseline_v53.py','train_stageB_v53.py','rfdetr_dev_selection_stageB_v53.py','rfdetr_partial_supervision_v53_r1.py','run_stageB_v53.py']:
            dest=source/name
            if not dest.exists():dest.write_bytes((CODE/name).read_bytes())
            assert dest.read_bytes()==(CODE/name).read_bytes(),f'Frozen source changed: {name}'
        while not free_gpu():status('waiting_GPU3');time.sleep(30)
        if not (B/'pilot/done.json').exists():run('pilot',[CODE/'train_stageB_v53.py','--pilot'])
        if not (B/'yolo_development_baseline.json').exists():
            while not free_gpu():status('waiting_GPU3');time.sleep(30)
            run('baseline',[CODE/'prepare_stageB_baseline_v53.py'])
        if not (B/'full/done.json').exists():
            while not free_gpu():status('waiting_GPU3');time.sleep(30)
            run('full',[CODE/'train_stageB_v53.py','--epochs','8'])
        status('training_complete_pending_final_review',full_done=str(B/'full/done.json'),default_replaced=False)
    except Exception as error:
        status('failed',error=repr(error));raise

if __name__=='__main__':main()

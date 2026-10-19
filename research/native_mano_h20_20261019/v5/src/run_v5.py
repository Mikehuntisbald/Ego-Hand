"""GPU3-only bounded pilot followed by frozen matched arms."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/why/native_mano_supervision_v5_20261019')
P='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python'

def main():
    lock=R/'controller.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Existing controller lock: inspect ownership before retry')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        # Only this GPU may be used. Never kill another reservation/worker.
        gpu=subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        assert int(gpu.strip())<100,('GPU3 occupied',gpu)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=f'{R}/src:/mnt/why/native_mano_dit_v2_20261019/src:/mnt/why/native_mano_rollout_v4_20261019/src:/mnt/why/public_headhand_h20_20261019/src')
        manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'src').glob('*.py')}
        (R/'receipts/training_source_seal.json').write_text(json.dumps(manifest,indent=2))
        jobs=[('pilot','dit','native',6),('native_regression','regression','native',600),('proxy_dit','dit','proxy',600),('native_dit','dit','native',600),('native_protected','protected','native',600)]
        for name,arm,supervision,steps in jobs:
            folder=R/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
            command=[P,str(R/'src/train_native_v5.py'),'--arm',arm,'--supervision',supervision,'--steps',str(steps),'--rollout','--output',str(folder)]
            if (folder/'last.pt').exists():command+=['--resume']
            status=dict(pid=os.getpid(),stage=name,arm=arm,supervision=supervision,steps=steps,started=time.time(),command=command)
            (R/'controller_status.json').write_text(json.dumps(status,indent=2))
            with (folder/'console.log').open('a') as log:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            assert result.returncode==0,(name,result.returncode)
            done=json.loads((folder/'done.json').read_text());assert done['complete']
            assert all(x>0 for x in done['gradient_max'].values()) and done['FK_buffers_unchanged']
            assert manifest=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'src').glob('*.py')},'Source changed during training'
        plans={json.loads((R/'runs'/name/'done.json').read_text())['batch_plan_sha'] for name,_,_,steps in jobs if steps==600};assert len(plans)==1
        (R/'controller_status.json').write_text(json.dumps(dict(complete=True,stages=[j[0] for j in jobs],batch_plan_sha=plans.pop(),sources_frozen=True,production_admitted=False),indent=2))
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()

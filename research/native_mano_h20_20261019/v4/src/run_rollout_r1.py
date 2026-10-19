"""Preregister then run matched 600-update deployment-path training pilots."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/why/native_mano_rollout_v4_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019');VALIDATION=Path('/mnt/why/native_mano_multiseq_v3_20261019')

def main():
    assert not (VALIDATION/'evaluation').exists(),'Validation labels must remain unopened until the extra arms are frozen'
    occupied=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not occupied,occupied
    protocol=dict(locked_unix=time.time(),complete=True,train_windows=304,dev_select_windows=52,validation_forbidden=str(VALIDATION/'evaluation'),
        change='replace FK/parameter/real-time velocity/protection source by actual 4-draw,10-step Gaussian-sampler mean; same loss coefficients, v objective, init and batch plan',
        train_steps=600,controls=str(V2/'runs'),arms=['dit','protected'],same_updates_not_equal_compute=True,decision='run only after sampler parity/gradient preflight and a 4-step finite pilot',
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'src').glob('*.py')})
    assert (R/'protocol.json').exists();protocol['amendment']='preflight isolates no-grad inference and gradient-enabled autocast contexts; training source unchanged';(R/'protocol_amendment_r1.json').write_text(json.dumps(protocol,indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(V2/'src'),OMP_NUM_THREADS='4')
    commands=[('preflight',[sys.executable,str(R/'src/preflight_rollout_r1.py')]),('pilot',[sys.executable,str(R/'src/train_native.py'),'--arm','dit','--steps','4','--rollout','--output',str(R/'runs/pilot')])]
    for arm in ['dit','protected']:commands.append((arm,[sys.executable,str(R/'src/train_native.py'),'--arm',arm,'--steps','600','--rollout','--output',str(R/f'runs/{arm}')]))
    stages=[];started=time.time()
    for stage,cmd in commands:
        with (R/'logs'/f'{stage}_r1.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if stage=='preflight':assert json.loads((R/'receipts/preflight.json').read_text())['complete']
        if stage in ['dit','protected']:
            new=json.loads((R/f'runs/{stage}/done.json').read_text());old=json.loads((V2/f'runs/cosine_{stage}/done.json').read_text());assert new['batch_plan_sha']==old['batch_plan_sha']
        stages.append(dict(stage=stage,seconds=time.time()-started));(R/'receipts/controller_progress.json').write_text(json.dumps(stages,indent=2));print(json.dumps(stages[-1]),flush=True)
    assert not (VALIDATION/'evaluation').exists()
    (R/'receipts/done.json').write_text(json.dumps(dict(complete=True,stages=stages,validation_labels_still_unopened=True),indent=2))

if __name__=='__main__':main()

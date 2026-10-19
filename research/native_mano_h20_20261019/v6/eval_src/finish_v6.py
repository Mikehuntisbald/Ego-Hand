"""Finish this bounded experiment after its existing controller, without a timer."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/why/native_mano_shape_v6_20261019')
def main():
    started=time.time()
    while True:
        status=json.loads((R/'controller_status.json').read_text())
        if status.get('complete'):break
        assert time.time()-started<900,'Training wait exceeded bounded timeout'
        assert (R/'controller.lock').exists(),'Controller exited without completion; inspect preserved logs'
        time.sleep(3)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=f'{R}/src:{R}/eval_src:/mnt/why/native_mano_dit_v2_20261019/src:/mnt/why/native_mano_rollout_v4_20261019/src:/mnt/why/public_headhand_h20_20261019/src')
    for script in ['verify_v6.py','infer_v6.py','evaluate_v6.py']:
        (R/'finish_status.json').write_text(json.dumps(dict(stage=script,pid=os.getpid(),started=time.time()),indent=2))
        with (R/'receipts'/f'{script}.log').open('w') as log:subprocess.run([sys.executable,str(R/'eval_src'/script)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(stage=script,complete=True)),flush=True)
    (R/'finish_status.json').write_text(json.dumps(dict(complete=True,production_admitted=False),indent=2))
if __name__=='__main__':main()

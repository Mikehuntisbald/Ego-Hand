import json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D');RUN=ROOT/'experiments/dit_lowconfidence_v1';PROJECT=Path('/mnt/why/hot3d_hand_residual')
def wait(path):
    while not path.exists():time.sleep(15)
def phase(name):
    (RUN/'pipeline_status.json').write_text(json.dumps(dict(stage=name,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2));print(name,flush=True)
def run(script,args,log):
    with (RUN/log).open('w') as out:subprocess.run([sys.executable,str(PROJECT/script),*args],stdout=out,stderr=subprocess.STDOUT,check=True)
try:
    phase('waiting_detector_crops_and_coarse_warmup');wait(RUN/'predicted_crops_done.json');wait(RUN/'coarse_warmup/done.json')
    phase('coarse_fine_training_on_predicted_boxes')
    run('train_coarse_pose.py',['--phase','fine','--epochs','40','--device','cuda:1','--resume',str(RUN/'coarse_warmup/best.pt')],'coarse_fine.log')
    phase('caching_frozen_coarse_predictions');run('cache_coarse_predictions.py',[],'cache_coarse.log')
    phase('training_dit_and_regression_control')
    processes=[]
    for kind,device in [('dit','cuda:2'),('regression','cuda:3')]:
        out=(RUN/f'train_{kind}.log').open('w')
        p=subprocess.Popen([sys.executable,str(PROJECT/'train_residual_proof.py'),'--kind',kind,'--device',device],stdout=out,stderr=subprocess.STDOUT)
        processes.append((p,out))
    for p,out in processes:
        code=p.wait();out.close()
        if code:raise RuntimeError(f'Residual job {p.pid} failed: {code}')
    phase('models_trained_ready_for_locked_evaluation')
except Exception as e:
    phase('failed');(RUN/'pipeline_error.txt').write_text(repr(e));raise

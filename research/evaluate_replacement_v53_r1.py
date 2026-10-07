"""Finite post-training evaluation; all candidates freeze before fresh targets open."""
import os,json,time,subprocess,fcntl,shutil,hashlib
from pathlib import Path
A=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')
B=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53B_20261007')
CODE=Path('/mnt/why/hot3d_hand_residual')
RF='/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007/venv/bin/python'
BASE='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python'
def gpu_free():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader']).decode()
    return 'GPU-2f45f698-b39e-0f82-204d-f3a7d925baa9' not in apps
def main():
    B.mkdir(exist_ok=True);root=B/'replacement_eval';root.mkdir(exist_ok=True)
    lock=(root/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);env=dict(os.environ,CUDA_VISIBLE_DEVICES='3')
    def status(phase,**kwargs):(root/'status.json').write_text(json.dumps(dict(phase=phase,pid=os.getpid(),time=time.time(),**kwargs),indent=2))
    def run(name,script,*args,base=False,gpu=True):
        if gpu:
            while not gpu_free():status('waiting_GPU3',next=name);time.sleep(30)
        status(name)
        with (root/(name+'.log')).open('a') as log:
            proc=subprocess.Popen([BASE if base else RF,str(CODE/script),*map(str,args)],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT);status(name,child_pid=proc.pid);ret=proc.wait()
        if ret:raise RuntimeError(f'{name} exited {ret}; evidence retained')
    try:
        while not (B/'full/done.json').exists():
            p=B/'controller_status.json'
            if p.exists() and json.loads(p.read_text())['phase']=='failed':raise RuntimeError('training failed; stop dependent evaluation')
            status('waiting_training');time.sleep(30)
        if not (B/'yolo_development_baseline_fixed_square.json').exists():run('fixed_square_baseline','prepare_stageB_baseline_v53_r1.py')
        if not (B/'strict_review_r1/selection.json').exists():run('strict_development_selection','strict_select_replacement_v53.py',gpu=False)
        selection=json.loads((B/'strict_review_r1/selection.json').read_text())
        (root/'selection.json').write_text(json.dumps(selection,indent=2))
        rows=[r for r in map(json.loads,(B/'domain_records.jsonl').read_text().splitlines()) if r['split']=='test'];rows+=list(map(json.loads,(B/'fresh_test_rgb_records.jsonl').read_text().splitlines()))
        rgb=dict(frames=[{k:r[k] for k in ['id','image','dataset','group','split']} for r in rows]);(root/'test_rgb.json').write_text(json.dumps(rgb))
        yolo='/mnt/why/HOT3D/experiments/full_model_gloves_multihand_v51_20261007/paired_protocol/detector/balanced/weights/epoch2.pt'
        for name,kind,checkpoint,threshold,policy in [('yolo','yolo',yolo,.05,'raw'),('stageA','rfdetr',A/'checkpoint_metadata_fix/best_admitted_ema.pth',.1,'raw'),('stageB','rfdetr',selection['checkpoint'],selection['threshold'],selection['policy'])]:
            if not (root/name/'freeze.json').exists():run(name,'predict_boxes_v53_r1.py','--input',root/'test_rgb.json','--output',root/name,'--kind',kind,'--checkpoint',checkpoint,'--threshold',threshold,'--policy',policy)
        if not (root/'statistics.json').exists():run('score','score_replacement_v53.py',gpu=False)
        old=Path('/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007')
        for name in ['glove_rgb.json','nail_rgb.json']:
            if (old/name).exists():shutil.copy2(old/name,B/name)
        if not (B/'glove_RF/freeze.json').exists():run('glove_instances','predict_instances_v53.py','--input',B/'glove_rgb.json','--output',B/'glove_RF','--checkpoint',selection['checkpoint'],'--threshold',selection['threshold'],'--policy',selection['policy'])
        if not (B/'glove_complete/freeze.json').exists():run('glove_3d','complete_rfdetr_v52.py','--predictions',B/'glove_RF/predictions.json','--output',B/'glove_complete/prediction.json','--threshold',selection['threshold'],base=True)
        if not (B/'full_glove_evaluation.json').exists():run('glove_score','score_full_rfdetr_v53.py',base=True,gpu=False)
        status('complete_pending_report',statistics=str(root/'statistics.json'),full_3d=str(B/'full_glove_evaluation.json'),default_replaced=False)
    except Exception as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()

"""Finite post-training benchmark, not an automation or recurring job."""
import os,json,subprocess,time,hashlib
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007')
CODE=Path('/mnt/why/hot3d_hand_residual')
BASE='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python'
RF=str(ROOT/'venv/bin/python')

def run(name,python,args,gpu=True):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3' if gpu else '',RF_HOME=str(ROOT/'weights'),OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_TELEMETRY='1')
    if gpu:
        uuid=subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=uuid','--format=csv,noheader']).decode().strip()
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader']).decode();assert uuid not in apps, 'GPU3 became occupied'
    (ROOT/'finalize_status.json').write_text(json.dumps(dict(stage=name,status='running',args=args)))
    with open(ROOT/(name+'.log'),'w') as log:subprocess.run([python,*args],env=env,stdout=log,stderr=subprocess.STDOUT,cwd=CODE,check=True)

def main():
    assert (ROOT/'full/done.json').exists() and json.loads((ROOT/'full/done.json').read_text())['visual_and_mask_updates_passed']
    run('training_verify',BASE,[str(CODE/'verify_rfdetr_v52.py')],gpu=False)
    for name in ['dev','test','glove','nail']:
        run(name+'_infer',RF,[str(CODE/'predict_rfdetr_v52.py'),'--input',str(ROOT/(name+'_rgb.json')),'--output',str(ROOT/(name+'_predictions')),'--threshold','.001' if name in ['glove','nail'] else '.05'])
        if name=='dev':run('select',BASE,[str(CODE/'benchmark_rfdetr_v52.py'),'select'],gpu=False)
    run('mask_baseline',BASE,[str(CODE/'benchmark_rfdetr_v52.py'),'baseline'])
    run('mask_score',BASE,[str(CODE/'benchmark_rfdetr_v52.py'),'score'],gpu=False)
    threshold=json.loads((ROOT/'threshold_selection.json').read_text())['selected']['threshold']
    run('glove_complete',BASE,[str(CODE/'complete_rfdetr_v52.py'),'--predictions',str(ROOT/'glove_predictions/predictions.json'),'--output',str(ROOT/'glove_complete/prediction.json'),'--threshold',str(threshold)])
    run('full_glove_score',BASE,[str(CODE/'score_full_rfdetr_v52.py')],gpu=False)
    coarse='/mnt/why/HOT3D/experiments/full_model_gloves_multihand_v51_20261007/complete_glove_test/candidate/coarse_observations.json'
    run('glove_yolo_rf_complete',BASE,[str(CODE/'complete_rfdetr_v52.py'),'--predictions',str(ROOT/'glove_predictions/predictions.json'),'--output',str(ROOT/'glove_yolo_rf_complete/prediction.json'),'--threshold',str(threshold),'--source-coarse',coarse])
    run('full_glove_mask_only_score',BASE,[str(CODE/'score_full_rfdetr_v52.py'),'--folder','glove_yolo_rf_complete','--name','full_glove_mask_only'],gpu=False)
    run('render',BASE,[str(CODE/'render_rfdetr_v52.py')],gpu=False)
    (ROOT/'finalize_status.json').write_text(json.dumps(dict(status='complete',mask_report=str(ROOT/'mask_evaluation.json'),full_report=str(ROOT/'full_glove_evaluation.json'))))

if __name__=='__main__':
    try:main()
    except Exception as e:
        path=ROOT/'finalize_status.json';data=json.loads(path.read_text()) if path.exists() else {};data.update(status='error',error=repr(e));path.write_text(json.dumps(data));raise

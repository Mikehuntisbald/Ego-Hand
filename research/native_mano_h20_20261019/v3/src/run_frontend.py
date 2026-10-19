"""Run locked RF and prediction-only coherent WiLoR on the six recordings."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/why/native_mano_multiseq_v3_20261019');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019')
RF='/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007/venv/bin/python'
CK='/mnt/why/HOT3D/experiments/rfdetr_multidata_v53B_20261007/strict_review_r1/best_development_ema.pth'

def main():
    protocol=json.loads((R/'protocol.json').read_text());started=time.time();done=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(PUBLIC/'src'),OMP_NUM_THREADS='4')
    occupied=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not occupied,occupied
    snapshot={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'src').glob('*.py')}
    (R/'receipts/frontend_source_lock.json').write_text(json.dumps(dict(files=snapshot,RF_checkpoint_sha256=hashlib.sha256(Path(CK).read_bytes()).hexdigest()),indent=2))
    for clip in protocol['clips']:
        view=Path(clip['view']);rf=view/'outputs/rf_clip2102';coarse=view/'outputs/wilor_clip2102'
        commands=[('RF',[RF,str(PUBLIC/'vendor/Ego-Hand/research/predict_instances_v53.py'),'--input',str(view/'data/frames.json'),'--output',str(rf),'--checkpoint',CK,'--threshold','.2','--policy','box_nms_0.7']),
                  ('WiLoR',[sys.executable,str(R/'src/run_wilor_coherent.py'),'--input',str(view/'data/clip-002102'),'--rf',str(rf),'--output',str(coarse)])]
        for stage,command in commands:
            with (R/'logs'/f'{clip["clip"]}_{stage}.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            row=dict(clip=clip['clip'],stage=stage,seconds=time.time()-started);done.append(row);(R/'receipts/frontend_progress.json').write_text(json.dumps(done,indent=2));print(json.dumps(row),flush=True)
    (R/'receipts/frontend_done.json').write_text(json.dumps(dict(complete=True,clips=len(protocol['clips']),hand_labels_read=False,supplied_camera_trajectory=True,stages=done),indent=2))

if __name__=='__main__':main()

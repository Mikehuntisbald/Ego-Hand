"""One-command isolated replay on the prepared H20 host; no private assets."""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
BASE=Path('/mnt/why/public_headhand_h20_20261019')
HAND='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python'
RF='/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007/venv/bin/python'
CK='/mnt/why/HOT3D/experiments/rfdetr_multidata_v53B_20261007/strict_review_r1/best_development_ema.pth'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',required=True,type=Path)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=args.run_root.resolve();assert root.is_relative_to(BASE/'runs'),root
    commands=[
        ('RF',[RF,str(BASE/'vendor/Ego-Hand/research/predict_instances_v53.py'),'--input',str(root/'data/frames.json'),'--output',str(root/'outputs/rf_clip2102'),'--checkpoint',CK,'--threshold','.2','--policy','box_nms_0.7'],root/'outputs/rf_clip2102/freeze.json'),
        ('WiLoR',[HAND,'src/run_wilor.py','--input','data/clip-002102','--rf','outputs/rf_clip2102','--output','outputs/wilor_clip2102'],root/'outputs/wilor_clip2102/seal.json'),
        ('surface',[HAND,'src/prepare_surface.py','--coarse','outputs/wilor_clip2102','--output','outputs/wilor_clip2102/surface.npz'],root/'outputs/wilor_clip2102/surface.json'),
        ('head',[str(BASE/'.venv-head/bin/python'),'src/run_headpose.py','--input','data/clip-002102','--output','outputs/head_clip2102','--rectified'],root/'outputs/head_clip2102/seal.json'),
        ('stereo',[str(BASE/'.venv-depth/bin/python'),'src/run_depth.py','--input','data/clip-002102','--coarse','outputs/wilor_clip2102','--rf','outputs/rf_clip2102','--output','outputs/depth_clip2102'],root/'outputs/depth_clip2102/seal.json'),
        ('mono',[str(BASE/'.venv-depth/bin/python'),'src/run_unidepth.py'],root/'outputs/depth_fused_clip2102/seal.json'),
        ('depth_protection',[str(BASE/'.venv-depth/bin/python'),'src/protect_depth.py'],root/'outputs/depth_protected_clip2102/seal.json'),
        ('BA',[HAND,'src/run_mano_ba.py','--depth','depth_protected_clip2102'],root/'outputs/mano_baseline_clip2102/seal.json'),
        ('DiT',[HAND,'src/run_dit.py'],root/'outputs/dit_clip2102/seal.json'),
        ('DiT_BA',[HAND,'src/run_mano_ba.py','--dit','--depth','depth_protected_clip2102'],root/'outputs/mano_dit_clip2102/seal.json'),
        ('freeze',[HAND,'src/verify_and_freeze.py'],root/'receipts/predictions_frozen_before_GT.json'),
        ('score',[HAND,'src/evaluate_public.py'],root/'outputs/review_clip2102/metrics.json')]
    plan=dict(run_root=str(root),GPU=3,commands=[dict(stage=n,argv=c,completion=str(p)) for n,c,p in commands],
              scope='same public diagnostic development clip; not a private GT9 or held-out experiment; no throughput benchmark')
    if not args.execute:print(json.dumps(plan,indent=2));return
    if root.exists():assert args.resume,'Existing run requires explicit --resume; outputs never overwritten'
    else:
        root.mkdir(parents=True)
        for name in ['assets','vendor']: (root/name).symlink_to(BASE/name,target_is_directory=True)
        for name in ['outputs','logs','receipts','data']: (root/name).mkdir()
        shutil.copytree(BASE/'src',root/'src',ignore=shutil.ignore_patterns('__pycache__'))
        (root/'data/clip-002102').symlink_to(BASE/'data/clip-002102',target_is_directory=True)
        source=json.loads((BASE/'data/frames.json').read_text())
        for frame in source['frames']:frame['image']=str(root/'data/clip-002102'/Path(frame['image']).name)
        (root/'data/frames.json').write_text(json.dumps(source))
        (root/'plan.json').write_text(json.dumps(plan,indent=2))
    status_path=root/'status.json'
    if args.resume and status_path.exists() and json.loads(status_path.read_text()).get('complete'):
        print(json.dumps(dict(complete=True,already_complete=True,status=str(status_path))));return
    env=dict(os.environ,PUBLIC_HEADHAND_ROOT=str(root),PUBLIC_DEVELOPMENT_REPLAY='1',CUDA_VISIBLE_DEVICES='3',LD_LIBRARY_PATH='/usr/local/cuda-12.8/lib64',OMP_NUM_THREADS='4')
    # Do not start while any other process owns the allowed GPU.
    occupied=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not occupied,f'GPU3 occupied: {occupied}'
    state=dict(complete=False,stages=[],started=time.time(),development_replay=True,GT_never_inference_input=True)
    for name,cmd,done in commands:
        if done.exists() and args.resume:
            # Verify every sealed artifact before trusting a completed stage.
            receipt=json.loads(done.read_text())
            for filename,expected in receipt.get('files',{}).items():
                assert hashlib.sha256((done.parent/filename).read_bytes()).hexdigest()==expected
            state['stages'].append(dict(stage=name,status='already-complete'));continue
        before=time.time()
        cmd=[str(root/value) if value.startswith(('src/','data/','outputs/')) else value for value in cmd]
        with (root/'logs'/f'{name}.log').open('w') as log:
            result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['stages'].append(dict(stage=name,exit_code=result.returncode,seconds=time.time()-before))
        (root/'status.json').write_text(json.dumps(state,indent=2))
        print(json.dumps(state['stages'][-1]),flush=True)
        assert result.returncode==0 and done.exists(),f'Stage {name} failed; see {root}/logs/{name}.log; partial output preserved'
    state.update(complete=True,seconds=time.time()-state['started'])
    (root/'status.json').write_text(json.dumps(state,indent=2));print(json.dumps(dict(complete=True,seconds=state['seconds'],report=str(root/'outputs/review_clip2102/report.html'))),flush=True)

if __name__=='__main__':main()

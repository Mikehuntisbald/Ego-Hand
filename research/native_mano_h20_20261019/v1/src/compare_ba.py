"""Create isolated post-freeze comparisons, keeping historical public outputs intact."""
import json,os,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path('/mnt/why/native_mano_dit_20261019');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019/runs/replay_r3')
PY='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python'

def main():
    for arm in ['raw','protected']:
        view=ROOT/'comparisons'/arm;view.mkdir(parents=True,exist_ok=False)
        for name in ['data','assets','vendor']:(view/name).symlink_to(PUBLIC/name,target_is_directory=True)
        for name in ['src','receipts','outputs']:(view/name).mkdir()
        for name in ['wilor_clip2102','head_clip2102','depth_clip2102','depth_fused_clip2102','depth_protected_clip2102','mano_baseline_clip2102']:
            (view/'outputs'/name).symlink_to(PUBLIC/'outputs'/name,target_is_directory=True)
        (view/'outputs/mano_dit_clip2102').symlink_to(ROOT/f'outputs/ba_native_{arm}',target_is_directory=True)
        verify=Path('/mnt/why/public_headhand_h20_20261019/src/verify_and_freeze.py').read_text()
        evaluate=Path('/mnt/why/public_headhand_h20_20261019/src/evaluate_public.py').read_text()
        evaluate=evaluate.replace('DiT为公开v48 UmeTrack20提案，经显式MANO拟合后进入同一BA','DiT直接输出原生MANO旋转参数，直接进入同一BA，推理无20点拟合桥接')
        (view/'src/verify.py').write_text(verify);(view/'src/evaluate.py').write_text(evaluate)
        env=dict(os.environ,PUBLIC_HEADHAND_ROOT=str(view),PUBLIC_DEVELOPMENT_REPLAY='1',CUDA_VISIBLE_DEVICES='3',PYTHONPATH='/mnt/why/public_headhand_h20_20261019/src')
        for script in ['verify.py','evaluate.py']:
            with (ROOT/'logs'/f'compare_{arm}_{script}.log').open('w') as log:subprocess.run([PY,str(view/'src'/script)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        metrics=json.loads((view/'outputs/review_clip2102/metrics.json').read_text());metrics['native_MANO_parameter_DiT']=True;metrics['inference_landmark_fitting']=False
        (view/'outputs/review_clip2102/metrics.json').write_text(json.dumps(metrics,indent=2))
        with (ROOT/'logs'/f'zoom_{arm}.log').open('w') as log:subprocess.run([PY,'/mnt/why/public_headhand_h20_20261019/src/make_zoom_review.py','--run-root',str(view)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        (view/'summary.json').write_text(json.dumps(dict(complete=True,native=True,old_bridge_reference=str(PUBLIC/'outputs/review_clip2102/metrics.json'),metrics=metrics['metrics'],harm=metrics['DiT_vs_BA_protection_camera']),indent=2))
        print(json.dumps(dict(arm=arm,metrics=metrics['metrics']['DiT_BA'],harm=metrics['DiT_vs_BA_protection_camera'])),flush=True)

if __name__=='__main__':main()

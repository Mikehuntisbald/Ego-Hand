"""Post-freeze preservation audit of the front-end repair itself, without DiT."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path('/mnt/why/native_mano_dit_v2_20261019');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019/runs/replay_r3');VIEW=R/'coherent_frontend'
SOURCE=Path('/mnt/why/public_headhand_h20_20261019/src')

def main():
    comp=R/'comparisons/frontend_repair';comp.mkdir(parents=True,exist_ok=False)
    for name in ['data','assets','vendor']:(comp/name).symlink_to(VIEW/name,target_is_directory=True)
    for name in ['src','receipts','outputs']:(comp/name).mkdir()
    for name in ['wilor_clip2102','head_clip2102','depth_clip2102','depth_fused_clip2102','depth_protected_clip2102']:
        (comp/'outputs'/name).symlink_to(VIEW/'outputs'/name,target_is_directory=True)
    (comp/'outputs/mano_baseline_clip2102').symlink_to(PUBLIC/'outputs/mano_baseline_clip2102',target_is_directory=True)
    (comp/'outputs/mano_dit_clip2102').symlink_to(VIEW/'outputs/mano_baseline_clip2102',target_is_directory=True)
    env=dict(os.environ,PUBLIC_HEADHAND_ROOT=str(comp),PUBLIC_DEVELOPMENT_REPLAY='1',PYTHONPATH=str(SOURCE),CUDA_VISIBLE_DEVICES='3')
    for old,new in [('verify_and_freeze.py','verify.py'),('evaluate_public.py','evaluate.py')]:
        source=(SOURCE/old).read_text().replace('DiT为公开v48 UmeTrack20提案，经显式MANO拟合后进入同一BA','该对照完全不使用DiT，候选列为一致性修复后的基础BA；检查前端修复本身的收益及损伤')
        (comp/'src'/new).write_text(source)
        with (R/'logs'/f'frontend_{new}.log').open('w') as log:subprocess.run([sys.executable,str(comp/'src'/new)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    p=comp/'outputs/review_clip2102/metrics.json';data=json.loads(p.read_text())
    data.update(candidate_is_DiT=False,candidate_method='coherent prediction-only handedness repair + same base BA',historical_metric_key='DiT_BA is the evaluator generic candidate column; this audit uses no DiT')
    p.write_text(json.dumps(data,indent=2))
    print(json.dumps(dict(metrics=data['metrics'],harm=data['DiT_vs_BA_protection_camera'],temporal=data['temporal'])))

if __name__=='__main__':main()

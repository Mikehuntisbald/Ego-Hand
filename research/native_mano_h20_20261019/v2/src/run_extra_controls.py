"""Complete like-for-like schedule and shared-noise sampling BA controls."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path('/mnt/why/native_mano_dit_v2_20261019');V1=Path('/mnt/why/native_mano_dit_20261019');VIEW=R/'coherent_frontend'
SOURCE=Path('/mnt/why/public_headhand_h20_20261019/src')

def run(args,name,env):
    with (R/'logs'/f'{name}.log').open('w') as log:subprocess.run([sys.executable,*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(stage=name,complete=True)),flush=True)

def main():
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PUBLIC_HEADHAND_ROOT=str(VIEW),PUBLIC_DEVELOPMENT_REPLAY='1',PYTHONPATH=str(SOURCE))
    arms={'linear_dit':(V1/'runs/dit','infer_native.py'),'cosine_shared':(R/'runs/cosine_dit','infer_shared.py')}
    for arm,(folder,script) in arms.items():
        out=VIEW/'outputs'/f'native_{arm}'
        run([R/'src'/script,'--public-root',VIEW,'--checkpoint',folder/'best_development.pt','--output',out,'--protect'],f'consistent_infer_{arm}',env)
        run([R/'src/native_ba.py','--depth','depth_protected_clip2102','--native',out/'native_parameters.npz','--output',VIEW/'outputs'/f'ba_{arm}','--suffix','coherent'],f'consistent_BA_{arm}',env)
    manifest={str(p.relative_to(VIEW)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (VIEW/'outputs').glob('*/seal.json')}
    (VIEW/'receipts/extra_candidates_frozen.json').write_text(json.dumps(dict(complete=True,seals=manifest,GT_inference=False,development_reuse=True),indent=2))
    for arm in arms:
        comp=R/'comparisons'/f'coherent_{arm}';comp.mkdir(parents=True,exist_ok=False)
        for name in ['data','assets','vendor']:(comp/name).symlink_to(VIEW/name,target_is_directory=True)
        for name in ['src','receipts','outputs']:(comp/name).mkdir()
        for name in ['wilor_clip2102','head_clip2102','depth_clip2102','depth_fused_clip2102','depth_protected_clip2102','mano_baseline_clip2102']:(comp/'outputs'/name).symlink_to(VIEW/'outputs'/name,target_is_directory=True)
        (comp/'outputs/mano_dit_clip2102').symlink_to(VIEW/'outputs'/f'ba_{arm}',target_is_directory=True)
        for old,new in [('verify_and_freeze.py','verify.py'),('evaluate_public.py','evaluate.py')]:
            source=(SOURCE/old).read_text().replace('DiT为公开v48 UmeTrack20提案，经显式MANO拟合后进入同一BA',f'原生MANO {arm}、同一一致前端和同一BA，推理无20点拟合')
            (comp/'src'/new).write_text(source)
        current=dict(env,PUBLIC_HEADHAND_ROOT=str(comp))
        for script in ['verify.py','evaluate.py']:run([comp/'src'/script],f'coherent_{arm}_{script}',current)
        run([SOURCE/'make_zoom_review.py','--run-root',comp],f'coherent_{arm}_zoom',current)
        m=json.loads((comp/'outputs/review_clip2102/metrics.json').read_text());print(json.dumps(dict(arm=arm,metrics=m['metrics']['DiT_BA'],harm=m['DiT_vs_BA_protection_camera'],temporal=m['temporal']['DiT_BA'])),flush=True)
    (R/'receipts/extra_done.json').write_text(json.dumps(dict(complete=True,arms=list(arms),safe_default_admitted=False),indent=2))

if __name__=='__main__':main()

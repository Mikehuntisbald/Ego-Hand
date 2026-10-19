"""Repair the front end coherently; then matched native-pose/BA controls."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path('/mnt/why/native_mano_dit_v2_20261019');V1=Path('/mnt/why/native_mano_dit_20261019')
PUBLIC=Path('/mnt/why/public_headhand_h20_20261019/runs/replay_r3');SOURCE=Path('/mnt/why/public_headhand_h20_20261019/src')
VIEW=R/'coherent_frontend'

def run(args,name,env):
    with (R/'logs'/f'{name}.log').open('w') as log:subprocess.run([sys.executable,*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(stage=name,complete=True)),flush=True)

def main():
    VIEW.mkdir(exist_ok=False)
    for name in ['data','assets','vendor']:(VIEW/name).symlink_to(PUBLIC/name,target_is_directory=True)
    for name in ['outputs','receipts','src']:(VIEW/name).mkdir()
    for name in ['rf_clip2102','head_clip2102']:(VIEW/'outputs'/name).symlink_to(PUBLIC/'outputs'/name,target_is_directory=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PUBLIC_HEADHAND_ROOT=str(VIEW),PUBLIC_DEVELOPMENT_REPLAY='1',PYTHONPATH=str(SOURCE))
    coarse=VIEW/'outputs/wilor_clip2102'
    run([R/'src/stabilize_wilor.py','--public-root',PUBLIC,'--output',coarse],'consistent_wilor',env)
    run([SOURCE/'prepare_surface.py','--coarse',coarse,'--output',coarse/'surface.npz'],'consistent_surface',env)
    run([SOURCE/'run_depth.py','--input',VIEW/'data/clip-002102','--coarse',coarse,'--rf',VIEW/'outputs/rf_clip2102','--output',VIEW/'outputs/depth_clip2102'],'consistent_stereo',env)
    run([SOURCE/'run_unidepth.py'],'consistent_unidepth',env)
    run([SOURCE/'protect_depth.py'],'consistent_depth_gate',env)
    run([R/'src/native_ba.py','--depth','depth_protected_clip2102','--output',VIEW/'outputs/mano_baseline_clip2102','--suffix','coherent'],'consistent_baseline_BA',env)
    arms={'regression':V1/'runs/regression','linear_protected':V1/'runs/protected','cosine_dit':R/'runs/cosine_dit','cosine_protected':R/'runs/cosine_protected'}
    for arm,folder in arms.items():
        out=VIEW/'outputs'/f'native_{arm}'
        run([R/'src/infer_native.py','--public-root',VIEW,'--checkpoint',folder/'best_development.pt','--output',out,'--protect'],f'consistent_infer_{arm}',env)
        run([R/'src/native_ba.py','--depth','depth_protected_clip2102','--native',out/'native_parameters.npz','--output',VIEW/'outputs'/f'ba_{arm}','--suffix','coherent'],f'consistent_BA_{arm}',env)
    # All arms are complete and sealed before any replay labels are opened.
    frozen={str(p.relative_to(VIEW)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (VIEW/'outputs').glob('*/seal.json')}
    (VIEW/'receipts/all_candidates_frozen.json').write_text(json.dumps(dict(complete=True,seals=frozen,GT_inference=False,development_reuse=True),indent=2))
    for arm in arms:
        comp=R/'comparisons'/f'coherent_{arm}';comp.mkdir(parents=True,exist_ok=False)
        for name in ['data','assets','vendor']:(comp/name).symlink_to(VIEW/name,target_is_directory=True)
        for name in ['src','receipts','outputs']:(comp/name).mkdir()
        for name in ['wilor_clip2102','head_clip2102','depth_clip2102','depth_fused_clip2102','depth_protected_clip2102','mano_baseline_clip2102']:
            (comp/'outputs'/name).symlink_to(VIEW/'outputs'/name,target_is_directory=True)
        (comp/'outputs/mano_dit_clip2102').symlink_to(VIEW/'outputs'/f'ba_{arm}',target_is_directory=True)
        for old,new in [('verify_and_freeze.py','verify.py'),('evaluate_public.py','evaluate.py')]:
            source=(SOURCE/old).read_text()
            if new=='evaluate.py':source=source.replace('DiT为公开v48 UmeTrack20提案，经显式MANO拟合后进入同一BA',f'原生MANO参数 {arm}，整轨迹左右手证据修复并实际重跑WiLoR；同一深度和BA')
            (comp/'src'/new).write_text(source)
        current=dict(env,PUBLIC_HEADHAND_ROOT=str(comp))
        for script in ['verify.py','evaluate.py']:run([comp/'src'/script],f'coherent_{arm}_{script}',current)
        run([SOURCE/'make_zoom_review.py','--run-root',comp],f'coherent_{arm}_zoom',current)
        m=json.loads((comp/'outputs/review_clip2102/metrics.json').read_text())
        print(json.dumps(dict(arm=arm,metrics=m['metrics']['DiT_BA'],baseline=m['metrics']['BA'],harm=m['DiT_vs_BA_protection_camera'],temporal=m['temporal']['DiT_BA'])),flush=True)
    (R/'receipts/coherent_done.json').write_text(json.dumps(dict(complete=True,arms=list(arms),safe_default_admitted=False),indent=2))

if __name__=='__main__':main()

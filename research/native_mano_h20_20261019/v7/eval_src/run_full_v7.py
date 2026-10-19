"""Matched small/full native heads in one fixed complete replay.

Every raw and guarded branch is frozen before any replay labels open.
"""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/why/native_mano_pool_v7_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019');VIEW=R/'full_pipeline'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(args,name,env):
    with (R/'logs'/f'full_{name}.log').open('w') as log:subprocess.run([sys.executable,*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(stage=name,complete=True)),flush=True)
def main():
    assert json.loads((R/'training_status.json').read_text())['complete']
    VIEW.mkdir(exist_ok=False);(R/'logs').mkdir(exist_ok=True);(R/'comparisons').mkdir(exist_ok=True)
    for n in ['outputs','src','receipts']:(VIEW/n).mkdir()
    for n in ['data','assets','vendor']:(VIEW/n).symlink_to(V2/'coherent_frontend'/n,target_is_directory=True)
    common=['wilor_clip2102','head_clip2102','depth_clip2102','depth_fused_clip2102','depth_protected_clip2102','mano_baseline_clip2102']
    for n in common:(VIEW/'outputs'/n).symlink_to(V2/'coherent_frontend/outputs'/n,target_is_directory=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PUBLIC_HEADHAND_ROOT=str(VIEW),PUBLIC_DEVELOPMENT_REPLAY='1',PYTHONPATH=os.pathsep.join([str(R/'eval_src'),str(V2/'src'),str(PUBLIC/'src')]))
    models={arm:R/'runs'/arm/'best_development.pt' for arm in ['small','full']}
    assert {a:sha(p) for a,p in models.items()}==json.loads((R/'receipts/full_predeclared.json').read_text())['models']
    (VIEW/'receipts/model_lock.json').write_text(json.dumps(dict(models={a:sha(p) for a,p in models.items()},same_frontend_depth_and_BA=True,historical_replay=True),indent=2))
    for arm,checkpoint in models.items():
        out=VIEW/f'outputs/native_{arm}'
        run([R/'eval_src/infer_full_right_v7.py','--public-root',VIEW,'--checkpoint',checkpoint,'--output',out,'--protect'],f'infer_{arm}',env)
        run([V2/'src/native_ba.py','--depth','depth_protected_clip2102','--native',out/'native_parameters.npz','--output',VIEW/f'outputs/ba_{arm}','--suffix','matched_data_v7'],f'BA_{arm}',env)
    for arm in models:run([R/'eval_src/guard_geometry_v7.py','--arm',arm],f'geometry_guard_{arm}',env)
    seals={}
    for arm in ['small','full','small_guarded','full_guarded']:
        p=VIEW/f'outputs/ba_{arm}/seal.json';assert p.exists();seals[str(p.relative_to(VIEW))]=sha(p)
    (VIEW/'receipts/all_predictions_frozen.json').write_text(json.dumps(dict(complete=True,freeze_unix=time.time(),seals=seals,GT_read=False,historical_clip=True),indent=2))
    # Labels are allowed only after all four variants are immutable.
    (VIEW/'receipts/label_open_order.json').write_text(json.dumps(dict(unix=time.time(),frozen_sha256=sha(VIEW/'receipts/all_predictions_frozen.json'),all_four_variants_complete=True,historical_exposure=True),indent=2))
    for arm in ['small','full','small_guarded','full_guarded']:
        comp=R/'comparisons'/f'full_{arm}';comp.mkdir()
        for n in ['data','assets','vendor']:(comp/n).symlink_to(VIEW/n,target_is_directory=True)
        for n in ['outputs','src','receipts']:(comp/n).mkdir()
        for n in common:(comp/'outputs'/n).symlink_to(VIEW/'outputs'/n,target_is_directory=True)
        (comp/'outputs/mano_dit_clip2102').symlink_to(VIEW/f'outputs/ba_{arm}',target_is_directory=True)
        current=dict(env,PUBLIC_HEADHAND_ROOT=str(comp))
        for old,new in [('verify_and_freeze.py','verify.py'),('evaluate_public.py','evaluate.py')]:
            source=(PUBLIC/'src'/old).read_text().replace('DiT为公开v48 UmeTrack20提案，经显式MANO拟合后进入同一BA',f'自然多数据原生MANO DiT v7 {arm}；固定相同物理前后端')
            (comp/'src'/new).write_text(source);run([comp/'src'/new],f'{arm}_{new}',current)
        run([PUBLIC/'src/make_zoom_review.py','--run-root',comp],f'{arm}_zoom',current)
    (R/'receipts/full_pipeline_done.json').write_text(json.dumps(dict(complete=True,all_four_predictions_frozen_before_labels=True,same_input_physics_and_BA=True,production_admitted=False,historical_diagnostic_replay=True),indent=2))
if __name__=='__main__':main()

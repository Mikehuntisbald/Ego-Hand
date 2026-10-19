"""Only swap the trained native head in the existing complete public chain."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path('/mnt/why/native_mano_shape_v6_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019');VIEW=R/'full_pipeline'

def run(args,name,env):
    with (R/'logs'/f'full_{name}.log').open('w') as log:subprocess.run([sys.executable,*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(stage=name,complete=True)),flush=True)

def main():
    lock=json.loads((R/'receipts/full_predeclared.json').read_text());assert hashlib.sha256((R/'runs/compatible/best_development.pt').read_bytes()).hexdigest()==lock['checkpoint_sha256']
    VIEW.mkdir(exist_ok=False);(R/'logs').mkdir(exist_ok=True);(R/'comparisons').mkdir(exist_ok=True)
    for name in ['outputs','src','receipts']:(VIEW/name).mkdir()
    for name in ['data','assets','vendor']:(VIEW/name).symlink_to(V2/'coherent_frontend'/name,target_is_directory=True)
    for name in ['wilor_clip2102','head_clip2102','depth_clip2102','depth_fused_clip2102','depth_protected_clip2102','mano_baseline_clip2102']:
        (VIEW/'outputs'/name).symlink_to(V2/'coherent_frontend/outputs'/name,target_is_directory=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PUBLIC_HEADHAND_ROOT=str(VIEW),PUBLIC_DEVELOPMENT_REPLAY='1',PYTHONPATH=os.pathsep.join([str(R/'eval_src'),str(V2/'src'),str(PUBLIC/'src')]))
    for arm in ['dit']:
        out=VIEW/f'outputs/native_rollout_{arm}';run([R/'eval_src/infer_full_right_v6.py','--public-root',VIEW,'--checkpoint',R/'runs/compatible/best_development.pt','--output',out,'--protect'],f'infer_{arm}',env)
        run([V2/'src/native_ba.py','--depth','depth_protected_clip2102','--native',out/'native_parameters.npz','--output',VIEW/f'outputs/ba_rollout_{arm}','--suffix','shape_compatible_native_pose_predeclared'],f'BA_{arm}',env)
    (VIEW/'receipts/all_native_outputs_frozen.json').write_text(json.dumps(dict(complete=True,files={str(p.relative_to(VIEW)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (VIEW/'outputs').glob('*/seal.json')},new_predictions_scored_only_after_freeze=True,clip_previously_evaluated=True),indent=2))
    # This also writes/redecodes/validates JSON/H5, freezes, scores and plots.
    run([R/'eval_src/guard_full_v6.py'],'post_BA_guard',env)
    for arm in ['dit']:
        comp=R/f'comparisons/full_rollout_{arm}';comp.mkdir(parents=True,exist_ok=False)
        for name in ['data','assets','vendor']:(comp/name).symlink_to(VIEW/name,target_is_directory=True)
        for name in ['src','receipts','outputs']:(comp/name).mkdir()
        for name in ['wilor_clip2102','head_clip2102','depth_clip2102','depth_fused_clip2102','depth_protected_clip2102','mano_baseline_clip2102']:(comp/'outputs'/name).symlink_to(VIEW/'outputs'/name,target_is_directory=True)
        (comp/'outputs/mano_dit_clip2102').symlink_to(VIEW/f'outputs/ba_rollout_{arm}',target_is_directory=True)
        for old,new in [('verify_and_freeze.py','verify.py'),('evaluate_public.py','evaluate.py')]:
            source=(PUBLIC/'src'/old).read_text().replace('DiT为公开v48 UmeTrack20提案，经显式MANO拟合后进入同一BA',f'姿态形状解耦右手DiT（网络左手零残差） {arm}，使用相同cuVSLAM预测轨迹、物理深度和BA')
            (comp/'src'/new).write_text(source)
        current=dict(env,PUBLIC_HEADHAND_ROOT=str(comp))
        for script in ['verify.py','evaluate.py']:run([comp/'src'/script],f'{arm}_{script}',current)
        run([PUBLIC/'src/make_zoom_review.py','--run-root',comp],f'{arm}_zoom',current)
        m=json.loads((comp/'outputs/review_clip2102/metrics.json').read_text());print(json.dumps(dict(arm=arm,metrics=m['metrics']['DiT_BA'],harm=m['DiT_vs_BA_protection_camera'])),flush=True)
    (R/'receipts/full_pipeline_done.json').write_text(json.dumps(dict(complete=True,reference='identical full public cuVSLAM/LightESS/UniDepth/BA inputs from v2',production_admitted=False,diagnostic_replay_only=True),indent=2))

if __name__=='__main__':main()

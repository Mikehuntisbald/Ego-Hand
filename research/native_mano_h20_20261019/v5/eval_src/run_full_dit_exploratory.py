"""Posthoc native-DiT replay, declared after six-clip development scoring."""
import hashlib,json,os,sys
from pathlib import Path
from run_full_v5 import R,V2,PUBLIC,VIEW,run
def main():
    assert json.loads((R/'finish_status.json').read_text())['complete']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PUBLIC_HEADHAND_ROOT=str(VIEW),PUBLIC_DEVELOPMENT_REPLAY='1',PYTHONPATH=os.pathsep.join([str(R/'eval_src'),str(V2/'src'),str(PUBLIC/'src')]))
    lock=dict(posthoc=True,reason='ordinary native DiT guard beat protection-loss arm on already-exposed six development recordings; full clip2102 also previously evaluated; not blind generalization',
        development_metrics_sha256=hashlib.sha256((R/'evaluation/metrics.json').read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256((R/'runs/native_dit/best_development.pt').read_bytes()).hexdigest())
    (R/'receipts/exploratory_full_dit_lock.json').write_text(json.dumps(lock,indent=2))
    out=VIEW/'outputs/native_rollout_dit'
    run([R/'eval_src/infer_full_right.py','--public-root',VIEW,'--checkpoint',R/'runs/native_dit/best_development.pt','--output',out,'--protect'],'infer_dit',env)
    run([V2/'src/native_ba.py','--depth','depth_protected_clip2102','--native',out/'native_parameters.npz','--output',VIEW/'outputs/ba_rollout_dit','--suffix','native_MANO_right_exploratory'],'BA_dit',env)
    run([R/'eval_src/guard_full_dit_v5.py'],'post_BA_guard_dit',env)
    comp=R/'comparisons/full_rollout_dit';comp.mkdir(parents=True,exist_ok=False)
    for name in ['data','assets','vendor']:(comp/name).symlink_to(VIEW/name,target_is_directory=True)
    for name in ['src','receipts','outputs']:(comp/name).mkdir()
    for name in ['wilor_clip2102','head_clip2102','depth_clip2102','depth_fused_clip2102','depth_protected_clip2102','mano_baseline_clip2102']:(comp/'outputs'/name).symlink_to(VIEW/'outputs'/name,target_is_directory=True)
    (comp/'outputs/mano_dit_clip2102').symlink_to(VIEW/'outputs/ba_rollout_dit',target_is_directory=True)
    current=dict(env,PUBLIC_HEADHAND_ROOT=str(comp))
    for old,new in [('verify_and_freeze.py','verify.py'),('evaluate_public.py','evaluate.py')]:
        source=(PUBLIC/'src'/old).read_text().replace('DiT为公开v48 UmeTrack20提案，经显式MANO拟合后进入同一BA','原生MANO右手DiT普通训练，开发验证后事后选择的完整回放对照')
        (comp/'src'/new).write_text(source);run([comp/'src'/new],f'dit_{new}',current)
    run([PUBLIC/'src/make_zoom_review.py','--run-root',comp],'dit_zoom',current)
    (R/'receipts/exploratory_full_dit_done.json').write_text(json.dumps(dict(complete=True,posthoc=True,production_admitted=False),indent=2))
if __name__=='__main__':main()

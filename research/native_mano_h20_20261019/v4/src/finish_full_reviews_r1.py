"""Finish reviews from sealed BA results; do not rerun geometry after mkdir error."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path('/mnt/why/native_mano_rollout_v4_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019');VIEW=R/'full_pipeline'

def main():
    manifests={}
    for arm in ['dit','protected','guarded']:
        folder=VIEW/f'outputs/ba_rollout_{arm}';seal=json.loads((folder/'seal.json').read_text())
        for name,h in seal['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h
        manifests[str((folder/'seal.json').relative_to(VIEW))]=hashlib.sha256((folder/'seal.json').read_bytes()).hexdigest()
    (VIEW/'receipts/all_final_outputs_frozen_r1.json').write_text(json.dumps(dict(complete=True,files=manifests,previous_review_error='missing comparisons parent; BA/guard computation already completed and preserved',GT_used_in_inference=False),indent=2))
    for arm in ['dit','protected','guarded']:
        comp=R/f'comparisons/full_rollout_{arm}';comp.mkdir(parents=True,exist_ok=False)
        for name in ['data','assets','vendor']:(comp/name).symlink_to(VIEW/name,target_is_directory=True)
        for name in ['src','receipts','outputs']:(comp/name).mkdir()
        for name in ['wilor_clip2102','head_clip2102','depth_clip2102','depth_fused_clip2102','depth_protected_clip2102','mano_baseline_clip2102']:(comp/'outputs'/name).symlink_to(VIEW/'outputs'/name,target_is_directory=True)
        (comp/'outputs/mano_dit_clip2102').symlink_to(VIEW/f'outputs/ba_rollout_{arm}',target_is_directory=True)
        for old,new in [('verify_and_freeze.py','verify.py'),('evaluate_public.py','evaluate.py')]:
            source=(PUBLIC/'src'/old).read_text().replace('DiT为公开v48 UmeTrack20提案，经显式MANO拟合后进入同一BA',f'实际采样训练原生MANO DiT {arm}，使用相同cuVSLAM预测轨迹、物理深度和BA；guarded为BA后原阈值的整轨迹MANO参数保护')
            (comp/'src'/new).write_text(source)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PUBLIC_HEADHAND_ROOT=str(comp),PUBLIC_DEVELOPMENT_REPLAY='1',PYTHONPATH=str(PUBLIC/'src'))
        for script in ['verify.py','evaluate.py']:
            with (R/'logs'/f'full_{arm}_{script}_r1.log').open('w') as log:subprocess.run([sys.executable,str(comp/'src'/script)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (R/'logs'/f'full_{arm}_zoom_r1.log').open('w') as log:subprocess.run([sys.executable,str(PUBLIC/'src/make_zoom_review.py'),'--run-root',str(comp)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        m=json.loads((comp/'outputs/review_clip2102/metrics.json').read_text());print(json.dumps(dict(arm=arm,metrics=m['metrics']['DiT_BA'],harm=m['DiT_vs_BA_protection_camera'])),flush=True)
    (R/'receipts/full_final_guard_done.json').write_text((VIEW/'outputs/ba_rollout_guarded/optimization.json').read_text())
    (R/'receipts/full_pipeline_done.json').write_text(json.dumps(dict(complete=True,reference='identical full public cuVSLAM/LightESS/UniDepth/BA inputs from v2',production_admitted=False,diagnostic_replay_only=True,BA_outputs_preserved_after_review_mkdir_repair=True),indent=2))

if __name__=='__main__':main()

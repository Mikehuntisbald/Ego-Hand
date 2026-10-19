"""Matched head-only / native-DiT+head BA; original gates and frozen physics."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
R=Path('/mnt/why/native_rgb_observation_v9_20261019');V7=Path('/mnt/why/native_mano_pool_v7_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019');VIEW=R/'full_pipeline'
ARMS=['BA_spatial','BA_pooled','DiT_spatial','DiT_pooled']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((R/'observations_frozen.json').read_text())['complete'];assert not (R/'full_status.json').exists()
    assert int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())==0,'GPU3 occupied'
    VIEW.mkdir(exist_ok=False)
    for n in ['outputs','receipts']:(VIEW/n).mkdir()
    for n in ['data','assets','vendor']:(VIEW/n).symlink_to(V2/'coherent_frontend'/n,target_is_directory=True)
    common=['wilor_clip2102','rf_clip2102','head_clip2102','depth_clip2102','depth_fused_clip2102','depth_protected_clip2102','mano_baseline_clip2102']
    for n in common:(VIEW/'outputs'/n).symlink_to(V2/'coherent_frontend/outputs'/n,target_is_directory=True)
    (VIEW/'outputs/native_full').symlink_to(V7/'full_pipeline/outputs/native_full',target_is_directory=True)
    native=VIEW/'outputs/native_full/native_parameters.npz';env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PUBLIC_HEADHAND_ROOT=str(VIEW),PYTHONPATH=os.pathsep.join([str(R/'src'),str(V2/'src'),str(PUBLIC/'src')]))
    def run(args,name):
        (R/'full_status.json').write_text(json.dumps(dict(active=True,pid=os.getpid(),stage=name,unix=time.time()),indent=2))
        with (R/'logs'/f'full_{name}.log').open('w') as log:subprocess.run([sys.executable,*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(stage=name,complete=True)),flush=True)
    run([R/'eval_src/prepare_ba_adapter.py'],'prepare_adapter')
    source=(V7/'eval_src/guard_geometry_v7.py').read_text();source=source.replace("R=Path('/mnt/why/native_mano_pool_v7_20261019')",f"R=Path('{R}')");assert "choices=['small','full']" in source;source=source.replace("choices=['small','full']",f'choices={ARMS!r}')
    guard=R/'eval_src/guard_rgb.py';assert not guard.exists();guard.write_text(source)
    lock=dict(native_source_sha256=sha(native),observation_seal_sha256=sha(R/'observations_frozen.json'),BA_adapter_sha256=sha(R/'eval_src/native_ba_rgb.py'),guard_source_sha256=sha(guard),arms=ARMS,unchanged_physics_and_weights=True,GT_inputs=False,production_admitted=False)
    (VIEW/'receipts/predeclared.json').write_text(json.dumps(lock,indent=2))
    # A no-observation run must reproduce the prior complete native DiT BA.
    run([R/'eval_src/native_ba_rgb.py','--depth','depth_protected_clip2102','--native',native,'--output',VIEW/'outputs/BA_parity'],'BA_parity')
    a=np.load(VIEW/'outputs/BA_parity/parameters.npz');b=np.load(V7/'full_pipeline/outputs/ba_full/parameters.npz');diff={k:float(np.max(np.abs(a[k]-b[k]))) for k in ['state_world51','betas_shared','joints_world','joints_camera']};assert max(diff.values())<1e-6,diff
    (VIEW/'receipts/BA_parity.json').write_text(json.dumps(dict(complete=True,difference_max=diff,source_output_sha256=sha(V7/'full_pipeline/outputs/ba_full/parameters.npz')),indent=2))
    for arm in ARMS:
        mode=arm.split('_')[1];command=[R/'eval_src/native_ba_rgb.py','--depth','depth_protected_clip2102','--image-observations',R/f'predictions/clip-002102/{mode}.npz','--output',VIEW/f'outputs/ba_{arm}','--suffix','independent_RGB_v9']
        if arm.startswith('DiT'):command+=['--native',native]
        run(command,arm)
    for arm in ARMS:run([guard,'--arm',arm],arm+'_guard')
    seals={}
    for arm in [a+s for a in ARMS for s in ['', '_guarded']]:
        p=VIEW/f'outputs/ba_{arm}/seal.json';assert p.exists();seals[str(p.relative_to(VIEW))]=sha(p)
    (VIEW/'receipts/predictions_frozen.json').write_text(json.dumps(dict(complete=True,freeze_unix=time.time(),GT_read=False,seals=seals,historical_replay=True),indent=2));(R/'full_status.json').write_text(json.dumps(dict(active=False,complete=True,all_eight_predictions_frozen=True,production_admitted=False),indent=2))
if __name__=='__main__':main()

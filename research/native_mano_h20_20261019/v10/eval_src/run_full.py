"""Same physical and independent-RGB BA after three native selectors."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
R=Path('/mnt/why/native_candidate_sequence_v10_20261019');V9=Path('/mnt/why/native_rgb_observation_v9_20261019');V7=Path('/mnt/why/native_mano_pool_v7_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019');VIEW=R/'full_pipeline'
ARMS=['frame_RGB','temporal_noRGB','temporal_RGB','temporal_noRGB_scaled','temporal_RGB_scaled']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((R/'all_predictions_frozen.json').read_text())['complete'];assert not (R/'full_status.json').exists();assert int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())==0
    VIEW.mkdir(exist_ok=False);(R/'eval_src').mkdir(exist_ok=True)
    for n in ['outputs','receipts']:(VIEW/n).mkdir()
    for n in ['data','assets','vendor']:(VIEW/n).symlink_to(V2/'coherent_frontend'/n,target_is_directory=True)
    common=['wilor_clip2102','rf_clip2102','head_clip2102','depth_clip2102','depth_fused_clip2102','depth_protected_clip2102','mano_baseline_clip2102']
    for n in common:(VIEW/'outputs'/n).symlink_to(V2/'coherent_frontend/outputs'/n,target_is_directory=True)
    source=(V9/'eval_src/guard_rgb.py').read_text();assert "R=Path('/mnt/why/native_rgb_observation_v9_20261019')" in source;source=source.replace("R=Path('/mnt/why/native_rgb_observation_v9_20261019')",f"R=Path('{R}')");assert "choices=['BA_spatial', 'BA_pooled', 'DiT_spatial', 'DiT_pooled']" in source;source=source.replace("choices=['BA_spatial', 'BA_pooled', 'DiT_spatial', 'DiT_pooled']",f'choices={ARMS!r}');guard=R/'eval_src/guard_sequence.py';assert not guard.exists();guard.write_text(source)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PUBLIC_HEADHAND_ROOT=str(VIEW),PYTHONPATH=os.pathsep.join([str(V2/'src'),str(PUBLIC/'src')]))
    def run(args,name):
        (R/'full_status.json').write_text(json.dumps(dict(active=True,pid=os.getpid(),stage=name,unix=time.time()),indent=2))
        with (R/'logs'/f'full_{name}.log').open('w') as log:subprocess.run([sys.executable,*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(stage=name,complete=True)),flush=True)
    observation=V9/'predictions/clip-002102/spatial.npz';native=V7/'full_pipeline/outputs/native_full/native_parameters.npz';adapter=V9/'eval_src/native_ba_rgb.py'
    (VIEW/'receipts/predeclared.json').write_text(json.dumps(dict(arms=ARMS,source_selection_seal_sha256=sha(R/'all_predictions_frozen.json'),image_observation_sha256=sha(observation),BA_adapter_sha256=sha(adapter),guard_sha256=sha(guard),same_all_physics_weights=True,GT_inputs=False,production_admitted=False),indent=2))
    run([adapter,'--depth','depth_protected_clip2102','--image-observations',observation,'--native',native,'--output',VIEW/'outputs/mean_parity'],'mean_parity')
    a=np.load(VIEW/'outputs/mean_parity/parameters.npz');b=np.load(V9/'full_pipeline/outputs/ba_DiT_spatial/parameters.npz');diff={k:float(np.abs(a[k]-b[k]).max()) for k in ['state_world51','betas_shared','joints_world','joints_camera']};assert max(diff.values())<1e-6,diff;(VIEW/'receipts/mean_BA_parity.json').write_text(json.dumps(dict(complete=True,difference=diff),indent=2))
    for arm in ARMS:
        selected=R/('likelihood_r1/predictions' if arm.endswith('_scaled') else 'predictions')/f'clip-002102/{arm}.npz'
        run([adapter,'--depth','depth_protected_clip2102','--image-observations',observation,'--native',selected,'--output',VIEW/f'outputs/ba_{arm}','--suffix','native_candidate_sequence_v10'],arm)
    for arm in ARMS:run([guard,'--arm',arm],arm+'_guard')
    seals={}
    for arm in [x+s for x in ARMS for s in ['', '_guarded']]:
        p=VIEW/f'outputs/ba_{arm}/seal.json';assert p.exists();seals[str(p.relative_to(VIEW))]=sha(p)
    (VIEW/'receipts/predictions_frozen.json').write_text(json.dumps(dict(complete=True,freeze_unix=time.time(),seals=seals,GT_read=False,historical_replay=True),indent=2));(R/'full_status.json').write_text(json.dumps(dict(active=False,complete=True,all_ten_predictions_frozen=True,production_admitted=False),indent=2))
if __name__=='__main__':main()

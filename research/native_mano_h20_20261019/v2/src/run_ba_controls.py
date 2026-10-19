"""Isolate the chirality flag repair with sealed v1 inputs and unchanged BA."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
R=Path('/mnt/why/native_mano_dit_v2_20261019')
V1=Path('/mnt/why/native_mano_dit_20261019')
PUBLIC=Path('/mnt/why/public_headhand_h20_20261019/runs/replay_r3')
SOURCE=Path('/mnt/why/public_headhand_h20_20261019/src')
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PUBLIC_HEADHAND_ROOT=str(PUBLIC),PUBLIC_DEVELOPMENT_REPLAY='1',PYTHONPATH=str(SOURCE))

def run(args,name,env=ENV):
    with (R/'logs'/f'{name}.log').open('w') as log:
        subprocess.run([sys.executable,*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
    for name,native in [('baseline',None),('native_raw',V1/'outputs/native_raw/native_parameters.npz'),('native_protected',V1/'outputs/native_protected/native_parameters.npz')]:
        output=R/'outputs'/f'ba_{name}'
        args=[R/'src/native_ba.py','--depth','depth_protected_clip2102','--output',output,'--suffix','chirality_repair']
        if native:args+=['--native',native]
        run(args,f'ba_{name}')
        print(json.dumps(dict(stage=name,complete=True)),flush=True)
    for arm in ['native_raw','native_protected']:
        view=R/'comparisons'/arm;view.mkdir(parents=True,exist_ok=False)
        for name in ['data','assets','vendor']:(view/name).symlink_to(PUBLIC/name,target_is_directory=True)
        for name in ['src','receipts','outputs']:(view/name).mkdir()
        for name in ['wilor_clip2102','head_clip2102','depth_clip2102','depth_fused_clip2102','depth_protected_clip2102']:
            (view/'outputs'/name).symlink_to(PUBLIC/'outputs'/name,target_is_directory=True)
        for alias,path in [('mano_baseline_clip2102',R/'outputs/ba_baseline'),('mano_dit_clip2102',R/f'outputs/ba_{arm}')]:
            (view/'outputs'/alias).symlink_to(path,target_is_directory=True)
        for old,new in [('verify_and_freeze.py','verify.py'),('evaluate_public.py','evaluate.py')]:
            source=(SOURCE/old).read_text()
            if new=='evaluate.py':source=source.replace('DiT为公开v48 UmeTrack20提案，经显式MANO拟合后进入同一BA','原生MANO参数DiT；本轮仅修复BA错误覆盖左右手标记，其他输入和损失不变')
            (view/'src'/new).write_text(source)
        env=dict(ENV,PUBLIC_HEADHAND_ROOT=str(view))
        for script in ['verify.py','evaluate.py']:run([view/'src'/script],f'{arm}_{script}',env)
        run([SOURCE/'make_zoom_review.py','--run-root',view],f'{arm}_zoom',env)
        metrics=json.loads((view/'outputs/review_clip2102/metrics.json').read_text())
        print(json.dumps(dict(arm=arm,metrics=metrics['metrics'],temporal=metrics['temporal'],harm=metrics['DiT_vs_BA_protection_camera'])),flush=True)
    (R/'receipts/chirality_done.json').write_text(json.dumps(dict(complete=True,GT_used_in_inference=False,source_sha256=hashlib.sha256((R/'src/native_ba.py').read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()

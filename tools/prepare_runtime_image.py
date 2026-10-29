"""Create a self-contained, hash-pinned OFFLINE build context, never copy weights."""
import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ego_hand_runtime.io import sha256,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--wheelhouse');p.add_argument('--output',required=True);p.add_argument('--code-only',action='store_true')
    p.add_argument('--wilor-source');p.add_argument('--unidepth-source');p.add_argument('--extra-wheel',action='append',default=[]);args=p.parse_args()
    source=Path(__file__).resolve().parents[1];out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'wheelhouse').mkdir()
    (out/'extra_wheels').mkdir()
    for name in args.extra_wheel:
        f=Path(name).resolve()
        if f.suffix!='.whl':p.error('extra dependencies must be pinned wheel entities')
        shutil.copy2(f,out/'extra_wheels'/f.name)
    atomic_json(out/'extra_wheels_sha256.json',dict(schema=1,scope='additional pinned dependencies',
                 files={p.name:sha256(p) for p in (out/'extra_wheels').iterdir()}))
    if not args.code_only:
        if not args.wheelhouse:p.error('--wheelhouse required unless --code-only')
        wheels=Path(args.wheelhouse).resolve()
        for f in wheels.iterdir():
            if f.suffix=='.whl' or f.name=='chumpy-0.70.tar.gz':shutil.copy2(f,out/'wheelhouse'/f.name)
    atomic_json(out/'wheelhouse_sha256.json',dict(schema=1,scope='pinned dependency wheel bytes',files={p.name:sha256(p) for p in (out/'wheelhouse').iterdir()}))
    for name in ['ego_hand_runtime','configs/runtime','tests/runtime']:
        shutil.copytree(source/name,out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (out/'tools').mkdir()
    for name in ['verify_runtime_release.py','preflight_runtime_backend.py','import_native_runtime_replay.py','build_runtime_ess.py','review_runtime_output.py','verified_runtime_entry.py']:
        shutil.copy2(source/'tools'/name,out/'tools'/name)
    (out/'research').mkdir();shutil.copy2(source/'research/hand3d_rollout_v8.py',out/'research/hand3d_rollout_v8.py')
    native=Path('research/native_mano_h20_20261019/v2/src/native_ba.py')
    (out/native).parent.mkdir(parents=True);shutil.copy2(source/native,out/native)
    (out/'vendor').mkdir()
    for name,folder in [('wilor',args.wilor_source),('unidepth',args.unidepth_source)]:
        if not folder:continue
        folder=Path(folder).resolve()
        for f in folder.rglob('*'):
            if not f.is_file() or any(x in {'.git','__pycache__','.venv','checkpoints','pretrained_models'} for x in f.relative_to(folder).parts):continue
            if f.suffix.lower() not in {'.py','.yaml','.yml','.json','.txt','.md','.toml','.ini','.cfg'} and not f.name.lower().startswith(('license','copying')):continue
            dst=out/'vendor'/name/f.relative_to(folder);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
    files={p.relative_to(out).as_posix():sha256(p) for folder in ['ego_hand_runtime','configs','tests','tools','research','vendor'] for p in sorted((out/folder).rglob('*')) if p.is_file()}
    atomic_json(out/'runtime_source_sha256.json',dict(schema=1,scope='independent runtime code and tests; no licensed weights',files=files))
    shutil.copy2(source/('docker/runtime/Dockerfile.code' if args.code_only else 'docker/runtime/Dockerfile'),out/'Dockerfile')
    print(dict(context=str(out),files=len(files),wheels=len(list((out/'wheelhouse').iterdir())),full_pipeline_accepted=False))


if __name__=='__main__':main()

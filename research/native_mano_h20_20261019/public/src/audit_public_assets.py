"""Record actually used public assets and installed runtimes; no model substitution."""
import hashlib,json,subprocess,time
from pathlib import Path
import os
ROOT=Path(os.environ.get('PUBLIC_HEADHAND_ROOT','/mnt/why/public_headhand_h20_20261019'))
PUBLIC=Path('/mnt/why/HOT3D/experiments')
paths=dict(
    LightESS_ONNX=ROOT/'assets/ess_legacy_public/dnn_stereo_disparity_v4.1.0_onnx/light_ess.onnx',
    ESS_plugin=ROOT/'assets/ess_legacy_public/dnn_stereo_disparity_v4.1.0_onnx/plugins/x86_64/ess_plugins.so',
    LightESS_H20_engine=ROOT/'assets/lightess_h20_fp16.engine',
    UniDepth_weights=ROOT/'assets/unidepth_public/model.safetensors',
    UniDepth_config=ROOT/'assets/unidepth_public/config.json',
    RF_SegSmall=PUBLIC/'rfdetr_multidata_v53B_20261007/strict_review_r1/best_development_ema.pth',
    WiLoR=PUBLIC/'yolo26_wilor_3d_20261003/assets/wilor_final.mirror.ckpt',
    WiLoR_side_helper=PUBLIC/'detector_compare_wilor_20261003/wilor_detector.pt',
    MANO_RIGHT=PUBLIC/'yolo26_wilor_3d_20261003/assets/MANO_RIGHT.pkl',
    DiT_protected_v48=PUBLIC/'online_rgb_iterative_v48/protected/best.pt',
    public_RGB_stereo=ROOT/'assets/clip-002102-full.tar')
assets={}
for name,path in paths.items():
    assert path.is_file(),path
    with path.open('rb') as file:sha=hashlib.file_digest(file,'sha256').hexdigest()
    assets[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha)
def command(cmd):return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
sources={name:command(['git','-C',str(ROOT/'vendor'/name),'rev-parse','HEAD']) for name in ['cuVSLAM','UniDepth']}
sources['Ego-Hand']='89b4764766e8c7d527eb4a228a87129283f53125 (verified git archive)'
envs={}
for name,python in dict(head=ROOT/'.venv-head/bin/python',depth=ROOT/'.venv-depth/bin/python',hand=PUBLIC/'yolo26_wilor_3d_20261003/venv/bin/python',RF=PUBLIC/'rfdetr_hand_instance_v52_20261007/venv/bin/python').items():
    envs[name]=command([str(python),'-c',"import sys,importlib.metadata as m,json; names=['torch','numpy','opencv-python','opencv-python-headless','tensorrt-cu12','cuvslam-cu12','rfdetr','smplx']; d={'python':sys.version,'executable':sys.executable}; exec('for n in names:\\n try:d[n]=m.version(n)\\n except m.PackageNotFoundError:pass'.replace('\\\\n','\\n')); print(json.dumps(d))"])
report=dict(complete=True,unix=time.time(),assets=assets,sources=sources,environments=envs,
            GPU=command(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv,noheader']),
            CPU=command(['lscpu']),source_root=str(ROOT/'src'),
            precision='RF public PyTorch; WiLoR BF16; cuVSLAM official H20-compatible binary; LightESS TRT FP16; UniDepth public FP16; MANO BA FP32',
            public_licenses='Public availability is not uniform commercial licensing. UniDepth source/weights CC-BY-NC-4.0; MANO retains its separate license; no license grants are implied.')
path=ROOT/'receipts/actual_assets_and_environment.json';path.write_text(json.dumps(report,indent=2));print(json.dumps(dict(complete=True,assets=len(assets))))

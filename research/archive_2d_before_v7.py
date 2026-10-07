"""Immutable record of the 2D branch before switching the task output to 3D."""
import json,hashlib,shutil,tarfile,time
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D/experiments');CODE=Path('/mnt/why/hot3d_hand_residual')
DEST=ROOT/'archive_2d_before_3d_v7_20261003';DEST.mkdir(exist_ok=True)
names=['offline_keypoint_rgb_v2','offline_keypoint_spatial_v3','natural_reliability_v4','natural_visual_ft_v5','temporal_sampling_v6']
manifest={};sources=[]
for name in names:
    run=ROOT/name
    for p in run.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        # Inference checkpoints, source, protocols, predictions and evidence.
        retain=(p.suffix in ['.json','.py','.html','.txt','.png','.npz'] or
                (p.suffix=='.pt' and (p.name in ['best.pt','risk_joint.pt','risk_box.pt','risk_projection.pt'] or 'sealed' in p.parts)))
        if not retain:continue
        rel=p.relative_to(ROOT);dest=DEST/'snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        digest=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
        manifest[str(rel)]=dict(bytes=p.stat().st_size,sha256=digest,original=str(p))
    sources.append(dict(run=str(run),cache_policy='Large training feature caches remain at their original paths; 2D source, inference checkpoints, metrics and review artifacts copied and hashed'))
code=DEST/'snapshot/code';code.mkdir(exist_ok=True)
for p in CODE.glob('*.py'):
    shutil.copy2(p,code/p.name);manifest['code/'+p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),original=str(p))
note='2D branch archived before 3D v7\n\nThe v3/v4/v5/v6 temporal completers predict 20x2 image-coordinate residuals, despite WiLoR upstream producing 3D. Their results quantify 2D refinement only and do not establish a 3D recovery limit.\n\nThe new task must output 20x3 camera-space joints in meters, with separate wrist translation and wrist-relative pose. Multi-frame XYZ must be aligned using camera extrinsics. Existing experiments remain unchanged.\n'
(DEST/'README.txt').write_text(note)
(DEST/'manifest.json').write_text(json.dumps(dict(created_unix=time.time(),files=manifest,sources=sources),indent=2))
archive=DEST/'snapshot.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
    for name in ['snapshot','manifest.json','README.txt']:tar.add(DEST/name,arcname=name)
(DEST/'archive_receipt.json').write_text(json.dumps(dict(files=len(manifest),archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2))
print((DEST/'archive_receipt.json').read_text())

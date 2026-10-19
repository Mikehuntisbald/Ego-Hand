"""Package native source, inference weights, real outputs and verification."""
import hashlib,json,shutil,tarfile
from pathlib import Path
import torch
ROOT=Path('/mnt/why/native_mano_dit_20261019');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019/runs/replay_r3')
models=ROOT/'models';models.mkdir(exist_ok=True)
for arm in ['regression','dit','protected']:
    ck=torch.load(ROOT/'runs'/arm/'best_development.pt',weights_only=False,map_location='cpu')
    torch.save(dict(model=ck['model'],config=ck['config'],arm=arm,step=ck['step'],GT_inference=False,
                    native_codec=dict(translation_scale=0.,root_rotation_scale=.2,local_rotation_scale=.8,shape='fixed WiLoR',geometry_dtype='float32'),
                    source='600-step pilot, best_development only; not safe default'),models/f'{arm}.pt')
paths=[]
for base in [ROOT/'src',ROOT/'receipts',ROOT/'models',ROOT/'evaluation']:
    paths.extend(p for p in base.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts)
for arm in ['regression','dit','protected']:
    paths.extend((ROOT/'runs'/arm/n) for n in ['done.json','selection.json','progress.json'])
for arm in ['raw','protected']:
    paths.extend(p for p in (ROOT/f'comparisons/{arm}/outputs/review_clip2102').rglob('*') if p.is_file())
    paths.append(ROOT/f'comparisons/{arm}/summary.json')
    paths.extend(p for p in (ROOT/f'comparisons/{arm}/receipts').glob('*') if p.is_file())
for name in ['native_raw','native_protected','ba_native_raw','ba_native_protected']:
    paths.extend(p for p in (ROOT/'outputs'/name).rglob('*') if p.is_file())
for n in ['selection.json','ready.json']:paths.append(ROOT/'data'/n)
paths.extend([ROOT/'README.md',ROOT/'report.html'])
paths=list(dict.fromkeys(paths))
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
receipt=ROOT/'delivery_files.json';receipt.write_text(json.dumps(dict(complete=True,files=manifest,
    pretrained_WiLoR_or_MANO_assets_included=False,optimizer_checkpoints='retained remotely',
    private_repository_published=False,native_parameter_DiT=True),indent=2));paths.append(receipt)
archive=ROOT/'delivery_native_mano_v1.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for path in paths:tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
output=dict(complete=True,archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(paths))
(ROOT/'delivery_archive.json').write_text(json.dumps(output,indent=2));print(json.dumps(output))

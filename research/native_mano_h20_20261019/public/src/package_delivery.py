"""Package source, receipts and real results. No weights or private source."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path('/mnt/why/public_headhand_h20_20261019');run=ROOT/'runs/replay_r3'
assert json.loads((run/'status.json').read_text())['complete']
paths=[ROOT/'README.md',ROOT/'receipts/actual_assets_and_environment.json',run/'status.json',run/'plan.json']
paths+=list((ROOT/'src').glob('*.py'))+list((run/'receipts').glob('*'))+list((run/'logs').glob('*.log'))
for name in ['mano_baseline_clip2102','mano_dit_clip2102','review_clip2102','head_clip2102','depth_clip2102','depth_fused_clip2102','depth_protected_clip2102','dit_clip2102']:
    folder=run/'outputs'/name
    for path in folder.rglob('*'):
        # A checkpoint adapter is an asset reference, not something to redistribute.
        if path.is_file() and not path.is_symlink() and 'checkpoint_adapter' not in path.parts:paths.append(path)
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
index=ROOT/'receipts/delivery_files.json';index.write_text(json.dumps(dict(complete=True,files=manifest,weights_included=False,private_source_included=False),indent=2));paths.append(index)
output=ROOT/'delivery_public_h20_r3.tar.gz'
with tarfile.open(output,'w:gz') as archive:
    for path in paths:archive.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
print(json.dumps(dict(path=str(output),bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),files=len(paths))))

"""Paths and immutable subject-out-of-fold study helpers."""
import hashlib,json
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D')
OLD=ROOT/'experiments/dit_lowconfidence_v1'
RUN=ROOT/'experiments/dit_subject_oof_v2'
PROJECT=Path('/mnt/why/hot3d_hand_residual')
SUBJECTS=['P0001','P0002','P0009','P0011','P0012','P0014']
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


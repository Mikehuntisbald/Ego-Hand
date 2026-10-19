"""Make the predeclared invalid->baseline rule explicit for every variant.

The raw sealed erode4 branch kept invalid values, while iter5 already fell back.
Preserve that evidence; create a separate final prediction before scoring.
"""
import hashlib,json
from pathlib import Path
R=Path('/mnt/why/native_depth_audit_v8_20261019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    seal=json.loads((R/'predictions_frozen.json').read_text());assert sha(R/'predictions.json')==seal['predictions_sha256']
    out=json.loads((R/'predictions.json').read_text());fallback=[]
    for row in out['observations']:
        for name,c in row['candidates'].items():
            if name=='baseline':continue
            c['raw_candidate_valid']=bool(c['valid']) and not c.get('fallback_to_baseline',False)
            if not c['valid']:
                c['invalid_raw_candidate']=dict(c);c.update(row['candidates']['baseline']);c['fallback_to_baseline']=True;fallback.append([row['observation'],name])
    p=R/'predictions_r1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2))
    (R/'predictions_frozen_r1.json').write_text(json.dumps(dict(complete=True,GT_used=False,predictions_sha256=sha(p),original_predictions_sha256=seal['predictions_sha256'],raw_seal_sha256=sha(R/'predictions_frozen.json'),source_sha256=sha(Path(__file__)),additional_fallbacks=fallback,note='Apply already-declared fallback to raw erode4 too; no labels opened; no parameter changes',production_admitted=False),indent=2))
if __name__=='__main__':main()

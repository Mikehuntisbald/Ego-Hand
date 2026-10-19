"""Unify natural-case captions with the already-19-point aggregate protocol."""
import hashlib,json,re,shutil,tarfile
from pathlib import Path
import numpy as np
R=Path('/mnt/why/native_mano_rollout_v4_20261019');V=Path('/mnt/why/native_mano_multiseq_v3_20261019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    src=R/'delivery';d=R/'delivery_r1';shutil.copytree(src,d)
    arrays=np.load(V/'evaluation_r3/labels_and_errors.npz');protocol=json.loads((V/'protocol.json').read_text());offset={};count=0
    for clip in protocol['clips']:
        offset[clip['clip']]=count;count+=len(json.loads((Path(clip['view'])/'outputs/wilor_clip2102/observations.json').read_text())['frames'])
    report=d/'reviews/validation/report.html';old=report.read_text();(report.parent/'report_legacy20point_captions.html').write_text(old);edits=[]
    def fix(m):
        name,caption=m.group(1),m.group(2);stem=Path(name).stem;clip=int(stem.split('_')[0]);observation=int(stem.rsplit('_',1)[1]);ix=offset[clip]+observation
        keep=np.arange(20)!=5;a=float(np.nanmean(arrays['WiLoR_error_relative_mm'][ix,keep]));b=float(np.nanmean(arrays['rollout_dit_displacement_error_relative_mm'][ix,keep]))
        corrected=re.sub(r'腕对齐 [\d.]+ → [\d.]+ mm',f'腕对齐19点 {a:.1f} → {b:.1f} mm',caption);assert corrected!=caption
        edits.append(dict(image=name,clip=clip,observation=observation,WiLoR_mm=a,rollout_guard_mm=b))
        return f'<figure><img src="images/{name}"><figcaption>{corrected}</figcaption></figure>'
    new=re.sub(r'<figure><img src="images/([^"]+)"><figcaption>(.*?)</figcaption></figure>',fix,old);assert len(edits)==18
    report.write_text(new)
    receipt=dict(complete=True,caption_protocol='exclude zero wrist, same 19-point wrist-aligned definition as tables',cases=len(edits),edits=edits,old_html_sha256=hashlib.sha256(old.encode()).hexdigest(),new_html_sha256=sha(report),all_model_predictions_and_statistics_unchanged=True)
    (d/'receipts/case_caption_protocol_r1.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    shutil.copy2(Path(__file__),d/'src/rollout/correct_delivery_captions_r1.py')
    original=json.loads((src/'delivery_files.json').read_text())['files']
    for name,h in original.items():
        if name.startswith(('models/','predictions/','outputs/')):assert sha(d/name)==h,name
    files={str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file() and p.name!='delivery_files.json'}
    (d/'delivery_files.json').write_text(json.dumps(dict(complete=True,files=files),indent=2))
    archive=R/'delivery_native_mano_v4_r1.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(d,arcname='delivery')
    result=dict(complete=True,archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files),source_delivery_preserved=True)
    (R/'delivery_archive_r1.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()

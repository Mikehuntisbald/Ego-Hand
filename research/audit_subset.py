"""Audit downloaded metadata and produce a compact report; --final verifies hashes."""
import argparse, collections, hashlib, json, time
from pathlib import Path

ROOT=Path('/mnt/why/HOT3D')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');a=ap.parse_args()
    manifest=json.loads((ROOT/'subset_manifest.json').read_text())
    assert len(manifest['train_subjects'])==6 and len(manifest['validation_subjects'])==3
    assert not set(manifest['train_subjects'])&set(manifest['validation_subjects'])
    subjects={};sequence_rows=[];missing=[];max_roundtrip=0.;archive_bytes=0;source_bytes=0;net_bytes=0;keypoints=valid_kp=0;total_hands=total_boxes=0;annotation_hashes=0;archive_hashes=0;zero_label_frames=0
    seen=set()
    for sequence in manifest['sequences']:
        split,subject,seq=sequence['split'],sequence['subject'],sequence['sequence'];complete=0;frames=0;hands=0
        assert subject==seq.split('_')[0]
        for clip in sequence['clips']:
            assert clip not in seen;seen.add(clip)
            archive=ROOT/'rgb_clips'/split/seq/f'clip-{clip:06d}.tar'
            receipt=archive.with_suffix('.receipt.json')
            exported=ROOT/'export/annotations'/split/seq/f'clip-{clip:06d}.receipt.json'
            annotation=exported.parent/f'clip-{clip:06d}.jsonl'
            if not receipt.exists() or not exported.exists():missing.append([split,seq,clip]);continue
            r=json.loads(receipt.read_text());e=json.loads(exported.read_text())
            assert r['completed'] and e['completed'] and r['sequence']==seq and e['subject']==subject
            assert r['source_revision']==manifest['revision']
            assert r['counts']==dict(rgb=150,camera=150,hands=150,info=150,shape=1)
            assert archive.stat().st_size==r['retained_archive_bytes']
            assert e['source_retained_sha256']==r['retained_archive_sha256']
            if a.final:
                assert sha(archive)==r['retained_archive_sha256'];archive_hashes+=1
                assert sha(annotation)==e['annotation_sha256'];annotation_hashes+=1
            complete+=1;frames+=e['frames'];hands+=e['hands'];total_hands+=e['hands'];total_boxes+=e['detector_boxes']
            zero_label_frames+=e['frames_without_detector_labels'];keypoints+=e['keypoints'];valid_kp+=e['valid_projected_keypoints']
            archive_bytes+=r['retained_archive_bytes'];source_bytes+=r['original_archive_bytes'];net_bytes+=r['network_bytes']
            max_roundtrip=max(max_roundtrip,e['xyz_roundtrip_max_error_m'])
        row=dict(split=split,subject=subject,sequence=seq,clips=complete,expected_clips=len(sequence['clips']),frames=frames,hands=hands)
        sequence_rows.append(row)
        if subject not in subjects:subjects[subject]=dict(split=split,sequences=0,clips=0,frames=0,hands=0)
        sub=subjects[subject];sub['sequences']+=1;sub['clips']+=complete;sub['frames']+=frames;sub['hands']+=hands
    for task in ['detect','pose']:
        for split in ['train','val']:
            listing=ROOT/'export'/task/f'{split}.txt'
            if not listing.exists():continue
            paths=listing.read_text().splitlines()
            assert len(paths)==len(set(paths))
            assert all('/'+split+'/' in p and Path(p).exists() for p in paths)
    report=dict(stage='complete' if not missing else 'in_progress',format=manifest['format'],subjects=subjects,sequences=sequence_rows,
        completed_clips=sum(s['clips'] for s in sequence_rows),expected_clips=len(seen),
        train_frames=sum(s['frames'] for s in sequence_rows if s['split']=='train'),
        val_frames=sum(s['frames'] for s in sequence_rows if s['split']=='val'),hands=total_hands,
        detector_boxes=total_boxes,frames_without_detector_labels=zero_label_frames,
        keypoints=keypoints,valid_projected_keypoints=valid_kp,
        retained_archive_bytes=archive_bytes,original_archive_bytes=source_bytes,
        selective_request_bytes=net_bytes,archive_hashes_verified=archive_hashes,
        annotation_hashes_verified=annotation_hashes,subject_disjoint=True,
        camera_roundtrip_max_error_m=max_roundtrip,missing=missing,
        keypoint_occlusion_labels_available=False,coarse_3d_or_dit_trained=False,
        source_full_archive_hash_verified=False)
    if a.final:
        assert not missing
        assert report['train_frames']==57000 and report['val_frames']==12000
    dest=ROOT/'audit_report.json';part=dest.with_suffix('.json.partial');part.write_text(json.dumps(report,indent=2));part.replace(dest)
    print(json.dumps({k:v for k,v in report.items() if k not in ['sequences','subjects','missing']},indent=2),flush=True)

if __name__=='__main__':main()

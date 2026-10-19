"""Lock new local recordings without hand labels or reference camera poses."""
import hashlib,json,re,shutil,tarfile,time
from pathlib import Path
R=Path('/mnt/why/native_candidate_risk_v11_20261019');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019');V3=Path('/mnt/why/native_mano_multiseq_v3_20261019')
TRAIN={'P0001','P0002','P0009','P0011','P0012','P0014'};OLD={'P0003','P0010','P0015'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()
def main():
    out=R/'new_recordings';out.mkdir(exist_ok=False);excluded={int(c['clip']) for c in json.loads((V3/'protocol.json').read_text())['clips']}|{2102,2114};available={}
    for split in ['train','val']:
        for seq in sorted((Path('/mnt/why/HOT3D/rgb_clips')/split).iterdir()):
            if not seq.is_dir() or not re.fullmatch(r'P[0-9]{4}_[0-9a-f]{8}',seq.name):continue
            person=seq.name[:5]
            if person in TRAIN or person=='P0003':continue
            choices=[p for p in sorted(seq.glob('clip-*.tar')) if re.fullmatch(r'clip-[0-9]{6}\.tar',p.name) and int(p.stem[5:]) not in excluded]
            if choices:available.setdefault(person,[]).append((seq,choices[len(choices)//2]))
    new_people=sorted(p for p in available if p not in OLD);people=(new_people+sorted(p for p in available if p in OLD))[:2];assert len(people)==2,'Two unused local recordings are not available';clips=[]
    for person in people:
        seq,archive=sorted(available[person],key=lambda x:x[0].name)[0];clip=int(archive.stem[5:]);view=out/'views'/archive.stem;view.mkdir(parents=True)
        for n in ['data','outputs','receipts','logs']:(view/n).mkdir()
        for n in ['assets','vendor']:(view/n).symlink_to(PUBLIC/n,target_is_directory=True)
        images=view/'data/clip-002102';images.mkdir();rows=[];files={}
        with tarfile.open(archive) as tar:
            names=sorted(m.name for m in tar.getmembers() if m.isfile() and re.fullmatch(r'[0-9]{6}\.info\.json',m.name))[:150];assert len(names)==150,'Expected150 input frames'
            for name in names:
                prefix=name[:6];info=json.load(tar.extractfile(name));cameras=json.load(tar.extractfile(f'{prefix}.cameras.json'));clean_info=dict(ref_timestamp_ns=int(info['ref_timestamp_ns']),image_timestamps_ns={key:int(info['image_timestamps_ns'][key]) for key in ['214-1','1201-1','1201-2']});clean_camera={key:dict(calibration=cameras[key]['calibration']) for key in ['214-1','1201-1','1201-2']}
                for filename,data in [(name,json.dumps(clean_info).encode()),(f'{prefix}.cameras.json',json.dumps(clean_camera).encode())]:
                    (images/filename).write_bytes(data);files[filename]=hashlib.sha256(data).hexdigest()
                for key in ['214-1','1201-1','1201-2']:
                    filename=f'{prefix}.image_{key}.jpg';member=tar.getmember(filename);assert member.isfile();data=tar.extractfile(member).read();(images/filename).write_bytes(data);files[filename]=hashlib.sha256(data).hexdigest()
                rows.append(dict(id=f'{archive.stem}_{prefix}',image=str(images/f'{prefix}.image_214-1.jpg'),group=seq.name,dataset='HOT3D',split='native_new_recording',timestamp_s=clean_info['ref_timestamp_ns']/1e9))
        (view/'data/frames.json').write_text(json.dumps(dict(frames=rows)));(view/'receipts/input_files_sha256.json').write_text(json.dumps(files,indent=2));clips.append(dict(clip=clip,sequence=seq.name,subject=person,archive=str(archive),archive_sha256=sha(archive),frames=150,view=str(view),hand_GT_read=False,reference_camera_pose_extracted=False,subject_new_to_native_diagnostics=person not in OLD))
    protocol=dict(complete=True,locked_unix=time.time(),selection='Local filenames/counts only; prefer new native-study people, otherwise unused recordings of existing optimizer/selection-excluded people; no hand labels or image error mining',clips=clips,scope='New full-pipeline recordings; original detector/backbone pretraining and whole-project historical exposure not established. Existing native-study people are explicitly flagged; do not claim fresh-person generalization.',no_hand_GT_or_reference_camera_pose_in_inputs=True,production_admitted=False);(out/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(dict(complete=True,clips=2,frames=300,new_native_study_people=sum(c['subject_new_to_native_diagnostics'] for c in clips),hand_GT_read=False,reference_camera_pose_in_inputs=False)))
if __name__=='__main__':main()

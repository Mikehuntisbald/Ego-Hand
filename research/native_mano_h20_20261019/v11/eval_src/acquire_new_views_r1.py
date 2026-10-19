"""Replace the failed RGB-only input attempt with official complete3-camera clips."""
import hashlib,json,re,shutil,tarfile,time
from pathlib import Path
import requests
R=Path('/mnt/why/native_candidate_risk_v11_20261019');OUT=R/'new_recordings_r1';PUBLIC=Path('/mnt/why/public_headhand_h20_20261019');V3=Path('/mnt/why/native_mano_multiseq_v3_20261019');TRAIN={'P0001','P0002','P0009','P0011','P0012','P0014'};OLD={'P0003','P0010','P0015'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()
def main():
    assert json.loads((R/'error_only_r1/controller_status.json').read_text())['complete'];OUT.mkdir(exist_ok=False);(OUT/'assets').mkdir();excluded={int(c['clip']) for c in json.loads((V3/'protocol.json').read_text())['clips']}|{2102,2114};available={}
    for split in ['train','val']:
        for seq in sorted((Path('/mnt/why/HOT3D/rgb_clips')/split).iterdir()):
            if not seq.is_dir() or not re.fullmatch(r'P[0-9]{4}_[0-9a-f]{8}',seq.name):continue
            person=seq.name[:5]
            if person in TRAIN or person=='P0003':continue
            choices=[p for p in sorted(seq.glob('clip-*.tar')) if re.fullmatch(r'clip-[0-9]{6}\.tar',p.name) and int(p.stem[5:]) not in excluded]
            if choices:available.setdefault(person,[]).append((seq,choices[len(choices)//2]))
    people=(sorted(p for p in available if p not in OLD)+sorted(p for p in available if p in OLD))[:2];assert len(people)==2;clips=[]
    for person in people:
        seq,archive=sorted(available[person],key=lambda x:x[0].name)[0];clip=int(archive.stem[5:]);clips.append(dict(clip=clip,sequence=seq.name,subject=person,local_RGB_source=str(archive),view=str(OUT/'views'/archive.stem),archive=str(OUT/'assets'/f'{archive.stem}.tar'),new_native_study_person=person not in OLD))
    locks={}
    for variant,base in [('initial',R),('error',R/'error_only_r1')]:
        for arm in ['pose_base','RGB_base','RGB_preserve']:locks[variant+'_'+arm]=sha(base/'runs'/arm/'best_development.pt')
    protocol=dict(complete=True,locked_unix=time.time(),clips=clips,model_sha256=locks,selection='Same deterministic filename-only choice as failed RGB-only attempt; no GT/error-based replacement',source='Official HOT3D-Clips Hugging Face repository linked by facebookresearch/hot3d; HTTPS GET assets only',no_hand_GT_or_reference_camera_pose_in_inputs=True,scope='New recordings relative to native optimization. Pretraining and whole-project exposure unestablished; existing held people explicitly flagged. Do not claim fresh-person generalization when flagged false.',production_admitted=False);(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));session=requests.Session();session.trust_env=False
    try:
        response=session.get('https://huggingface.co/api/datasets/bop-benchmark/hot3d',timeout=(15,45));assert response.status_code==200;commit=response.json()['sha'];assert re.fullmatch(r'[0-9a-f]{40}',commit);(OUT/'source_version.json').write_text(json.dumps(dict(commit=commit),indent=2))
        for clip in clips:
            ident=int(clip['clip']);assert 0<=ident<1000000;target=OUT/'assets'/f'clip-{ident:06}.tar';url=f'https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/{commit}/train_aria/clip-{ident:06}.tar';download=session.get(url,stream=True,timeout=(15,60));status=int(download.status_code);assert status==200,'Official train_aria asset unavailable';temporary=target.with_suffix('.tar.partial');h=hashlib.sha256();size=0
            with temporary.open('wb') as f:
                for block in download.iter_content(1024*1024):
                    if block:f.write(block);h.update(block);size+=len(block)
            temporary.replace(target);assert size>1000000;view=OUT/'views'/f'clip-{ident:06}';view.mkdir(parents=True)
            for n in ['data','outputs','receipts','logs']:(view/n).mkdir()
            for n in ['assets','vendor']:(view/n).symlink_to(PUBLIC/n,target_is_directory=True)
            images=view/'data/clip-002102';images.mkdir();frames=[];hashes={}
            with tarfile.open(target) as tar,tarfile.open(Path(clip['local_RGB_source'])) as local:
                names=sorted(m.name for m in tar.getmembers() if m.isfile() and re.fullmatch(r'[0-9]{6}\.info\.json',m.name));assert len(names)==150
                original_first=json.load(local.extractfile(names[0]));official_first=json.load(tar.extractfile(names[0]));assert int(original_first['ref_timestamp_ns'])==int(official_first['ref_timestamp_ns']),'Source identity mismatch';first=names[0][:6];assert hashlib.sha256(local.extractfile(f'{first}.image_214-1.jpg').read()).hexdigest()==hashlib.sha256(tar.extractfile(f'{first}.image_214-1.jpg').read()).hexdigest(),'Source RGB mismatch'
                for name in names:
                    prefix=name[:6];info=json.load(tar.extractfile(name));camera=json.load(tar.extractfile(f'{prefix}.cameras.json'));clean_info=dict(ref_timestamp_ns=int(info['ref_timestamp_ns']),image_timestamps_ns={k:int(info['image_timestamps_ns'][k]) for k in ['214-1','1201-1','1201-2']});clean_camera={k:dict(calibration=camera[k]['calibration']) for k in ['214-1','1201-1','1201-2']}
                    for filename,data in [(name,json.dumps(clean_info).encode()),(f'{prefix}.cameras.json',json.dumps(clean_camera).encode())]:
                        (images/filename).write_bytes(data);hashes[filename]=hashlib.sha256(data).hexdigest()
                    for k in ['214-1','1201-1','1201-2']:
                        filename=f'{prefix}.image_{k}.jpg';member=tar.getmember(filename);assert member.isfile();data=tar.extractfile(member).read();(images/filename).write_bytes(data);hashes[filename]=hashlib.sha256(data).hexdigest()
                    frames.append(dict(id=f'clip-{ident:06}_{prefix}',image=str(images/f'{prefix}.image_214-1.jpg'),group=clip['sequence'],dataset='HOT3D',split='native_locked_new_recording',timestamp_s=clean_info['ref_timestamp_ns']/1e9))
            (view/'data/frames.json').write_text(json.dumps(dict(frames=frames)));(view/'receipts/input_files_sha256.json').write_text(json.dumps(hashes,indent=2));(view/'receipts/acquisition.json').write_text(json.dumps(dict(complete=True,bytes=size,sha256=h.hexdigest(),source_commit=commit,HTTP_status=status,source_RGB_and_timestamp_identity=True,images=450,frames=150,hand_GT_extracted=False,reference_camera_pose_extracted=False),indent=2));print(json.dumps(dict(clip=ident,complete=True,bytes=size,frames=150,hand_GT_read=False)),flush=True)
        (OUT/'acquisition_done.json').write_text(json.dumps(dict(complete=True,clips=2,frames=300,hand_GT_read=False),indent=2))
    except Exception as exc:
        (OUT/'acquisition_failed.json').write_text(json.dumps(dict(complete=False,error_type=type(exc).__name__,evidence_preserved=True),indent=2));raise RuntimeError('Official complete-clip acquisition failed; local evidence preserved') from None
if __name__=='__main__':main()

"""Download real author-released data; keep provenance and hashes."""
import hashlib,json,os,ssl,socket,time,tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT=Path('/mnt/why/HOT3D/domain_data_v51')
LEGACY_LEAF='030cdc0b37d89bc39354898d9d53ca518b559b587b66f8b61cc297b06d2abef6'
SOURCES=[
    dict(name='surgical_hands',url='https://prism.eecs.umich.edu/natlouis/surgical_hands/surgical_hands_release.tar.gz',file='surgical_hands_release.tar.gz',expected_bytes=4868515840,source='https://github.com/MichiganCOG/Surgical_Hands_RELEASE',labels='real RGB, 2D hand keypoints, boxes, tracking IDs; no calibrated 3D ground truth',legacy=True),
    dict(name='cppe5',url='https://github.com/Rishit-dagli/CPPE-Dataset/releases/download/v0.1.0/dataset.tar.gz',file='cppe5.tar.gz',md5='f4e043f983cff94ef82ef7d57a879212',source='https://github.com/Rishit-dagli/CPPE-Dataset',labels='real PPE RGB; glove bounding boxes only; no 3D or finger GT'),
]

def download(source):
    folder=ROOT/source['name'];folder.mkdir(parents=True,exist_ok=True)
    dst=folder/source['file'];part=dst.with_suffix(dst.suffix+'.part')
    note=dict(source,started=time.time(),synthetic=False)
    verify=True
    if source.get('legacy'):
        # Only this public, unauthenticated, certificate-expired author endpoint.
        # Pin the leaf seen during the access audit; do not alter global TLS.
        host='prism.eecs.umich.edu'
        with socket.create_connection((host,443),timeout=20) as raw:
            with ssl._create_unverified_context().wrap_socket(raw,server_hostname=host) as sock:
                leaf=hashlib.sha256(sock.getpeercert(binary_form=True)).hexdigest()
        assert leaf==LEGACY_LEAF,('legacy_certificate_changed',leaf)
        verify=False;note.update(tls_exception='author certificate expired; pinned public endpoint only',leaf_sha256=leaf)
    if not dst.exists():
        start=part.stat().st_size if part.exists() else 0
        r=requests.get(source['url'],headers={'Range':f'bytes={start}-'} if start else {},stream=True,timeout=(25,90),verify=verify)
        r.raise_for_status()
        if start and r.status_code!=206:start=0
        with part.open('ab' if start else 'wb') as f:
            last=time.time()
            for chunk in r.iter_content(2**20):
                f.write(chunk);start+=len(chunk)
                if time.time()-last>20:
                    (folder/'progress.json').write_text(json.dumps(dict(bytes=start,time=time.time(),stage='download')))
                    print(json.dumps(dict(dataset=source['name'],bytes=start)),flush=True);last=time.time()
        os.replace(part,dst)
    sha=hashlib.sha256();md5=hashlib.md5()
    with dst.open('rb') as f:
        for chunk in iter(lambda:f.read(8*2**20),b''):sha.update(chunk);md5.update(chunk)
    if 'expected_bytes' in source:assert dst.stat().st_size==source['expected_bytes']
    if 'md5' in source:assert md5.hexdigest()==source['md5']
    note.update(bytes=dst.stat().st_size,sha256=sha.hexdigest(),md5_actual=md5.hexdigest())
    if not (folder/'extracted.json').exists():
        with tarfile.open(dst,'r:*') as t:t.extractall(folder,filter='data')
        (folder/'extracted.json').write_text(json.dumps(dict(complete=True,time=time.time())))
    note.update(complete=True,finished=time.time());(folder/'provenance.json').write_text(json.dumps(note,indent=2))
    print(json.dumps(note),flush=True);return note

def main():
    ROOT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(download,s) for s in SOURCES]
        results=[]
        for source,f in zip(SOURCES,futures):
            try:results.append(f.result())
            except Exception as exc:
                result=dict(name=source['name'],complete=False,error=repr(exc));results.append(result);print(json.dumps(result),flush=True)
    (ROOT/'acquisition_status.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()

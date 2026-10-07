"""Resume slow public archive transfers as validated HTTP range requests."""
import argparse,concurrent.futures,hashlib,json,os,socket,ssl,time,zipfile
from pathlib import Path
import requests
from acquire_domain_data_v51 import LEGACY_LEAF

ROOT=Path('/mnt/why/HOT3D/domain_data_v51')

def transfer(url,folder,start,length,verify=True,prefix=None,workers=6,chunk_size=4*2**20):
    folder.mkdir(exist_ok=True,parents=True);completed=0
    if prefix:
        with Path(prefix).open('rb') as f:
            f.seek(start);data=f.read(length)
        if data:(folder/'prefix.bin').write_bytes(data);completed=len(data)
    pieces=[(start+completed+i,min(chunk_size,length-completed-i)) for i in range(0,length-completed,chunk_size)]
    def get(piece):
        offset,size=piece;path=folder/f'{offset:012d}_{size:010d}.bin'
        if path.exists() and path.stat().st_size==size:return path
        for attempt in range(3):
            try:
                r=requests.get(url,headers={'Range':f'bytes={offset}-{offset+size-1}'},timeout=(20,180),verify=verify)
                assert r.status_code==206 and r.headers.get('Content-Range','').startswith(f'bytes {offset}-{offset+size-1}/'),(r.status_code,r.headers.get('Content-Range'))
                assert len(r.content)==size,(len(r.content),size)
                tmp=path.with_suffix('.tmp');tmp.write_bytes(r.content);os.replace(tmp,path);return path
            except Exception:
                if attempt==2:raise
                time.sleep(2*(attempt+1))
    begin=time.time();paths=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(get,p):p for p in pieces}
        for n,f in enumerate(concurrent.futures.as_completed(futures)):
            paths.append(f.result())
            if n%10==0:print(json.dumps(dict(stage='ranges',folder=folder.name,done=n+1,total=len(pieces),seconds=time.time()-begin)),flush=True)
    dst=folder/'assembled.bin'
    with dst.open('wb') as f:
        if completed:f.write((folder/'prefix.bin').read_bytes())
        for p in sorted(paths):
            with p.open('rb') as src:
                for data in iter(lambda:src.read(2**20),b''):f.write(data)
    assert dst.stat().st_size==length
    return dst

def surgical():
    host='prism.eecs.umich.edu'
    with socket.create_connection((host,443),timeout=20) as raw:
        with ssl._create_unverified_context().wrap_socket(raw,server_hostname=host) as sock:leaf=hashlib.sha256(sock.getpeercert(binary_form=True)).hexdigest()
    assert leaf==LEGACY_LEAF
    u='https://prism.eecs.umich.edu/natlouis/surgical_hands/surgical_hands_release.tar.gz'
    folder=ROOT/'surgical_hands';dst=transfer(u,folder/'annotation_ranges',6656,19331728,False,folder/'surgical_hands_release.tar.gz.part',workers=6,chunk_size=256*2**10)
    value=json.loads(dst.read_text());target=folder/'annotations.json';os.replace(dst,target)
    note=dict(complete=True,source=u,member='surgical_hands_release/annotations.json',tar_data_offset=6656,length=19331728,sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),legacy_leaf_sha256=leaf,images_pending=True,keys=list(value) if isinstance(value,dict) else None)
    (folder/'annotation_provenance.json').write_text(json.dumps(note,indent=2));print(json.dumps(note),flush=True)

def ego():
    folder=ROOT/'egohands';u='https://www.kaggle.com/api/v1/datasets/download/himaniishah/egohands'
    r=requests.get(u,stream=True,timeout=(20,30));r.raise_for_status();size=int(r.headers['Content-Length']);location=r.url;r.close()
    dst=transfer(location,folder/'ranges',0,size,prefix=folder/'egohands.zip.part');target=folder/'egohands.zip';os.replace(dst,target)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for p in z.namelist():assert (folder/p).resolve().is_relative_to(folder.resolve())
        z.extractall(folder)
    images=list(folder.rglob('*.jpg'));metadata=list(folder.rglob('metadata.mat'));assert len(images)>=4800 and metadata
    note=dict(complete=True,public_mirror=u,author_source='https://public.roboflow.com/object-detection/hands',original_author_citation='Bambach et al, ICCV 2015',sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),images=len(images),metadata=[str(p) for p in metadata],real_RGB=True,synthetic=False,GT_3D=False,finished=time.time())
    (folder/'provenance.json').write_text(json.dumps(note,indent=2));print(json.dumps(note),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['surgical','ego']);a=p.parse_args();globals()[a.kind]()

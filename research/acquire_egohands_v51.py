"""Retrieve an anonymous public mirror of original EgoHands, not new labels."""
from pathlib import Path
import hashlib,json,os,time,zipfile
import requests
ROOT=Path('/mnt/why/HOT3D/domain_data_v51/egohands')
URL='https://www.kaggle.com/api/v1/datasets/download/himaniishah/egohands'

def main():
    ROOT.mkdir(exist_ok=True,parents=True);dst=ROOT/'egohands.zip';part=ROOT/'egohands.zip.part'
    if not dst.exists():
        response=requests.get(URL,stream=True,timeout=(25,90));response.raise_for_status()
        assert 'text/html' not in response.headers.get('Content-Type',''), 'Mirror returned a page instead of a dataset'
        with part.open('wb') as f:
            last=time.time()
            for chunk in response.iter_content(2**20):
                f.write(chunk)
                if time.time()-last>20:
                    print(json.dumps(dict(bytes=f.tell(),stage='download')),flush=True);last=time.time()
        os.replace(part,dst)
    digest=hashlib.file_digest(dst.open('rb'),'sha256').hexdigest()
    with zipfile.ZipFile(dst) as z:
        assert z.testzip() is None
        for item in z.infolist():
            assert (ROOT/item.filename).resolve().is_relative_to(ROOT.resolve())
        z.extractall(ROOT)
    imgs=list(ROOT.rglob('*.jpg'));metadata=list(ROOT.rglob('metadata.mat'))
    note=dict(complete=True,public_mirror=URL,source='https://public.roboflow.com/object-detection/hands',original='http://vision.soic.indiana.edu/projects/egohands/',license_reference='Original EgoHands authors and author-authorized CC BY 4.0 derivative; preserve author citation',sha256=digest,images=len(imgs),metadata=[str(x) for x in metadata],real_RGB=True,synthetic=False,GT_3D=False,finished=time.time())
    assert len(imgs)>=4800 and metadata,'Incomplete original EgoHands mirror'
    (ROOT/'provenance.json').write_text(json.dumps(note,indent=2));print(json.dumps(note),flush=True)

if __name__=='__main__':main()

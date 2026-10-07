"""Repair public URL acquisition; preserve original transfer and partial bytes."""
from pathlib import Path
import json,os,zipfile,hashlib,time
from acquire_domain_ranges_v51 import ROOT,transfer

def main():
    folder=ROOT/'egohands';location=json.loads((folder/'public_location.json').read_text())
    dst=transfer(location['url'],folder/'ranges',0,location['bytes'],prefix=folder/'egohands.zip.part')
    target=folder/'egohands.zip';os.replace(dst,target)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for p in z.namelist():assert (folder/p).resolve().is_relative_to(folder.resolve())
        z.extractall(folder)
    images=list(folder.rglob('*.jpg'));metadata=list(folder.rglob('metadata.mat'));assert len(images)>=4800 and metadata
    note=dict(complete=True,public_mirror='https://www.kaggle.com/datasets/himaniishah/egohands',author_source='https://public.roboflow.com/object-detection/hands',sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),images=len(images),metadata=[str(p) for p in metadata],real_RGB=True,synthetic=False,GT_3D=False,finished=time.time())
    (folder/'provenance.json').write_text(json.dumps(note,indent=2));print(json.dumps(note),flush=True)

if __name__=='__main__':main()

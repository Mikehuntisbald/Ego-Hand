"""Index a public tar with small Range reads, fetch a real-video pilot subset."""
import json,time,hashlib,concurrent.futures,io,tarfile
from pathlib import Path
import requests
import numpy as np
from acquire_domain_ranges_v51 import ROOT

URL='https://prism.eecs.umich.edu/natlouis/surgical_hands/surgical_hands_release.tar.gz'
TOTAL=4868515840
FOLDER=ROOT/'surgical_hands'

def split(group):
    x=int(hashlib.sha256(('v51-surgical:'+group).encode()).hexdigest()[:8],16)%10
    return 'test' if x<2 else 'dev' if x==2 else 'train'

def main():
    annotations=json.loads((FOLDER/'annotations.json').read_text());chosen={};counts={}
    for clip,row in annotations.items():
        group=clip[:11];role=split(group);frames=[r for r in row['images'] if r['is_labeled']]
        for i in np.linspace(0,len(frames)-1,min(8,len(frames))).round().astype(int):
            record=frames[i];path=f"surgical_hands_release/images/{record['video_dir']}/{record['file_name']}";chosen[path]=dict(clip=clip,group=group,split=role,image=record)
    session=requests.Session();index_path=FOLDER/'tar_index.jsonl';done=FOLDER/'tar_index_done.json'
    existing=[json.loads(x) for x in index_path.read_text().splitlines()] if index_path.exists() else []
    offset=existing[-1]['next_offset'] if existing else 19342336
    started=time.time();thread=concurrent.futures.ThreadPoolExecutor(max_workers=6);downloads=[]
    extracted=FOLDER/'selected_images';extracted.mkdir(exist_ok=True)
    def extract(entry):
        dst=extracted/Path(entry['name']).relative_to('surgical_hands_release/images');dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists() and dst.stat().st_size==entry['size']:return
        for attempt in range(3):
            try:
                r=requests.get(URL,headers={'Range':f"bytes={entry['offset']+512}-{entry['offset']+512+entry['size']-1}"},timeout=(20,180),verify=False)
                assert r.status_code==206 and len(r.content)==entry['size']
                dst.write_bytes(r.content);return
            except Exception:
                if attempt==2:raise
                time.sleep(2)
    for row in existing:
        if row['name'] in chosen:downloads.append(thread.submit(extract,row))
    if not done.exists():
        with index_path.open('a') as index:
            while offset<TOTAL-1024:
                r=session.get(URL,headers={'Range':f'bytes={offset}-{offset+511}'},timeout=(20,60),verify=False)
                assert r.status_code==206 and len(r.content)==512
                b=r.content
                if not b.strip(b'\0'):break
                checksum=int(b[148:156].strip(b'\0 '),8);assert checksum==sum(b[:148])+8*32+sum(b[156:])
                name=b[:100].split(b'\0')[0].decode();prefix=b[345:500].split(b'\0')[0].decode()
                if prefix:name=prefix+'/'+name
                size=int(b[124:136].strip(b'\0 ') or b'0',8);next_offset=offset+512+((size+511)//512)*512
                entry=dict(name=name,offset=offset,size=size,next_offset=next_offset);existing.append(entry);index.write(json.dumps(entry)+'\n');index.flush()
                if name in chosen:downloads.append(thread.submit(extract,entry))
                offset=next_offset
                if len(existing)%40==0:print(json.dumps(dict(stage='surgical_tar_index',headers=len(existing),archive_fraction=offset/TOTAL,selected_images=len(downloads),seconds=time.time()-started)),flush=True)
        done.write_text(json.dumps(dict(complete=True,headers=len(existing),time=time.time())))
    for future in concurrent.futures.as_completed(downloads):future.result()
    thread.shutdown();records=[]
    for name,meta in chosen.items():
        path=extracted/Path(name).relative_to('surgical_hands_release/images')
        if not path.exists():continue
        image=meta['image'];clip=meta['clip'];hands=[a for a in annotations[clip]['annotations'] if a['image_id']==image['id']]
        records.append(dict(image=str(path),dataset='surgical_hands',id=image['id'],clip=clip,group=meta['group'],split=meta['split'],image_size=[image['frame_width'],image['frame_height']],hands=hands,GT_3D=False,real_RGB=True))
    (FOLDER/'selected_records.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    (FOLDER/'selected_provenance.json').write_text(json.dumps(dict(complete=True,real_images=len(records),source_archive=URL,source='https://github.com/MichiganCOG/Surgical_Hands_RELEASE',selection='8 uniformly sampled labelled frames per clip, source-video group split',GT_3D=False,original_RGB_unaltered=True,seconds=time.time()-started),indent=2))

if __name__=='__main__':main()

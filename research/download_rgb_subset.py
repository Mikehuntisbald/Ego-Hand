"""Read uncompressed official clip tar via HTTP ranges; retain RGB/calibration/hands.

No object or monochrome payloads are written. Small header prefetches may contain
unretained neighboring bytes. Full original archive SHA cannot be verified with
selective range access; receipts distinguish it from verified output SHA.
"""
import argparse,hashlib,io,json,os,re,time,tarfile,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests
from PIL import Image
ROOT=Path('/mnt/why/HOT3D')
SUFFIXES=('.image_214-1.jpg','.cameras.json','.hands.json','.info.json')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):h.update(part)
    return h.hexdigest()

class Ranges:
    def __init__(self,url,size):
        self.url=url;self.cdn=None;self.size=size;self.start=-1;self.cache=b'';self.session=requests.Session();self.bytes=0;self.requests=0
    def read(self,offset,size,header=False):
        if self.start<=offset and offset+size<=self.start+len(self.cache):return self.cache[offset-self.start:offset-self.start+size]
        # Frame metadata occupies <32KB. RGB-sized reads include a small tail
        # so the following info/object header can be parsed without its payload.
        amount=max(size,32768 if header and not self.cdn else 2048) if header else size+8192
        amount=min(amount,self.size-offset);end=offset+amount-1
        error=None
        for attempt in range(6):
            try:
                r=self.session.get(self.cdn or self.url,headers={'Range':f'bytes={offset}-{end}'},timeout=(10,60),stream=True)
                if r.status_code in [401,403,404] and self.cdn:self.cdn=None;continue
                r.raise_for_status();expected=f'bytes {offset}-{end}/{self.size}'
                if r.status_code!=206 or r.headers.get('Content-Range')!=expected:
                    r.close();raise RuntimeError('Server returned an unexpected range')
                if len(r.content)!=amount:raise RuntimeError('Incomplete HTTP range')
                self.cdn=r.url;self.start=offset;self.cache=r.content;self.bytes+=amount;self.requests+=1
                return self.cache[:size]
            except Exception as e:error=e;self.cdn=None;time.sleep(min(8,.5*2**attempt))
        raise RuntimeError(f'Range access failed at {offset}: {type(error).__name__}')
    def prefetch_metadata(self,offset):
        if self.start<=offset and offset+32768<=self.start+len(self.cache):return
        self.read(offset,min(32768,self.size-offset))

def download(job,m,tree):
    split,subject,sequence,clip=job;name=f'clip-{clip:06d}.tar';source='train_aria/'+name
    folder=ROOT/'rgb_clips'/split/sequence;folder.mkdir(parents=True,exist_ok=True);out=folder/name;receipt=out.with_suffix('.receipt.json');tmp=out.with_suffix('.tar.partial')
    if out.exists() and receipt.exists():
        r=json.loads(receipt.read_text())
        if r['completed'] and out.stat().st_size==r['retained_archive_bytes'] and sha(out)==r['retained_archive_sha256']:return r
        raise RuntimeError(f'Existing output failed verification: {out}')
    info=tree[source];size=info['size'];url=f"{m['mirror']}/datasets/{m['official_repo']}/resolve/{m['revision']}/{source}";reader=Ranges(url,size)
    offset=0;counts={k:0 for k in ['rgb','camera','hands','info','shape']};member_digest=hashlib.sha256();start=time.time();pax=None;long_name=None;subject_check=False;model_shape={}
    with tarfile.open(tmp,'w',format=tarfile.PAX_FORMAT) as target:
        while offset+512<=size:
            block=reader.read(offset,512,header=True)
            if block==b'\0'*512:break
            t=tarfile.TarInfo.frombuf(block,'utf-8','surrogateescape');data_offset=offset+512;next_offset=data_offset+((t.size+511)//512)*512
            if t.type in [tarfile.XHDTYPE,tarfile.XGLTYPE]:
                # Only mtime fields are present in current archives; preserve
                # standard path extensions if a future upstream uses them.
                raw=reader.read(data_offset,t.size);pax={}
                for line in raw.decode('utf-8').splitlines():
                    payload=line.partition(' ')[2];key,_,value=payload.partition('=');pax[key]=value
                offset=next_offset;continue
            if t.type==tarfile.GNUTYPE_LONGNAME:long_name=reader.read(data_offset,t.size).rstrip(b'\0').decode();offset=next_offset;continue
            if long_name:t.name=long_name;long_name=None
            if pax and 'path' in pax:t.name=pax['path']
            pax=None
            if '/' in t.name or '\\' in t.name or t.name.startswith('.'):raise RuntimeError('Unexpected archive member path')
            keep=t.name.endswith(SUFFIXES) or t.name=='__hand_shapes.json__'
            if keep:
                payload=reader.read(data_offset,t.size)
                member_digest.update(t.name.encode());member_digest.update(hashlib.sha256(payload).digest())
                if t.name.endswith('.jpg'):
                    category='rgb';assert payload[:2]==b'\xff\xd8' and payload[-2:]==b'\xff\xd9'
                    if counts['rgb']==0:
                        image=Image.open(io.BytesIO(payload));assert image.size==(1408,1408);image.verify()
                else:
                    value=json.loads(payload)
                    if t.name.endswith('.cameras.json'):category='camera';assert '214-1' in value
                    elif t.name.endswith('.hands.json'):category='hands';assert isinstance(value,dict)
                    elif t.name.endswith('.info.json'):
                        category='info';assert value['sequence_id']==sequence and value['participant_id']==subject;assert value['device']=='Aria';subject_check=True
                    else:category='shape';assert 'umetrack' in value;model_shape={k:len(v) if isinstance(v,list) else None for k,v in value['umetrack'].items()}
                counts[category]+=1
                new=tarfile.TarInfo(t.name);new.size=len(payload);new.mode=0o644;new.mtime=0;target.addfile(new,io.BytesIO(payload))
            offset=next_offset
            # Each frame starts with camera metadata; prefetch only that small
            # region when the preceding object payload has been skipped.
            if t.name.endswith('.objects.json') and offset<size:reader.prefetch_metadata(offset)
    assert counts==dict(rgb=150,camera=150,hands=150,info=150,shape=1),(clip,counts)
    assert subject_check
    os.replace(tmp,out)
    r=dict(completed=True,clip=clip,split=split,subject=subject,sequence=sequence,source_repo=m['official_repo'],source_revision=m['revision'],source_path=source,original_archive_bytes=size,original_archive_lfs_sha256=info.get('lfs',{}).get('oid'),original_archive_full_sha256_verified=False,retained_archive_bytes=out.stat().st_size,retained_archive_sha256=sha(out),member_digest_sha256=member_digest.hexdigest(),counts=counts,hand_model_shape=model_shape,http_requests=reader.requests,network_bytes=reader.bytes,seconds=time.time()-start)
    receipt.write_text(json.dumps(r,indent=2));return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=48);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();m=json.loads((ROOT/'subset_manifest.json').read_text());tree={x['path']:x for x in json.loads((ROOT/'provenance/train_aria_tree.json').read_text())}
    # Round-robin sequences so validation subjects and diverse training subjects
    # arrive early; the immutable split itself is unchanged.
    jobs=[(s['split'],s['subject'],s['sequence'],s['clips'][i]) for i in range(max(len(s['clips']) for s in m['sequences'])) for s in m['sequences'] if i<len(s['clips'])]
    if a.limit:jobs=jobs[:a.limit]
    started=time.time();rows=[];errors=[];status=ROOT/('download_pilot_status.json' if a.limit else 'download_status.json')
    status.write_text(json.dumps(dict(stage='starting',completed=0,expected=len(jobs),workers=a.workers,errors=[]),indent=2))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        tasks={pool.submit(download,j,m,tree):j for j in jobs}
        for future in as_completed(tasks):
            job=tasks[future]
            try:rows.append(future.result())
            except Exception as e:errors.append(dict(job=job,error=str(e)[:250]))
            report=dict(stage='downloading',completed=len(rows),expected=len(jobs),errors=errors,seconds=time.time()-started,network_bytes=sum(r['network_bytes'] for r in rows),retained_bytes=sum(r['retained_archive_bytes'] for r in rows));status.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    report.update(stage='complete' if not errors else 'failed',seconds=time.time()-started);status.write_text(json.dumps(report,indent=2))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()

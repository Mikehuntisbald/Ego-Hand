"""All available labelled TRAIN frames; locked development/test remain unchanged."""
import json,hashlib,ssl,http.client,time,concurrent.futures
from pathlib import Path
from acquire_surgical_images_v51 import split
ROOT=Path('/mnt/why/HOT3D/domain_data_v51/surgical_hands')
RUN=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')
HOST='prism.eecs.umich.edu';URL='/natlouis/surgical_hands/surgical_hands_release.tar.gz'
LEAF='030cdc0b37d89bc39354898d9d53ca518b559b587b66f8b61cc297b06d2abef6'

def get(entry,dest):
    for attempt in range(4):
        try:
            c=http.client.HTTPSConnection(HOST,timeout=60,context=ssl._create_unverified_context());c.connect();assert hashlib.sha256(c.sock.getpeercert(binary_form=True)).hexdigest()==LEAF
            start=entry['offset']+512;end=start+entry['size']-1;c.request('GET',URL,headers={'Range':f'bytes={start}-{end}'});r=c.getresponse();assert r.status==206 and r.getheader('Content-Range').startswith(f'bytes {start}-{end}/');data=r.read();c.close();assert len(data)==entry['size'] and data.startswith(b'\x89PNG\r\n\x1a\n')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);return hashlib.sha256(data).hexdigest()
        except Exception:
            if attempt==3:raise
            time.sleep(1+attempt)

def main():
    annotations=json.loads((ROOT/'annotations.json').read_text());index={e['name']:e for e in map(json.loads,(ROOT/'tar_index.jsonl').read_text().splitlines())};chosen=[]
    for clip,data in annotations.items():
        group=clip[:11]
        if split(group)!='train':continue
        by_id={}
        for a in data['annotations']:by_id.setdefault(a['image_id'],[]).append(a)
        for image in data['images']:
            if not image['is_labeled']:continue
            name=f"surgical_hands_release/images/{image['video_dir']}/{image['file_name']}"
            if name not in index:continue
            old=ROOT/'selected_images'/image['video_dir']/image['file_name'];dest=old if old.exists() else RUN/'surgical_images'/image['video_dir']/image['file_name'];entry=index[name]
            row=dict(id=image['id'],dataset='surgical_hands',group=group,clip=clip,split='train',image=str(dest),image_size=[image['frame_width'],image['frame_height']],hands=[dict(instance_id=str(a['track_id']),box_xyxy=a['bbox'],has_keypoint_GT=True) for a in by_id.get(image['id'],[])],GT_3D=False,real_RGB=True)
            if row['hands']:chosen.append((entry,dest,row))
    started=time.time();records=[];hashes={}
    def worker(item):
        entry,dest,row=item
        digest=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest() if dest.exists() and dest.stat().st_size==entry['size'] else get(entry,dest)
        return row,digest
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for f in concurrent.futures.as_completed([pool.submit(worker,v) for v in chosen]):
            row,digest=f.result();records.append(row);hashes[row['id']]=digest
            if len(records)%100==0:
                p=dict(images=len(records),total=len(chosen),seconds=time.time()-started);(RUN/'surgical_expand_progress.json').write_text(json.dumps(p));print(json.dumps(p),flush=True)
    records.sort(key=lambda r:r['id']);(RUN/'surgical_expanded_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));(RUN/'surgical_expanded_hashes.json').write_text(json.dumps(hashes));(RUN/'surgical_expand_done.json').write_text(json.dumps(dict(complete=True,train_images=len(records),source='https://github.com/MichiganCOG/Surgical_Hands_RELEASE',mask_GT=False,GT_3D=False,held_groups_unchanged=True)));print('COMPLETE',len(records),flush=True)

if __name__=='__main__':main()

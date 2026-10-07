"""Diverse real hand interactions, one frame/video, official source-disjoint splits."""
import json,struct,zlib,hashlib,re,time,concurrent.futures
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
from remote_range_zip_v53 import fetch
ROOT=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')
PREFIX='pascal_voc_format/VOCdevkit2007_handobj_100K/VOC2007/'

def extract(entry,dest):
    if dest.exists() and dest.stat().st_size==entry['size']:
        data=dest.read_bytes()
        if zlib.crc32(data)&0xffffffff==entry['crc']:return data
    raw=fetch(entry['offset'],entry['offset']+512+entry['compressed']-1);h=struct.unpack_from('<IHHHHHIIIHH',raw);assert h[0]==0x04034b50;name=raw[30:30+h[-2]].decode();assert name==entry['name'];offset=30+h[-2]+h[-1];payload=raw[offset:offset+entry['compressed']];assert len(payload)==entry['compressed'];data=zlib.decompress(payload,-15) if entry['compression']==8 else payload
    assert len(data)==entry['size'] and zlib.crc32(data)&0xffffffff==entry['crc'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);return data

def group(name):
    m=re.search(r'_v_(.{11})_frame',name);return m.group(1) if m else name.rsplit('_frame',1)[0]

def choose(names,count,role):
    by_group={}
    for name in sorted(names,key=lambda n:hashlib.sha256(('v53-doh:'+n).encode()).hexdigest()):by_group.setdefault(group(name),name)
    bins={}
    for name in by_group.values():bins.setdefault(name.split('_v_',1)[0],[]).append(name)
    result=[]
    while len(result)<count and any(bins.values()):
        for kind in sorted(bins):
            if bins[kind] and len(result)<count:result.append((bins[kind].pop(),role))
    return result

def main():
    index={e['name']:e for e in json.loads((ROOT/'100doh_zip_index.json').read_text())};sets={r:(ROOT/(r+'.txt')).read_text().splitlines() for r in ['train','val','test']};groups={r:set(map(group,v)) for r,v in sets.items()};assert not groups['train']&(groups['val']|groups['test']) and not groups['val']&groups['test']
    selected=choose(sets['train'],6000,'train')+choose(sets['val'],300,'dev')+choose(sets['test'],400,'test');records=[];test=[];hashes={};started=time.time();folder=ROOT/'100doh_files'
    def worker(item):
        name,role=item;assert '/' not in name and chr(92) not in name and name not in ['.','..']; assert Path(name).name==name
        image=folder/'images'/(name+'.jpg');xml=folder/'annotations'/(name+'.xml');im=extract(index[PREFIX+'JPEGImages/'+name+'.jpg'],image);ann=extract(index[PREFIX+'Annotations/'+name+'.xml'],xml);w,h=Image.open(image).size;row=dict(id='100doh_'+name,dataset='100doh',group=group(name),split=role,image=str(image),image_size=[w,h],real_RGB=True,GT_3D=False,annotation_xml=str(xml))
        if role!='test':
            target=ET.fromstring(ann);assert [int(target.findtext('size/width')),int(target.findtext('size/height'))]==[w,h];hands=[]
            for i,obj in enumerate(target.findall('object')):
                if obj.findtext('name')!='hand':continue
                b=obj.find('bndbox');x,y,z,t=[float(b.findtext(k)) for k in ['xmin','ymin','xmax','ymax']];hands.append(dict(instance_id=str(i),box_xyxy=[max(x-1,0),max(y-1,0),min(z,w),min(t,h)],has_keypoint_GT=False,hand_side=obj.findtext('handside'),contact_state=obj.findtext('contactstate'),difficult=int(obj.findtext('difficult') or 0)))
            row['hands']=hands
        return row,hashlib.sha256(im).hexdigest()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for f in concurrent.futures.as_completed([pool.submit(worker,x) for x in selected]):
            row,digest=f.result();(test if row['split']=='test' else records).append(row);hashes[row['id']]=digest
            n=len(records)+len(test)
            if n%100==0:
                data=dict(downloaded=n,total=len(selected),seconds=time.time()-started);(ROOT/'100doh_progress.json').write_text(json.dumps(data));print(json.dumps(data),flush=True)
    records.sort(key=lambda r:r['id']);test.sort(key=lambda r:r['id']);(ROOT/'100doh_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));(ROOT/'100doh_test_rgb_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in test));(ROOT/'100doh_hashes.json').write_text(json.dumps(hashes));(ROOT/'100doh_done.json').write_text(json.dumps(dict(complete=True,training=6000,development=300,test=400,one_frame_per_video=True,groups_disjoint=True,test_boxes_not_parsed=True,CRC_verified=True,human_boxes_only=True,GT_masks=False,source='https://github.com/ddshan/hand_object_detector',license='Author states non-commercial research only'),indent=2));print('COMPLETE',len(records),len(test),flush=True)

if __name__=='__main__':main()

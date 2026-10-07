"""Mixed true masks/boxes, declared domain IDs and split-safe real RGB repeats."""
import os,json,hashlib
from pathlib import Path
import numpy as np
from pycocotools import mask as cm

RUN=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')
PAIRED=Path('/mnt/why/HOT3D/experiments/full_model_gloves_multihand_v51_20261007/paired_protocol')
DOMAIN=dict(egohands=0,hot3d_preservation=1,surgical_hands=2,cppe5=3,**{'100doh':4})
REPEATS=dict(egohands=1,hot3d_preservation=2,surgical_hands=2,cppe5=2,**{'100doh':1})

def export(rows,dest,repeats=True):
    stats={}
    for split,name in [('train','train'),('dev','valid'),('test','test')]:
        folder=dest/name;folder.mkdir(parents=True,exist_ok=True);data=dict(info=dict(description='Human mask supervision only where actually available'),images=[],annotations=[],categories=[dict(id=1,name='hand',supercategory='none')]);counts={}
        for row in (r for r in rows if r['split']==split):
            count=REPEATS[row['dataset']] if repeats and split=='train' else 1
            for repeat in range(count):
                image_id=DOMAIN[row['dataset']]*1_000_000+len(data['images'])+1;src=Path(row['image']);filename=row['id']+f'_r{repeat}'+src.suffix;dst=folder/filename
                if not dst.exists():os.symlink(src,dst)
                w,h=row['image_size'];data['images'].append(dict(id=image_id,file_name=filename,width=w,height=h,dataset=row['dataset'],group=row['group'],record_id=row['id'],partial_glove_labels=row['dataset']=='cppe5'));masks=np.load(row['mask_gt'])['masks'] if row.get('mask_gt') else None
                for hand in row['hands']:
                    x,y,z,t=hand['box_xyxy'];x=max(0,min(x,w));z=max(0,min(z,w));y=max(0,min(y,h));t=max(0,min(t,h))
                    if z<=x or t<=y:continue
                    mask=[];area=(z-x)*(t-y)
                    if masks is not None:
                        mask=cm.encode(np.asfortranarray(masks[hand['mask_index']].astype(np.uint8)));area=float(cm.area(mask));mask['counts']=mask['counts'].decode('ascii')
                    data['annotations'].append(dict(id=len(data['annotations'])+1,image_id=image_id,category_id=1,bbox=[x,y,z-x,t-y],area=area,iscrowd=0,segmentation=mask,mask_GT_available=masks is not None))
                counts[row['dataset']]=counts.get(row['dataset'],0)+1
        (folder/'_annotations.coco.json').write_text(json.dumps(data));stats[split]=dict(images=len(data['images']),instances=len(data['annotations']),domain_images=counts)
    return stats

def main():
    RUN.mkdir(exist_ok=True);rows=list(map(json.loads,(PAIRED/'domain_records.jsonl').read_text().splitlines()))
    more=RUN/'surgical_expanded_records.jsonl'
    if more.exists():
        extra=list(map(json.loads,more.read_text().splitlines()));ids={r['id'] for r in rows};rows.extend(r for r in extra if r['id'] not in ids)
    extra=RUN/'100doh_records.jsonl'
    if extra.exists():rows.extend(map(json.loads,extra.read_text().splitlines()))
    pilot=[]
    for dataset in DOMAIN:
        for split,n in [('train',24),('dev',8)]:
            candidates=sorted((r for r in rows if r['dataset']==dataset and r['split']==split),key=lambda r:hashlib.sha256(('pilot53:'+r['id']).encode()).hexdigest());pilot.extend(candidates[:n])
    small=export(pilot,RUN/'pilot_data',False);full=export(rows,RUN/'data');(RUN/'domain_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    protocol=dict(full=full,pilot=small,repeats=REPEATS,mask_loss='only nonempty true human masks; box-only mask matching/loss excluded',CPPE_unmatched_queries='unknown, classification negative loss omitted',real_RGB=True,synthetic_occlusion=False,train_test_disjoint='source-video/interaction pair; CPPE original official test',test_from_v51_reused_for_existing_domains=True,default_changed=False)
    (RUN/'dataset_protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol),flush=True)

if __name__=='__main__':main()

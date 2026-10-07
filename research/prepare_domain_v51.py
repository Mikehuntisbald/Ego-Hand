"""Canonical real-image instance records and group-held-out detector data."""
import json,hashlib,os
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import cv2

ROOT=Path('/mnt/why/HOT3D/domain_data_v51')
RUN=Path('/mnt/why/HOT3D/experiments/full_model_gloves_multihand_v51_20261007')

def group_split(group):
    value=int(hashlib.sha256(('v51-20261007:'+group).encode()).hexdigest()[:8],16)%10
    return 'test' if value<2 else 'dev' if value==2 else 'train'

def egohands():
    folder=ROOT/'egohands';path=next(folder.rglob('metadata.mat'))
    metadata=loadmat(path,simplify_cells=True)['video']
    if isinstance(metadata,dict):metadata=[metadata]
    frames_lookup={p.parent.name: p.parent for p in folder.rglob('frame_*.jpg')}
    output=[];gt_folder=RUN/'instance_gt';gt_folder.mkdir(exist_ok=True)
    for video in metadata:
        name=str(video['video_id']);split=group_split('egohands:'+name)
        records=video['labelled_frames'];records=[records] if isinstance(records,dict) else records
        directory=frames_lookup[name]
        for f in records:
            index=int(f['frame_num']);image=directory/f'frame_{index:04d}.jpg'
            assert image.exists(),image
            rgb=cv2.imread(str(image));h,w=rgb.shape[:2];hands=[];masks=[]
            for key in ['myleft','myright','yourleft','yourright']:
                polygon=np.asarray(f[key])
                if polygon.size<6:continue
                polygon=polygon.reshape(-1,2).astype(np.float32)-1 # original MATLAB coordinates
                polygon[:,0]=polygon[:,0].clip(0,w-1);polygon[:,1]=polygon[:,1].clip(0,h-1)
                mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],1)
                masks.append(mask)
                hands.append(dict(instance_id=key,person_id='viewer' if key.startswith('my') else 'partner',right=key.endswith('right'),box_xyxy=[float(polygon[:,0].min()),float(polygon[:,1].min()),float(polygon[:,0].max()+1),float(polygon[:,1].max()+1)],mask_index=len(masks)-1,has_keypoint_GT=False))
            if not hands:continue
            uid=f'egohands_{name}_{index:04d}';maskpath=gt_folder/f'{uid}.npz';np.savez_compressed(maskpath,masks=np.stack(masks))
            output.append(dict(id=uid,dataset='egohands',group=name,split=split,image=str(image),image_size=[w,h],hands=hands,mask_gt=str(maskpath),GT_3D=False,real_RGB=True,dense_video=False,frame_index=index))
    return output

def cppe5():
    folder=ROOT/'cppe5';out=[]
    for split_file in ['train','test']:
        data=json.loads((folder/f'annotations/{split_file}.json').read_text());images={x['id']:x for x in data['images']};ann={}
        for a in data['annotations']:
            if a['category_id']==3:ann.setdefault(a['image_id'],[]).append(a)
        for key,hands in ann.items():
            row=images[key];name=row['file_name'];found=list((folder/'images').rglob(name));assert len(found)==1,(name,len(found))
            group=f'cppe5:{key}';split='test' if split_file=='test' else group_split(group)
            out.append(dict(id=f'cppe5_{key}',dataset='cppe5',group=group,split=split,image=str(found[0]),image_size=[row['width'],row['height']],hands=[dict(instance_id=str(a['id']),box_xyxy=[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]],has_keypoint_GT=False) for a in hands],GT_3D=False,real_RGB=True,glove=True,dense_video=False))
    return out

def hot3d_preservation():
    import torch
    source=ROOT.parent/'experiments/online_rgb_3d_v47'
    data=torch.load(source/'inputs.pt',weights_only=False,mmap=True);records=json.loads((source/'records.json').read_text());annotation_cache={};rows=[];seen=set()
    for role,count,split in [('train',256,'train'),('dev_select',96,'dev')]:
        candidates=[i for i,r in enumerate(data['roles']) if r==role]
        chosen=np.linspace(0,len(candidates)-1,min(count,len(candidates))).round().astype(int)
        for index in chosen:
            wi=candidates[index];fid=int(data['feature_ids'][wi,8]);r=records[fid-1];image=Path(r['image'])
            if str(image) in seen:continue
            seen.add(str(image));parts=image.relative_to(ROOT.parent/'export/images').parts
            path=ROOT.parent/'export/annotations'/parts[0]/parts[1]/(parts[2]+'.jsonl')
            if path not in annotation_cache:annotation_cache[path]=[json.loads(x) for x in path.read_text().splitlines()]
            frame=annotation_cache[path][int(image.stem)];hands=[]
            for hand in frame['hands']:
                box=hand['box_amodal_xyxy']
                if box is None or (hand.get('modeled_hand_visible_fraction') or 0)<=0:continue
                hands.append(dict(instance_id=hand['side'],box_xyxy=box,has_keypoint_GT=True))
            rows.append(dict(id=f'hot3d_{parts[1]}_{parts[2]}_{image.stem}',dataset='hot3d_preservation',group=parts[1],split=split,image=str(image),image_size=[1408,1408],hands=hands,GT_3D=True,real_RGB=True,labels='clipped projected amodal hand boxes; visibility proxy, not pixel-occlusion ground truth'))
    return rows

def write_detection(rows):
    from collections import Counter
    folder=RUN/'detector_data';folder.mkdir(exist_ok=True);counts=Counter()
    for split in ['train','dev','test']:
        (folder/f'{split}/images').mkdir(parents=True,exist_ok=True);(folder/f'{split}/labels').mkdir(parents=True,exist_ok=True)
    for row in rows:
        if not row['hands']:continue
        split=row['split'];src=Path(row['image']);dst=folder/split/'images'/f"{row['id']}{src.suffix}"
        if not dst.exists():os.symlink(src,dst)
        w,h=row['image_size'];lines=[]
        for hand in row['hands']:
            x,y,z,t=hand['box_xyxy'];x=np.clip(x,0,w);z=np.clip(z,0,w);y=np.clip(y,0,h);t=np.clip(t,0,h)
            if z>x and t>y:lines.append(f'0 {(x+z)/(2*w):.8f} {(y+t)/(2*h):.8f} {(z-x)/w:.8f} {(t-y)/h:.8f}')
        (folder/split/'labels'/f"{row['id']}.txt").write_text('\n'.join(lines));counts[(row['dataset'],split)]+=1
    (folder/'data.yaml').write_text(f'path: {folder}\ntrain: train/images\nval: dev/images\ntest: test/images\nnames:\n  0: hand\n')
    return {f'{a}:{b}':n for (a,b),n in counts.items()}

def main():
    RUN.mkdir(exist_ok=True)
    rows=egohands()+cppe5()+hot3d_preservation()
    (RUN/'domain_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    stats=write_detection(rows)
    (RUN/'dataset_protocol.json').write_text(json.dumps(dict(complete=True,counts=stats,split='EgoHands whole video groups held out; CPPE official test preserved plus train image development split',GT_masks_loss_and_evaluation_only=True,dense_timestamp_not_fabricated=True,no_GT_3D_from_these_datasets=True,manicure_coverage_verified=False,natural_only=True),indent=2))
    print(json.dumps(stats),flush=True)

if __name__=='__main__':main()

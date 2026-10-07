"""Lock synchronized EgoHands partners together and balance native rehearsal."""
import json,hashlib,os
from pathlib import Path
import numpy as np,torch
from scipy.io import loadmat
import prepare_domain_v51 as original
from cache_instance_conditions_v51 import RUN

PAIRED=RUN/'paired_protocol'

def more_native(existing):
    source=Path('/mnt/why/HOT3D/experiments/online_rgb_3d_v47');data=torch.load(source/'inputs.pt',weights_only=False,mmap=True);records=json.loads((source/'records.json').read_text())
    candidates=[i for i,r in enumerate(data['roles']) if r=='train'];selected=np.linspace(0,len(candidates)-1,min(1600,len(candidates))).round().astype(int)
    seen={r['image'] for r in existing};annotations={};result=[]
    root=Path('/mnt/why/HOT3D/export')
    for index in selected:
        fid=int(data['feature_ids'][candidates[index],8]);row=records[fid-1];image=Path(row['image'])
        if str(image) in seen:continue
        seen.add(str(image));parts=image.relative_to(root/'images').parts;path=root/'annotations'/parts[0]/parts[1]/(parts[2]+'.jsonl')
        if path not in annotations:annotations[path]=[json.loads(x) for x in path.read_text().splitlines()]
        sourceframe=annotations[path][int(image.stem)];hands=[]
        for hand in sourceframe['hands']:
            if hand['box_amodal_xyxy'] is not None and (hand.get('modeled_hand_visible_fraction') or 0)>0:hands.append(dict(instance_id=hand['side'],box_xyxy=hand['box_amodal_xyxy'],has_keypoint_GT=True))
        result.append(dict(id=f'hot3d_{parts[1]}_{parts[2]}_{image.stem}',dataset='hot3d_preservation',group=parts[1],split='train',image=str(image),image_size=[1408,1408],hands=hands,GT_3D=True,real_RGB=True))
    return result

def main():
    PAIRED.mkdir(exist_ok=True)
    if (PAIRED/'dataset_protocol.json').exists():return
    rows=[json.loads(x) for x in (RUN/'domain_records.jsonl').read_text().splitlines()]
    raw=Path('/mnt/why/HOT3D/domain_data_v51/egohands');videos=loadmat(next(raw.rglob('metadata.mat')),simplify_cells=True)['video']
    pairs={v['video_id']:'__'.join(sorted([str(v['video_id']),str(v['partner_video_id'])])) for v in videos}
    for row in rows:
        if row['dataset']=='egohands':
            video=row['group'];pair=pairs[video];row.update(clip=video,group=pair,split=original.group_split('egohands_pair:'+pair))
    rows+=more_native(rows)
    surgery=Path('/mnt/why/HOT3D/domain_data_v51/surgical_hands/selected_records.jsonl')
    if surgery.exists():
        for line in surgery.read_text().splitlines():
            source=json.loads(line);source=dict(source,hands=[dict(instance_id=str(a['track_id']),box_xyxy=a['bbox'],has_keypoint_GT=True) for a in source['hands']]);rows.append(source)
    (PAIRED/'domain_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));original.RUN=PAIRED;counts=original.write_detection(rows)
    trainpaths=[]
    for row in rows:
        if row['split']!='train' or not row['hands']:continue
        path=PAIRED/'detector_data/train/images'/f"{row['id']}{Path(row['image']).suffix}"
        # Exact real images are resampled, without manufacturing occlusions.
        trainpaths.extend([str(path)]*(3 if row['dataset']=='hot3d_preservation' else 1))
    (PAIRED/'detector_data/train_balanced.txt').write_text('\n'.join(trainpaths))
    (PAIRED/'detector_data/data.yaml').write_text(f'path: {PAIRED / "detector_data"}\ntrain: train_balanced.txt\nval: dev/images\ntest: test/images\nnames:\n  0: hand\n')
    split_groups={role:sorted({r['group'] for r in rows if r['dataset']=='egohands' and r['split']==role}) for role in ['train','dev','test']}
    assert all(not set(split_groups[a])&set(split_groups[b]) for a,b in [('train','dev'),('train','test'),('dev','test')])
    (PAIRED/'dataset_protocol.json').write_text(json.dumps(dict(complete=True,counts=counts,EgoHands_partner_pair_disjoint=True,pair_groups=split_groups,native_rehearsal_repeat=3,weighted_training_images=len(trainpaths),reset_from_pre_external_adaptation_checkpoints=True,test_not_used_for_supervision=True,default_changed=False),indent=2));print(json.dumps(counts),flush=True)

if __name__=='__main__':main()

"""Exact human EgoHands masks, synchronized-pair split; no generated mask targets."""
import json, hashlib, os
from pathlib import Path
import numpy as np
from pycocotools import mask as coco_mask

RUN = Path('/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007')
SOURCE = Path('/mnt/why/HOT3D/experiments/full_model_gloves_multihand_v51_20261007/paired_protocol/domain_records.jsonl')

def export(rows, folder):
    stats = {}
    for split, dirname in [('train', 'train'), ('dev', 'valid'), ('test', 'test')]:
        dest = folder / dirname; dest.mkdir(parents=True, exist_ok=True)
        data = dict(info=dict(description='Human EgoHands polygons rasterized with MATLAB origin corrected'), images=[], annotations=[], categories=[dict(id=1, name='hand', supercategory='none')])
        selected = [r for r in rows if r['split'] == split]
        for i, row in enumerate(selected, 1):
            name = row['id'] + '.jpg'; path = dest / name
            if not path.exists(): os.symlink(row['image'], path)
            w, h = row['image_size']
            data['images'].append(dict(id=i, file_name=name, width=w, height=h, group=row['group'], record_id=row['id']))
            masks = np.load(row['mask_gt'])['masks']
            for hand in row['hands']:
                m = masks[hand['mask_index']].astype(np.uint8)
                rle = coco_mask.encode(np.asfortranarray(m)); area = int(coco_mask.area(rle)); box = coco_mask.toBbox(rle).tolist()
                assert area > 0 and (coco_mask.decode(rle) == m).all()
                rle['counts'] = rle['counts'].decode('ascii')
                data['annotations'].append(dict(id=len(data['annotations'])+1, image_id=i, category_id=1, segmentation=rle, bbox=box, area=area, iscrowd=0, source_instance=hand['instance_id']))
        (dest / '_annotations.coco.json').write_text(json.dumps(data))
        stats[split] = dict(images=len(selected), instances=len(data['annotations']), groups=sorted(set(r['group'] for r in selected)))
    return stats

def main():
    rows = [json.loads(s) for s in SOURCE.read_text().splitlines()]
    rows = [r for r in rows if r['dataset'] == 'egohands']
    groups = {s: set(r['group'] for r in rows if r['split'] == s) for s in ['train', 'dev', 'test']}
    assert not groups['train'] & (groups['dev'] | groups['test']) and not groups['dev'] & groups['test']
    full = export(rows, RUN / 'data')
    # Pilot remains within the declared train/dev groups, and never reads test masks.
    pilot = []
    for split, count in [('train', 96), ('dev', 32)]:
        selected = sorted((r for r in rows if r['split'] == split), key=lambda r: hashlib.sha256(('pilot52:'+r['id']).encode()).hexdigest())[:count]
        pilot.extend(selected)
    small = export(pilot, RUN / 'pilot_data')
    protocol = dict(full=full, pilot=small, mask_targets='human polygons only; exact RLE roundtrip checked', no_synthetic_occlusion=True, no_SAM_targets=True, glove_mask_GT_available=False, train_dev_test_pair_disjoint=True, test_reused_from_v51_diagnostic=True)
    (RUN / 'dataset_protocol.json').write_text(json.dumps(protocol, indent=2)); print(json.dumps(protocol), flush=True)

if __name__ == '__main__': main()
